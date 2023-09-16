import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.encoders import build_encoder
from einops import rearrange, repeat
import functools
import numpy as np

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def derive_matrix_from_vertexes(init_vertex, final_vertex):
    '''
    init_vertex, final_vertex: tensor of shape (B,2,H,W) or (B,N,2)
    '''
    init_vertex  = init_vertex.flatten(2)
    final_vertex = final_vertex.flatten(2)
    if init_vertex.shape[1] != 2:
        init_vertex = init_vertex.permute(0,2,1)
    if final_vertex.shape[1] != 2:
        final_vertex = final_vertex.permute(0,2,1)

    X, Y = init_vertex[:,0], init_vertex[:,1]
    U, V = final_vertex[:,0], final_vertex[:,1]
    O, I = torch.zeros_like(X), torch.ones_like(X)
    A = torch.cat([torch.stack([X, Y, I, O, O, O, -U*X, -U*Y, O], dim=-1),
                   torch.stack([O, O, O, X, Y, I, -V*X, -V*Y, O], dim=-1)], dim=1)
    B = torch.cat([U,V],dim=1)
    X = torch.linalg.lstsq(A,B,driver='gels').solution
    # X = torch.linalg.lstsq(A, B).solution
    I = torch.ones(size=(X.shape[0],1), dtype=X.dtype, device=X.device)
    X = torch.cat([X[:,:8], I], dim=1).reshape(X.shape[0], 3, -1)
    return X


def warp_vertex_by_matrix(vertex, matrix):
    '''
    :param vertex: tensor(b,n,2)
    :param matrix: tensor(b,3,3)
    :return: final_vertex: tensor(b,n,2)
    '''
    if matrix.dim() == 2:
        matrix = matrix.reshape(matrix.shape[0], 3, 3)
    ones_t = torch.ones(size=(vertex.shape[0], vertex.shape[1], 1),
                        dtype=vertex.dtype, device=vertex.device)
    homo_vertex = torch.cat([vertex, ones_t], dim=-1) # b,n,3
    # warp_vertex = torch.bmm(homo_vertex, matrix.permute(0,2,1))
    warp_vertex = torch.bmm(matrix, homo_vertex.permute(0,2,1)).permute(0,2,1)
    warp_vertex = warp_vertex / (warp_vertex[:,:,-1].unsqueeze(-1) + 1e-8)
    return warp_vertex[:,:,:2]

# warp the image
def composite_batch_images(bg, fg, matrix):
    '''
    :param bg: tensor(b, 3, h, w)
    :param fg: tensor(b, 4, h, w)
    :param matrix: tensor(b, 3, 3)
    :return:
    '''
    assert bg.dim() == 4 and bg.shape[1] == 3
    assert fg.dim() == 4 and fg.shape[1] == 4
    device, dtype = bg.device, bg.dtype
    B, C, H, W = bg.shape
    # warp the canonical coordinates# warp the canonical coordinates
    shifts_x = torch.arange(0, W, dtype=dtype, device=device)
    shifts_y = torch.arange(0, H, dtype=dtype, device=device)
    shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y)
    coords = torch.stack((shift_x, shift_y), dim=-1) # w,h,2
    ones_t = torch.ones(size=(W, H, 1), dtype=dtype, device=device)
    homo_coords = torch.cat([coords, ones_t], dim=-1).unsqueeze(0) # 1,w,h,3
    homo_coords = repeat(homo_coords, '1 w h c -> b w h c', b=B)
    homo_coords = rearrange(homo_coords, 'b w h c -> b (h w) c')
    if matrix.dim() == 2:
        matrix = matrix.reshape(matrix.shape[0], 3, 3)
    try:
        matrix = torch.linalg.inv(matrix)
    except:
        matrix = torch.linalg.pinv(matrix)
    warp_coords = torch.bmm(matrix, homo_coords.permute(0,2,1)).permute(0,2,1)
    warp_x, warp_y, warp_z = torch.chunk(warp_coords, 3, dim=-1)
    warp_x = torch.clip(warp_x / (warp_z + 1e-8), min=0, max=W)
    norm_x = (warp_x - W / 2) / (W/2) # on a range of [-1,1]
    warp_y = torch.clip(warp_y / (warp_z + 1e-8), min=0, max=H)
    norm_y = (warp_y - H / 2) / (H/2)

    grid_coords = torch.stack([norm_x.squeeze(-1), norm_y.squeeze(-1)], dim=-1)
    grid_coords = rearrange(grid_coords, 'b (h w) c -> b h w c', w=W)
    warp_fg   = F.grid_sample(fg, grid_coords, mode='bilinear', align_corners=True)
    sample_mask = warp_fg[:,3:]
    sample_fg = warp_fg[:,:3]
    composite   = sample_mask * sample_fg + (1 - sample_mask) * bg
    return composite, warp_fg


def invert_image_transformation(im_tensor, value_range=(0,255), permute=True):
    '''
    input im_tensor: b,c,h,w
    return: b,c,h,w
    '''
    if im_tensor.dim() == 3:
        im_tensor = im_tensor.unsqueeze(1)
    mask = torch.tensor([])
    if im_tensor.shape[1] == 4:
        mask = im_tensor[:,-1].unsqueeze(1)
        im_tensor = im_tensor[:,:3]
    dtype, device = im_tensor.dtype, im_tensor.device
    mean_t = torch.tensor(IMAGE_NET_MEAN, dtype=dtype, device=device).reshape(1,-1,1,1)
    std_t  = torch.tensor(IMAGE_NET_STD,  dtype=dtype, device=device).reshape(1,-1,1,1)
    im = (im_tensor * std_t + mean_t)
    if mask.shape[0] > 0:
        im = torch.cat([im, mask], dim=1)
    if value_range != (0,1):
        val_min, val_max = min(value_range), max(value_range)
        im = (val_max - val_min) * im + val_min
    if permute:
        im = im.permute(0,2,3,1)
    return im

