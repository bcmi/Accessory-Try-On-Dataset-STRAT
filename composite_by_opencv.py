import torch
import numpy as np
import cv2
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def warp_vertex_by_matrix(vertex, matrix, im_w, im_h):
    if isinstance(vertex, torch.Tensor):
        vertex = vertex.squeeze()
        vertex = vertex.numpy().reshape((-1,2))
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.squeeze()
        matrix = matrix.numpy().reshape((3,3))
    ones = np.ones(shape=(vertex.shape[0], 1), dtype=vertex.dtype)
    init_vertex = np.concatenate([vertex, ones], axis=-1) # [N,3]
    final_vertex = init_vertex.dot(matrix.transpose(1,0))
    w = final_vertex[:,2:] + 1e-8
    final_vertex = final_vertex[:,:2] / w# [N,2]
    final_vertex[:,0] = np.clip(final_vertex[:,0], a_min=0, a_max=im_w-1)
    final_vertex[:,1] = np.clip(final_vertex[:,1], a_min=0, a_max=im_h-1)
    return final_vertex


def composite_image_by_matrix(bg, fg, matrix):
    if matrix.shape != (3,3):
        matrix = matrix.reshape((3,3))
    h, w = fg.shape[:2]
    if matrix[-1,-1] != 1:
        matrix /= (matrix[-1,-1] + 1e-8)
    warp = cv2.warpPerspective(fg, matrix, (w, h))
    mask = warp[:, :, 3:].astype(np.float32) / 255
    warp = warp[:, :, :3].astype(np.float32)
    bg = bg.astype(np.float32)
    composite = mask * warp + (1 - mask) * bg
    composite = composite.astype(np.uint8)
    return composite

# derive homography matrix from pairwise vertexes
# H, _ = cv2.findHomography(init_vertex, final_vertex)
# composite = composite_image_by_matrix(src_bg, src_fg, H)