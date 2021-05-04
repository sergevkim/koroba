# adapted from https://github.com/lilanxiao/Rotated_IoU

import torch
import numpy as np
from .ops.cuda_ext import sort_v

EPSILON = 1e-8


def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """find intersection points of rectangles
    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, N, 4, 2
    Returns:
        intersectons (torch.Tensor): B, N, 4, 4, 2
        mask (torch.Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3) # B, N, 4, 4: Batch, Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
    mask = mask_t * mask_u
    t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a                  # (B, N, 1, 2)
    am = corners1 - a           # (B, N, 4, 2)
    ad = d - a                  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)   # (B, N, 4)
    cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)   # (B, N, 4)
    return cond1*cond2

def box_in_box_th(corners1: torch.Tensor, corners2:torch.Tensor):
    """check if corners of two boxes lie in each other
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: torch.Tensor, corners2: torch.Tensor,
                   c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor):
    """find vertices of intersection area
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, 4)
        inters (torch.Tensor): (B, N, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, 4, 4)

    Returns:
        vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2)  # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2)  # Bool (B, N, 4+4+16)
    return vertices, mask


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """[summary]
    Args:
        vertices (torch.Tensor): float (B, N, 24, 2)
        mask (torch.Tensor): bool (B, N, 24)
    Returns:
        sorted_index: bool (B, N, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()  # (B, N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return sort_v(vertices_normalized, mask, num_valid).long()


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """calculate area of intersection
    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)

    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1: torch.Tensor, corners2: torch.Tensor):
    """calculate intersection area of 2d rectangles
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners_th(box: torch.Tensor) -> torch.Tensor:
    """convert box coordinate to corners
    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha
    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5]  # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device)  # (1,1,4)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def calculate_iou(
        box1: torch.Tensor,
        box2: torch.Tensor,
    ):
    """calculate iou
    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)

    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, polygon = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


def calculate_3d_iou(
        box3d1: torch.Tensor,
        box3d2: torch.Tensor,
        verbose: bool = False,
    ):
    """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis
    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.)
    iou_2d, corners1, corners2, u = calculate_iou(box1, box2)  # (B, N)
    intersection_3d = iou_2d * u * z_overlap
    v1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    v2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    u3d = v1 + v2 - intersection_3d
    if verbose:
        z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.)
        return intersection_3d / u3d, corners1, corners2, z_range, u3d
    else:
        return intersection_3d / u3d


def generate_table():
    """generate candidates of hull polygon edges and the the other 6 points
    Returns:
        lines: (24, 2)
        points: (24, 6)
    """
    skip = [[0, 2], [1, 3], [5, 7], [4, 6]]  # impossible hull edge
    line = []
    points = []

    def all_except_two(o1, o2):
        a = []
        for i in range(8):
            if i != o1 and i != o2:
                a.append(i)
        return a

    for i in range(8):
        for j in range(i + 1, 8):
            if [i, j] not in skip:
                line.append([i, j])
                points.append(all_except_two(i, j))
    return line, points


LINES, POINTS = generate_table()
LINES = np.array(LINES).astype(np.int)
POINTS = np.array(POINTS).astype(np.int)


def gather_lines_points(corners: torch.Tensor):
    """get hull edge candidates and the rest points using the index
    Args:
        corners (torch.Tensor): (..., 8, 2)

    Return:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)
        idx_lines (torch.Tensor): Long (..., 24, 2, 2)
        idx_points (torch.Tensor): Long (..., 24, 6, 2)
    """
    dim = corners.dim()
    idx_lines = torch.LongTensor(LINES).to(corners.device).unsqueeze(-1)  # (24, 2, 1)
    idx_points = torch.LongTensor(POINTS).to(corners.device).unsqueeze(-1)  # (24, 6, 1)
    idx_lines = idx_lines.repeat(1, 1, 2)  # (24, 2, 2)
    idx_points = idx_points.repeat(1, 1, 2)  # (24, 6, 2)
    if dim > 2:
        for _ in range(dim - 2):
            idx_lines = torch.unsqueeze(idx_lines, 0)
            idx_points = torch.unsqueeze(idx_points, 0)
        idx_points = idx_points.repeat(*corners.size()[:-2], 1, 1, 1)  # (..., 24, 2, 2)
        idx_lines = idx_lines.repeat(*corners.size()[:-2], 1, 1, 1)  # (..., 24, 6, 2)
    corners_ext = corners.unsqueeze(-3).repeat(*([1] * (dim - 2)), 24, 1, 1)  # (..., 24, 8, 2)
    lines = torch.gather(corners_ext, dim=-2, index=idx_lines)  # (..., 24, 2, 2)
    points = torch.gather(corners_ext, dim=-2, index=idx_points)  # (..., 24, 6, 2)

    return lines, points, idx_lines, idx_points


def point_line_distance_range(lines: torch.Tensor, points: torch.Tensor):
    """calculate the maximal distance between the points in the direction perpendicular to the line
    methode: point-line-distance
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)

    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]  # (..., 24, 1)
    y1 = lines[..., 0:1, 1]  # (..., 24, 1)
    x2 = lines[..., 1:2, 0]  # (..., 24, 1)
    y2 = lines[..., 1:2, 1]  # (..., 24, 1)
    x = points[..., 0]  # (..., 24, 6)
    y = points[..., 1]  # (..., 24, 6)
    den = (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
    num = torch.sqrt((y2 - y1).square() + (x2 - x1).square()) + 1e-8
    d = den / num  # (..., 24, 6)
    d_max = d.max(dim=-1)[0]  # (..., 24)
    d_min = d.min(dim=-1)[0]  # (..., 24)
    d1 = d_max - d_min  # suppose points on different side
    d2 = torch.max(d.abs(), dim=-1)[0]  # or, all points are on the same side
    # NOTE: if x1 = x2 and y1 = y2, this will return 0
    return torch.max(d1, d2)


def point_line_projection_range(lines: torch.Tensor, points: torch.Tensor):
    """calculate the maximal distance between the points in the direction parallel to the line
    methode: point-line projection
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)

    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]  # (..., 24, 1)
    y1 = lines[..., 0:1, 1]  # (..., 24, 1)
    x2 = lines[..., 1:2, 0]  # (..., 24, 1)
    y2 = lines[..., 1:2, 1]  # (..., 24, 1)
    k = (y2 - y1) / (x2 - x1 + 1e-8)  # (..., 24, 1)
    vec = torch.cat([torch.ones_like(k, dtype=k.dtype, device=k.device), k], dim=-1)  # (..., 24, 2)
    vec = vec.unsqueeze(-2)  # (..., 24, 1, 2)
    points_ext = torch.cat([lines, points], dim=-2)  # (..., 24, 8), consider all 8 points
    den = torch.sum(points_ext * vec, dim=-1)  # (..., 24, 8)
    proj = den / torch.norm(vec, dim=-1, keepdim=False)  # (..., 24, 8)
    proj_max = proj.max(dim=-1)[0]  # (..., 24)
    proj_min = proj.min(dim=-1)[0]  # (..., 24)
    return proj_max - proj_min


def smallest_bounding_box(corners: torch.Tensor, verbose=False):
    """return width and length of the smallest bouding box which encloses two boxes.
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        verbose (bool, optional): If True, return area and index. Defaults to False.
    Returns:
        (torch.Tensor): width (..., 24)
        (torch.Tensor): height (..., 24)
        (torch.Tensor): area (..., )
        (torch.Tensor): index of candiatae (..., )
    """
    lines, points, _, _ = gather_lines_points(corners)
    proj = point_line_projection_range(lines, points)   # (..., 24)
    dist = point_line_distance_range(lines, points)     # (..., 24)
    area = proj * dist
    # remove area with 0 when the two points of the line have the same coordinates
    zero_mask = (area == 0).type(corners.dtype)
    fake = torch.ones_like(zero_mask, dtype=corners.dtype, device=corners.device)* 1e8 * zero_mask
    area += fake        # add large value to zero_mask
    area_min, idx = torch.min(area, dim=-1, keepdim=True)     # (..., 1)
    w = torch.gather(proj, dim=-1, index=idx)
    h = torch.gather(dist, dim=-1, index=idx)          # (..., 1)
    w = w.squeeze(-1).float()
    h = h.squeeze(-1).float()
    area_min = area_min.squeeze(-1).float()
    if verbose:
        return w, h, area_min, idx.squeeze(-1)
    else:
        return w, h


def enclosing_box(
        corners1: torch.Tensor,
        corners2: torch.Tensor,
        enclosing_type: str = 'smallest',
    ):
    assert enclosing_type == 'smallest'
    return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))


def calculate_3d_giou(
        box3d1: torch.Tensor,
        box3d2: torch.Tensor,
        enclosing_type: str = 'smallest',
    ):
    """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis
    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
    Returns:
        (torch.Tensor): (B, N) 3d GIoU loss
        (torch.Tensor): (B, N) 3d IoU
    """
    iou3d, corners1, corners2, z_range, u3d = calculate_3d_iou(
        box3d1,
        box3d2,
        verbose=True,
    )
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    v_c = z_range * w * h
    giou_loss = 1.0 - iou3d + (v_c - u3d)/v_c

    return giou_loss, iou3d


def calculate_2d_giou(
        box1: torch.Tensor,
        box2: torch.Tensor,
        enclosing_type: str = 'smallest',
    ):
    iou, corners1, corners2, u = calculate_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    area_c = w * h
    giou_loss = 1.0 - iou + ( area_c - u )/area_c
    return giou_loss, iou