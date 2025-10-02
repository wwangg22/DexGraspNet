"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: energy functions
"""

import torch

def _normalize(v, eps=1e-9):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def rot6d_to_matrix(x6: torch.Tensor, treat_as_columns: bool = True) -> torch.Tensor:
    """
    Convert 6D rotation representation to a 3x3 rotation matrix.
    x6: (B, 6). Split into two 3D vectors a1, a2.
    If treat_as_columns=True, returns R with those as (orthonormalized) first two COLUMNS.
    If treat_as_columns=False, returns R with those as (orthonormalized) first two ROWS.
    """
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]
    b1 = _normalize(a1)
    # remove component of a2 along b1, then normalize
    proj = (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = _normalize(a2 - proj)
    b3 = torch.cross(b1, b2, dim=1)

    if treat_as_columns:
        # R[:, :, 0]=b1, R[:, :, 1]=b2, R[:, :, 2]=b3
        R = torch.stack([b1, b2, b3], dim=2)
    else:
        # put them as first two ROWS; third row via cross of rows
        R = torch.stack([b1, b2, b3], dim=1)
    return R

def palm_normal_from_hand_pose(hand_pose: torch.Tensor) -> torch.Tensor:
    """
    hand_pose: (B, 3 + 6 + n_dofs), where [:, 3:9] is the 6D rotation block
               created by: rotation.transpose(1, 2)[:, :2].reshape(-1, 6)

    Returns: (B,3) unit world-space palm normal.
    """
    device = hand_pose.device
    x6 = hand_pose[:, 3:9]                    # (B,6)

    # Most pipelines (including your earlier usage) expect the 6D to produce
    # the first TWO COLUMNS of R. That’s what we do here:
    R_final = rot6d_to_matrix(x6, treat_as_columns=True)   # (B,3,3)

    # Fixed “hand frame” correction used at init:
    # rotation_hand = Rz(-pi/2) @ Ry(-pi/2) @ Rz(0) = Rz(-pi/2) @ Ry(-pi/2)
    # This multiplies out to the constant matrix below.
    R_hand = torch.tensor([[0., 1., 0.],
                           [0., 0., 1.],
                           [1., 0., 0.]], dtype=torch.float, device=device)  # (3,3)

    # Undo the constant correction on the RIGHT, then take +Z of that frame:
    R_global_local = R_final @ R_hand.t()     # (B,3,3)
    n_world = R_global_local[:, :, 2]         # third column == ... @ [0,0,1]
    n_world = _normalize(n_world)
    return n_world

def cal_energy(hand_model, object_model, dst=0.1, w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0, verbose=False):
    
    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)], 
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_joints
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
        torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)

    # E_pen
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_spen
    E_spen = hand_model.self_penetration()
    # print('ds
    palm_norm = palm_normal_from_hand_pose(hand_model.hand_pose)
    s = (hand_model.hand_pose[:, :3] * palm_norm).sum(dim=-1, keepdim=True).view(-1)

    E_dst = 100*torch.abs(dst + s)
    # print('dst', E_dst.item())

    if verbose:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + E_dst, E_fc, E_dis, E_pen, E_spen, E_joints
    else:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + E_dst
