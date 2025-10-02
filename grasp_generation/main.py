"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: Entry of the program
"""

import os

os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d

from utils.hand_model import HandModel
from utils.hand_model_experimental import ExperimentalHandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.optimizer_experimental import ExperimentalAnnealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--object_code_list', default=
    [
        'sem-Car-2f28e2bd754977da8cfac9da0ff28f62',
        'sem-Car-27e267f0570f121869a949ac99a843c4',
        'sem-Hammer-5d4da30b8c0eaae46d7014c7d6ce68fc',
        'core-mug-1a1c0a8d4bad82169f0594e65f756cf5',
        'core-bottle-1ffd7113492d375593202bf99dddc268',
    ], type=list)
parser.add_argument('--name', default='exp_7', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--n_iter', default=10000, type=int)
# hyper parameters
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--noise_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=30.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.03, type=float)
parser.add_argument('--distance_upper', default=0.03, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

parser.add_argument("--experimental", action="store_true")
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# path = "path/to/result_dir/OBJECT_CODE.npy"
# arr = np.load(path, allow_pickle=True)
# data_list = arr.tolist()
# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)
if args.experimental:
    print('using experimental hand model')
    hand_model = ExperimentalHandModel(
        urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
        contact_points_path='allegro_hand_description/contact_points.json', 
        n_surface_points=1000, 
        device=device
    )
else:
    hand_model = HandModel(
        urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
        contact_points_path='allegro_hand_description/contact_points.json', 
        n_surface_points=1000, 
        device=device
    )

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)


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

# object_model.initialize(args.object_code_list)
object_model.initialize("sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98")
# object_model.initialize("mujoco-Marvel_Avengers_Titan_Hero_Series_Doctor_Doom")

initialize_convex_hull(hand_model, object_model, args)
# palm_axis = 'x'
dst = (hand_model.hand_pose[:, :3] ** 2).sum(dim=1).sqrt()
print('initial distance',  dst)

print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()
palm_norm = palm_normal_from_hand_pose(hand_model.hand_pose)
s = (hand_model.hand_pose[:, :3] * palm_norm).sum(dim=-1, keepdim=True)

print('proj', s)


optim_config = {
    'switch_possibility': args.switch_possibility,
    'starting_temperature': args.starting_temperature,
    'temperature_decay': args.temperature_decay,
    'annealing_period': args.annealing_period,
    'noise_size': args.noise_size,
    'stepsize_period': args.stepsize_period,
    'mu': args.mu,
    'device': device
}
if args.experimental:
    print('using experimental optimizer')
    optimizer = ExperimentalAnnealing(hand_model, **optim_config)
else:
    optimizer = Annealing(hand_model, **optim_config)

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)
# dst_goal = torch.linspace(0.04, 0.12, steps=args.batch_size, dtype=float, device=device)
dst_goal = torch.linspace(0.05, 0.12, steps=args.batch_size, dtype=float, device=device)
print("goal dst:", dst_goal)
energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, dst=dst_goal, verbose=True, **weight_dict)

print("energy.requires_grad:", energy.requires_grad)
print("rot.requires_grad:", hand_model.rotation.requires_grad)
print("jnt.requires_grad:", hand_model.joint_angles.requires_grad)
energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()
    hand_model.rebuild_hand_pose()
    if step % 100 == 0:
        dst = (hand_model.hand_pose[:, :3] ** 2).sum(dim=1).sqrt()
        print('distance',  dst)
        palm_norm = palm_normal_from_hand_pose(hand_model.hand_pose)
        s = (hand_model.hand_pose[:, :3] * palm_norm).sum(dim=-1, keepdim=True)
        print('proj', s)
    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, dst=dst_goal, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    
    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]
try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(object_model.object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        hand_pose = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler)))
        qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        data_list.append(dict(
            scale=scale,
            qpos=qpos,
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
        ))
    np.save(os.path.join(result_path, object_model.object_code_list[i] + '.npy'), data_list, allow_pickle=True)
