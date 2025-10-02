# verify_grasp_in_gym.py
# Replays an optimized Allegro pose on a random trained object in Isaac Gym.
# Layout expected:
#   this_script.py
#   allegro_hand_description/allegro_hand_description_right.urdf
#   ../data/experiments/<EXP_NAME>/results/*.npy
#   ../data/meshdata/<object_code>/**/coacd.urdf   (or any *.urdf under the object dir)
#
# Run:  python verify_grasp_in_gym.py --exp exp_33 --gpu_pipeline 0 --best_of 0

from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import os, glob, math, random, argparse, re
import numpy as np
import xml.etree.ElementTree as ET

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="exp_5", type=str, help="experiment name under ../data/experiments")
parser.add_argument("--results_root", default="../data/experiments", type=str)
parser.add_argument("--mesh_root", default="../data/meshdata", type=str)
parser.add_argument("--gpu_pipeline", default=0, type=int)
parser.add_argument("--raise_object_z", default=0.0, type=float)  # keep object off plane
parser.add_argument("--raise_all_z", default=0.0, type=float)      # global lift (applied to both object + hand)
parser.add_argument("--best_of", default=0, type=int, help=">0 to sample k random entries and choose min energy")
parser.add_argument("--no_ground", action="store_true")
parser.add_argument("--show_forces", action="store_true", help="print contact force magnitudes")
parser.add_argument("--reset_key", default="r", type=str, help="Keyboard key to reset object pose (default: 'r')")
parser.add_argument("--save_file", default="saved_poses.npy", type=str,
                    help="Filename (under results_dir) to append saved poses to")
parser.add_argument("--indices_json", type=str, default=None,
    help="Path to a JSON with {'indices':[...], 'index_base':0|1}. If set with --trajectory_json, "
         "reorders/subsets waypoints accordingly.")
parser.add_argument("--save_json", default="saved_poses.json", type=str,   
                    help="Filename (under results_dir) to append saved poses to")
parser.add_argument("--trajectory_json", type=str, default=None,
                    help="Path to a VLM-planned trajectory JSON. If set, skips .npy loading.")
parser.add_argument("--object_rpy_deg", default="0 0 0",
                    help="Rotate object frame (sxyz) in degrees, e.g. '0 90 0' if model is Y-up.")
parser.add_argument("--object_scale", type=float, default=None,
                    help="Override object scale (final scale). If unset, uses entry['scale'] or 1.0.")
args = parser.parse_args()

# ----------------- Small utilities -----------------
def key_enum_from_char(ch: str):
    ch = (ch or "r").strip()
    if not ch:
        return None
    ch = ch[0].upper()
    enum_name = f"KEY_{ch}"
    return getattr(gymapi, enum_name, gymapi.KEY_R)
def load_indices_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "indices" not in data:
        raise ValueError(f"[indices_json] Expected object with 'indices' in {path}")
    idxs = data["indices"]
    if not isinstance(idxs, list) or not all(isinstance(i, int) for i in idxs):
        raise ValueError(f"[indices_json] 'indices' must be a list[int] in {path}")
    base = int(data.get("index_base", 0))
    if base not in (0, 1):
        raise ValueError(f"[indices_json] index_base must be 0 or 1 (got {base})")
    return idxs, base

def euler_sxyz_to_quat(rx, ry, rz):
    """Convert extrinsic sxyz Euler (rx,ry,rz in radians) to (x,y,z,w) quat."""
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    qw = cx*cy*cz + sx*sy*sz
    qx = sx*cy*cz - cx*sy*sz
    qy = cx*sy*cz + sx*cy*sz
    qz = cx*cy*sz - sx*sy*cz
    return (qx, qy, qz, qw)

def map_name_to_index(names):
    return {n:i for i,n in enumerate(names)}

def _json_path():
    return os.path.join(results_dir, args.save_json)

def _load_waypoints_json(path):
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("waypoints", None), list):
                return data
        except Exception:
            pass
    return {"waypoints": []}

def _save_waypoints_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _palm_object_dist(entry):
    qpos = entry.get("qpos", {})
    try:
        tx, ty, tz = float(qpos["WRJTx"]), float(qpos["WRJTy"]), float(qpos["WRJTz"])
    except Exception:
        return float("inf")
    # return (tx*tx + ty*ty + tz*tz) ** 0.5
    # return ty
    # return entry.get("energy", float(10000))
    return 0

def load_sorted_entries(results_dir):
    npy_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"[results] No .npy files in {results_dir}")
    npy = npy_files[0]

    raw = np.load(npy, allow_pickle=True)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    entries = []
    for e in raw:
        if isinstance(e, dict):
            entries.append(e)
        elif isinstance(e, np.ndarray):
            entries.append(e.item())
        else:
            entries.append(e)
    if not entries:
        raise ValueError(f"[results] {npy} appears empty")

    entries.sort(key=_palm_object_dist)
    obj_code = os.path.splitext(os.path.basename(npy))[0]
    return obj_code, entries, npy
import json

def entries_from_trajectory_json(json_path, joint_names):
    """
    JSON schema expected:
    {
      "waypoints": [
        {
          "OBJTx": <float>, "OBJTy": <float>, "OBJTz": <float>,
          "OBJRx": <float>, "OBJRy": <float>, "OBJRz": <float>,
          "joints": [<16 floats>],           # Allegro order matching joint_names
          "scale": <float>,                  # optional; default 1.0
          "energy": <float>,                 # optional; default 0.0
          "note": "<optional>"
        }, ...
      ]
    }
    Returns a list[dict] in your internal “object_in_palm” entry format.
    """
    with open(json_path, "r") as f:
        traj = json.load(f)

    wps = traj.get("waypoints", [])
    if not isinstance(wps, list) or not wps:
        raise ValueError(f"[trajectory_json] No waypoints in {json_path}")

    out = []
    for i, w in enumerate(wps):
        # Build qpos map
        qpos = {
            "OBJTx": float(w["OBJTx"]),
            "OBJTy": float(w["OBJTy"]),
            "OBJTz": float(w["OBJTz"]),
            "OBJRx": float(w["OBJRx"]),
            "OBJRy": float(w["OBJRy"]),
            "OBJRz": float(w["OBJRz"]),
        }
        joints = w.get("joints", None)
        if joints is None or len(joints) != len(joint_names):
            raise ValueError(f"[trajectory_json] waypoint {i} has invalid joints length "
                             f"(got {0 if joints is None else len(joints)}; expected {len(joint_names)})")
        for jn, val in zip(joint_names, joints):
            qpos[jn] = float(val)

        out.append({
            "qpos": qpos,
            "frame": "object_in_palm",
            "scale": float(w.get("scale", 1.0)),
            "energy": float(w.get("energy", 0.0)),
            "note":  w.get("note", f"wp_{i}")
        })
    return out

# ----------------- URDF helpers (locate + baked scale read) -----------------
def find_object_urdf(obj_code, mesh_root):
    """Prefer coacd/coacd.urdf; otherwise first *.urdf anywhere under the object folder."""
    base = os.path.join(mesh_root, obj_code)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"[mesh] Directory not found for object {obj_code}: {base}")
    pref = os.path.join(base, "coacd", "coacd.urdf")
    if os.path.isfile(pref):
        return pref
    hits = glob.glob(os.path.join(base, "**", "*.urdf"), recursive=True)
    if hits:
        hits.sort(key=lambda p: (p.count(os.sep), len(p)))
        return hits[0]
    raise FileNotFoundError(f"[mesh] No URDF found under {base}")

_ws = re.compile(r"\s+")
def _parse_mesh_scale_attr(val: str) -> float:
    if not val or not val.strip():
        return 1.0
    parts = [p for p in _ws.split(val.strip()) if p]
    if len(parts) == 1:
        return float(parts[0])
    xs = list(map(float, parts[:3]))
    if len(xs) < 3:
        return float(xs[0])
    return (xs[0] * xs[1] * xs[2]) ** (1.0 / 3.0)

def get_urdf_effective_scale(urdf_path: str) -> float:
    with open(urdf_path, "r") as f:
        txt = f.read()
    root = ET.fromstring(txt)
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith("mesh"):
            return _parse_mesh_scale_attr(elem.get("scale"))
    return 1.0

# ---------- minimal quat & vector helpers ----------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vec(q, v):
    # Rotate v by unit quaternion q
    vq = np.array([0.0, *v])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]
def object_in_hand_from_hand_in_object(t_oh, q_oh):
    q_ho = quat_conj(q_oh)
    t_ho = -rotate_vec(q_ho, t_oh)
    return t_ho, q_ho
def quat_conj(q):  # q = [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_inv(q):  # (w,x,y,z)
    q = np.asarray(q, dtype=float)
    return quat_conj(q) / (np.dot(q, q) + 1e-12)

def quat_apply(q, v3):
    return rotate_vec(q, v3)

def compose(t_a, q_a, t_b, q_b):
    """Compose A*B: world-from-b = A, then B; returns (t, q) with q=(w,x,y,z)."""
    t = np.array(t_a) + quat_apply(q_a, np.array(t_b))
    q = quat_mul(q_a, q_b)
    return t, q

def relative(from_t, from_q, to_t, to_q):
    """Return transform of 'to' **in** 'from' frame: T_from^-1 * T_to."""
    q_inv = quat_inv(from_q)
    t = quat_apply(q_inv, np.array(to_t) - np.array(from_t))
    q = quat_mul(q_inv, to_q)
    return t, q
def quat_normed(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)
def quat_to_euler_sxyz(qw, qx, qy, qz):
    """Extrinsic sxyz Euler (radians) from (qw,qx,qy,qz)."""
    # Use intrinsic XYZ equivalent (same as extrinsic sxyz) math
    # Reference: standard conversions
    # Guard against numerical drift
    qw, qx, qy, qz = quat_normed([qw, qx, qy, qz])
    # Rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ])
    # intrinsic XYZ
    sy = -R[2,0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = math.asin(sy)
    rx = math.atan2(R[2,1], R[2,2])
    rz = math.atan2(R[1,0], R[0,0])
    return rx, ry, rz

COLLISION_GROUP  = 0
COLLISION_FILTER = -1
ALIGN_Q = gymapi.Quat(0.0, 0.0, 0.0, 1.0)   # computed once to align palm->+Z
_key_is_down = {"next_pose": False, "prev_pose": False, "save_pose": False, "reset_obj": False}
translation_names = ['WRJTx','WRJTy','WRJTz']
rot_names = ['WRJRx','WRJRy','WRJRz']
joint_names = [
    'joint_0.0','joint_1.0','joint_2.0','joint_3.0',
    'joint_4.0','joint_5.0','joint_6.0','joint_7.0',
    'joint_8.0','joint_9.0','joint_10.0','joint_11.0',
    'joint_12.0','joint_13.0','joint_14.0','joint_15.0'
]
# ----------------- Isaac setup -----------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0/120.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(-9.81,0.0,0.0)
sim_params.use_gpu_pipeline = bool(args.gpu_pipeline)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 12
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.005
sim_params.physx.rest_offset = 0.0

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("Failed to create sim")

if not args.no_ground:
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0,0,1)
    plane.static_friction = 1.0
    plane.dynamic_friction = 1.0
    gym.add_ground(sim, plane)

# Viewer
vp = gymapi.CameraProperties()
vp.width, vp.height = 1280, 800
vp.use_collision_geometry = False
viewer = gym.create_viewer(sim, vp)
if viewer is None:
    raise RuntimeError("Failed to create viewer")
RESET_KEY_ENUM = key_enum_from_char(args.reset_key)
gym.subscribe_viewer_keyboard_event(viewer, RESET_KEY_ENUM, "reset_obj")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_N, "next_pose")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "prev_pose")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "save_pose")

print(f"[keys] Press '{args.reset_key}' to reset the OBJECT pose")

SAVE_OFFSET = 0.01  # undo the +0.05 applied when setting targets

def _read_root_pose(env, actor):
    rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)[0]
    p = (float(rb['pose']['p']['x']), float(rb['pose']['p']['y']), float(rb['pose']['p']['z']))
    q = (float(rb['pose']['r']['w']), float(rb['pose']['r']['x']),
         float(rb['pose']['r']['y']), float(rb['pose']['r']['z']))
    return p, q

def save_current_pose():
    # 1) Read world poses
    obj_p, obj_q   = _read_root_pose(env, obj_actor)    # object in world
    hand_p, hand_q = _read_root_pose(env, hand_actor)   # palm (hand base) in world

    # 2) Object in palm frame (relative transform): T_palm^-1 * T_obj
    t_op, q_op = relative(hand_p, hand_q, obj_p, obj_q)  # (tx,ty,tz), (w,x,y,z)
    rx, ry, rz = quat_to_euler_sxyz(q_op[0], q_op[1], q_op[2], q_op[3])

    # 3) Read current hand DOF positions
    dof_state = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
    cur_pos = np.array(dof_state["pos"], dtype=float)

    # 4) Build entry (object-in-palm). Keep joints consistent with your names; undo +0.05
    qpos_save = {
        "OBJTx": float(t_op[0]),
        "OBJTy": float(t_op[1]),
        "OBJTz": float(t_op[2]),
        "OBJRx": float(rx),
        "OBJRy": float(ry),
        "OBJRz": float(rz),
    }
    for jn in joint_names:
        idx = hand_dof_map.get(jn, None)
        if idx is not None:
            qpos_save[jn] = float(cur_pos[idx] - SAVE_OFFSET)
    joints_arr = []
    for jn in joint_names:
        idx = hand_dof_map.get(jn, None)
        if idx is not None:
            val = float(cur_pos[idx] - SAVE_OFFSET)
            qpos_save[jn] = val
            joints_arr.append(val)


    new_entry = {
        "qpos": qpos_save,
        "frame": "object_in_palm",   # <— key to maintain backward compatibility
        "energy": 0.0,
        "scale": float(target_scale),
    }

    # 5) Append to results_dir/save_file
    out_path = os.path.join(results_dir, args.save_file)
    if os.path.isfile(out_path):
        existing = np.load(out_path, allow_pickle=True)
        existing = existing.tolist() if isinstance(existing, np.ndarray) else [existing]
        existing.append(new_entry)
        np.save(out_path, np.array(existing, dtype=object), allow_pickle=True)
        print(f"[save] Appended pose #{len(existing)} → {out_path}")
    else:
        np.save(out_path, np.array([new_entry], dtype=object), allow_pickle=True)
        print(f"[save] Created {out_path} with 1 entry")
    
    out_json = _json_path()
    data = _load_waypoints_json(out_json)
    data["waypoints"].append({
        "OBJTx": qpos_save["OBJTx"],
        "OBJTy": qpos_save["OBJTy"],
        "OBJTz": qpos_save["OBJTz"],
        "OBJRx": qpos_save["OBJRx"],
        "OBJRy": qpos_save["OBJRy"],
        "OBJRz": qpos_save["OBJRz"],
        "joints": joints_arr,                 # 16-length array in Allegro order
        "scale": new_entry["scale"],
        "energy": new_entry["energy"],
        "note": f"saved_step_{len(data['waypoints'])}"
    })
    _save_waypoints_json(out_json, data)
    print(f"[save] JSON waypoint #{len(data['waypoints'])} → {out_json}")

# Hand asset
here = os.path.dirname(os.path.abspath(__file__))
asset_root = os.path.join(here, "allegro_hand_description")
asset_file = "allegro_hand_description_right.urdf"

hand_opts = gymapi.AssetOptions()
hand_opts.fix_base_link = True
hand_opts.disable_gravity = True
hand_opts.collapse_fixed_joints = True
hand_opts.use_mesh_materials = True
hand_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
hand_opts.thickness = 0.0005
hand_opts.angular_damping = 0.02

hand_asset = gym.load_asset(sim, asset_root, asset_file, hand_opts)
if hand_asset is None:
    raise RuntimeError("Failed to load Allegro asset")

hand_dof_names = gym.get_asset_dof_names(hand_asset)
hand_dof_map = map_name_to_index(hand_dof_names)
num_hand_dofs = gym.get_asset_dof_count(hand_asset)

# Pick an object + pose
# results_dir = os.path.join(args.results_root, args.exp, "results")
# obj_code, entries, which_npy = load_sorted_entries(results_dir)
# cur_idx = 0
# entry = entries[cur_idx]
results_dir = os.path.join(args.results_root, args.exp, "results")

if args.trajectory_json:
    # # Build entries from JSON; skip .npy completely
    # entries = entries_from_trajectory_json(args.trajectory_json, joint_names)
    # # obj_code = os.path.splitext(os.path.basename(args.trajectory_json))[0]
    # obj_code = "sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98"
    # which_src = args.trajectory_json
    entries = entries_from_trajectory_json(args.trajectory_json, joint_names)
    
    which_src = args.trajectory_json
    # Hardcode or make this a flag if you prefer
    obj_code = "sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98"

    # If indices JSON is provided, subset/reorder now
    if args.indices_json:
        idxs, base = load_indices_json(args.indices_json)
        # normalize to 0-based
        if base == 1:
            idxs = [i - 1 for i in idxs]
        # bounds check
        n = len(entries)
        bad = [i for i in idxs if i < 0 or i >= n]
        if bad:
            raise IndexError(f"[indices_json] Out-of-range indices {bad}; have {n} waypoints in {args.trajectory_json}")
        # reorder/subset
        entries = [entries[i] for i in idxs]
        which_src += f" + {args.indices_json}"
else:
    obj_code, entries, which_src = load_sorted_entries(results_dir)
# SKIP_1BASED = {2, 4}  # e.g., skip poses 2 and 4 (human counting)

# orig_n = len(entries)
# entries = [e for k, e in enumerate(entries, start=1) if k not in SKIP_1BASED]
# obj_code = "mujoco-Marvel_Avengers_Titan_Hero_Series_Doctor_Doom"
cur_idx = 0
entry = entries[cur_idx]

# Decide target scale from first entry (or override)
target_scale = float(entry.get("scale", 1.0))
if args.object_scale is not None:
    target_scale = float(args.object_scale)

print(f"[pick] Object: {obj_code}\n"
      f"       From: {which_src}\n"
      f"       Energy: {entry.get('energy', float('nan')): .6f}\n"
      f"       Target scale: {target_scale:g}")

# Decide target scale from first entry (or override)
target_scale = float(entry.get("scale", 1.0))
# target_scale = 0.08
if args.object_scale is not None:
    target_scale = float(args.object_scale)
# print(f"[pick] Object: {obj_code}\n"
#       f"       From: {which_npy}\n"
#       f"       Energy: {entry.get('energy', float('nan')): .6f}\n"
#       f"       Target scale: {target_scale:g}")

env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)



# ----------------- Hand initial placement from first entry -----------------


hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
hand_pose.r = gymapi.Quat(1.0, 0.0, 0.0, 0.0)
hand_actor = gym.create_actor(env, hand_asset, hand_pose, "allegro", COLLISION_GROUP, COLLISION_FILTER)

# ---- Load object URDF (no file edits) + runtime actor scaling ----
orig_obj_urdf = find_object_urdf(obj_code, args.mesh_root)
baked = get_urdf_effective_scale(orig_obj_urdf)

ao = gymapi.AssetOptions()
ao.fix_base_link = False
ao.disable_gravity = False
ao.use_mesh_materials = True
ao.collapse_fixed_joints = True
ao.override_com = True
ao.override_inertia = True
ao.density = 500.0

obj_asset = gym.load_asset(sim, os.path.dirname(orig_obj_urdf), os.path.basename(orig_obj_urdf), ao)
if obj_asset is None:
    raise RuntimeError(f"Failed to load object URDF: {orig_obj_urdf}")
# qpos0 = entry["qpos"]
# tx0, ty0, tz0 = (float(qpos0[n]) for n in translation_names)
# rx0, ry0, rz0 = (float(qpos0[n]) for n in rot_names)
# qx0, qy0, qz0, qw0 = euler_sxyz_to_quat(rx0, ry0, rz0)

# loc_np = np.array([tx0, ty0, tz0])
# ori_np = np.array([qw0, qx0, qy0, qz0])

# obj_loc, obj_ori= object_in_hand_from_hand_in_object(loc_np, ori_np)

def _extract_object_in_palm(entry):
    qpos0 = entry["qpos"]
    if entry.get("frame", "") == "object_in_palm" or ("OBJTx" in qpos0 and "OBJRx" in qpos0):
        # New format: directly object-in-palm
        tx, ty, tz = (float(qpos0["OBJTx"]), float(qpos0["OBJTy"]), float(qpos0["OBJTz"]))
        rx, ry, rz = (float(qpos0["OBJRx"]), float(qpos0["OBJRy"]), float(qpos0["OBJRz"]))
        qw, qx, qy, qz = euler_sxyz_to_quat(rx, ry, rz)  # (x,y,z,w) from your helper
        # convert to (w,x,y,z)
        q = (qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=float)
        return t, q, qpos0
    else:
        # Legacy format: hand-in-object (WRJ*). Convert to object-in-palm.
        tx0, ty0, tz0 = (float(qpos0['WRJTx']), float(qpos0['WRJTy']), float(qpos0['WRJTz']))
        rx0, ry0, rz0 = (float(qpos0['WRJRx']), float(qpos0['WRJRy']), float(qpos0['WRJRz']))
        qx0, qy0, qz0, qw0 = euler_sxyz_to_quat(rx0, ry0, rz0)  # (x,y,z,w)
        # hand-in-object → object-in-hand (then hand=palm, so that's our target)
        loc_np = np.array([tx0, ty0, tz0])
        ori_np = np.array([qw0, qx0, qy0, qz0])  # (w,x,y,z)
        t_oh, q_oh = object_in_hand_from_hand_in_object(loc_np, ori_np)
        return t_oh, q_oh, qpos0
def apply_entry(entry):
    global init_obj_rb_states
    obj_loc, obj_ori, qpos0 = _extract_object_in_palm(entry)
    # qpos0 = entry["qpos"]
    # tx0, ty0, tz0 = (float(qpos0[n]) for n in translation_names)
    # rx0, ry0, rz0 = (float(qpos0[n]) for n in rot_names)
    # qx0, qy0, qz0, qw0 = euler_sxyz_to_quat(rx0, ry0, rz0)

    # loc_np = np.array([tx0, ty0, tz0])
    # ori_np = np.array([ qw0, qx0, qy0, qz0])
    # obj_loc, obj_ori= object_in_hand_from_hand_in_object(loc_np, ori_np)

    init_obj_rb_states = [(((obj_loc[0], obj_loc[1], obj_loc[2]),
                            (obj_ori[3], obj_ori[0], obj_ori[1], obj_ori[2])),
                           ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))]
    

    # Set finger joints & PD targets
    dof_state = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
    dof_state["pos"][:] = 0.0
    dof_state["vel"][:] = 0.0
    for jn in joint_names:
        if jn in hand_dof_map:
            dof_state["pos"][hand_dof_map[jn]] = float(qpos0[jn]) + SAVE_OFFSET
    gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)
    targets = np.array(dof_state["pos"], dtype=np.float32)
    gym.set_actor_dof_position_targets(env, hand_actor, targets)

    gym.set_actor_rigid_body_states(env, obj_actor, init_obj_rb_states, gymapi.STATE_ALL)


    print(f"[pose] {cur_idx+1}/{len(entries)}  energy={entry.get('energy', float('nan')):.6f}  dist={_palm_object_dist(entry):.4f}")


obj_loc, obj_ori, qpos0 = _extract_object_in_palm(entry)
# Spawn object (origin-aligned) + raise
obj_pose = gymapi.Transform()
obj_pose.p = gymapi.Vec3(obj_loc[0], obj_loc[1], obj_loc[2])
obj_pose.r = gymapi.Quat(obj_ori[3], obj_ori[0], obj_ori[1], obj_ori[2])
obj_actor  = gym.create_actor(env, obj_asset,  obj_pose,  f"obj:{obj_code}", COLLISION_GROUP, COLLISION_FILTER)
gym.set_actor_scale(env, obj_actor, float(target_scale))
print(f"[object] URDF={orig_obj_urdf}\n"
      f"         baked≈{baked:g}  actor_scale={target_scale:g}  final≈{baked*target_scale:g}")
init_obj_rb_states = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_ALL).copy()
print("init obj rb states", init_obj_rb_states)


# Tweak shape frictions
def tune_friction(actor, val=1.5):
    props = gym.get_actor_rigid_shape_properties(env, actor)
    for p in props:
        p.friction = val
        p.restitution = 0.0
        p.torsion_friction = val
        p.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, actor, props)
tune_friction(hand_actor, 2.0)
tune_friction(obj_actor, 1.0)


# Hand DOF PD holds at target angles
dprops = gym.get_actor_dof_properties(env, hand_actor)
for i in range(num_hand_dofs):
    dprops["driveMode"][i] = int(gymapi.DOF_MODE_POS)
    dprops["stiffness"][i]  = 300.0
    dprops["damping"][i]    = 900.0
    dprops["friction"][i]   = 0.1
    dprops["armature"][i]   = 0.1
    dprops["effort"][i]     = 20.0
    dprops["velocity"][i]   = 1.0
gym.set_actor_dof_properties(env, hand_actor, dprops)

# ---------- Compute fixed alignment: palm -> +Z, then apply to OBJECT once ----------


# qpos = entry["qpos"]
# dof_state = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
# dof_state["pos"][:] = 0.0
# dof_state["vel"][:] = 0.0
# for jn in joint_names:
#     if jn in hand_dof_map:
#         dof_state["pos"][hand_dof_map[jn]] = float(qpos[jn]) + 0.05
# gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)
# targets = np.array(dof_state["pos"], dtype=np.float32)
# gym.set_actor_dof_position_targets(env, hand_actor, targets)

# ----------------- Pose application (no respawn) -----------------

# Apply the first (aligned) pose now
apply_entry(entry)

# Finalize & camera
gym.prepare_sim(sim)
cam_pos = gymapi.Vec3(0.55, 0.45, 0.35)
cam_tgt = gymapi.Vec3(0.00, 0.00, obj_pose.p.z + 0.05)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_tgt)

# Optional contact force readout
if args.show_forces:
    net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf_t = gymtorch.wrap_tensor(net_cf)  # shape [num_nodes, 3]

print("[sim] Running. Close the viewer window to exit.")
steps = 0
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if args.show_forces and steps % 2 == 0:
        gym.refresh_net_contact_force_tensor(sim)
        cf = net_cf_t.clone()
        total = float(cf.norm(dim=1).sum().cpu())
        print(f"[contact] net |F| sum ≈ {total: .3f}")

    for evt in gym.query_viewer_action_events(viewer):
        name = evt.action
        val  = evt.value > 0
        if name in _key_is_down:
        # Rising edge: act once
            if val and not _key_is_down[name]:
                if name == "reset_obj":
                    gym.set_actor_rigid_body_states(env, obj_actor, init_obj_rb_states, gymapi.STATE_ALL)
                    print("[reset] Object pose reset to aligned spawn.")
                elif name == "next_pose":
                    cur_idx = (cur_idx + 1) % len(entries)
                    entry = entries[cur_idx]
                    apply_entry(entry)
                elif name == "prev_pose":
                    cur_idx = (cur_idx - 1) % len(entries)
                    entry = entries[cur_idx]
                    apply_entry(entry)
                elif name == "save_pose":
                    save_current_pose()
            # Update state on both down and up
            _key_is_down[name] = val

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    steps += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
