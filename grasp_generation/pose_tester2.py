# verify_grasp_in_gym.py
# Replays an optimized Allegro pose on a random trained object in Isaac Gym.
# Layout expected:
#   this_script.py
#   allegro_hand_description/allegro_hand_description_right.urdf
#   ../data/experiments/<EXP_NAME>/results/*.npy
#   ../data/meshdata/<object_code>/**/coacd.urdf   (or any *.urdf under the object dir)
#
# Run:  python verify_grasp_in_gym.py --exp exp_33 --gpu_pipeline 0 --best_of 0
#       (set --best_of N to pick the lowest-energy among N randoms)
#
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import os, glob, math, random, argparse, re
import numpy as np
import xml.etree.ElementTree as ET

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="exp_1", type=str, help="experiment name under ../data/experiments")
parser.add_argument("--results_root", default="../data/experiments", type=str)
parser.add_argument("--mesh_root", default="../data/meshdata", type=str)
parser.add_argument("--gpu_pipeline", default=0, type=int)
parser.add_argument("--raise_object_z", default=0.0, type=float)  # keep object off plane
parser.add_argument("--raise_all_z", default=0.0, type=float)      # global lift (applied to both object + hand)
parser.add_argument("--best_of", default=0, type=int, help=">0 to sample k random entries and choose min energy")
parser.add_argument("--no_ground", action="store_true")
parser.add_argument("--show_forces", action="store_true", help="print contact force magnitudes")
parser.add_argument("--reset_key", default="r", type=str, help="Keyboard key to reset object pose (default: 'r')")
parser.add_argument("--object_rpy_deg", default="0 0 0",
                    help="Rotate object frame (sxyz) in degrees, e.g. '0 90 0' if model is Y-up.")
parser.add_argument("--object_scale", type=float, default=None,
                    help="Override object scale (final scale). If unset, uses entry['scale'] or 0.001.")
args = parser.parse_args()

# ----------------- Small utilities -----------------
def key_enum_from_char(ch: str):
    ch = (ch or "r").strip()
    if not ch:
        return None
    ch = ch[0].upper()
    enum_name = f"KEY_{ch}"
    return getattr(gymapi, enum_name, gymapi.KEY_R)

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

def _palm_object_dist(entry):
    qpos = entry.get("qpos", {})
    try:
        tx, ty, tz = float(qpos["WRJTx"]), float(qpos["WRJTy"]), float(qpos["WRJTz"])
    except Exception:
        # If anything is missing, fall back to "far" so it won't be chosen
        return float("inf")
    return (tx*tx + ty*ty + tz*tz) ** 0.5 + entry.get("energy", 0.0)*0.0

def load_sorted_entries(results_dir):
    npy_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"[results] No .npy files in {results_dir}")
    npy = npy_files[0]   # or pick by name; keeping your choice

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

    entries.sort(key=_palm_object_dist)  # closest first
    obj_code = os.path.splitext(os.path.basename(npy))[0]
    return obj_code, entries, npy

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
    """Return a uniform scalar from a <mesh scale> attribute.
       - empty/None -> 1.0
       - single value -> that value
       - 3+ values -> geometric mean (approx uniformization)
    """
    if not val or not val.strip():
        return 1.0
    parts = [p for p in _ws.split(val.strip()) if p]
    if len(parts) == 1:
        return float(parts[0])
    # take first three if more are present
    xs = list(map(float, parts[:3]))
    if len(xs) < 3:
        return float(xs[0])
    return (xs[0] * xs[1] * xs[2]) ** (1.0 / 3.0)

def get_urdf_effective_scale(urdf_path: str) -> float:
    """Read the first <mesh> tag and return an effective baked scale."""
    with open(urdf_path, "r") as f:
        txt = f.read()
    # Parse without touching the file on disk
    root = ET.fromstring(txt)
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith("mesh"):
            return _parse_mesh_scale_attr(elem.get("scale"))
    return 1.0


COLLISION_GROUP  = 0  
COLLISION_FILTER = -1
# ----------------- Isaac setup -----------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0/120.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0,0,-9.81)
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
print(f"[keys] Press '{args.reset_key}' to reset the OBJECT pose")

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
results_dir = os.path.join(args.results_root, args.exp, "results")
obj_code, entries, which_npy = load_sorted_entries(results_dir)
cur_idx = -20
entry = entries[cur_idx]

# Decide target scale: CLI override -> entry['scale'] -> default mm->m correction

target_scale = float(entry.get("scale", 1.0))
print(target_scale)

print(f"[pick] Object: {obj_code}\n"
      f"       From: {which_npy}\n"
      f"       Energy: {entry.get('energy', float('nan')): .6f}\n"
      f"       Target scale: {target_scale:g}")

env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)

# ---- Load object URDF (no file edits) + runtime actor scaling ----
orig_obj_urdf = find_object_urdf(obj_code, args.mesh_root)
# orig_obj_urdf = "/home/william/Desktop/USC/DexGraspNet/data/meshdata/sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98/coacd/coacd.urdf"
baked = get_urdf_effective_scale(orig_obj_urdf)
actor_scale = target_scale  #0.05 

ao = gymapi.AssetOptions()
ao.fix_base_link = False
ao.disable_gravity = False
ao.use_mesh_materials = True
ao.collapse_fixed_joints = True
ao.override_com = True            # ← match paper
ao.override_inertia = True        # ← match paper
ao.density = 500.0

obj_asset = gym.load_asset(sim, os.path.dirname(orig_obj_urdf), os.path.basename(orig_obj_urdf), ao)
if obj_asset is None:
    raise RuntimeError(f"Failed to load object URDF: {orig_obj_urdf}")

# Spawn object (origin-aligned) + raise
obj_pose = gymapi.Transform()
obj_pose.p = gymapi.Vec3(0.0, 0.0, args.raise_object_z + args.raise_all_z)

# Optional extra rotation
# rx_deg, ry_deg, rz_deg = map(float, args.object_rpy_deg.split())
# rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
oqx, oqy, oqz, oqw = euler_sxyz_to_quat(0, 0, 0)
obj_pose.r = gymapi.Quat(oqx, oqy, oqz, oqw)

obj_actor  = gym.create_actor(env, obj_asset,  obj_pose,  f"obj:{obj_code}", COLLISION_GROUP, COLLISION_FILTER)

# 👉 Apply runtime scaling so final scale == target_scale
gym.set_actor_scale(env, obj_actor, float(actor_scale))
print(f"[object] URDF={orig_obj_urdf}\n"
      f"         baked≈{baked:g}  actor_scale={actor_scale:g}  final≈{baked*actor_scale:g}")

init_obj_rb_states = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_ALL).copy()

# ----------------- Hand placement -----------------
# Pull qpos (WRJ + 16 joints)
translation_names = ['WRJTx','WRJTy','WRJTz']
rot_names = ['WRJRx','WRJRy','WRJRz']
joint_names = [
    'joint_0.0','joint_1.0','joint_2.0','joint_3.0',
    'joint_4.0','joint_5.0','joint_6.0','joint_7.0',
    'joint_8.0','joint_9.0','joint_10.0','joint_11.0',
    'joint_12.0','joint_13.0','joint_14.0','joint_15.0'
]
qpos = entry["qpos"]

# Hand root pose = object pose + WRJ translation (object frame == world), rotation from Euler (sxyz, radians)
tx, ty, tz = (float(qpos[n]) for n in translation_names)
rx, ry, rz = (float(qpos[n]) for n in rot_names)
qx, qy, qz, qw = euler_sxyz_to_quat(rx, ry, rz)

hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(obj_pose.p.x + tx, obj_pose.p.y + ty, obj_pose.p.z + tz)
hand_pose.r = gymapi.Quat(qx, qy, qz, qw)

hand_actor = gym.create_actor(env, hand_asset, hand_pose, "allegro", COLLISION_GROUP, COLLISION_FILTER)



wrj_pos = hand_pose.p  # gymapi.Vec3

# A tiny fixed-base sphere at WRJ so you can see the point
marker_opts = gymapi.AssetOptions()
marker_opts.fix_base_link = True
marker_opts.disable_gravity = True
marker_radius = 0.007  # 7 mm
marker_asset = gym.create_sphere(sim, marker_radius, marker_opts)

marker_tf = gymapi.Transform()
marker_tf.p = gymapi.Vec3(wrj_pos.x, wrj_pos.y, wrj_pos.z)
marker_tf.r = gymapi.Quat(0,0,0,1)
wrj_marker = gym.create_actor(env, marker_asset, marker_tf, "WRJ_marker", COLLISION_GROUP, COLLISION_FILTER)

# Helper to (re)draw the object-origin→WRJ line
def draw_wrj_line():
    # two vertices: [object origin], [wrj]
    verts = np.array([
        [obj_pose.p.x, obj_pose.p.y, obj_pose.p.z],
        [wrj_pos.x,    wrj_pos.y,    wrj_pos.z   ],
    ], dtype=np.float32)

    cols = np.array([
        [1.0, 0.0, 0.0],  # red
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)

    gym.clear_lines(viewer)
    gym.add_lines(viewer, env, 1, verts, cols)

# Initial draw
draw_wrj_line()


# Tweak shape frictions (optional: stickier fingers)
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
    dprops["friction"][i] = 0.1
    dprops["armature"][i] = 0.1
    dprops["effort"][i] = 20.0
    dprops["velocity"][i] = 1.0
gym.set_actor_dof_properties(env, hand_actor, dprops)
def _despawn_actor(env, actor_handle_name: str):
    """Destroy actor by name of the global handle variable, flush sim/graphics, and clear the handle."""
    global viewer
    handle = globals().get(actor_handle_name, None)
    if handle is None:
        return
    try:
        gym.destroy_actor(env, handle)
    except Exception:
        pass
    # Important: clear our reference so we don't accidentally use a dead handle
    globals()[actor_handle_name] = None

    # Flush physics & graphics so the viewer updates and internal buffers rebuild
    # (one tiny step is enough; do both graphics+simulate to be safe)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
def apply_entry(entry):
    global obj_actor, hand_actor, wrj_marker, wrj_pos, init_obj_rb_states, actor_scale

    # ------------------ parse pose from entry ------------------
    qpos = entry["qpos"]
    tx, ty, tz = float(qpos["WRJTx"]), float(qpos["WRJTy"]), float(qpos["WRJTz"])
    rx, ry, rz = float(qpos["WRJRx"]), float(qpos["WRJRy"]), float(qpos["WRJRz"])
    qx, qy, qz, qw = euler_sxyz_to_quat(rx, ry, rz)

    # Optional per-entry scale (CLI override wins, fall back to previous actor_scale)
    actor_scale = float(entry.get("scale", actor_scale))

    # ------------------ destroy previous actors ------------------
    # (Safe if they don't exist yet.)
    try:

        if hand_actor is not None:
            print("destroying hand")
            _despawn_actor(env, "hand_actor")            
    except Exception:
        pass
    try:
        if wrj_marker is not None:
            print("destroying wrj marker")
            _despawn_actor(env, "WRJ_marker")
    except Exception:
        pass
    try:
        if obj_actor is not None:
            print("destroying object")
            _despawn_actor(env, "obj_actor")
    except Exception:
        pass

    # ------------------ respawn object ------------------
    # Spawn at the canonical obj_pose and re-apply runtime scale.
    obj_actor = gym.create_actor(env, obj_asset, obj_pose, f"obj:{obj_code}", COLLISION_GROUP, COLLISION_FILTER)
    gym.set_actor_scale(env, obj_actor, float(actor_scale))
    tune_friction(obj_actor, 1.0)

    # Capture the new object's initial rigid-body states for the reset handler.
    init_obj_rb_states = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_ALL).copy()

    # ------------------ respawn hand ------------------
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(obj_pose.p.x + tx, obj_pose.p.y + ty, obj_pose.p.z + tz)
    hand_pose.r = gymapi.Quat(qx, qy, qz, qw)

    hand_actor = gym.create_actor(env, hand_asset, hand_pose, "allegro", COLLISION_GROUP, COLLISION_FILTER)
    tune_friction(hand_actor, 2.0)

    # Re-apply PD/drive props to the new hand actor
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

    # Seed DOF states from qpos and set PD targets
    dof_state = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
    dof_state["pos"][:] = 0.0
    dof_state["vel"][:] = 0.0
    for jn in joint_names:
        idx = hand_dof_map.get(jn, None)
        if idx is not None:
            dof_state["pos"][idx] = float(qpos[jn]) + 0.05
    gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)
    targets = np.array(dof_state["pos"], dtype=np.float32)
    gym.set_actor_dof_position_targets(env, hand_actor, targets)

    # ------------------ respawn WRJ marker + redraw line ------------------
    wrj_pos = gymapi.Vec3(hand_pose.p.x, hand_pose.p.y, hand_pose.p.z)
    marker_tf = gymapi.Transform()
    marker_tf.p = gymapi.Vec3(wrj_pos.x, wrj_pos.y, wrj_pos.z)
    marker_tf.r = gymapi.Quat(0, 0, 0, 1)
    wrj_marker = gym.create_actor(env, marker_asset, marker_tf, "WRJ_marker", COLLISION_GROUP, COLLISION_FILTER)

    # If showing forces, reacquire the tensor because node layout may change after respawn
    if getattr(args, "show_forces", False):
        try:
            global net_cf_t
            net_cf = gym.acquire_net_contact_force_tensor(sim)
            net_cf_t = gymtorch.wrap_tensor(net_cf)
        except Exception:
            pass

    draw_wrj_line()

    # ------------------ debug / HUD ------------------
    try:
        print(f"[object] actor_scale={actor_scale:g}")
    except Exception:
        pass
    print("[debug] hand shapes:", gym.get_actor_rigid_shape_count(env, hand_actor))
    print("[debug]  obj  shapes:", gym.get_actor_rigid_shape_count(env, obj_actor))
    print(f"[pose] {cur_idx+1}/{len(entries)}  energy={entry.get('energy', float('nan')):.6f}  dist={_palm_object_dist(entry):.4f}")

qpos = entry["qpos"]
tx, ty, tz = float(qpos["WRJTx"]), float(qpos["WRJTy"]), float(qpos["WRJTz"])
rx, ry, rz = float(qpos["WRJRx"]), float(qpos["WRJRy"]), float(qpos["WRJRz"])
qx, qy, qz, qw = euler_sxyz_to_quat(rx, ry, rz)
dof_state = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
dof_state["pos"][:] = 0.0
dof_state["vel"][:] = 0.0
for jn in joint_names:
    idx = hand_dof_map.get(jn, None)
    if idx is not None:
        dof_state["pos"][idx] = float(qpos[jn]) + 0.05
gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)
targets = np.array(dof_state["pos"], dtype=np.float32)
gym.set_actor_dof_position_targets(env, hand_actor, targets)

# Set joint positions from qpos
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
    draw_wrj_line()

    if args.show_forces and steps % 2 == 0:
        gym.refresh_net_contact_force_tensor(sim)
        cf = net_cf_t.clone()
        total = float(cf.norm(dim=1).sum().cpu())
        print(f"[contact] net |F| sum ≈ {total: .3f}")

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset_obj" and evt.value > 0:
            gym.set_actor_rigid_body_states(env, obj_actor, init_obj_rb_states, gymapi.STATE_ALL)
            print("[reset] Object pose reset to spawn transform.")
        elif evt.action == "next_pose":
            cur_idx = (cur_idx + 1) % len(entries)
            entry = entries[cur_idx]
            apply_entry(entry)

        elif evt.action == "prev_pose":
            cur_idx = (cur_idx - 1) % len(entries)
            entry = entries[cur_idx]
            apply_entry(entry)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    steps += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)