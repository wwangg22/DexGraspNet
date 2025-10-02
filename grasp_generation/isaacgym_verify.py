# minimal_allegro_viewer_bulletproof.py
from isaacgym import gymapi, gymtorch
import os, xml.etree.ElementTree as ET

# ------------------ toggles ------------------
DYNAMIC_BASE = False         # start with False; set True ONLY after this works
USE_GROUND   = True          # keep True; if debugging, you can set to False
USE_COLLISION_GEOM = True    # so you see something even if visuals fail
RAISE_Z = 0.15               # spawn above plane (hand collision box has -0.0475 z offset)
# ---------------------------------------------

def patch_urdf(asset_root, asset_file):
    """Fix package:// URIs and the link_0.0 path; report missing meshes (case/path)."""
    urdf_abs = os.path.join(asset_root, asset_file)
    with open(urdf_abs, "r") as f:
        txt = f.read()

    changed = False
    if "package://" in txt:
        txt2 = txt.replace("package://allegro_hand_description/", "")
        if txt2 != txt:
            txt = txt2; changed = True
            print("[patch] Rewrote package://allegro_hand_description/ → ''")

    # Fix the one-off bad path for link_0.0 if present
    txt2 = txt.replace('filename="link_0.0.STL"', 'filename="meshes/link_0.0.STL"')
    if txt2 != txt:
        txt = txt2; changed = True
        print('[patch] Fixed link_0.0 mesh path → meshes/link_0.0.STL')

    if changed:
        with open(urdf_abs, "w") as f:
            f.write(txt)

    # Report missing mesh files (helps with case sensitivity)
    missing = []
    try:
        tree = ET.parse(urdf_abs)
        root = tree.getroot()
        for mesh in root.findall(".//geometry/mesh"):
            fn = mesh.attrib.get("filename", "")
            # resolve relative to the URDF file's folder (what Gym does)
            rel_dir = os.path.dirname(asset_file)
            cand1 = os.path.join(asset_root, rel_dir, fn)
            cand2 = os.path.join(asset_root, fn)
            if not os.path.exists(cand1) and not os.path.exists(cand2):
                missing.append(fn)
    except Exception as e:
        print("[warn] Could not parse URDF to check meshes:", e)

    if missing:
        print("\n[ERROR] Missing mesh files (check path *and case*):")
        for fn in missing:
            print("   -", fn)
        print("Fix these and re-run. Collision geometry will still render in the meantime.\n")

def main():
    gym = gymapi.acquire_gym()

    # --- sim params ---
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    # PhysX sanity
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # --- ground ---
    if USE_GROUND:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0,0,1)
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        gym.add_ground(sim, plane)

    # --- viewer ---
    vp = gymapi.CameraProperties()
    vp.width, vp.height = 960, 720
    vp.use_collision_geometry = USE_COLLISION_GEOM
    viewer = gym.create_viewer(sim, vp)
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    # --- assets root & file ---
    here = os.path.dirname(os.path.abspath(__file__))
    asset_root = os.path.join(here, "allegro_hand_description")
    asset_file = "allegro_hand_description_right.urdf"

    print("Asset root:", asset_root)
    # patch_urdf(asset_root, asset_file)

    # --- asset options ---
    ao = gymapi.AssetOptions()
    ao.fix_base_link = not DYNAMIC_BASE
    ao.disable_gravity = False if DYNAMIC_BASE else True
    ao.collapse_fixed_joints = True
    ao.use_mesh_materials = True
    ao.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    ao.thickness = 0.001
    ao.angular_damping = 0.01

    # If you want a dynamic base, give it physical properties
    # if DYNAMIC_BASE:
    #     ao.override_com = True
    #     ao.override_inertia = True
    #     ao.density = 800.0    # kg/m^3-ish; tune if you like

    asset = gym.load_asset(sim, asset_root, asset_file, ao)
    if asset is None:
        raise RuntimeError("Failed to load Allegro asset")

    # --- diagnostics ---
    num_dofs = gym.get_asset_dof_count(asset)
    num_shapes = gym.get_asset_rigid_shape_count(asset)
    rb_names = gym.get_asset_rigid_body_names(asset)
    print("Num DOFs:", num_dofs)
    print("Shapes (colliders):", num_shapes)
    print("Bodies:", rb_names)

    # --- env & actor ---
    env = gym.create_env(sim, gymapi.Vec3(-0.75,-0.75,0.0), gymapi.Vec3(0.75,0.75,0.75), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, RAISE_Z)  # *** above plane ***
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "allegro", 0, -1)

    # Tint visuals & collisions
    for n in rb_names:
        h = gym.find_actor_rigid_body_handle(env, actor, n)
        if h >= 0:
            gym.set_rigid_body_color(env, actor, h,
                                     gymapi.MESH_VISUAL_AND_COLLISION,
                                     gymapi.Vec3(0.85, 0.85, 0.85))

    # # DOF props
    dprops = gym.get_asset_dof_properties(asset)
    for i in range(num_dofs):
        dprops["driveMode"][i] = int(gymapi.DOF_MODE_EFFORT)
        dprops["stiffness"][i] = 0.0
        dprops["damping"][i] = 0.0
        dprops["friction"][i] = 0.1
        dprops["armature"][i] = 0.1
        # helpful if you start controlling:
        dprops["effort"][i] = 20.0
        dprops["velocity"][i] = 3.14
    gym.set_actor_dof_properties(env, actor, dprops)

    # Zero DOF state
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"][:] = 0.0
    dof_states["vel"][:] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

    # Sanity cube (far away)
    box = gym.create_box(sim, 0.05, 0.05, 0.05, gymapi.AssetOptions())
    gym.create_actor(env, box, gymapi.Transform(p=gymapi.Vec3(-0.6, 0.6, 0.3)), "box", 0, -1)

    # Finalize
    gym.prepare_sim(sim)
    cam_pos = gymapi.Vec3(0.8, 0.6, 0.5)
    cam_tgt = gymapi.Vec3(0.0, 0.0, 0.2)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_tgt)

    # Root-state NaN watch (helps catch dynamic-base issues immediately)
    root = gym.acquire_actor_root_state_tensor(sim)
    root_t = gymtorch.wrap_tensor(root)

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # check NaNs once sim is running
        if (root_t.isnan().any().item()):
            print("[ERROR] NaN detected in root state → likely dynamic base without valid inertia or starting interpenetration.")
            break

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
