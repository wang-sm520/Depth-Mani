"""Isaac Lab scene: M1 lying still, AIRBOT Play on its back, a can on the floor, head + wrist cameras.

Import only after the Isaac app has been launched.
"""

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.mjcf")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from airbot_rl.assets import ARM_MJCF, ASSETS, CARPET_USD, FLOOR_CENTER_X, FLOOR_HALF, M1_URDF  # noqa: E402

# All poses are in the env frame: x forward, z up, floor at z=0.
# Lying BASE_LINK height. Three estimates agree on a head lens ~0.185 m above the floor:
# can perspective in the recorded head frames, the demo grasp depth (arm base - 0.128 m
# puts the pads ~2.5 cm below the can top), and the lens sitting level with the wheel tops.
BASE_Z = 0.147
HEAD_CAM_POS = (0.4123, 0.0, BASE_Z + 0.0378)  # vendor sensor table, level, looking +x
ARM_BASE_POS = (HEAD_CAM_POS[0] - 0.25, 0.0, BASE_Z + 0.0695)  # plate flush on the back deck
CAN_RADIUS, CAN_HEIGHT, CAN_MASS = 0.0325, 0.114, 0.25
WALL_AHEAD = 7.0

HEAD_SIZE = (1920, 1080)
HEAD_K = np.array([[793.1139, 0.0, 949.9812], [0.0, 792.7117, 508.7500], [0.0, 0.0, 1.0]])  # IMX415
HEAD_DIST = np.array([-0.108285, 0.013539, -0.000107, -0.0000745, -0.000573])
WRIST_SIZE = (848, 480)
WRIST_K = np.array([[435.3573, 0.0, 418.3114], [0.0, 434.2661, 247.5603], [0.0, 0.0, 1.0]])  # D405 (colour intrinsics)
# D405 pose in link6 (OpenGL), fitted to the closed fingertips in the recorded home-pose wrist frames:
# 14 cm behind / 14 cm above the fingertips, 1 cm left, pitched 29 deg down, 2 deg yaw.
WRIST_CAM_POS = (-0.1399, 0.01, -0.1145)
WRIST_CAM_ROT = (-0.18897, -0.68139, 0.68757, 0.16508)


# Both cameras render at 1/4 of the real resolution: the BC sees 128 x 128, and DA2 on a 480 x 270 head image
# moves the BC chunk by 0.0016 on recorded frames, below the 0.004 frame-to-frame difference.
SCALE = 4


PHYSICS_HZ, CONTROL_HZ = 100, 25
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ


def _pinhole_canvas(K, dist, size, scale=SCALE):
    """Omniverse renders centred pinholes only: a canvas covering the real camera's (undistorted) view, and the
    grid_sample grid (align_corners=False) resampling it onto the real pixel grid downsampled by `scale`."""
    w, h = size[0] // scale, size[1] // scale
    u, v = np.meshgrid((np.arange(w) + 0.5) * scale - 0.5, (np.arange(h) + 0.5) * scale - 0.5)
    xy = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), K, dist).reshape(h, w, 2)
    f = (K[0, 0] + K[1, 1]) / 2 / scale
    canvas = tuple(int(np.ceil(2 * np.abs(xy[..., i]).max() * f)) + 2 for i in range(2))
    return canvas, f, torch.tensor(xy * f * 2 / np.array(canvas), dtype=torch.float32)


HEAD_CANVAS, HEAD_F, HEAD_GRID = _pinhole_canvas(HEAD_K, HEAD_DIST, HEAD_SIZE)
WRIST_CANVAS, WRIST_F, WRIST_GRID = _pinhole_canvas(WRIST_K, None, WRIST_SIZE)  # D405 depth: distortion ignored
HEAD_GAIN = 0.65  # matches the sim head image's mean grey to the recorded frames (118)


def _camera(canvas, f, near):
    # An explicit focal length: the default 1/width scaling yields tiny values Omniverse clamps to ~89 deg HFOV.
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        [f, 0.0, canvas[0] / 2, 0.0, f, canvas[1] / 2, 0.0, 0.0, 1.0], width=canvas[0], height=canvas[1],
        clipping_range=(near, 30.0), focal_length=24.0)


def _resample(image, grid, mode):
    return F.grid_sample(image, grid.to(image.device)[None].expand(len(image), -1, -1, -1), mode=mode,
                         align_corners=False)


def head_image(rgb):
    """Rendered uint8 [N,H,W,3] head canvas -> the real head image at 1/SCALE, with its exposure."""
    image = _resample(rgb.permute(0, 3, 1, 2).float(), HEAD_GRID, "bilinear") * HEAD_GAIN
    return image.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def wrist_depth(depth):
    """Rendered [N,H,W,1] distance_to_image_plane -> [N,h,w] metres on the D405 grid at 1/SCALE.
    Nearest sampling: no blended depths across the gripper's edges."""
    return _resample(depth.permute(0, 3, 1, 2), WRIST_GRID, "nearest")[:, 0]


ARM_LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "left", "right")
DOG_LINKS = ("BASE_LINK",) + tuple(f"{leg}_{part}_LINK" for leg in ("FAR", "FBL", "RAR", "RBL")
                                   for part in ("ABAD", "HIP", "KNEE", "FOOT"))
LYING_LEGS = {"F.._HIP_JOINT": 1.2, "F.._KNEE_JOINT": -2.75, "R.._HIP_JOINT": -1.2, "R.._KNEE_JOINT": 2.75}


@sim_utils.clone
def _spawn_mounted_arm(prim_path, cfg, translation=None, orientation=None):
    """Fix the imported root at its installation pose before cloning."""
    prim = sim_utils.spawn_from_mjcf(prim_path, cfg.replace(articulation_props=None), translation, orientation)
    stage = prim.GetStage()
    # The MJCF importer also marks an empty sibling as an articulation root.
    world_body = stage.GetPrimAtPath(f"{prim_path}/worldBody")
    world_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    world_body.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    # The origin-anchored root joint is replaced by Isaac Lab's fix_root_link joint at the mount pose.
    stage.GetPrimAtPath(f"{prim_path}/joints/rootJoint_arm_base").SetActive(False)
    sim_utils.modify_articulation_root_properties(f"{prim_path}/arm_base/arm_base", cfg.articulation_props)
    # The bolted mount plate overlaps the coarse dog body collider.
    root_body = stage.GetPrimAtPath(f"{prim_path}/arm_base/arm_base")
    UsdPhysics.FilteredPairsAPI.Apply(root_body).CreateFilteredPairsRel().AddTarget(
        f"{prim_path.rsplit('/', 1)[0]}/Dog/BASE_LINK")
    return prim


@configclass
class CanSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 20.0  # wider than the floor, so no other env shows up in either camera
    # Collision floor (invisible) plus the visual carpet: the recorded carpet tile, tiled (assets.build_carpet).
    floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(size=(2 * FLOOR_HALF, 2 * FLOOR_HALF, 0.02), visible=False,
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(FLOOR_CENTER_X, 0.0, -0.01)))
    carpet = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Carpet", spawn=sim_utils.UsdFileCfg(usd_path=str(CARPET_USD)),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(FLOOR_CENTER_X, 0.0, 0.0)))
    # The recording room's white wall: the head view puts the wall-floor line ~7 m ahead of the lens.
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2 * FLOOR_HALF, 3.0),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.88))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(HEAD_CAM_POS[0] + WALL_AHEAD + 0.05, 0.0, 1.5)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))

    dog = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Dog",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(M1_URDF), usd_dir=str(ASSETS / "usd/m1"), fix_base=True, merge_fixed_joints=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1000.0, damping=50.0))),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_Z), joint_pos={".*(ABAD|FOOT)_JOINT": 0.0, **LYING_LEGS}),
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1000.0, damping=50.0)})

    # Drives identified against the recordings (each action held 40 ms): no gravity (the real controller
    # compensates it; static error 0.00054 vs real 0.00064 rad), plus env.COMMAND_DELAY. Validation MAE 0.0039 rad.
    arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Arm",
        articulation_root_prim_path="/arm_base",
        spawn=sim_utils.MjcfFileCfg(
            func=_spawn_mounted_arm, asset_path=str(ARM_MJCF), usd_dir=str(ASSETS / "usd/arm"), fix_base=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            activate_contact_sensors=True, import_sites=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=ARM_BASE_POS, joint_pos={".*": 0.0}),
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["joint[1-3]"], stiffness=1000.0, damping=40.0,
                                            effort_limit_sim=24.0),
            "wrist": ImplicitActuatorCfg(joint_names_expr=["joint[4-6]"], stiffness=80.0, damping=5.0,
                                         effort_limit_sim=8.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["end.*"], stiffness=1000.0, damping=20.0,
                                           effort_limit_sim=15.0)})

    can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Can",
        spawn=sim_utils.MeshCylinderCfg(
            radius=CAN_RADIUS, height=CAN_HEIGHT, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CAN_MASS), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.70, 0.05))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(HEAD_CAM_POS[0] + 0.22, 0.0, CAN_HEIGHT / 2)))  # reset moves it

    # Rendered on centred canvases; head_image() / wrist_depth() map them onto the real cameras' pixels.
    head_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/head_cam", data_types=["rgb"], width=HEAD_CANVAS[0], height=HEAD_CANVAS[1],
        spawn=_camera(HEAD_CANVAS, HEAD_F, 0.02), offset=TiledCameraCfg.OffsetCfg(pos=HEAD_CAM_POS, convention="world"))
    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Arm/arm_base/link6/wrist_cam", data_types=["distance_to_image_plane"],
        width=WRIST_CANVAS[0], height=WRIST_CANVAS[1], spawn=_camera(WRIST_CANVAS, WRIST_F, 0.005),
        offset=TiledCameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=WRIST_CAM_ROT, convention="opengl"))

    def __post_init__(self):
        # No-go zones = an arm link touching the dog or the floor. Filtered contacts need one body per sensor.
        for link in ARM_LINKS:
            setattr(self, f"contact_{link}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Arm/arm_base/{link}",
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Dog/{name}" for name in DOG_LINKS] + ["{ENV_REGEX_NS}/Floor"],
                history_length=SUBSTEPS))
