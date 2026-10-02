"""Offline asset builders: the flattened AIRBOT MJCF, a carpet floor textured from the recorded head video,
and the can start positions. Run from the project root: python -m airbot_rl.assets"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

THIRD_PARTY = Path("/home/user/locomotion_zh/ATEC-locomotion/third_party")
M1_URDF = THIRD_PARTY / "genisom-m1-model/urdf/ZG_M1_A0_V1_0.urdf"
P6 = THIRD_PARTY / "p6_model"
ASSETS = Path(__file__).resolve().parent / "assets"
ARM_MJCF = ASSETS / "airbot_play.xml"
CARPET_USD = ASSETS / "carpet.usda"
CAN_POSITIONS = ASSETS / "can_positions_fk.json"
RECORDING = Path("/home/user/wang-sm/pi0.5/openpi_v0.2.0/data/pick_can_bd/2026-09-29/session_000001")

# Head lens in the env frame (x forward, height above the floor); matches scene.HEAD_CAM_POS.
LENS_X, LENS_Z = 0.4123, 0.1848
HEAD_K = np.array([[793.1139, 0, 949.9812], [0, 792.7117, 508.75], [0, 0, 1]])
HEAD_DIST = np.array([-0.108285, 0.013539, -0.000107, -0.0000745, -0.000573])
# One 0.5 m carpet tile, cut where the recorded head view of episode 0 is sharp and free of the can:
# 0.31-0.81 m ahead of the lens, 0.01-0.49 m to its right. Real tile seams sit ~0.33/0.83 m ahead.
TILE, TILE_AHEAD, TILE_RIGHT = 0.5, (0.31, 0.81), (0.01, 0.49)
FLOOR_CENTER_X, FLOOR_HALF = 2.0, 8.0  # same footprint as the collision floor; reaches the back wall


def build_arm_mjcf():
    """Drop tendon/equality/actuators (Isaac Lab owns the drives) and the upstream eye_arm camera."""
    import mujoco

    (ASSETS / "meshes").mkdir(parents=True, exist_ok=True)
    link = ASSETS / "meshes" / "airbot_play"
    if not link.exists():
        link.symlink_to(P6 / "meshes")
    mjcf = P6 / "mjcf"
    source = ASSETS / "source.xml"
    source.write_text(f"""<mujoco model="airbot_play">
  <compiler angle="radian" meshdir="{ASSETS / "meshes"}"/>
  <include file="{mjcf / "airbot_play_dependencies.xml"}"/>
  <worldbody><include file="{mjcf / "airbot_play.xml"}"/></worldbody>
</mujoco>""")
    mujoco.mj_saveLastXML(str(ARM_MJCF), mujoco.MjModel.from_xml_path(str(source)))
    source.unlink()
    tree = ET.parse(ARM_MJCF)
    tree.getroot().find("compiler").set("meshdir", "meshes/")
    for body in tree.getroot().iter("body"):
        for child in body.findall("body"):
            if child.find("camera") is not None:
                body.remove(child)
    tree.write(ARM_MJCF)
    return ARM_MJCF


def _recorded_head_frame(episode=0, offset_s=0.1):
    import av
    import pandas as pd

    key = "observation.images.head"
    meta = pd.read_parquet(RECORDING / "meta/episodes/chunk-000/file-000.parquet")
    row = meta[meta.episode_index == episode].iloc[0]
    start = float(row[f"videos/{key}/from_timestamp"]) + offset_s
    path = RECORDING / f"videos/{key}/chunk-000/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        return next(f for f in container.decode(stream) if f.time >= start - 0.02).to_ndarray(format="rgb24")


def build_carpet(size_px=512):
    """Rectify one carpet tile from the recorded head video and write a tiled, textured floor USD."""
    import cv2

    # Texture row 0 = far edge, column 0 = left edge (the tile's lens-relative footprint).
    ahead = np.linspace(TILE_AHEAD[1], TILE_AHEAD[0], size_px)
    right = np.linspace(TILE_RIGHT[0], TILE_RIGHT[1], size_px)
    x, y = np.meshgrid(ahead, right, indexing="ij")
    points = np.stack([y, np.full_like(x, LENS_Z), x], -1).reshape(-1, 3)  # OpenCV camera frame
    uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), HEAD_K, HEAD_DIST)
    uv = uv.reshape(size_px, size_px, 2).astype(np.float32)
    tile = cv2.remap(_recorded_head_frame(), uv[..., 0], uv[..., 1], cv2.INTER_AREA).astype(np.float32)
    # Remove the recorded light falloff across the tile so repeats do not form a brightness checkerboard.
    tile = np.clip(tile / cv2.GaussianBlur(tile, (0, 0), size_px / 8) * tile.mean((0, 1)), 0, 255).astype(np.uint8)
    ASSETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ASSETS / "carpet.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    # st = (metres to the right of the tile's left edge, metres ahead of its near edge) / TILE, so a wrapped
    # texture repeats the tile exactly; corners are in the env frame, points relative to the floor centre.
    corners = [(FLOOR_CENTER_X + sx * FLOOR_HALF, sy * FLOOR_HALF) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    st = [((-cy - TILE_RIGHT[0]) / TILE, (cx - LENS_X - TILE_AHEAD[0]) / TILE) for cx, cy in corners]
    pts = ", ".join(f"({cx - FLOOR_CENTER_X}, {cy}, 0)" for cx, cy in corners)
    sts = ", ".join(f"({u:.6f}, {v:.6f})" for u, v in st)
    looks = "</Carpet/Looks/Carpet"
    CARPET_USD.write_text(f"""#usda 1.0
(
    defaultPrim = "Carpet"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Carpet"
{{
    def Mesh "Plane" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [{pts}]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
        texCoord2f[] primvars:st = [{sts}] (interpolation = "vertex")
        rel material:binding = {looks}>
    }}

    def Scope "Looks"
    {{
        def Material "Carpet"
        {{
            token outputs:surface.connect = {looks}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = {looks}/Texture.outputs:rgb>
                float inputs:roughness = 1
                float inputs:metallic = 0
                token outputs:surface
            }}
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @carpet.png@
                float2 inputs:st.connect = {looks}/St.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
""")
    return CARPET_USD


def _grasp_points(states):
    """Finger-pad midpoint (arm-base frame) for recorded [N,7] native states, by MuJoCo forward kinematics."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ARM_MJCF))
    data = mujoco.MjData(model)
    joints = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
    fingers = [model.joint(name).qposadr[0] for name in ("endleft", "endright")]
    pads = [model.body(name).id for name in ("left", "right")]
    points = []
    for state in states:
        data.qpos[:] = 0
        data.qpos[joints], data.qpos[fingers] = state[:6], (state[6] / 2, -state[6] / 2)
        mujoco.mj_kinematics(model, data)
        points.append(np.mean([data.xpos[b] + data.xmat[b].reshape(3, 3) @ (0, 0, 0.005) for b in pads], 0))
    return np.array(points)


def find_grasp_frame(width):
    """First frame >= 1 cm closed from the widest opening whose next two widths change < 1 mm, else None."""
    widest = int(np.argmax(width))
    for t in range(widest, len(width) - 2):
        if width[widest] - width[t] >= 0.01 and np.abs(np.diff(width[t:t + 3])).max() < 0.001:
            return t
    return None


def build_can_positions(data="data/airbot-can100-da2-20260929", arm_base=(0.1623, 0.0, 0.2165)):
    """Can start (env frame) per recording = FK grasp midpoint at the grasp frame, where the demo closed on it.
    Head-image measurements disagree with this by 2.2 cm median (7.7 cm max); FK is what the arm reached.
    Recordings whose gripper never closes (65, 76) or never grasp a grounded can (10) are skipped.
    arm_base = scene.ARM_BASE_POS (importing scene needs Isaac)."""
    import json

    import h5py

    manifest = json.loads((Path(data) / "manifest.json").read_text())
    positions = []
    for episode in manifest["episodes"]:
        with h5py.File(Path(data) / episode["path"]) as stream:
            states = stream["state"][:]
        frame = find_grasp_frame(states[:, 6])
        if frame is None or episode["episode_index"] == 10:
            continue
        x, y = _grasp_points(states[frame:frame + 1])[0, :2] + arm_base[:2]
        positions.append({"episode": episode["episode_index"], "grasp_frame": frame, "x": round(float(x), 6),
                          "y": round(float(y), 6)})
    CAN_POSITIONS.write_text(json.dumps(positions, indent=1))
    return CAN_POSITIONS


if __name__ == "__main__":
    print(build_arm_mjcf(), build_carpet(), build_can_positions())
