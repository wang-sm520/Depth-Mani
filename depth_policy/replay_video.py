import cv2
import numpy as np


def labeled_tile(image, label):
    tile = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
    header = np.full((32, 384, 3), 24, dtype=np.uint8)
    cv2.putText(header, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([header, tile])


def color_depth(depth, maximum):
    normalized = np.clip(depth / maximum, 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def replay_frame(observation, index, time_seconds, instruction, phase, success=False):
    tiles = [
        labeled_tile(observation["rgb_agentview"], "External RGB (debug only)"),
        labeled_tile(color_depth(observation["depth_agentview"], 3.2), "External depth: 0-3.2 m"),
        labeled_tile(observation["rgb_wrist"], "Wrist RGB (debug only)"),
        labeled_tile(color_depth(observation["depth_wrist"], 0.5), "Wrist depth: 0-0.5 m"),
    ]
    grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
    footer = np.full((64, grid.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(footer, instruction, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
    status = "SUCCESS (environment predicate)" if success else phase
    cv2.putText(footer, f"Step {index} | sim {time_seconds:.2f}s | {status}", (10, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (110, 240, 150), 1, cv2.LINE_AA)
    return np.concatenate([grid, footer])
