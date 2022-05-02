# modify from PointGroup
# Written by Li Jiang
import os
import os.path as osp
import logging
from typing import Optional
from operator import itemgetter
from copy import deepcopy

import gorilla
import torch
import numpy as np
import open3d as o3d
import colorsys
import matplotlib.colors

preset_colors  = ["#fabed4", "#FF6D00", "#00C853", "#0091EA",  "#00ced1", "#D50000", "#FFD600",
              "#673AB7", "#3F51B5", "#795548", "#AEEA00", "#009688", "#ba55d3", "#e9967a", "#607D8B", "#E91E63",
              "#7fff00", "#AA00FF"]
preset_colors = [list(matplotlib.colors.to_rgb(color)) for color in preset_colors]


color_cache = {}
sem_color_cache = {}

def increase_s(r, g, b, adjust_value):
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s+adjust_value)


def get_inst_color(sem_label, color_len):
    if sem_label not in sem_color_cache:
        sem_color_cache[sem_label] = preset_colors[sem_label].copy()
        color = sem_color_cache[sem_label]
    else:
        adjust_value = +0.5 / color_len if color_len > 1 else 0
        sem_color_cache[sem_label][0], sem_color_cache[sem_label][1], sem_color_cache[sem_label][2] = increase_s(sem_color_cache[sem_label][0], sem_color_cache[sem_label][1], sem_color_cache[sem_label][2], adjust_value)
        color = sem_color_cache[sem_label]
    
    for i in range(len(color)):
        color[i] = min(1, color[i])
        color[i] = max(0, color[i])
    #color_cache[(sem_label, instance_id)] = color
    return color
# COLOR28 = np.array((
#                 (174, 199, 232),
#                 (152, 223, 138),
#                 (31, 119, 180),
#                 (255, 187, 120),
#                 (188, 189, 34),
#                 (140, 86, 75),
#                 (255, 152, 150),
#                 (214, 39, 40),
#                 (197, 176, 213),
#                 (148, 103, 189),
#                 (196, 156, 148),
#                 (23, 190, 207),
#                 (178, 76, 76),
#                 (247, 182, 210),
#                 (66, 188, 102),
#                 (219, 219, 141),
#                 (140, 57, 197),
#                 (202, 185, 52),
#                 (51, 176, 203),
#                 (200, 54, 131),
#                 (92, 193, 61),
#                 (78, 71, 183),
#                 (172, 114, 82),
#                 (255, 127, 14),
#                 (91, 163, 138),
#                 (153, 98, 156),
#                 (140, 153, 101),
#                 (158, 218, 229)
#                 ))

SEMANTIC_IDXS = np.array(range(1, 21))
SEMANTIC_NAMES = np.array(['floor', 'ceiling', 'wall', 'door', 'table', 'chair', 'cabinet', 'window', 'sofa', 'microwave', 'pillow',
'tv_monitor', 'curtain', 'trash_can', 'suitcase', 'sink', 'backpack', 'bed', 'refrigerator','toilet'])
# CLASS_COLOR = {
#     "unannotated": (0, 0, 0),
#     "floor": (174, 199, 232),
#     "ceiling": (152, 223, 138),
#     "wall": (31, 119, 180),
#     "door": (255, 187, 120),
#     "table": (188, 189, 34),
#     "chair": (140, 86, 75),
#     "cabinet": (255, 152, 150),
#     "window": (214, 39, 40),
#     "sofa": (197, 176, 213),
#     "microwave": (148, 103, 189),
#     "pillow": (196, 156, 148),
#     "tv_monitor": (23, 190, 207),
#     "curtain": (178, 76, 76),
#     "trash_can": (247, 182, 210),
#     "suitcase": (66, 188, 102),
#     "sink": (219, 219, 141),
#     "backpack": (140, 57, 197),
#     "bed": (202, 185, 52),
#     "refrigerator": (51, 176, 203),
#     "toilet": (200, 54, 131)
# }
SEMANTIC_IDX2NAME = {
    1: "floor",
    2: "ceiling",
    3: "wall",
    4: "door",
    5: "table",
    6: "chair",
    7: "cabinet",
    8: "window",
    9: "sofa",
    10: "microwave",
    11: "pillow",
    12: "tv_monitor",
    13: "curtain",
    14: "trash_can",
    15: "suitcase",
    16: "sink",
    17: "backpack",
    18: "bed",
    19: "refrigerator",
    20: "toilet"
}


def visualize_instance_mask(clusters: np.ndarray,
                            room_name: str,
                            visual_dir: str,
                            data_root: str,
                            cluster_scores: Optional[np.ndarray] = None,
                            semantic_pred: Optional[np.ndarray] = None,
                            color: int = 28,
                            **kwargs):
    global color_cache
    global sem_color_cache
    color_cache = {}
    sem_color_cache = {}
    logger = gorilla.derive_logger(__name__)
    # colors = globals()[f"COLOR{color}"]
    mesh_file = osp.join(data_root, room_name, room_name + "_clean.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    points = np.array(pred_mesh.vertices)
    inst_label_pred_rgb = np.asarray(mesh.vertex_colors)  # np.ones(rgb.shape) * 255 #
    logger.info(f"room_name: {room_name}")
    inst_tmp = np.zeros(len(inst_label_pred_rgb))
    for cluster_id, cluster in enumerate(clusters):
        if logger is not None:
            # NOTE: remove the handlers are not FileHandler to avoid
            #       outputing this message on console(StreamHandler)
            #       and final will recover the handlers of logger
            handler_storage = []
            for handler in logger.handlers:
                if not isinstance(handler, logging.FileHandler):
                    handler_storage.append(handler)
                    logger.removeHandler(handler)
            message = f"{cluster_id:<4}: pointnum: {int(cluster.sum()):<7} "
            if semantic_pred is not None:
                semantic_label = np.argmax(
                    np.bincount(semantic_pred[np.where(cluster == 1)[0]]))
                semantic_id = int(SEMANTIC_IDXS[semantic_label])
                # if semantic_id in [1, 2, 3]:
                #     continue
                semantic_name = SEMANTIC_IDX2NAME[semantic_id]
                message += f"semantic: {semantic_id:<3}-{semantic_name:<15} "
            if cluster_scores is not None:
                score = float(cluster_scores[cluster_id])
                message += f"score: {score:.4f} "
            logger.info(message)
            for handler in handler_storage:
                logger.addHandler(handler)
        
        inst_tmp[cluster == 1] = cluster_id + 1
        
    
    for i in np.unique(inst_tmp):
        if i == 0:
            continue
        idx = inst_tmp == i
        sem_idx = semantic_pred == (semantic_pred[idx][0])
        color_len = len(np.unique(inst_tmp[sem_idx]))
        inst_label_pred_rgb[idx] = get_inst_color(semantic_pred[idx][0], color_len)

    rgb = inst_label_pred_rgb
    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    # points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    o3d.io.write_triangle_mesh(osp.join(visual_dir, room_name + ".ply"), pred_mesh)


# TODO: add the semantic visualization


def visualize_pts_rgb(rgb, room_name, data_root, output_dir, mode="test"):
    split = "scans"
    mesh_file = osp.join(data_root, split, room_name,
                         room_name + "_clean.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255)
    points = np.array(pred_mesh.vertices)
    # points[:, 2] += 3
    points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh += pred_mesh
    o3d.io.write_triangle_mesh(osp.join(output_dir, room_name + ".ply"), mesh)


def get_coords_color(data_root: str,
                     result_root: str,
                     room_split: str,
                     room_name: str,
                     task: str = "instance_pred"):
    input_file = os.path.join(data_root, room_split,
                              room_name + "_inst_nostuff.pth")
    assert os.path.isfile(input_file), f"File not exist - {input_file}."
    if "test" in room_split:
        xyz, rgb, edges, scene_idx = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (task == "semantic_gt"):
        assert "test" not in room_split
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (task == "instance_gt"):
        assert "test" not in room_split
        label = label.astype(np.int)
        inst_label = inst_label.astype(np.int)
        print(f"Instance number: {inst_label.max() + 1}")
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        sem_idx = label == label[object_idx][0]
        color_len = len(np.unique(inst_label[sem_idx]))
        inst_label_rgb[object_idx] = get_inst_color(label[object_idx][0], color_len)
        rgb = inst_label_rgb

    elif (task == "semantic_pred"):
        assert room_split != "train"
        semantic_file = os.path.join(result_root, room_split, "semantic",
                                     room_name + ".npy")
        assert os.path.isfile(
            semantic_file), f"No semantic result - {semantic_file}."
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(
            itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (task == "instance_pred"):
        assert room_split != "train"
       
        semantic_file = os.path.join(result_root, room_split, "semantic",
                                     room_name + ".npy")
        assert os.path.isfile(
            semantic_file), f"No semantic result - {semantic_file}."
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        instance_file = os.path.join(result_root, room_split,
                                     room_name + ".txt")
        assert os.path.isfile(
            instance_file), f"No instance result - {instance_file}."
        f = open(instance_file, "r")
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(result_root, room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            # print(
            #     f"{i} {masks[i][2]}: {SEMANTIC_IDX2NAME[int(masks[i][1])]} pointnum: {mask.sum()}"
            # )
            sem_idx = label_pred == label_pred[mask == 1][0]
            color_len = len(np.unique(inst_label_pred_rgb[sem_idx]))
            inst_label_pred_rgb[mask == 1] = get_inst_color(label_pred[mask == 1], color_len)
        rgb = inst_label_pred_rgb

    if "test" not in room_split:
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


def visualize_instance_mask_lite(
    clusters: np.ndarray,
    points: np.ndarray,
    visual_path: str,
    color: int = 28,
):
    colors = globals()[f"COLOR{color}"]
    inst_label_pred_rgb = np.zeros_like(points)  # np.ones(rgb.shape) * 255 #
    for cluster_id, cluster in enumerate(clusters):
        inst_label_pred_rgb[cluster == 1] = colors[cluster_id % len(colors)]
    rgb = inst_label_pred_rgb

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(rgb / 255)
    o3d.io.write_point_cloud(visual_path, pc)
