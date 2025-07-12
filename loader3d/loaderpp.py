# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Adapted by Ayca Takmaz, July 2023

import bisect
import copy
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d


def scaling_mapping(mapping, a, b, c, d):
    # Calculate scaling factors
    scale_x = c / a
    scale_y = d / b

    mapping[:, 0] = mapping[:, 0] * scale_x
    mapping[:, 1] = mapping[:, 1] * scale_y
    return mapping


class ScanNetPPReader(object):
    def __init__(
        self,
        root_path,
        cfg = None,
    ):
        self.root_path = root_path
        self.scene_id = os.path.basename(root_path)  # til iphone

        self.depth_folder = os.path.join(self.root_path, "depth")
        self.image_folder = os.path.join(self.root_path, "images")
        if not os.path.exists(self.depth_folder):
            self.frame_ids = []
        else:
            depth_images = os.listdir(self.depth_folder)
            color_images = os.listdir(self.image_folder)
            self.frame_ids = sorted([x.split(".")[0] for x in color_images])


        print("Number of original frames:", len(self.frame_ids))

        self.depth_scale = 1000.0

        # self.scene_pcd_path = os.path.join(cfg.data.original_ply, f"{self.scene_id}_vh_clean_2.ply")
        # # video_path / f"{args.video}_vh_clean_2.ply"

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def read_depth(self, depth_path):

        depth_image = cv2.imread(depth_path, -1)
        depth_image = depth_image / self.depth_scale  # rescale to obtain depth in meters

        return depth_image

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def read_pose(self, pose_path, raw_frame_id):
        tmp = json.load(open(pose_path))
        pose = np.array(tmp[raw_frame_id]["aligned_pose"])
        inst = np.array(tmp[raw_frame_id]["intrinsic"])
        return pose, inst

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        frame["raw_frame_id"] = "frame_" + frame_id.split("_")[1].zfill(6)
        fnamedepth = "{}.png".format(frame["raw_frame_id"])
        fnamecolor = "{}.jpg".format(frame["frame_id"])

        depth_image_path = os.path.join(self.depth_folder, fnamedepth)
        image_path = os.path.join(self.image_folder, fnamecolor)
        pose_path = os.path.join(self.root_path, "pose_intrinsic_imu.json")

        frame["depth_path"] = depth_image_path
        frame["image_path"] = image_path
        frame["pose_path"] = pose_path
        return frame
