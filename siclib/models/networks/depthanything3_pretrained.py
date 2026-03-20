"""Wrapper for Depth Anything 3 model to estimate focal length."""

import torch
import numpy as np
from depth_anything_3.api import DepthAnything3

from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.gravity import Gravity
from siclib.models import BaseModel
from PIL import Image


class DepthAnything3Focal(BaseModel):
    """Depth Anything 3 model for focal length estimation."""

    default_conf = {
        "model_name": "depth-anything/DA3NESTED-GIANT-LARGE",
        "device": "cuda",
        "show_scene": False,
    }

    required_data_keys = ["path"]

    def _init(self, conf):
        """Initialize Depth Anything 3 model."""
        device = torch.device(conf["device"])
        self.device = device
        self.model = DepthAnything3.from_pretrained(conf["model_name"])
        self.model = self.model.to(device=device)

    def _forward(self, data):
        """Forward pass to estimate focal length."""

        assert len(data["path"]) == 1, "Only batch size of 1 is supported."

        path = data["path"][0]

        # Inference
        prediction = self.model.inference([path])

        # intrinsics: [N, 3, 3]
        K = prediction.intrinsics[0]  # shape (3,3)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # f = (fx + fy) / 2.0
        f = fx

        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        h = torch.tensor([H], dtype=torch.float32, device=self.device)
        w = torch.tensor([W], dtype=torch.float32, device=self.device)
        f_tensor = torch.tensor([f], dtype=torch.float32, device=self.device)

        camera = BaseCamera.from_dict({
            "height": h,
            "width": w,
            "f": f_tensor,
        })

        gravity = Gravity.from_rp([0.0], [0.0])

        return {
            "camera": camera,
            "gravity": gravity,
        }

    def loss(self, pred, data):
        return {}, {}


if __name__ == "__main__":
    from pathlib import Path

    path = "/mnt/nas_9/group/chenguyuan/GeoCalib/data/monovo2k/images/sequence_35_00008.jpg"

    model = DepthAnything3Focal({})
    output = model({"path": [path]})
    print(output["camera"].K)
    print(output["camera"].vfov)
    print(output["camera"].size)