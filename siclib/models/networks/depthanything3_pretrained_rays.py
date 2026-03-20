"""Wrapper for Depth Anything 3 model to estimate focal length and rays."""

import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3
from PIL import Image

from siclib.models import BaseModel


class DepthAnything3Focal(BaseModel):
    """Depth Anything 3 model for focal length estimation."""

    default_conf = {
        "model_name": "depth-anything/DA3NESTED-GIANT-LARGE",
        "device": "cuda",
    }

    required_data_keys = ["path"]

    def _init(self, conf):
        self.device = torch.device(conf["device"])
        self.model = DepthAnything3.from_pretrained(conf["model_name"])
        self.model = self.model.to(device=self.device)

    def _forward(self, data):

        assert len(data["path"]) == 1, "Only batch size of 1 is supported."

        path = data["path"][0]
        cam_id = data.get("cam_id", ["pinhole"])[0]

        # 读取原图尺寸
        if "image" in data:
            ho, wo = data["image"].shape[-2:]
        else:
            ho, wo = Image.open(path).size[::-1]

        # ========== Depth Anything 3 推理 ==========
        prediction = self.model.inference([path])

        # 取第一个 batch
        K = prediction.intrinsics[0]  # (3,3)
        depth = prediction.depth[0]  # (H,W)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # intrinsics tensor
        if "simple" in cam_id:
            intrinsics = torch.tensor([fx, cx, cy], device=self.device)
        else:
            # intrinsics = torch.tensor([fx, fy, cx, cy], device=self.device)
            intrinsics = torch.tensor([fx, fx, cx, cy], device=self.device)

        intrinsics = intrinsics[None]  # (1,3) or (1,4)

        # ========== 构造 rays ==========
        # H, W = depth.shape

        # # 像素网格
        # u = torch.arange(W, device=self.device)
        # v = torch.arange(H, device=self.device)
        # grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

        # # 转换为 camera 坐标
        # x = (grid_u - cx) / fx
        # y = (grid_v - cy) / fy
        # z = torch.ones_like(x)

        # rays = torch.stack([x, y, z], dim=0)  # (3,H,W)
        # rays = F.normalize(rays, dim=0)

        # rays = rays.view(3, -1).permute(1, 0).contiguous()  # (H*W,3)
        # rays = rays[None]  # (1,H*W,3)

        # return {
        #     "intrinsics": intrinsics,
        #     "rays": rays,
        # }
        return {
            "intrinsics": intrinsics,
        }

    def loss(self, pred, data):
        raise NotImplementedError


if __name__ == "__main__":

    path = "/mnt/nas_9/group/chenguyuan/GeoCalib/data/tartanair/images/abandonedfactory_night-abandonedfactory_night-Easy-P001-image_left-000059_left.png"

    model = DepthAnything3Focal({})
    output = model({"path": [path], "cam_id": ["pinhole"]})

    print(output["intrinsics"])
    # print(output["rays"].shape)