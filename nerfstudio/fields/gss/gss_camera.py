from dataclasses import dataclass

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.fields.gss.utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


@dataclass
class GssCamera:
    R: np.ndarray
    T: np.ndarray
    focalX: float
    focalY: float
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor

    @staticmethod
    def convert_from(camera: Cameras, device: str, xshift: float=0, yshift: float=0):
        H, W = int(camera.height), int(camera.width)

        fovx = focal2fov(camera.fx, W)
        fovy = focal2fov(camera.fy, H)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3] = camera.camera_to_worlds.cpu().numpy()
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        xshift *= W / 2 / camera.fx * 0.01
        yshift *= H / 2 / camera.fy * 0.01

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(device)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fovx, fovY=fovy, xshift=xshift, yshift=yshift).transpose(0, 1).to(device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        return GssCamera(
            R,
            T,
            float(camera.fx),
            float(camera.fy),
            fovx,
            fovy,
            W,
            H,
            world_view_transform,
            projection_matrix,
            full_proj_transform,
            camera_center,
        )


def get_gss_radius(cameras: Cameras):
    def get_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return diagonal

    cam_info = []
    for i in range(len(cameras)):
        cam_info.append(GssCamera.convert_from(cameras[i], "cpu"))

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    diagonal = get_diag(cam_centers)
    radius = diagonal * 1.1

    return radius
