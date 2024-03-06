import math

import numpy as np
import torch
import viser.transforms as vtf
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.fields.gss.gaussian_model import GaussianModel


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


def render(
    gaussians: GaussianModel,
    camera: Cameras,
    color: Tensor,
) -> dict[str, Tensor]:
    camera = camera.to(gaussians.get_xyz.device)

    # shift the camera to center of scene looking at center
    R = camera.camera_to_worlds[:3, :3]  # 3 x 3
    T = camera.camera_to_worlds[:3, 3:4]  # 3 x 1
    # flip the z axis to align with gsplat conventions
    R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=R.device, dtype=R.dtype)
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    # calculate the FOV of the camera given fx and fy, width and height
    cx = camera.cx.item()
    cy = camera.cy.item()
    fovx = 2 * math.atan(camera.width / (2 * camera.fx))
    fovy = 2 * math.atan(camera.height / (2 * camera.fy))
    projmat = projection_matrix(0.01, 100, fovx, fovy, device=camera.device)

    W, H = camera.width.item(), camera.height.item()
    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = (
        (W + BLOCK_X - 1) // BLOCK_X,
        (H + BLOCK_Y - 1) // BLOCK_Y,
        1,
    )

    xys, depths, radii, conics, num_tiles_hit, _ = project_gaussians(
        gaussians.get_xyz,
        gaussians.get_scaling,
        1,
        gaussians.get_rotation,
        viewmat.squeeze()[:3, :],
        projmat.squeeze() @ viewmat.squeeze(),
        camera.fx.item(),
        camera.fy.item(),
        cx,
        cy,
        H,
        W,
        tile_bounds,
    )

    if xys.requires_grad:
        xys.retain_grad()

    out_img = rasterize_gaussians(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        color,
        gaussians.get_opacity,
        H,
        W,
        background=torch.zeros_like(color[0]),
    )

    if not xys.requires_grad:
        out_depth = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            depths[..., None],
            gaussians.get_opacity,
            H,
            W,
            background=torch.zeros_like(depths[0, None]),
        )
    else:
        out_depth = None

    return {
        "render": out_img,
        "depth": out_depth,
        "radii": radii,
        "viewspace_points": xys,
        "visibility_filter": radii > 0,
    }
