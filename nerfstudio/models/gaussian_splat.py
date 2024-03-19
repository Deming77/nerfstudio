import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Type

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.image import PeakSignalNoiseRatio

import nerfstudio.fields.gss.gsplat_renderer as gsplat_renderer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.fields.gss.appearance_model import GssAppearanceModel, GssAppearanceModelConfig
from nerfstudio.fields.gss.gaussian_model import GaussianModel
from nerfstudio.fields.gss.gaussian_renderer import RenderConfig, render
from nerfstudio.fields.gss.gss_camera import GssCamera, get_gss_radius
from nerfstudio.fields.gss.utils.graphics_utils import BasicPointCloud
from nerfstudio.fields.gss.utils.loss_utils import ssim
from nerfstudio.fields.gss.utils.sh_utils import SH2RGB
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.timer import timer
from typing import Union, Dict

@dataclass
class GaussianSplatConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: GaussianSplatModel)
    enable_collider: bool = False
    """Overrides base model"""
    appearance: GssAppearanceModelConfig = field(default_factory=GssAppearanceModelConfig)
    renderer: Literal["gss", "gsplat"] = "gss"
    rendering: Literal["gaussian", "pixel"] = "gaussian"
    """How pixel colors are rendererd.
        gaussian: decode per-Gaussian color first, then rasterize
        pixel: rasterize a feature map, then decode per-pixel
    """

    sh_degree: int = 3
    ply_file: Union[str, None] = None
    npz_file: Union[str, None] = None
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002

    depth_loss_type: str = "none"
    lambda_depth_loss_start: float = 0.05
    lambda_depth_loss_full: float = 0.001
    depth_loss_start_step: int = 500
    depth_loss_full_step: int = 10000
    proximity_end_step: int = 0

    pseudo_loss_start_step: int = 1000
    pseudo_loss_end_step: int = 0
    pseudo_loss_max_factor: float = 1
    pseudo_loss_interval: int = 10

    sky_loss_type: Literal["render_ones", "render_ones_opaque"] = "render_ones"
    sky_loss_lambda: float = 0.1

    num_initial_points: int = 100000
    initial_points_range: float = 1.3

    mip_kernel_size: float = 0
    mip_filter_update_until_iter: int = 99900

    mask_in_metrics: bool = False
    save_ply_interval: int = 0
    log_debug: str = ""


@dataclass
class RenderingExtras:
    viewspace_point_tensor: Tensor = None
    visibility_filter: Tensor = None
    radii: Tensor = None
    max_size: int = 0


class GaussianSplatModel(Model):
    config: GaussianSplatConfig

    def __init__(
        self,
        config: GaussianSplatConfig,
        scene_box: SceneBox,
        num_train_data: int,
        metadata: Union[Dict[str, Tensor]],
        device: str,
        **kwargs,
    ):
        super().__init__(config, scene_box, num_train_data, **kwargs)
        assert config.depth_loss_type in {"none", "median", "fsgs"}

        cameras: Cameras = metadata["train_cameras"]
        self.cameras = cameras
        self.nearest_camera_indices = find_nearest_cameras(cameras, 1)

        cameras_extent = get_gss_radius(cameras)
        CONSOLE.log(f"Cameras extent: {cameras_extent}")

        train_timestamps: Tensor = metadata["train_timestamps"]

        model_init_kwargs = {}
        ckpt_file = kwargs.get("checkpoint_file")
        if ckpt_file:
            CONSOLE.log(f"Gaussians from ckpt: {ckpt_file}")
            ckpt: dict[str, Tensor] = torch.load(ckpt_file, map_location="cpu")["pipeline"]
            model_init_kwargs["state_dict"] = {
                k.replace("_model.gaussians.", ""): v for k, v in ckpt.items() if k.startswith("_model.gaussians.")
            }
        elif config.ply_file:
            CONSOLE.log(f"Gaussians from ply: {config.ply_file}")
            model_init_kwargs["ply_file"] = config.ply_file
        elif config.npz_file:
            CONSOLE.log(f"Gaussians from npz: {config.npz_file}")
            npz = np.load(config.npz_file)
            npz_pts = npz["points"]
            model_init_kwargs["point_cloud"] = BasicPointCloud(
                npz_pts,
                npz["colors"],
                np.zeros([len(npz_pts), 3]),
            )
        else:
            n_pts = self.config.num_initial_points
            CONSOLE.log(f"Gaussians from {n_pts} random points")
            model_init_kwargs["point_cloud"] = BasicPointCloud(
                np.random.random([n_pts, 3]) * 2 * config.initial_points_range - config.initial_points_range,
                SH2RGB(np.random.random([n_pts, 3]) / 255.0),
                np.zeros([n_pts, 3]),
            )

        self.gaussians = GaussianModel(
            config.sh_degree,
            config.percent_dense,
            feature_mode="sh" if config.appearance.features == "none" else "feature",
            feature_size=config.appearance.feature_size,
            **model_init_kwargs,
        ).to(device)
        self.appearance: GssAppearanceModel = config.appearance.setup(
            num_train_data=num_train_data,
            max_sh_degree=config.sh_degree,
            train_timestamps=train_timestamps,
        ).to(device)
        if config.mip_kernel_size > 0:
            self.gaussians.filter_3D = compute_3D_filter(self.gaussians.get_xyz, cameras)
        self.cameras_extent = cameras_extent
        self.bg_color = torch.FloatTensor([0, 0, 0]).to(device)
        self.render_config = RenderConfig()
        self.previous_rendering_extras = RenderingExtras()
        self.base_dir = Path("")
        self.global_step = 0
        self.depth_model = None

    def is_image_based(self):
        return True

    def populate_modules(self):
        super().populate_modules()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure

    def get_param_groups(self):
        """Returns the parameter groups needed to optimizer your model components."""
        groups = self.gaussians.get_parameter_groups()
        appearance_params = list(self.appearance.parameters())
        if len(appearance_params) > 0:
            CONSOLE.log(f"Optimizing appearance params ({len(appearance_params)})")
            groups.update({"appearance": appearance_params})
        else:
            CONSOLE.log(f"No appearance params")
        return groups

    def render(
        self,
        camera: Cameras,
        gss_camera: GssCamera,
        color: Tensor,
        set_extras: bool = False,
        opacity: Union[Tensor, None] = None,
    ) -> Tensor:
        if self.config.renderer == "gss":
            outputs = render(
                gss_camera,
                self.gaussians,
                self.render_config,
                self.bg_color,
                mip_kernel_size=self.config.mip_kernel_size,
                mip_subpixel_offset=None,
                override_color=color,
                override_opacity=opacity,
            )
            result = {
                "render": torch.permute(outputs["render"], (1, 2, 0)),
            }
        elif self.config.renderer == "gsplat":
            outputs = gsplat_renderer.render(self.gaussians, camera, color)
            result = {
                "render": outputs["render"],
                "depth": outputs["depth"],
            }
        else:
            raise ValueError()

        if set_extras:
            self.previous_rendering_extras = RenderingExtras(
                outputs["viewspace_points"],
                outputs["visibility_filter"],
                outputs["radii"],
                max(int(camera.image_width), int(camera.image_height)),
            )

        return result

    def get_outputs(self, ray_bundle: RayBundle, set_extras: bool = True):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""
        timer.start("gss_get_outputs")
        config = self.config
  
        camera: Cameras = ray_bundle.metadata["camera"]
        gss_camera = GssCamera.convert_from(camera, self.device)
        timestamp = ray_bundle.metadata["timestamp"].view(-1)[0].item() if "timestamp" in ray_bundle.metadata else 0

        # Decode per-Gaussian color / SH
        color = None
        if config.rendering == "gaussian":
            timer.start("render_per_gaussian")
            color = self.appearance.render_per_gaussian(
                ray_bundle, self.gaussians, camera.camera_to_worlds[..., :3, -1], timestamp
            )
            timer.end("render_per_gaussian")
        else:
            color = self.gaussians.get_features

        # Rasterize
        timer.start("render")
        render_output = self.render(camera, gss_camera, color, set_extras=set_extras)
        timer.end("render")

        # Decode per-pixel color / SH
        if config.rendering == "gaussian":
            rgb = render_output["render"]
        else:
            timer.start("render_per_pixel")
            rgb = self.appearance.render_per_pixel(ray_bundle, self.gaussians, render_output["render"], timestamp)
            timer.end("render_per_pixel")

        # compute distance for depth
        if "depth" in render_output:
            depth = render_output["depth"]
        else:
            location = camera.camera_to_worlds[..., :3, -1].view(1, 3).to(self.device)

            dists = torch.linalg.vector_norm(location - self.gaussians.get_xyz, dim=1, keepdim=True)
            with torch.no_grad():
                d_min, d_max = dists.min(), dists.max()

            dists = (dists - d_min) / (d_max - d_min)
            dists = dists.expand((-1, 3))

            depth = self.render(camera, gss_camera, dists)["render"][..., 0:1]
            depth = depth * (d_max - d_min) + d_min

        timer.end("gss_get_outputs")
        return {
            "rgb": rgb,
            "depth": depth,
            "camera_idx": ray_bundle.metadata.get("camera_idx"),
        }

    def get_loss_dict(self, outputs: dict[str, Tensor], batch: dict[str, Tensor], metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        config = self.config
        loss_dict = {}
        gt_rgb = batch["image"].to(self.device)
        if gt_rgb.shape[-1] == 4:
            gt_rgb = gt_rgb[..., :3]

        pred_rgb = outputs["rgb"]
        mask = batch.get("mask")
        lambda_dssim = config.lambda_dssim

        if mask is not None:
            mask = mask[..., 0].to(self.device)
            loss_dict["l1_loss"] = (1.0 - lambda_dssim) * F.l1_loss(pred_rgb[mask], gt_rgb[mask])

            if lambda_dssim > 0:
                loss_dict["ssim_loss"] = lambda_dssim * (
                    1.0 - ssim(pred_rgb.permute(2, 0, 1), gt_rgb.permute(2, 0, 1), mask=mask)
                )
        else:
            loss_dict["l1_loss"] = (1.0 - lambda_dssim) * F.l1_loss(pred_rgb, gt_rgb)
            if lambda_dssim > 0:
                loss_dict["ssim_loss"] = lambda_dssim * (1.0 - ssim(pred_rgb.permute(2, 0, 1), gt_rgb.permute(2, 0, 1)))

        # Compute depth loss and psudo novel view loss
        if config.depth_loss_type != "none":
            loss_dict.update(self._compute_depth_loss(outputs["depth"], batch["depth_image"].to(self.device), mask))
            loss_dict.update(self._compute_pseudo_loss(outputs))

        # Supervise sky
        loss_dict.update(self._compute_sky_loss(outputs, batch))

        return loss_dict

    def _compute_depth_loss(self, pred_depth: Tensor, gt_depth: Tensor, mask: Tensor) -> dict[str, Tensor]:
        config = self.config
        depth_loss_type = config.depth_loss_type

        if depth_loss_type == "none" or self.global_step < config.depth_loss_start_step:
            return {}

        weight = config.lambda_depth_loss_start
        if config.depth_loss_full_step > config.depth_loss_start_step:
            weight += (config.lambda_depth_loss_full - config.lambda_depth_loss_start) * min(
                1,
                (self.global_step - config.depth_loss_start_step)
                / (config.depth_loss_full_step - config.depth_loss_start_step),
            )

        if mask is not None:
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

        result = {}
        if depth_loss_type == "median":
            result["depth_loss"] = weight * compute_depth_loss(pred_depth, gt_depth)
        elif depth_loss_type == "fsgs":
            result["depth_loss"] = weight * (1 - pearson_corrcoef(gt_depth.view(-1), pred_depth.view(-1)))

        return result

    def _compute_pseudo_loss(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        config = self.config
        if self.global_step < config.pseudo_loss_start_step or self.global_step > config.pseudo_loss_end_step:
            return {}
        if (self.global_step + 1) % config.pseudo_loss_interval != 0:
            return {}

        camera_idx = outputs["camera_idx"]
        train_camera = self.cameras[camera_idx]
        nearest_camera = self.cameras[self.nearest_camera_indices[camera_idx, 0]]

        sampled_camera = Cameras(
            train_camera.camera_to_worlds.clone(),
            train_camera.fx,
            train_camera.fy,
            train_camera.cx,
            train_camera.cy,
            train_camera.width,
            train_camera.height,
            train_camera.distortion_params,
            train_camera.camera_type,
            train_camera.times,
            train_camera.metadata,
        )

        t = random.random() * config.pseudo_loss_max_factor
        sampled_camera.camera_to_worlds[:, 3] = (
            t * nearest_camera.camera_to_worlds[:, 3] + (1 - t) * sampled_camera.camera_to_worlds[:, 3]
        )
        pseudo_outputs = self.get_outputs(
            RayBundle(None, None, None, metadata={"camera": sampled_camera}), set_extras=False
        )

        depth_model = self.depth_model
        if depth_model is None:
            depth_model = torch_compile(torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(self.device))
            self.depth_model = depth_model
        with torch.no_grad():
            image = torch.permute(pseudo_outputs["rgb"], (2, 0, 1))[None]  # (1, 3, H, W)
            monodepth = self.depth_model.infer(image).squeeze().unsqueeze(-1)  # (H, W, 1)

        return {"pseudo_loss": self._compute_depth_loss(pseudo_outputs["depth"], monodepth, None)["depth_loss"]}

    def _compute_sky_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        config = self.config
        device = self.device
        if "sky_mask" not in batch:
            return {}

        sky_mask = ~batch["sky_mask"][..., 0].to(device)
        camera = self.cameras[outputs["camera_idx"]]
        gss_camera = GssCamera.convert_from(camera, device)

        color = torch.ones(len(self.gaussians.get_xyz), 3, device=device)
        opacity = None
        if config.sky_loss_type == "render_ones_opaque":
            opacity = torch.ones(len(self.gaussians.get_xyz), 1, device=device)
        sky_render = self.render(camera, gss_camera, color, opacity=opacity)["render"]

        return {"sky_loss": torch.mean(sky_render[sky_mask]) * config.sky_loss_lambda}

    def get_image_metrics_and_images(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        gt_rgb = batch["image"].to(self.device)
        if gt_rgb.shape[-1] == 4:
            gt_rgb = gt_rgb[..., :3]

        pred_rgb = outputs["rgb"].clone()
        pred_rgb.clip_(0, 1)

        depth = colormaps.apply_depth_colormap(outputs["depth"])

        pred_image = torch.cat([pred_rgb], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        if gt_rgb.shape[0] != pred_rgb.shape[0]:
            metrics_dict = {}
        elif self.config.mask_in_metrics:
            mask = batch["mask"][..., 0].to(self.device)
            metrics_dict = {
                "psnr": float(self.psnr(gt_rgb[mask], pred_rgb[mask])),
                "ssim": float(ssim(pred_rgb.permute(2, 0, 1), gt_rgb.permute(2, 0, 1), mask=mask)),
            }
        else:
            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            pred_rgb = torch.moveaxis(pred_rgb, -1, 0)[None, ...]

            # all of these metrics will be logged as scalars
            metrics_dict = {
                "psnr": float(self.psnr(gt_rgb, pred_rgb)),
                "ssim": float(self.ssim(gt_rgb, pred_rgb)),
            }  # type: ignore

        images_dict = {"img": pred_image, "depth": combined_depth}
        return metrics_dict, images_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> list[TrainingCallback]:
        self.base_dir = training_callback_attributes.base_dir
        self.gaussians.optimizers = training_callback_attributes.optimizers
        return [
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.before_train_iteration,
                update_every_num_iters=1,
            ),
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_OPTIMIZER_STEP],
                self.before_optimizer_step,
                update_every_num_iters=1,
            ),
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train_iteration,
                update_every_num_iters=1,
            ),
        ]

    def before_train_iteration(self, step: int):
        self.global_step = step

        gaussians = self.gaussians
        iteration = step + 1

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

    @torch.no_grad()
    def before_optimizer_step(self, step: int):
        config = self.config
        iteration = step + 1
        if iteration >= self.config.densify_until_iter:
            return

        gaussians = self.gaussians
        extras = self.previous_rendering_extras

        visibility_filter = extras.visibility_filter
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], extras.radii[visibility_filter]
        )

        factor = 1
        if config.renderer == "gsplat":
            factor *= extras.max_size * 0.5
        gaussians.add_densification_stats(extras.viewspace_point_tensor, visibility_filter, factor)

        self.log_debug_densify(step)

        if iteration > config.densify_from_iter and iteration % config.densification_interval == 0:
            size_threshold = 20 if iteration > config.opacity_reset_interval else None
            gaussians.densify_and_prune(
                config.densify_grad_threshold,
                0.005,
                self.cameras_extent,
                size_threshold,
                run_proximity=iteration <= config.proximity_end_step,
            )
            if config.mip_kernel_size > 0:
                gaussians.filter_3D = compute_3D_filter(gaussians.get_xyz, self.cameras)

        if iteration > config.densify_from_iter and iteration % config.opacity_reset_interval == 0:
            gaussians.reset_opacity()

        if (
            config.mip_kernel_size > 0
            and iteration > config.densify_until_iter
            and iteration <= config.mip_filter_update_until_iter
            and iteration % 100 == 0
        ):
            gaussians.filter_3D = compute_3D_filter(gaussians.get_xyz, self.cameras)

    def after_train_iteration(self, step: int):
        gaussians = self.gaussians
        iteration = step + 1

        writer.put_scalar(
            "info/num_gaussians",
            len(gaussians.get_xyz),
            step,
        )

        if self.config.save_ply_interval > 0 and iteration % self.config.save_ply_interval == 0:
            ply_file = str(self.base_dir / f"point_clouds/{iteration}.ply")
            CONSOLE.log(f"Save ply to {ply_file}")
            gaussians.save_ply(ply_file)

    def modify_optimizer_config(self, config: dict[str, dict[str, Any]]):
        xyz_optim: OptimizerConfig = config["xyz"]["optimizer"]
        xyz_optim.lr *= self.cameras_extent
        xyz_scheduler: ExponentialDecaySchedulerConfig = config["xyz"]["scheduler"]
        xyz_scheduler.lr_final *= self.cameras_extent

        CONSOLE.log(f"Set lr_init to {xyz_optim.lr} and lr_final to {xyz_scheduler.lr_final}")

    @torch.no_grad()
    def log_debug_densify(self, step: int):
        if "densify" not in self.config.log_debug:
            return

        gaussians = self.gaussians
        extras = self.previous_rendering_extras
        visibility_filter = extras.visibility_filter

        writer.put_scalar(
            "debug/max_radii2D",
            float(torch.median(gaussians.max_radii2D[visibility_filter])),
            step,
        )
        xyz_grad_norm = torch.norm(extras.viewspace_point_tensor.grad[visibility_filter, :2], dim=-1)
        xyz_grad_norm_median = float(torch.median(xyz_grad_norm))
        xyz_grad_norm_max = float(torch.max(xyz_grad_norm))

        writer.put_scalar(
            "debug/xys_grad_norm_median",
            xyz_grad_norm_median,
            step,
        )
        writer.put_scalar(
            "debug/xys_grad_norm_max",
            xyz_grad_norm_max,
            step,
        )


def compute_depth_loss(dyn_depth, gt_depth):
    # https://github.com/gaochen315/DynamicNeRF/blob/c417fb207ef352f7e97521a786c66680218a13af/run_nerf_helpers.py#L483

    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)


def find_nearest_cameras(cameras: Cameras, num: int) -> Tensor:
    N = len(cameras)
    positions = cameras.camera_to_worlds[:, :, 3]
    distances = torch.linalg.vector_norm(positions.view(N, 1, 3) - positions.view(1, N, 3), dim=-1)
    indices = torch.argsort(distances, dim=-1)
    return indices[:, 1 : 1 + num]


@torch.no_grad()
def compute_3D_filter(xyz: torch.Tensor, cameras: Cameras):
    # TODO consider focal length and image width
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

    # we should use the focal length of the highest resolution camera
    focal_length = 0.0
    for i in range(len(cameras)):
        nfs_camera = cameras[i]
        focal_x = float(nfs_camera.fx)
        focal_y = float(nfs_camera.fy)

        camera = GssCamera.convert_from(nfs_camera, xyz.device)

        # transform points to camera space
        R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
        xyz_cam = xyz @ R + T[None, :]

        # xyz_to_cam = torch.norm(xyz_cam, dim=1)

        # project to screen space
        valid_depth = xyz_cam[:, 2] > 0.2

        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)

        x = x / z * focal_x + camera.image_width / 2.0
        y = y / z * focal_y + camera.image_height / 2.0

        # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))

        # use similar tangent space filtering as in the paper
        in_screen = torch.logical_and(
            torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15),
            torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height),
        )

        valid = torch.logical_and(valid_depth, in_screen)

        # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
        distance[valid] = torch.min(distance[valid], z[valid])
        valid_points = torch.logical_or(valid_points, valid)
        if focal_length < focal_x:
            focal_length = focal_x

    distance[~valid_points] = distance[valid_points].max()

    # TODO remove hard coded value
    # TODO box to gaussian transform
    filter_3D = distance / focal_length * (0.2**0.5)
    return filter_3D[..., None]
