from pathlib import Path
from typing import Literal

import numpy as np
import simple_knn_fsgs._C as knn_fsgs
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import Tensor, nn
from torch.optim import Optimizer

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.fields.gss.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from nerfstudio.fields.gss.utils.graphics_utils import BasicPointCloud
from nerfstudio.fields.gss.utils.sh_utils import RGB2SH
from nerfstudio.utils.rich_utils import CONSOLE


class GaussianModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        max_sh_degree: int,
        percent_dense: float,
        feature_mode: Literal["sh", "feature"] = "sh",
        feature_size: int = 0,
        state_dict: dict[str, Tensor] | None = None,
        ply_file: str | None = None,
        point_cloud: BasicPointCloud | None = None,
    ):
        super().__init__()
        assert (
            int(state_dict is not None) + int(ply_file is not None) + int(point_cloud is not None) == 1
        ), f"Must provide either state_dict, ply_file, or point_cloud"
        assert (feature_mode == "sh" and feature_size == 0) or feature_size > 1, "Invalid configuration"

        self.max_sh_degree = max_sh_degree
        self.percent_dense = percent_dense
        self.feature_mode = feature_mode
        self.feature_size = feature_size
        self.setup_functions()
        self.optimizers: Optimizers = None
        self.filter_3D = torch.zeros(0)

        if state_dict is not None:
            self.create_from_checkpoint(state_dict)
        elif ply_file is not None:
            self.load_ply(ply_file)
        else:
            self.create_from_pcd(point_cloud)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        if len(self.filter_3D) == 0:
            return scales

        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        if len(self.filter_3D) == 0:
            return opacity

        # apply 3D filter
        scales = self.get_scaling

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud):
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float()
        if self.feature_mode == "sh":
            fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float())
            features = torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            features = torch.randn(len(fused_point_cloud), self.feature_size)
            self._features_dc = nn.Parameter(features[:, 0:1].requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, 1:].requires_grad_(True))

        n_pts = len(fused_point_cloud)

        CONSOLE.log(f"Number of points at initialisation : {n_pts}")

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()).cpu(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros(n_pts, 4)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((n_pts, 1), dtype=torch.float))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = nn.Parameter(torch.zeros(n_pts), requires_grad=False)
        self.xyz_gradient_accum = nn.Parameter(torch.zeros(n_pts, 1), requires_grad=False)
        self.denom = nn.Parameter(torch.zeros(n_pts, 1), requires_grad=False)
        self.active_sh_degree = nn.Parameter(torch.IntTensor([0]), requires_grad=False)

    def create_from_checkpoint(self, state_dict: dict[str, Tensor]):
        self._xyz = nn.Parameter(state_dict["_xyz"].requires_grad_(True))
        self._features_dc = nn.Parameter(state_dict["_features_dc"].requires_grad_(True))
        self._features_rest = nn.Parameter(state_dict["_features_rest"].requires_grad_(True))
        self._scaling = nn.Parameter(state_dict["_scaling"].requires_grad_(True))
        self._rotation = nn.Parameter(state_dict["_rotation"].requires_grad_(True))
        self._opacity = nn.Parameter(state_dict["_opacity"].requires_grad_(True))
        self.max_radii2D = nn.Parameter(state_dict["max_radii2D"].requires_grad_(False))
        self.xyz_gradient_accum = nn.Parameter(state_dict["xyz_gradient_accum"].requires_grad_(False))
        self.denom = nn.Parameter(state_dict["denom"].requires_grad_(False))
        self.active_sh_degree = nn.Parameter(state_dict["active_sh_degree"], requires_grad=False)

    def get_parameter_groups(self):
        return {
            "xyz": [self._xyz],
            "features_dc": [self._features_dc],
            "features_rest": [self._features_rest],
            "opacity": [self._opacity],
            "scaling": [self._scaling],
            "rotation": [self._rotation],
        }

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        if self.feature_mode == "sh":
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append("f_dc_{}".format(i))
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append("f_rest_{}".format(i))
        else:
            for i in range(self._features_dc.shape[1]):
                l.append(f"f_dc_{i}")
            for i in range(self._features_rest.shape[1]):
                l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        if self.feature_mode == "sh":
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().cpu().numpy()
            f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        if len(self.filter_3D) == 0:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        else:
            # reset opacity to by considering 3D filter
            current_opacity_with_filter = self.get_opacity_with_3D_filter
            opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * 0.01)

            # apply 3D filter
            scales = self.get_scaling

            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)

            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = opacities_new / coef[..., None]
            opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @staticmethod
    def _load_ply_prefix(plydata: PlyData, prefix: str):
        names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
        names = sorted(names, key=lambda x: int(x.split("_")[-1]))
        size = len(np.asarray(plydata.elements[0][names[0]]))
        result = np.zeros((size, len(names)))
        for idx, attr_name in enumerate(names):
            result[:, idx] = np.asarray(plydata.elements[0][attr_name])
        return result

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        n_pts = xyz.shape[0]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if self.feature_mode == "sh":
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            features_extra = self._load_ply_prefix(plydata, "f_rest_")
            assert features_extra.shape[1] == 3 * (self.max_sh_degree + 1) ** 2 - 3
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._features_dc = nn.Parameter(
                torch.from_numpy(features_dc).float().transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.from_numpy(features_extra).float().transpose(1, 2).contiguous().requires_grad_(True)
            )
        else:
            features_dc = self._load_ply_prefix(plydata, "f_dc_")
            features_rest = self._load_ply_prefix(plydata, "f_rest_")
            self._features_dc = nn.Parameter(torch.from_numpy(features_dc).float().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.from_numpy(features_rest).float().requires_grad_(True))

        scales = self._load_ply_prefix(plydata, "scale_")
        rots = self._load_ply_prefix(plydata, "rot_")

        self._xyz = nn.Parameter(torch.from_numpy(xyz).float().requires_grad_(True))
        self._opacity = nn.Parameter(torch.from_numpy(opacities).float().requires_grad_(True))
        self._scaling = nn.Parameter(torch.from_numpy(scales).float().requires_grad_(True))
        self._rotation = nn.Parameter(torch.from_numpy(rots).float().requires_grad_(True))
        self.max_radii2D = nn.Parameter(torch.zeros(n_pts), requires_grad=False)
        self.xyz_gradient_accum = nn.Parameter(torch.zeros(n_pts, 1), requires_grad=False)
        self.denom = nn.Parameter(torch.zeros(n_pts, 1), requires_grad=False)
        self.active_sh_degree = nn.Parameter(torch.IntTensor([self.max_sh_degree]), requires_grad=False)

    def replace_tensor_to_optimizer(self, tensor: Tensor, name: str):
        optimizers = self.optimizers
        assert isinstance(optimizers, Optimizers)

        optimizable_tensors = {}
        optimizer = optimizers.optimizers[name]
        assert isinstance(optimizer, Optimizer)
        assert len(optimizer.param_groups) == 1

        group = optimizer.param_groups[0]
        assert len(group["params"]) == 1

        stored_state = optimizer.state.get(group["params"][0], None)
        stored_state["exp_avg"] = torch.zeros_like(tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

        del optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        optimizer.state[group["params"][0]] = stored_state

        param = group["params"][0]
        optimizable_tensors[name] = param
        optimizers.parameters[name] = param

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizers = self.optimizers
        assert isinstance(optimizers, Optimizers)

        optimizable_tensors = {}
        for name, optimizer in optimizers.optimizers.items():
            if name not in self.get_parameter_groups():
                continue

            assert isinstance(optimizer, Optimizer)
            assert len(optimizer.param_groups) == 1

            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))

            param = group["params"][0]
            optimizable_tensors[name] = param
            optimizers.parameters[name] = param

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = nn.Parameter(self.xyz_gradient_accum[valid_points_mask], requires_grad=False)
        self.denom = nn.Parameter(self.denom[valid_points_mask], requires_grad=False)
        self.max_radii2D = nn.Parameter(self.max_radii2D[valid_points_mask], requires_grad=False)

    def cat_tensors_to_optimizer(self, tensors_dict: dict[str, Tensor]):
        optimizers = self.optimizers
        assert isinstance(optimizers, Optimizers)

        optimizable_tensors = {}
        for name, optimizer in optimizers.optimizers.items():
            if name not in tensors_dict:
                continue

            assert isinstance(optimizer, Optimizer)
            assert len(optimizer.param_groups) == 1

            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1

            extension_tensor = tensors_dict[name]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )

            param = group["params"][0]
            optimizable_tensors[name] = param
            optimizers.parameters[name] = param

        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
    ):
        d = {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        n_pts = self._xyz.shape[0]
        device = self._xyz.device
        self.xyz_gradient_accum = nn.Parameter(torch.zeros(n_pts, 1, device=device), requires_grad=False)
        self.denom = nn.Parameter(torch.zeros(n_pts, 1, device=device), requires_grad=False)
        self.max_radii2D = nn.Parameter(torch.zeros(n_pts, device=device), requires_grad=False)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        device = self._xyz.device
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        if self.feature_mode == "sh":
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
        )

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        run_proximity: bool,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        if run_proximity:
            self.proximity(extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, factor: float):
        self.xyz_gradient_accum[update_filter] += (
            torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True) * factor
        )
        self.denom[update_filter] += 1

    def proximity(self, scene_extent, N=3):
        dist, nearest_indices = knn_fsgs.distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(
            dist > (5.0 * scene_extent), torch.max(self.get_scaling, dim=1).values > (scene_extent)
        )

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
