from dataclasses import dataclass, field
from typing import Literal, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.sh import spherical_harmonics
from sklearn.preprocessing import RobustScaler
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.gss.gaussian_model import GaussianModel


@dataclass
class GssAppearanceModelConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: GssAppearanceModel)

    features: Literal["none", "hash_grid", "gaussians"] = "none"
    feature_size: int = 0

    appearance: Literal["none", "embedding"] = "none"
    appearance_embed_dim: int = 32
    test_appearance: str = "mean"

    temporal: Literal["none", "robust_standardization"] = "none"
    test_time: str = "metadata"

    decoder: Literal["mlp"] = "mlp"
    decode_to: Literal["sh", "rgb"] = "rgb"


class GssAppearanceModel(nn.Module):
    def __init__(
        self, config: GssAppearanceModelConfig, num_train_data: int, max_sh_degree: int, train_timestamps: Tensor
    ) -> None:
        super().__init__()

        # Validate config
        assert config.test_appearance in ["zeros", "mean", "trained"] or config.test_appearance.startswith("trained-")
        assert (
            config.test_time in ["metadata"]
            or config.test_time.startswith("fixed-")
            or config.test_time.startswith("trained-")
        )
        if config.appearance != "none" or config.temporal != "none":
            assert config.features != "none"

        self.config = config
        need_decode = config.features != "none"
        decoder_input_dim = 0

        self.direction_encoding = None
        if need_decode and config.decode_to == "rgb":
            self.direction_encoding = SHEncoding(
                levels=4,
                implementation="tcnn",
            )
            decoder_input_dim += self.direction_encoding.get_out_dim()

        self.features = None
        if config.features == "hash_grid":
            self.features = HashEncoding(implementation="tcnn")
            decoder_input_dim += self.features.get_out_dim()
        elif config.features == "gaussians":
            decoder_input_dim += config.feature_size

        self.appearance = None
        if config.appearance == "embedding":
            self.appearance = Embedding(num_train_data, config.appearance_embed_dim)
            decoder_input_dim += config.appearance_embed_dim

        self.temporal = None
        self.time_scaler = None
        self.train_timestamps = train_timestamps
        if config.temporal == "robust_standardization":
            self.time_scaler = RobustScaler()
            self.time_scaler.fit(train_timestamps.numpy().reshape([-1, 1]))
            self.temporal = NeRFEncoding(
                in_dim=1, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3, implementation="tcnn"
            )
            decoder_input_dim += self.temporal.get_out_dim()

        self.decode = None
        if need_decode:
            out_dim = 3 if config.decode_to == "rgb" else 3 * (max_sh_degree + 1) ** 2
            self.decoder = self._create_decoder(config, decoder_input_dim, out_dim)

    def _create_decoder(self, config: GssAppearanceModelConfig, in_dim: int, out_dim: int):
        if config.decoder == "mlp":
            return MLP(
                in_dim=in_dim,
                num_layers=3,
                layer_width=64,
                out_dim=out_dim,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="torch",
            )

    def _get_appearance_feats(self, ray_bundle: RayBundle, expand_to: int, device) -> list[Tensor]:
        config = self.config
        results = []

        if config.appearance == "embedding":
            assert isinstance(self.appearance, Embedding)
            if self.training or config.test_appearance == "trained":
                results.append(self.appearance(ray_bundle.camera_indices.view(-1)[0]).expand((expand_to, -1)))
            else:
                if config.test_appearance == "mean":
                    results.append(self.appearance.mean(dim=0).view(1, -1).expand((expand_to, -1)))
                elif config.test_appearance == "zeros":
                    results.append(torch.zeros((expand_to, config.appearance_embed_dim), device=device))
                elif config.test_appearance.startswith("trained-"):
                    idx_input = torch.LongTensor([1]).to(device) * int(config.test_appearance[len("trained-") :])
                    results.append(self.appearance(idx_input).expand((expand_to, -1)))
        return results

    def _get_temporal_feats(self, time: float, expand_to: int, device) -> list[Tensor]:
        config = self.config
        results = []

        if not self.training:
            if config.test_time.startswith("fixed-"):
                time = float(config.test_time[len("fixed-") :])
            if config.test_time.startswith("trained-"):
                time = float(self.train_timestamps[int(config.test_time[len("trained-") :])])

        if config.temporal == "robust_standardization":
            assert isinstance(self.temporal, NeRFEncoding)
            assert isinstance(self.time_scaler, RobustScaler)
            time = self.time_scaler.transform(np.array([[time]]))
            results.append(self.temporal(torch.from_numpy(time).to(device)).expand((expand_to), -1))

        return results

    def render_per_gaussian(
        self, ray_bundle: RayBundle, gaussians: GaussianModel, camera_location: Tensor, time: float
    ) -> Tensor:
        config = self.config
        directions = F.normalize(
            gaussians.get_xyz.detach() - camera_location[None].to(gaussians.get_xyz.device), dim=-1
        )

        if config.features == "none":
            return decode_sh(gaussians, directions, gaussians.get_features)

        assert isinstance(self.decoder, MLP)

        n_gaussians = len(gaussians.get_xyz)
        if config.features == "hash_grid":
            assert isinstance(self.features, HashEncoding)
            image_feats = self.features(gaussians.get_xyz)
        elif config.features == "gaussians":
            image_feats = gaussians.get_features
        else:
            raise NotImplementedError()

        input_feats = [image_feats]

        if config.decode_to == "rgb":
            assert isinstance(self.direction_encoding, SHEncoding)
            input_feats += [self.direction_encoding(directions)]

        input_feats += self._get_appearance_feats(ray_bundle, n_gaussians, image_feats.device)
        input_feats += self._get_temporal_feats(time, n_gaussians, image_feats.device)

        input_feats = torch.cat(input_feats, dim=1).float()
        output = self.decoder(input_feats)

        if config.decode_to == "rgb":
            return output
        return decode_sh(gaussians, directions, output.view(n_gaussians, -1, 3))

    def render_per_pixel(
        self, ray_bundle: RayBundle, gaussians: GaussianModel, features: Tensor, time: float
    ) -> Tensor:
        assert isinstance(self.decoder, MLP)

        config = self.config
        H, W, _ = features.shape
        input_feats = [features.view(H * W, -1)]
        directions = F.normalize(ray_bundle.directions.view(-1, 3), dim=-1)

        if config.decode_to == "rgb":
            assert isinstance(self.direction_encoding, SHEncoding)
            input_feats += [self.direction_encoding(directions)]

        input_feats += self._get_appearance_feats(ray_bundle, len(directions), features.device)
        input_feats += self._get_temporal_feats(time, len(directions), features.device)

        input_feats = torch.cat(input_feats, dim=1).float()
        output = self.decoder(input_feats)

        if config.decode_to == "rgb":
            return output.view(H, W, 3)
        return decode_sh(gaussians, directions, output.view(H * W, -1, 3)).view(H, W, 3)


def decode_sh(
    gaussians: GaussianModel,
    viewdirs: Tensor,
    feats: Tensor,
) -> Tensor:
    colors = spherical_harmonics(gaussians.active_sh_degree.item(), viewdirs, feats)
    return torch.clamp_min(colors + 0.5, 0.0)
