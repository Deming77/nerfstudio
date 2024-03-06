import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import yaml
from tqdm import tqdm

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint, eval_find_checkpoint
from nerfstudio.utils.io import replace_data_root
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class OurEval:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    load_config: Path

    name: str = ""
    """Name of output folder (created under training folder). If not set, defaults to eval-[step] (or eval-[step]-train when using train split)"""
    suffix: str = ""
    """Suffix to append to name, if provided"""
    num_images: int = 0
    """Limit number of images to run, mostly for debugging"""
    img_indices: str = ""
    """Limit indices of images to run, mostly for debugging"""
    use_train_split: bool = False
    """Use train split, for checking train view performance"""
    load_checkpoint: bool = True
    """Use --no-load-checkpoint to prevent checkpoint loading, just for debugging"""
    downsample: int = 1
    """Downsample cameras"""
    image_format: Literal["png", "jpg"] = "png"
    """Image format to save"""
    save_raw_depth: bool = False
    """Save raw depth as npy files"""

    def main(self, extra_args: list[str]) -> None:
        config_path = self.load_config

        # load save config
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)

        if self.load_checkpoint:
            config.load_dir = config.get_checkpoint_dir()
            ckpt_path, ckpt_step = eval_find_checkpoint(config)
        else:
            ckpt_path, ckpt_step = "", 0

        if extra_args is not None and len(extra_args) > 0:
            config = tyro.cli(TrainerConfig, args=extra_args, default=config)
            CONSOLE.print("------------\nOverriden config:\n" + yaml.dump(config) + "------------")

        data_root = os.environ.get("DATA_ROOT", "/data")
        config.data = replace_data_root(config.data, data_root)
        config.pipeline.datamanager.data = config.data
        CONSOLE.log(f"Data: {config.data}")

        mask_path = config.pipeline.datamanager.dataparser.mask_root
        if mask_path is not None and len(mask_path) > 0:
            config.pipeline.datamanager.dataparser.mask_root = replace_data_root(mask_path, data_root)
        # TODO need to replace special mask roots too

        # create pipeline
        pipeline = config.pipeline.setup(device="cuda", test_mode="test", checkpoint_file=ckpt_path)
        assert isinstance(pipeline, VanillaPipeline)

        # load data
        datamanager = pipeline.datamanager
        assert isinstance(datamanager, VanillaDataManager)
        datamanager.setup_train(load_data=False)
        datamanager.setup_eval()

        # load ckpt
        if self.load_checkpoint:
            ckpt_path, ckpt_step = eval_load_checkpoint(config, pipeline)
            CONSOLE.log(f"Loaded checkpoint at step {ckpt_step}")
        else:
            CONSOLE.log(f"Not using any checkpoint")

        # setup output
        output_name = self.name
        if len(output_name) == 0:
            output_name = f"eval-{ckpt_step}"
            if self.use_train_split:
                output_name += "-train"
        if len(self.suffix) > 0:
            output_name += f"-{self.suffix}"
        output_dir = config_path.parent / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.log(f"Write to {output_dir}")

        if self.use_train_split:
            loader = FixedIndicesEvalDataloader(
                datamanager.train_dataset,
                device=pipeline.device,
                num_workers=datamanager.world_size * 4,
            )
        else:
            loader = datamanager.fixed_indices_eval_dataloader

        if self.downsample != 1:
            CONSOLE.log(f"Downsample by {self.downsample}x")
            loader.cameras.rescale_output_resolution(1 / self.downsample)

        num_images = len(loader.dataset)
        if len(self.img_indices) > 0:
            image_indices = [int(v) for v in self.img_indices.split(",")]
            assert all(v >= 0 and v < num_images for v in image_indices)
        elif self.num_images > 0:
            image_indices = range(min(self.num_images, num_images))
        else:
            image_indices = range(num_images)

        all_metrics = defaultdict(list)
        pbar = tqdm(total=len(image_indices))
        for idx in image_indices:
            metrics = pipeline.eval_single_image(
                idx,
                output_dir,
                loader,
                image_format=self.image_format,
                raw_data_keys=["depth"] if self.save_raw_depth else [],
            )
            for k, v in metrics.items():
                all_metrics[k].append(v)

            pbar.update(1)
            pbar.set_postfix_str(", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

        # save metrics
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(ckpt_path),
            "results": {k: float(np.mean(np.array(v))) for k, v in all_metrics.items()},
            "raw_metrics": all_metrics,
        }
        output_file = output_dir / "output.json"
        output_file.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.log(f"Saved results to: {output_file}")

        # print metrics
        CONSOLE.print(
            "Average metrics\n" + "\n".join([f"{k:>8}: {v:.4f}" for k, v in benchmark_info["results"].items()])
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")

    argv = list(sys.argv[1:])
    extra_args = []

    if "--" in argv:
        indicator_idx = argv.index("--")
        extra_args = argv[indicator_idx + 1 :]
        argv = argv[:indicator_idx]

    tyro.cli(OurEval, args=argv).main(extra_args)


if __name__ == "__main__":
    entrypoint()
