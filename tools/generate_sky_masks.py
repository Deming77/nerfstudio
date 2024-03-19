from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from PIL import Image, ImageOps
from tqdm import tqdm

from nerfstudio.utils.io import globs_sorted

mask_color = np.array([[0, 1, 0]], dtype=np.float32)
mask_alpha = 0.4


@torch.no_grad()
def predict_sky_detectron(model, device, file: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(ImageOps.exif_transpose(Image.open(file)), dtype=np.float32)
    image = image[..., :3]  # RGBA to RGB
    image = torch.from_numpy(image.transpose(2, 0, 1))
    inputs = [
        {
            "image": image.to(device),
            "height": image.shape[1],
            "width": image.shape[2],
        }
    ]
    predictions, segment_info = model(inputs)[0]["panoptic_seg"]

    sky_segs = [info for info in segment_info if info["category_id"] == 40]
    if len(sky_segs) == 0:
        mask = predictions == -100
    else:
        mask = predictions == sky_segs[0]["id"]
        for seg in sky_segs[1:]:
            mask |= predictions == seg["id"]

    return torch.permute(image / 255, (1, 2, 0)).numpy(), mask.cpu().numpy()


@torch.no_grad()
def predict_sky_xiongzhu(model, file: Path) -> tuple[np.ndarray, np.ndarray]:
    def load_image(path):
        img_original = ImageOps.exif_transpose(Image.open(path))
        img = img_original.resize((320, 320), resample=Image.Resampling.LANCZOS)
        img_data = np.array(img).astype(np.float32)
        img_data = np.transpose(img_data, (2, 0, 1))

        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype("float32")
        for i in range(3):
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

        return img_original, norm_img_data[None].astype(np.float32)

    img_orig, img_input = load_image(file)
    mask = torch.from_numpy(model.run([], {"input.1": img_input})[0])  # [1, 1, 320, 320]
    mask = F.interpolate(mask, size=(img_orig.height, img_orig.width), mode="bilinear", align_corners=True)
    mask = mask[0, 0] > 0.5

    return np.array(img_orig, dtype=np.float32) / 255, mask.numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--visualization", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", default="xiongzhu", choices=["xiongzhu", "detectron"])
    parser.add_argument("--xiongzhu_model", default="/data/Sky-Segmentation-and-Post-processing/skyseg.onnx")
    parser.add_argument("--detectron_model", default="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    args = parser.parse_args()

    device = args.device

    if args.method == "xiongzhu":
        model = ort.InferenceSession(args.xiongzhu_model)
    else:
        model: torch.nn.Module = model_zoo.get(args.detectron_model, trained=True)
        model = model.eval().to(device)

    in_root = Path(args.input)
    out_root = Path(args.output)
    vis_root = Path(args.visualization) if args.visualization else None
    for root in [out_root, vis_root]:
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    for file in tqdm(globs_sorted(in_root, ["*.png", "*.jpg"])):
        if args.method == "xiongzhu":
            img, mask = predict_sky_xiongzhu(model, file)
        else:
            img, mask = predict_sky_detectron(model, device, file)

        Image.fromarray((~mask).astype(np.uint8) * 255).save(out_root / f"{file.stem}.png")

        if vis_root is not None:
            vis = img.copy()
            vis[mask] = img[mask] * (1 - mask_alpha) + mask_color * mask_alpha
            vis = Image.fromarray((vis * 255).astype(np.uint8))
            vis.save(vis_root / f"{file.stem}.jpg", quality=90)


if __name__ == "__main__":
    main()
