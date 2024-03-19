from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from PIL import Image, ImageDraw
from torch import Tensor
from tqdm import tqdm

from nerfstudio.utils.io import globs_sorted

mask_color = np.array([[0, 1, 0]], dtype=np.float32)
mask_alpha = 0.4


@torch.no_grad()
def run_segmentation(model, device, file: Path) -> 'dict[str, Tensor]':
    image = np.array(Image.open(file), dtype=np.float32)
    image = image[..., :3]  # RGBA to RGB
    image = torch.from_numpy(image.transpose(2, 0, 1))
    inputs = [
        {
            "image": image.to(device),
            "height": image.shape[1],
            "width": image.shape[2],
        }
    ]
    with torch.no_grad():
        instances = model(inputs)[0]["instances"].to("cpu")

    fields = instances.get_fields()
    return {
        "pred_boxes": fields["pred_boxes"].tensor.cpu().numpy(),
        **{k: fields[k].cpu().numpy() for k in ["scores", "pred_classes", "pred_masks"]},
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--visualization", default=None)
    parser.add_argument("-c", "--classes", action="append", type=int, default=None)
    parser.add_argument("--box_dilation", type=int, default=20)
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    args = parser.parse_args()

    if args.classes is None:
        args.classes = [0, 2, 3, 5]

    device = args.device
    box_dilation = args.box_dilation

    model = model_zoo.get(args.model, trained=True).eval().to(device)

    in_root = Path(args.input)
    out_root = Path(args.output)
    vis_root = Path(args.visualization) if args.visualization else None

    for root in [out_root, vis_root]:
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    for file in tqdm(globs_sorted(in_root, ["*.png", "*.jpg"])):
        seg_data = run_segmentation(model, device, file)

        img = np.array(Image.open(file)).astype(np.float32) / 255
        mask = np.zeros_like(img[..., 0], dtype=bool)
        H, W = mask.shape

        classes = seg_data["pred_classes"]
        pred_boxes = seg_data["pred_boxes"]
        pred_masks = seg_data["pred_masks"]
        scores = seg_data["scores"]
        applied_masks = []
        for i, cls in enumerate(classes):
            if cls not in args.classes:
                continue
            score = scores[i]
            if score < args.threshold:
                continue

            if args.bbox:
                x0, y0, x1, y1 = pred_boxes[i]
                x0, y0 = (int(np.floor(v)) for v in (x0, y0))
                x1, y1 = (int(np.ceil(v)) for v in (x1, y1))
                applied_masks.append([x0, y0, cls, score])

                x0 = max(0, x0 - box_dilation)
                y0 = max(0, y0 - box_dilation)
                x1 = min(W - 1, x1 + box_dilation)
                y1 = min(H - 1, y1 + box_dilation)

                mask[y0:y1, x0:x1] = True
            else:
                pred_mask = pred_masks[i].astype(np.float32)
                ys, xs = np.nonzero(pred_mask)
                applied_masks.append([xs.min(), ys.min(), cls, score])

                pred_mask = 1 - cv2.erode(
                    (1 - pred_mask),
                    kernel=np.ones((box_dilation, box_dilation)),
                    iterations=1,
                )
                pred_mask = pred_mask > 0.5
                mask |= pred_mask

        Image.fromarray((~mask).astype(np.uint8) * 255).save(out_root / f"{file.stem}.png")

        if vis_root is not None:
            vis = img.copy()
            vis[mask] = img[mask] * (1 - mask_alpha) + mask_color * mask_alpha
            vis = Image.fromarray((vis * 255).astype(np.uint8))

            draw = ImageDraw.Draw(vis)
            for x0, y0, cls, score in applied_masks:
                draw.text((x0, y0), f"{cls} ({score:.2f})")
            vis.save(vis_root / f"{file.stem}.jpg", quality=95)


if __name__ == "__main__":
    main()
