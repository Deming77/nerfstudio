from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from nerfstudio.utils.io import globs_sorted

mask_color = np.array([[0, 1, 0]], dtype=np.float32)
mask_alpha = 0.4


def main():
    parser = ArgumentParser()
    parser.add_argument("input", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--visualization")
    parser.add_argument("--image")
    args = parser.parse_args()

    in_dirs = args.input
    print(f"Combine masks from {len(in_dirs)} sources")

    out_root = Path(args.output)
    vis_root = Path(args.visualization) if args.visualization else None
    img_root = Path(args.image) if args.image else None
    img_files = globs_sorted(img_root, ["*.png", "*.jpg"]) if img_root is not None else []
    if vis_root is not None:
        assert img_root is not None
    for root in [out_root, vis_root]:
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    mask_files = [globs_sorted(Path(d), "*.png") for d in args.input]
    for i_file, first_file in enumerate(tqdm(mask_files[0])):
        mask = np.array(Image.open(first_file)) > 0.5

        for file in [files[i_file] for files in mask_files[1:]]:
            mask &= np.array(Image.open(file)) > 0.5

        Image.fromarray(mask.astype(np.uint8) * 255).save(out_root / f"{first_file.stem}.png")

        if vis_root is not None:
            img = ImageOps.exif_transpose(Image.open(img_files[i_file]))
            img = np.array(img, dtype=np.float32) / 255

            vis = img.copy()
            vis[~mask] = img[~mask] * (1 - mask_alpha) + mask_color * mask_alpha
            vis = Image.fromarray((vis * 255).astype(np.uint8))
            vis.save(vis_root / f"{first_file.stem}.jpg", quality=90)


if __name__ == "__main__":
    main()
