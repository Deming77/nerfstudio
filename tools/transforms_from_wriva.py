import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    sfm_file = "transforms.json"
    ref_file = "reference/transforms.json"
    out_file = "transforms_full.json"

    root = Path(args.root)
    sfm_data = json.loads((root / sfm_file).read_text())
    ref_data = json.loads((root / ref_file).read_text())

    sfm_frames = sfm_data["frames"]
    sfm_frames = sorted(sfm_frames, key=lambda x: x["file_path"])
    for frame in sfm_frames:
        del frame["mask_path"]
        tr = frame["transform_matrix"]
        if len(tr) == 3:
            tr.append([0.0, 0.0, 0.0, 1.0])

    sfm_data["frames"] = sfm_frames

    sfm_filenames = [f["file_path"] for f in sfm_frames]
    sfm_data["train_filenames"] = sfm_filenames

    ref_frames = ref_data["frames"]
    ref_frames = sorted(ref_frames, key=lambda x: x["file_path"])
    for frame in ref_frames:
        frame["file_path"] = frame["file_path"].replace("./images", "./reference/images")

    ref_filenames = [f["file_path"] for f in ref_frames]

    sfm_data["frames"] += ref_frames
    sfm_data["test_filenames"] = ref_filenames
    sfm_data["val_filenames"] = ref_filenames
    (root / out_file).write_text(json.dumps(sfm_data, sort_keys=True, indent=2), "utf-8")

    # not needed - input images already downsampled
    # for frame in tqdm(sfm_data["frames"]):
    #     in_path = frame["file_path"]
    #     out_path = in_path.replace("./", "./images_4/")
    #     in_file = root / in_path
    #     out_file: Path = root / out_path
    #     out_file = out_file.parent / f"{out_file.stem}.png"
    #     if out_file.exists():
    #         continue

    #     out_file.parent.mkdir(parents=True, exist_ok=True)
    #     img = Image.open(in_file)
    #     img = img.resize((img.width // 4, img.height // 4), resample=Image.Resampling.LANCZOS)
    #     img.save(out_file)

if __name__ == "__main__":
    main()
