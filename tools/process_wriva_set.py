import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pymap3d as pm
import tqdm
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
# from wriva_utils import compute_centroid, qvec2rotmat, sharpness
import cv2
Image.MAX_IMAGE_PIXELS = None

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])


def downsample_images(folder: Path, factor: int): #-> list[Path]:
    print(f"Downsample {folder}")
    outdir = folder.parent / "images_4" / folder.name
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for file in tqdm(sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))):
        files.append(file)
        dst = outdir / f"{file.stem}.png"
        if dst.exists():
            continue

        img = ImageOps.exif_transpose(Image.open(file))
        img = img.resize(
            (int(np.round(img.width / factor)), int(np.round(img.height / factor))), resample=Image.Resampling.LANCZOS
        )
        img.save(dst, optimize=True)
    return files


def main():
    parser = ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--no_reuse_sharpness", dest="reuse_sharpness", action="store_false")
    args = parser.parse_args()

    root = Path(args.root)
    result_file = root / "transforms.json"
    
    # if there is a transforms.json file, read it
    existing_sharpness = {}
    if result_file.exists() and args.reuse_sharpness:
        frames = json.loads(result_file.read_text())["frames"]
        for frame in frames:
            existing_sharpness[frame["file_path"]] = frame["sharpness"]
        print(f"Using existing sharpness for {len(existing_sharpness)} files")

    
    train_image_dir = root / "reference" / "ta2_images"
    train_meta_dir = root / "reference" / "ta2_metadata"
    test_image_dir = root / "reference" / "images"
    test_meta_dir = root / "reference"  / "metadata"

    train_files = downsample_images(train_image_dir, 4)
    train_metadata = [json.loads((train_meta_dir / f"{file.stem}.json").read_text()) for file in train_files]
    test_files = downsample_images(test_image_dir, 4)
    test_metadata = [json.loads((test_meta_dir / f"{file.stem}.json").read_text()) for file in test_files]

    # origin = compute_centroid(train_metadata)
    frames = []
    train_filenames = []
    test_filenames = []
    transforms_dict = {
        "aabb_scale": 16,
        # "origin": origin.tolist(),
        "frames": frames,
        "train_filenames": train_filenames,
        "test_filenames": test_filenames,
        "val_filenames": test_filenames,
    }

    for file, metadata_dict in tqdm(list(zip(train_files + test_files, train_metadata + test_metadata))):
        frame = {}
        file_path = f"./{file.parent.name}/{file.name}"
        frame["file_path"] = file_path
        if file_path in existing_sharpness:
            frame["sharpness"] = existing_sharpness[file_path]
        else:
            frame["sharpness"] = sharpness(str(file))

        # intrinsics
        frame["camera_model"] = "OPENCV"
        frame["fl_x"] = metadata_dict["intrinsics"]["fx"]
        frame["fl_y"] = metadata_dict["intrinsics"]["fy"]
        frame["cx"] = metadata_dict["intrinsics"]["cx"]
        frame["cy"] = metadata_dict["intrinsics"]["cy"]
        frame["w"] = metadata_dict["intrinsics"]["columns"]
        frame["h"] = metadata_dict["intrinsics"]["rows"]
        frame["k1"] = metadata_dict["intrinsics"]["k1"]
        frame["k2"] = metadata_dict["intrinsics"]["k2"]
        frame["k3"] = metadata_dict["intrinsics"]["k3"]
        frame["p1"] = metadata_dict["intrinsics"]["p1"]
        frame["p2"] = metadata_dict["intrinsics"]["p2"]
        frame["camera_angle_x"] = math.atan(frame["w"] / (frame["fl_x"] * 2)) * 2
        frame["camera_angle_y"] = math.atan(frame["h"] / (frame["fl_y"] * 2)) * 2
        frame["timestamp"] = metadata_dict["timestamp"]

        # extrinsics
        d = metadata_dict["extrinsics"]
        r = (
            R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            * R.from_euler(
                "zyx",
                [d["kappa"], d["phi"], d["omega"]],
                degrees=True,
            ).inv()
        )
        qvec = np.roll(r.as_quat(), 1)
        tvec = r.apply(-np.array(pm.geodetic2enu(d["lat"], d["lon"], d["alt"], *origin)))

        rotation = qvec2rotmat(qvec)
        translation = tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        # Remove this so viewer is not upside-down
        # c2w[2, :] *= -1
        frame["transform_matrix"] = c2w.tolist()

        frames.append(frame)
        if file in test_files:
            test_filenames.append(frame["file_path"])
        else:
            train_filenames.append(frame["file_path"])

    result_file.write_text(json.dumps(transforms_dict, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
