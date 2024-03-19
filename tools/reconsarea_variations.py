import json
from argparse import ArgumentParser
from pathlib import Path


def make_subset(root: Path, out_name: str, z_filter):
    assert Path(out_name).name != "transforms.json"

    data = json.loads((root / "transforms.json").read_text())
    train_names = set(data["train_filenames"])
    excluded = set()
    for frame in data["frames"]:
        if frame["file_path"] not in train_names:
            continue

        z = frame["transform_matrix"][2][3]
        if not z_filter(z):
            excluded.add(frame["file_path"])

    data["frames"] = [f for f in data["frames"] if f["file_path"] not in excluded]
    data["train_filenames"] = [f for f in data["train_filenames"] if f not in excluded]
    (root / out_name).write_text(json.dumps(data, sort_keys=True, indent=2))
    print(f"{out_name} has {len(data['frames'])} frames, {len(data['train_filenames'])} train views")


def main():
    parser = ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    make_subset(Path(args.root), "transforms_ground.json", lambda z: z < 0)
    make_subset(Path(args.root), "transforms_drones.json", lambda z: z > 0 and z < 500)
    make_subset(Path(args.root), "transforms_ground+drones.json", lambda z: z < 500)


if __name__ == "__main__":
    main()
