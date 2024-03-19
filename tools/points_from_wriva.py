import itertools
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from typing import Union, List
import nerfstudio.data.utils.colmap_parsing_utils as colmap_parsing_utils


def save_all_combinations(root: Path, points: np.ndarray, colors: np.ndarray):
    folder = root / "sfm_configs"
    folder.mkdir(exist_ok=True)

    idx = 0
    for pm in itertools.permutations(range(3)):
        for mult_idx in range(8):
            xmult = -1 if mult_idx & 0b100 else 1
            ymult = -1 if mult_idx & 0b10 else 1
            zmult = -1 if mult_idx & 0b1 else 1

            print(f"{idx:>2} {pm} [{xmult}, {ymult}, {zmult}]")
            pp = np.stack(
                [
                    points[:, pm[0]] * xmult,
                    points[:, pm[1]] * ymult,
                    points[:, pm[2]] * zmult,
                ],
                axis=1,
            )
            np.savez(
                folder / f"sfm_points_{idx}.npz",
                points=pp,
                colors=colors,
            )
            idx += 1


def save_point_cloud(root: Path, data: 'list[colmap_parsing_utils.Point3D]', mode: str, save_everything: bool):
    points = np.stack([p.xyz for p in data]).astype(np.float32)
    colors = np.stack([p.rgb for p in data]).astype(np.float32) / 255

    if save_everything:
        save_all_combinations(root, points, colors)
        return

    if mode == "docker":
        pm = [0, 2, 1]
        xmult, ymult, zmult = 1, -1, -1
    elif mode == "colmap":
        pm = [1, 0, 2]
        xmult, ymult, zmult = 1, 1, 1
    else:
        raise ValueError()

    pp = np.stack(
        [
            points[:, pm[0]] * xmult,
            points[:, pm[1]] * ymult,
            points[:, pm[2]] * zmult,
        ],
        axis=1,
    )

    pc_file = root / "sfm_points.npz"
    np.savez(
        pc_file,
        points=pp,
        colors=colors,
    )
    print(f"Save PC with {len(points)} points to: {pc_file}")


def main():
    parser = ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("-m", "--mode", required=True, choices=["docker", "colmap"])
    parser.add_argument("--path", default="sparse/points3D.txt")
    parser.add_argument("--everything", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    if args.path.endswith(".txt"):
        reader = colmap_parsing_utils.read_points3D_text
    else:
        reader = colmap_parsing_utils.read_points3D_binary
    pt_id_to_data = reader(root / args.path)
    save_point_cloud(root, list(pt_id_to_data.values()), args.mode, save_everything=args.everything)


if __name__ == "__main__":
    main()
