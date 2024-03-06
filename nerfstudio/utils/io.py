# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input/output utils.
"""

import itertools
import json
from pathlib import Path
from typing import TypeVar

TPath = TypeVar("TPath", str, Path)

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def globs_sorted(path: str | Path, appendices: str | list[str]) -> list[Path]:
    """Query globs of the same root path, returns sorted result."""
    if not isinstance(appendices, list):
        appendices = [appendices]

    path = Path(path)
    return sorted(itertools.chain(*[path.glob(app) for app in appendices]))


def replace_data_root(path: TPath, new_root: str) -> TPath:
    indicator = "/data/"
    idx = str(path).find(indicator)
    assert idx >= 0

    if not new_root.endswith("/"):
        new_root += "/"
    new_path = new_root + str(path)[idx+len(indicator):]

    if isinstance(path, Path):
        return Path(new_path)

    return new_path
