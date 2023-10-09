from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from simple_parsing import ArgumentParser


class NuscenesCameras(str, Enum):
    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"


@dataclass
class CLIParameter:
    nuscenes_data_root: Path
    nuscenes_version: str

    def parse_args(argv: list[str]) -> CLIParameter:
        parser = ArgumentParser()
        parser.add_arguments(CLIParameter, dest="opt")
        args: CLIParameter = parser.parse_args(argv).opt
        return args
