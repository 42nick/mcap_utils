from __future__ import annotations

import sys

import matplotlib
from nuscenes.nuscenes import NuScenes

from mcap_utils.nuscenes_visu.definitions import CLIParameter, NuscenesCameras

matplotlib.use("TkAgg")


def load_nuscenes(argv: list[str]):
    args = CLIParameter.parse_args(argv=argv)

    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_data_root, verbose=True)

    my_scene = nusc.scene[0]
    my_sample = nusc.get("sample", my_scene["first_sample_token"])

    for cam in NuscenesCameras:
        cam_front_data = nusc.get("sample_data", my_sample["data"][cam])
        a = nusc.render_sample_data(cam_front_data["token"])
        print()
    cam_front_data = nusc.get("sample_data", my_sample["data"][NuscenesCameras.CAM_BACK_RIGHT])
    a = nusc.render_sample_data(cam_front_data["token"])

    my_annotation_token = my_sample["anns"][18]
    my_annotation_metadata = nusc.get("sample_annotation", my_annotation_token)
    my_annotation_metadata

    nusc.render_annotation(my_annotation_token)

    my_instance = nusc.instance[599]
    my_instance
    nusc.render_annotation(my_instance["first_annotation_token"])
    nusc.render_annotation(my_instance["last_annotation_token"])

    print()


def main(argv: list[str]):
    load_nuscenes(argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
