from __future__ import annotations
from io import BytesIO

import sys

import matplotlib
from mcap_protobuf.writer import Writer
from nuscenes.nuscenes import NuScenes

from mcap_utils.nuscenes_visu.definitions import CLIParameter, NuscenesCameras
from mcap_utils.nuscenes_visu.mcap_writer import McapWriter

matplotlib.use("TkAgg")


def load_nuscenes(argv: list[str]):
    args = CLIParameter.parse_args(argv=argv)

    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_data_root, verbose=True)

    with open("z.mcap", "wb") as f, Writer(f) as writer:
        mcap_writer = McapWriter(writer=writer)

        my_scene = nusc.scene[0]
        sample = None

        while True:
            if sample is None:
                sample = nusc.get("sample", my_scene["first_sample_token"])
            elif sample["next"] == "":
                break
            else:
                sample = nusc.get("sample", sample["next"])

            sample_data = nusc.get("sample_data", sample["data"][NuscenesCameras.CAM_FRONT])

            # get ego pose
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

            mcap_writer.add_nuscenes_ego_pose(nuscenes_egopose_data=ego_pose, flag_add_point_cloud=True)

    print()


def main(argv: list[str]):
    load_nuscenes(argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
