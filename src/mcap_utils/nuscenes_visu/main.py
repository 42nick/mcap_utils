from __future__ import annotations

import sys

import matplotlib
from mcap_protobuf.writer import Writer
from nuscenes.nuscenes import NuScenes

from mcap_utils.nuscenes_visu.definitions import CLIParameter, NuscenesCameras
from mcap_utils.nuscenes_visu.mcap_writer import McapWriterNuscenes

matplotlib.use("TkAgg")


def load_nuscenes(argv: list[str]):
    args = CLIParameter.parse_args(argv=argv)

    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_data_root, verbose=True)

    with open("z.mcap", "wb") as f, Writer(f) as writer:
        mcap_writer = McapWriterNuscenes(writer=writer, nusc=nusc)

        my_scene = nusc.scene[0]
        sample = None

        frame_cnt = 0

        while True:
            # if frame_cnt > 10:
            #     break

            if sample is None:
                sample = nusc.get("sample", my_scene["first_sample_token"])
            elif sample["next"] == "":
                break
            else:
                sample = nusc.get("sample", sample["next"])

            for camera_name in NuscenesCameras:
                sample_data = nusc.get("sample_data", sample["data"][camera_name])

                # add nuscenes camera information
                mcap_writer.add_nuscenes_camera_pose(
                    sample_data=sample_data,
                    camera_topic_name=sample_data["channel"],
                )

                # add image to mcap
                mcap_writer.add_nuscenes_image(sample_data=sample_data)

                # get ego pose
                ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
                mcap_writer.add_nuscenes_ego_pose(nuscenes_egopose_data=ego_pose, flag_add_point_cloud=True)

            # add 3d box annotations to mcap
            mcap_writer.add_nuscenes_3d_box_annotations(sample=sample)

            frame_cnt += 1

    print()


def main(argv: list[str]):
    load_nuscenes(argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
