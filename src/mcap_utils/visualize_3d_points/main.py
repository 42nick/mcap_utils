import pandas as pd
from mcap_protobuf.writer import Writer

from mcap_utils.nuscenes_visu.mcap_writer import McapWriter


def main():
    print("Hello World!")

    df = pd.read_csv(".vscode/test3d_points.csv")
    df["z"] = 0

    with open("z.mcap", "wb") as f, Writer(f) as writer:
        mcap_writer = McapWriter(writer=writer)

        mcap_writer.add_ego_pose_point_cloud(
            timestamp_ns=0,
            points=df.values,
        )


if __name__ == "__main__":
    main()
