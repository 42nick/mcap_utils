"""Contains functionality to simplify writing of mcap files with focus on visualizing nuscenes data."""
import struct
from io import BytesIO
from typing import Union

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CircleAnnotation_pb2 import CircleAnnotation
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer


def get_protobuf_timestamp(timestamp_ns: int) -> Timestamp:
    return Timestamp(seconds=timestamp_ns // 1_000_000_000, nanos=timestamp_ns % 1_000_000_000)


class McapWriter:
    WORLD_FRAME_ID = "world"

    def __init__(self, writer: Writer) -> None:
        self.writer = writer

        self.point_cloud_fields = [
            PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
            PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
            PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
            PackedElementField(name="intensity", offset=12, type=PackedElementField.FLOAT32),
        ]
        self.centerpose = Pose(
            position=Vector3(x=0, y=0, z=0),
            orientation=Quaternion(w=1, x=0, y=0, z=0),
        )
        self.ego_pose_data = BytesIO()

    def add_world_frame(self, timestamp: int):
        self.writer.write_message(
            topic="/frame_transform",
            message=FrameTransform(
                timestamp=get_protobuf_timestamp(timestamp),
                frame_id=self.WORLD_FRAME_ID,
                parent_frame_id="",
                translation=Vector3(x=0, y=0, z=0),
                rotation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            log_time=timestamp,
        )

    def add_nuscenes_ego_pose(
        self, nuscenes_egopose_data: dict[str, Union[str, int, list[float]]], flag_add_point_cloud: bool = False
    ):
        timestamp_ns = nuscenes_egopose_data["timestamp"] * 1_000

        frame_transform_msg = FrameTransform(
            timestamp=get_protobuf_timestamp(timestamp_ns=timestamp_ns),
            child_frame_id="ego_pose",
            parent_frame_id=self.WORLD_FRAME_ID,
            translation=Vector3(
                x=nuscenes_egopose_data["translation"][0],
                y=nuscenes_egopose_data["translation"][1],
                z=nuscenes_egopose_data["translation"][2],
            ),
            rotation=Quaternion(
                x=nuscenes_egopose_data["rotation"][0],
                y=nuscenes_egopose_data["rotation"][1],
                z=nuscenes_egopose_data["rotation"][2],
                w=nuscenes_egopose_data["rotation"][3],
            ),
        )

        self.writer.write_message(
            topic="/frame_transform",
            message=frame_transform_msg,
            log_time=timestamp_ns,
        )

        if flag_add_point_cloud:
            self.add_ego_pose_point_track(
                timestamp_ns=timestamp_ns,
                points=nuscenes_egopose_data["translation"],
            )

    def add_ego_pose_point_track(self, timestamp_ns: int, points: np.ndarray):
        self.ego_pose_data.write(
            struct.pack(
                "<ffff",
                points[0],
                points[1],
                points[2],
                0.5,
            )
        )

        msg = PointCloud(
            frame_id=self.WORLD_FRAME_ID,
            pose=self.centerpose,
            data=self.ego_pose_data.getvalue(),
            fields=self.point_cloud_fields,
            timestamp=get_protobuf_timestamp(timestamp_ns),
            point_stride=16,
        )

        self.writer.write_message(
            topic="/ego_vehicle_track",
            message=msg,
            log_time=timestamp_ns,
        )

    def add_image(
        self,
        image: np.ndarray,
        timestamp: int,
        frame_id: str,
        encoding: str = "rgb8",
    ):
        img = RawImage(
            timestamp=get_protobuf_timestamp(timestamp),
            frame_id=frame_id,
            width=image.shape[1],
            height=image.shape[0],
            encoding=encoding,
            step=image.shape[1] * 3,
            data=image.tobytes(),
        )

        self.writer.write_message(
            topic="/camera/image",
            message=img,
            log_time=timestamp,
        )
