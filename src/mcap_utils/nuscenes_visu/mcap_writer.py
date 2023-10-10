"""Contains functionality to simplify writing of mcap files with focus on visualizing nuscenes data."""
import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CircleAnnotation_pb2 import CircleAnnotation
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer


def get_protobuf_timestamp(timestamp: int) -> Timestamp:
    return Timestamp(seconds=timestamp // 1_000_000_000, nanos=timestamp % 1_000_000_000)


class McapWriter:
    WORLD_FRAME_ID = "world"

    def __init__(self, writer: Writer) -> None:
        self.writer = writer

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
        pass
