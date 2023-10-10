# This example writes a single point cloud message.
import struct
import time
from io import BytesIO
from random import random
from typing import List

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CircleAnnotation_pb2 import CircleAnnotation
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer


def main():
    with open("z.mcap", "wb") as f, Writer(f) as writer:
        start = time.time_ns()
        for i in range(10):
            now = start + i * 100_000_000
            write_frame(writer, now)


def write_frame(writer: Writer, now: int):
    width = 128
    height = 128
    data = BytesIO()
    # Generate a geometric pattern using `now` as a seed
    seed = int(now / 100_000_000)
    for y in range(height):
        for x in range(width):
            r = (x + seed) % 256
            g = (y + seed) % 256
            b = (x * y + seed) % 256
            data.write(struct.pack("BBB", r, g, b))

    circles: List[CircleAnnotation] = []
    for _ in range(10):
        circles.append(
            CircleAnnotation(
                timestamp=timestamp(now),
                position=Point2(x=random() * width, y=random() * height),
                diameter=random() * (width / 2),
                thickness=random() * 2,
                outline_color=Color(r=random(), g=random(), b=random(), a=1),
            )
        )

    # /camera/image
    img = RawImage(
        timestamp=timestamp(now),
        frame_id="camera",
        width=width,
        height=height,
        encoding="rgb8",
        step=width * 3,
        data=data.getvalue(),
    )
    writer.write_message(
        topic="/camera/image",
        log_time=now,
        message=img,
        publish_time=now,
    )

    # /camera/calibration
    focal_length_mm = 35.0
    sensor_width_mm = 10.0
    fx = (focal_length_mm / sensor_width_mm) * width
    fy = (focal_length_mm / sensor_width_mm) * height
    cx = width / 2
    cy = height / 2
    cal = CameraCalibration(
        timestamp=timestamp(now),
        frame_id="camera",
        width=width,
        height=height,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 10.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    writer.write_message(
        topic="/camera/calibration",
        log_time=now,
        message=cal,
        publish_time=now,
    )

    # /camera/annotations
    ann = ImageAnnotations(circles=circles)
    writer.write_message(
        topic="/camera/annotations",
        log_time=now,
        message=ann,
        publish_time=now,
    )
    world_frame = FrameTransform(
        timestamp=timestamp(now),
        child_frame_id="camera",
        parent_frame_id="world",
        translation=Vector3(x=np.random.randint(0, 10), y=0, z=0),
        rotation=Quaternion(x=0, y=0, z=0, w=1),
    )
    writer.write_message(
        topic="/tf",
        log_time=now,
        message=world_frame,
        publish_time=now,
    )


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


if __name__ == "__main__":
    main()
