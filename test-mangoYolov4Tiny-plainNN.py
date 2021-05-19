#!/usr/bin/env python3

"""
Tiny-yolo-v4 device side decoding demo
The code is the same as for Tiny-yolo-V3, the only difference is the blob file.
The blob was compiled following this tutorial: https://github.com/TNTWEN/OpenVINO-YOLOV4

这里我使用的是普通nn的输出去获得结果
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import math
import os
from utils.yolo_utils import yolo_res
from PIL import Image, ImageDraw, ImageFont
import time


# tiny yolo v4 label texts
labelMap = [
    "M"
]

syncNN = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/yolov4TinyMango16.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nnBlobPath)

print("Creating Color Camera...")
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(416, 416)
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("rgb")
cam_rgb.preview.link(cam_xout.input)
cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

frame = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)


font = ImageFont.truetype(font='font/simhei.ttf',
                          size=np.floor(3e-2 * 416 + 0.5).astype('int32'))
colors = [(255, 0, 0)]
# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    def get_frame():
        in_rgb = q_rgb.get()
        new_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        return True, np.ascontiguousarray(new_frame)

    while True:
        start_time = time.time()
        read_correctly, frame = get_frame()

        if not read_correctly:
            break

        in_nn = q_nn.tryGet()

        if in_nn is not None:
            all_layer_names = in_nn.getAllLayerNames()
            all_layer_data_l = []
            for i_layer in range(len(all_layer_names)):
                single_layer_data = in_nn.getLayerFp16(all_layer_names[i_layer])
                all_layer_data_l.append(single_layer_data)

            # (1, 18, 13, 13) conv17
            # (1, 18, 26, 26) conv20
            conv17 = np.array(all_layer_data_l[0], dtype=np.float32).reshape([1, 18, 13, 13])
            conv20 = np.array(all_layer_data_l[1], dtype=np.float32).reshape([1, 18, 26, 26])

            conv17_trans = np.transpose(conv17, (0, 2, 3, 1))
            conv20_trans = np.transpose(conv20, (0, 2, 3, 1))
            print("--- %s seconds ---" % (time.time() - start_time))

            # out_boxes, out_scores, out_classes = yolo_res([conv17_trans, conv20_trans, (416, 416)])
            # for i, c in list(enumerate(out_classes)):
            #     box = out_boxes[i]
            #     score = out_scores[i]
            #
            #     top, left, bottom, right = box
            #     top = top - 5
            #     left = left - 5
            #     bottom = bottom + 5
            #     right = right + 5
            #     top = max(0, np.floor(top + 0.5).astype('int32'))
            #     left = max(0, np.floor(left + 0.5).astype('int32'))
            #     bottom = min(416, np.floor(bottom + 0.5).astype('int32'))
            #     right = min(416, np.floor(right + 0.5).astype('int32'))
            #
            #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            #     cv2.putText(frame, 'Mango', (left + 10, top + 20), cv2.FONT_HERSHEY_TRIPLEX,
            #                 0.5, 255)
            #     cv2.putText(frame, f"{int(score * 100)}%", (left + 10, top + 40),
            #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imshow("rgb", frame)
        start_time = 0
        if cv2.waitKey(1) == ord('q'):
            break