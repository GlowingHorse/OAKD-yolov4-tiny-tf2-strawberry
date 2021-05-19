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
from utils.yolo_utils import yolo_res
from PIL import ImageFont
import time


# tiny yolo v4 label texts
labelMap = [
    "MatureStrawberry", "GreenStrawberry"
]

syncNN = True
subpixel = False
downscaleColor = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/prunedYolov4TinyStrawberry.blob')).resolve().absolute())
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

cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.initialControl.setManualFocus(140)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("rgb")

cam_rgb.preview.link(cam_xout.input)
cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# ------------------------------------------------------------------------
# cam = pipeline.createColorCamera()
# cam.setBoardSocket(dai.CameraBoardSocket.RGB)
# cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# if downscaleColor:
#     cam.setIspScale(10, 27)
#
# cam.initialControl.setManualFocus(140)
#
# rgbOut = pipeline.createXLinkOut()
# rgbOut.setStreamName("rgbisp")
# cam.isp.link(rgbOut.input)

# ------------------------------------------------------------
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)
# LR-check is required for depth alignment
# stereo.setLeftRightCheck(True)
# stereo.setRectifyMirrorFrame(True)
stereo.setSubpixel(subpixel)
# stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

left.out.link(stereo.left)
right.out.link(stereo.right)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depth")
# Currently we use the 'disparity' output. TODO 'depth'
stereo.disparity.link(depthOut.input)
# ------------------------------------------------------------


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


frame = None
depth_frame = None
q_depth = None

font = ImageFont.truetype(font='font/simhei.ttf',
                          size=np.floor(3e-2 * 416 + 0.5).astype('int32'))
colors = [(255, 0, 0)]
# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frameRgb = None
    frameDepth = None

    def get_frame():
        in_rgb = q_rgb.get()
        new_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        return True, np.ascontiguousarray(new_frame)

    def get_depthframe():
        packets = device.getOutputQueue("depth").tryGetAll()
        if len(packets) > 0:
            frameDepth = packets[-1].getFrame()
            maxDisparity = 95
            if subpixel:
                maxDisparity *= 32
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1:
                frameDepth = (frameDepth * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 0:
                frameDepth = cv2.applyColorMap(frameDepth, cv2.COLORMAP_HOT)

            # frameDepth = np.array(frameDepth)
            frameDepth = np.ascontiguousarray(frameDepth)
            return True, frameDepth
        else:
            return False, 0

    while True:
        start_time = time.time()
        read_correctly, frame = get_frame()
        read_depth_correctly, depth_frame = get_depthframe()

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
            conv17 = np.array(all_layer_data_l[0], dtype=np.float32).reshape([1, 21, 13, 13])
            conv20 = np.array(all_layer_data_l[1], dtype=np.float32).reshape([1, 21, 26, 26])

            conv17_trans = np.transpose(conv17, (0, 2, 3, 1))
            conv20_trans = np.transpose(conv20, (0, 2, 3, 1))
            print("--- {:.8f} microsecond ---".format(1000000*(time.time() - start_time)))
            print("--- fps: {} ---".format(1/(time.time() - start_time + 0.0000001)))

            out_boxes, out_scores, out_classes = yolo_res([conv17_trans, conv20_trans, (416, 416)], num_classes=2)
            for i, c in list(enumerate(out_classes)):
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(416, np.floor(bottom + 0.5).astype('int32'))
                right = min(416, np.floor(right + 0.5).astype('int32'))

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, labelMap[c], (left + 10, top + 20), cv2.FONT_HERSHEY_TRIPLEX,
                            0.5, 255)
                cv2.putText(frame, f"{int(score * 100)}%", (left + 10, top + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        if read_depth_correctly:
            cv2.imshow("rgb", frame)
            cv2.imshow("depth", depth_frame)

            # if len(depth_frame.shape) < 3:
            #     depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
            # # TODO add a slider to adjust blending ratio
            # blended = cv2.addWeighted(frame, 0.6, depth_frame, 0.4 , 0)
            # cv2.imshow("rgb-depth", blended)
            # frame = None
            # depth_frame = None

        start_time = 0
        if cv2.waitKey(1) == ord('q'):
            break