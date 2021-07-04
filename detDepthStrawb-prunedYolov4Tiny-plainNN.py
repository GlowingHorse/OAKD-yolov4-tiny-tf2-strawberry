#!/usr/bin/env python3

"""
Detect strawberry and label its location in spatial
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
from utils.yolo_utils import yolo_res
from PIL import ImageFont
import time
from datetime import datetime, timedelta


# tiny yolo v4 label texts
labelMap = [
    "MatureStrawberry", "GreenStrawberry"
]

syncNN = True
subpixel = False
downscaleColor = True
TARGET_SHAPE = (416, 416)

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/prunedYolov4TinyStrawberry.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# -------彩色图像-----------------------------------------------------
print("Creating Color Camera...")
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Color cam: 1920x1080
# Mono cam: 640x400
cam_rgb.setIspScale(2, 3)  # To match 400P mono cameras
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.initialControl.setManualFocus(130)

# for yolo-v4-tiny
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setPreviewSize(TARGET_SHAPE[0], TARGET_SHAPE[1])
cam_rgb.setInterleaved(False)

# isp output linked to XLinkOut
isp_xout = pipeline.createXLinkOut()
isp_xout.setStreamName("cam")
cam_rgb.isp.link(isp_xout.input)

# -------网络-----------------------------------------------------
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nnBlobPath)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam_rgb.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# xout_passthrough = pipeline.createXLinkOut()
# xout_passthrough.setStreamName("pass")
# xout_passthrough.setMetadataOnly(True)
# detection_nn.passthrough.link(xout_passthrough.input)

# -------深度图像-----------------------------------------------------
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
stereo = pipeline.createStereoDepth()

stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
stereo.setConfidenceThreshold(255)

stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
# stereo.depth.link(xout_depth.input)

# -------空间位置计算-----------------------------------------------------
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

spatialLocationCalculator.passthroughDepth.link(xout_depth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.setWaitForConfigInput(False)

# --------这部分可以放到实时生成的部分---------
topLeft_ori = dai.Point2f(0.5, 0.5)
bottomRight_ori = dai.Point2f(0.51, 0.51)
config_ori = dai.SpatialLocationCalculatorConfigData()
config_ori.depthThresholds.lowerThreshold = 100
config_ori.depthThresholds.upperThreshold = 10000
config_ori.roi = dai.Rect(topLeft_ori, bottomRight_ori)
spatialLocationCalculator.initialConfig.addROI(config_ori)
# ---------------------------------------

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


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


def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]


def dispay_depth(frame, name):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    cv2.imshow(name, frame_colored)
    return frame_colored


font = ImageFont.truetype(font='font/simhei.ttf',
                          size=np.floor(3e-2 * 416 + 0.5).astype('int32'))

colors = [(255, 0, 0)]
color = (255, 255, 255)
stepSize = 0.05

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Output queue will be used to get the depth frames from the outputs defined above
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    frameRgb = None
    frameDepth = None

    while True:
        start_time = time.time()
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

            # Get NN output timestamp from the passthrough
            in_rgb = q_color.get()
            frame = in_rgb.getCvFrame()
            frame = crop_to_square(frame)
            frame = cv2.resize(frame, TARGET_SHAPE)
            frame = np.ascontiguousarray(frame)

            cfg = dai.SpatialLocationCalculatorConfig()
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

                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000

                topLeft = dai.Point2f(left/416, top/416)
                bottomRight = dai.Point2f(right/416, bottom/416)
                config.roi = dai.Rect(topLeft, bottomRight)
                cfg.addROI(config)

            # if cfg is not None:
            if len(cfg.getConfigData()) != 0:
                spatialCalcConfigInQueue.send(cfg)
            else:
                config_ori = dai.SpatialLocationCalculatorConfigData()
                config_ori.depthThresholds.lowerThreshold = 100
                config_ori.depthThresholds.upperThreshold = 10000
                config_ori.roi = dai.Rect(topLeft_ori, bottomRight_ori)
                cfg.addROI(config_ori)
                spatialCalcConfigInQueue.send(cfg)

            in_depth = q_depth.tryGet()
            inDepthAvg = spatialCalcQueue.tryGet()
            if in_depth is not None and inDepthAvg is not None:
                depth_frame = in_depth.getFrame()
                maxDisparity = 95
                if subpixel:
                    maxDisparity *= 32
                depth_frame_processed = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depth_frame_processed = cv2.equalizeHist(depth_frame_processed)

                depth_frame_processed = crop_to_square(depth_frame_processed)
                depth_frame_processed = cv2.resize(depth_frame_processed, TARGET_SHAPE)
                depth_frame_processed = np.ascontiguousarray(depth_frame_processed)

                spatialData = inDepthAvg.getSpatialLocations()
                for depthData in spatialData:
                    roi = depthData.config.roi
                    roi = roi.denormalize(width=depth_frame_processed.shape[1], height=depth_frame_processed.shape[0])
                    xmin = int(roi.topLeft().x)
                    ymin = int(roi.topLeft().y)
                    xmax = int(roi.bottomRight().x)
                    ymax = int(roi.bottomRight().y)

                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.rectangle(depth_frame_processed, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                    cv2.putText(depth_frame_processed, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                                fontType,
                                0.5, color)
                    cv2.putText(depth_frame_processed, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                                fontType,
                                0.5, color)
                    cv2.putText(depth_frame_processed, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                                fontType,
                                0.5, color)

                cv2.imshow("rgb", frame)
                cv2.imshow("depth", depth_frame_processed)

        start_time = 0
        newConfig = False
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # elif key == ord('w'):
        #     if topLeft.y - stepSize >= 0:
        #         topLeft.y -= stepSize
        #         bottomRight.y -= stepSize
        #         newConfig = True
        # elif key == ord('a'):
        #     if topLeft.x - stepSize >= 0:
        #         topLeft.x -= stepSize
        #         bottomRight.x -= stepSize
        #         newConfig = True
        # elif key == ord('s'):
        #     if bottomRight.y + stepSize <= 1:
        #         topLeft.y += stepSize
        #         bottomRight.y += stepSize
        #         newConfig = True
        # elif key == ord('d'):
        #     if bottomRight.x + stepSize <= 1:
        #         topLeft.x += stepSize
        #         bottomRight.x += stepSize
        #         newConfig = True
        #
        # if newConfig:
        #     cfg = dai.SpatialLocationCalculatorConfig()
        #
        #     config = dai.SpatialLocationCalculatorConfigData()
        #     config.depthThresholds.lowerThreshold = 100
        #     config.depthThresholds.upperThreshold = 10000
        #
        #     topLeft = dai.Point2f(0.4, 0.4)
        #     bottomRight = dai.Point2f(0.6, 0.6)
        #     config.roi = dai.Rect(topLeft, bottomRight)
        #     cfg.addROI(config)
        #
        #     config2 = dai.SpatialLocationCalculatorConfigData()
        #     config2.depthThresholds.lowerThreshold = 100
        #     config2.depthThresholds.upperThreshold = 10000
        #
        #     topLeft2 = dai.Point2f(0.6, 0.6)
        #     bottomRight2 = dai.Point2f(0.7, 0.7)
        #     config2.roi = dai.Rect(topLeft2, bottomRight2)
        #     cfg.addROI(config2)
        #
        #     spatialCalcConfigInQueue.send(cfg)
