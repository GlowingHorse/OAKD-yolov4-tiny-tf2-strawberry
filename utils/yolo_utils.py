import colorsys
import copy
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
# from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model


# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    feats = tf.convert_to_tensor(feats)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    # ---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   获取每个box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    # -----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    boxes = K.concatenate([
        box_mins[..., 0:1] * image_shape[0],  # y_min
        box_mins[..., 1:2] * image_shape[1],  # x_min
        box_maxes[..., 0:1] * image_shape[0],  # y_max
        box_maxes[..., 1:2] * image_shape[1]  # x_max
    ])
    # -----------------------------------------------------------------#
    #   获得最终得分和框的位置
    # -----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_res(yolo_outputs,
             num_classes=1,
             image_shape=np.array((416, 416)),
             max_boxes=20,
             score_threshold=.45,
             iou_threshold=.45,
             eager=True):
    anchors = np.array([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]])
    anchor_mask = [[3, 4, 5], [1, 2, 3]]
    input_shape = image_shape
    boxes = []
    box_scores = []
    # -----------------------------------------------------------#
    #   对每个特征层进行处理
    # -----------------------------------------------------------#
    for l in range(2):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                    num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # -----------------------------------------------------------#
    #   将每个特征层的结果进行堆叠
    # -----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    box_scores_np = box_scores.numpy()
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # -----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        # -----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # -----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        # -----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # -----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        # -----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_.numpy(), scores_.numpy(), classes_.numpy()