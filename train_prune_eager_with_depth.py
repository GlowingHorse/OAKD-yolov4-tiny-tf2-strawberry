import os
import time
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.loss import yolo_loss
from nets.pruned_yolo4_tiny import yolo_body
from utils.utils_depth import get_random_data


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape,
                   anchors, num_classes, random=True,
                   depth_names=None):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                seed = np.random.randint(0, 10000)
                np.random.seed(seed)
                np.random.shuffle(annotation_lines)
                np.random.seed(seed)
                np.random.shuffle(depth_names)
                assert annotation_lines[0][108:108+5] == depth_names[0][116:116+5]

            image, box = get_random_data(annotation_lines[i], input_shape,
                                         random=random, depth_name=depth_names[i])
            i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield image_data, y_true[0], y_true[1]


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]


    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # -----------------------------------------------------------#
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # -----------------------------------------------------------#
    #   [6,2] -> [1,6,2]
    # -----------------------------------------------------------#
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # -----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # -----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        # -----------------------------------------------------------#
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # -----------------------------------------------------------#
        #   intersect_area  [n,6]
        #   box_area        [n,1]
        #   anchor_area     [1,6]
        #   iou             [n,6]
        # -----------------------------------------------------------#
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # -----------------------------------------------------------#
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # -----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    # -----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    # -----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def get_train_step_fn():
    @tf.function
    def train_step(imgs, yolo_loss, targets, net, optimizer, regularization, normalize):
        with tf.GradientTape() as tape:
            P5_output, P4_output = net(imgs, training=True)
            args = [P5_output, P4_output] + targets
            loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                loss_value = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value

    return train_step


def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors,
                  num_classes, label_smoothing, regularization=False, train_step=None,
                  save_flag=False, log_dir='./logs_depth'):
    loss = 0
    val_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]
            loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization, normalize=normalize)
            loss = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1),
                                'lr': optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]

            P5_output, P4_output = net(images)
            args = [P5_output, P4_output] + targets
            loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                loss_value = tf.reduce_sum(net.losses) + loss_value
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Curr lr is: {:.8f}'.format(optimizer._decayed_lr(tf.float32).numpy()))
    if save_flag:
        net.save_weights(log_dir + '/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5' % (
        (epoch + 1), loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
        net.save(log_dir + '/yolov4TinyStrawberry')


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    flag_read_weight = False
    Freeze_epoch = None

    batch_size = 32
    batch_size_eval = 4
    annotation_path = '2007_train.txt'
    log_dir = './logs_depth'
    classes_path = 'model_data/strawberry_class.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    weights_path_my = log_dir + '/Epoch570-Total_Loss1.9499-Val_Loss2.2745.h5'
    weights_path = 'model_data/yolov4_tiny_weights_coco.h5'
    input_shape = (416, 416)
    normalize = True

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    label_smoothing = 0

    regularization = True

    image_input = Input(shape=(None, None, 4))
    h, w = input_shape
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 2, num_classes)

    if flag_read_weight:
        print('Load weights {}.'.format(weights_path_my))
        model_body.load_weights(weights_path_my, by_name=True, skip_mismatch=True)
    else:
        print('Load weights {}.'.format(weights_path))
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    rgb_position = lines[0].find('rgb-')

    path_to_depth = os.listdir(r"./VOCdevkit/VOC2007/JPEGImages/depth/")
    total_depth_names = []
    for depth_dir in path_to_depth:
        if depth_dir.endswith(".jpeg"):
            total_depth_names.append(lines[0][:rgb_position] + 'depth/' + depth_dir)

    lines.sort()
    total_depth_names.sort()
    # depth_lines = []
    # for i_single_line in range(len(lines)):
    #     single_line = lines[i_single_line]
    #     single_line.split()[0].replace('rgb-', 'depth-')
    #     depth_name_prefix = single_line[:10]
    #     depth_lines.append()

    AUTOTUNE = tf.data.AUTOTUNE

    if True:
        Freeze_epoch = Freeze_epoch if Freeze_epoch is not None else 0
        Epoch = 50000

        # start from 0
        # learning_rate_base = 1e-4

        # retrain
        learning_rate_base = 1e-5

        gen = data_generator(lines[:num_train], batch_size, input_shape,
                             anchors, num_classes, depth_names=total_depth_names[:num_train])
        gen_val = data_generator(lines[num_train:], batch_size_eval, input_shape,
                                 anchors, num_classes, depth_names=total_depth_names[num_train:])

        epoch_size = num_train // batch_size
        epoch_size_val = num_val // batch_size_eval

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_base,
            decay_steps=50,
            decay_rate=0.98,
            staircase=True
        )
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for epoch in range(Freeze_epoch, Epoch):
            if epoch % 500 == 0 or epoch == Epoch-1:
                save_flag = True
            else:
                save_flag = False
            fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val,
                          Epoch, anchors, num_classes, label_smoothing, regularization, get_train_step_fn(),
                          save_flag=save_flag, log_dir=log_dir)
