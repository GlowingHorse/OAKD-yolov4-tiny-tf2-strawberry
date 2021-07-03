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

from nets.yolo4_tiny import yolo_eval
from utils.utils import letterbox_image
from nets.pruned_yolo4_tiny import yolo_body

class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch5001-Total_Loss1.9758-Val_Loss2.2896.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/strawberry_class.txt',
        "score": 0.5,
        "iou": 0.3,
        "eager": True,
        "max_boxes": 100,
        "model_image_size": (416, 416),
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        if self.eager:
            self.input_image_shape = Input([2, ], batch_size=1)
            # (1, 18, 13, 13) conv17
            # (1, 18, 26, 26) conv20
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                             arguments={'anchors': self.anchors, 'num_classes': len(self.class_names),
                                        'image_shape': self.model_image_size,
                                        'max_boxes': self.max_boxes, 'score_threshold': self.score, 'eager': True,
                                        'letterbox_image': self.letterbox_image})(inputs)
            self.ori_yolo_model = self.yolo_model
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2,))

            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                              num_classes, self.input_image_shape,
                                                              max_boxes=self.max_boxes,
                                                              score_threshold=self.score, iou_threshold=self.iou,
                                                              letterbox_image=self.letterbox_image)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    def detect_image(self, image):
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        aaa = image_data[:10, :10, 0]
        # image_data /= 255.
        # ---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # ---------------------------------------------------------#
        if self.eager:
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
            ori_two_conv_output = self.ori_yolo_model(image_data, training=False)
            ori_two_conv_output = [ori_conv_res.numpy() for ori_conv_res in ori_two_conv_output]
        else:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        aaa = ori_two_conv_output[0][0, :, :, 0]
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def close_session(self):
        self.sess.close()
