#----------------------------------------------------#
#   get test set detection-result and images-optional
#----------------------------------------------------#
import colorsys
import copy
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm

from nets.pruned_yolo4_tiny import yolo_body, yolo_eval
from utils.utils import letterbox_image
from yolo import YOLO

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   get all classes
    #---------------------------------------------------#
    def generate(self):
        self.score = 0.01
        self.iou = 0.5
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   Calculate the number of prior boxes and the number of types
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   load model
        #---------------------------------------------------------#
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        #   bounding box color
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        #---------------------------------------------------------#
        #   post process for yolo_eval func, e.g., NMS, decode
        #---------------------------------------------------------#
        if self.eager:
            self.input_image_shape = Input([2,],batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size, 
                'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes, 'letterbox_image': self.letterbox_image})(inputs)
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2, ))
            
            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                    num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                    score_threshold=self.score, iou_threshold=self.iou, letterbox_image=self.letterbox_image)

    def detect_image(self, image_id, image):
        f = open("./input/detection-results/"+image_id+".txt","w") 

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.

        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        if self.eager:

            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        else:

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(float(out_scores[i]))

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

yolo = mAP_YOLO()

image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    yolo.detect_image(image_id,image)
    
print("Conversion completed!")
