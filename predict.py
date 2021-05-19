import tensorflow as tf
from PIL import Image

from yolo import YOLO

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO()

# while True:
img_path = './img/gear1n_(14)_03_03.jpg'
image = Image.open(img_path)
r_image = yolo.detect_image(image)
r_image.show()
print('finished')
