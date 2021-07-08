# depth-yolov4-tiny-tf2-strawberry-git
### To detect strawberry with [OAK-D](https://opencv.org/opencv-ai-competition-2021/#introSection)
### 0. Prepare your python environment.
tensorflow2 and depthai (newest)
### 1. Prepare your fruit dataset.
My strawberry dataset contains two classes, _i.e._, MatureStrawberry and GreenStrawberry. All data are collected from [Kobayashi Farm](https://kobayashifarm-mitaka.tokyo/) in Mure Mitaka City, Japan. Thank the orchard owner very much for supporting us in collecting fruit data.<br><br>
<img src="https://github.com/GlowingHorse/depth-yolov4-tiny-tf2-strawberry-git/blob/main/img/rgb-00007-16160356916470.jpeg" alt="drawing" width="400"/>
<img src="https://github.com/GlowingHorse/depth-yolov4-tiny-tf2-strawberry-git/blob/main/img/rgb-00357-16160404199341.jpeg" alt="drawing" width="400"/>
<br><br>
You can prepare your dataset by downloading some open dataset or collecting them by yourself. Although we collected color and depth images roughly aligned with the color images, I found that adding a depth channel to the neural network's input does not seem to increase the accuracy obviously. May be due to the imprecise alignment between the depth and rgb images.<br><br>
I manually adjust the camera internal parameters and then use image processing methods to align depth images. But recently the official has published the code for depth and color image alignment. If you want to collect two types of images at the same time, you can refer to [depthai-python](https://github.com/luxonis/depthai-python/blob/main/examples/31_rgb_depth_aligned.py).
### 2. Change files in model_data directory.
Change class information and anchor sizes which can be calculated using `kmeans_for_anchors.py`
### 3. Finetune params in `train_prune_eager.py` based on your needs
Batch size, inital learning rate and learning rate decay should be the parts that need adjustment the most.<br><br>
I analyzed the number of channels that can be pruned for each layer according to the attribution pruning method ([paper](https://www.sciencedirect.com/science/article/pii/S0168169919313717), [code](https://github.com/GlowingHorse/Fast-Mango-Detection)), so the network size is much smaller than original yolov4-tiny. If more image classes need to be detected, the network also needs to be redesigned.
### 4. Save best trained model manually or automatically. (empirically valid loss less than 2)
Our training code mainly refers to the code of [here](https://github.com/bubbliiiing/yolov4-tiny-tf2/blob/master/train.py). The model trained in the non-eager mode always makes an error when trying to convert it to the IR model, so I only train the model in eager mode. However, the non-eager mode can be faster, thus, you can also use the non-eager mode to train to improve efficiency, and then load the weights in eager mode and save it.
### 5. Upload model files to your Googledrive, like:
<img src="https://github.com/GlowingHorse/depth-yolov4-tiny-tf2-strawberry-git/blob/main/img/drive_files.png" alt="drawing" width="800"/>

### 6. Run `convertPbModel-evalData-yolov4.ipynb` in your colab to generate `.blob` file.
Before running, you need to modify the directory where the uploaded model files are stored.
Or you can also refer to [**Converting model to MyriadX blob**](https://docs.luxonis.com/en/latest/pages/model_conversion/) to convert your model. 

### 7. Download .blob file to models directory and use `detDepthStrawb-prunedYolov4Tiny-plainNN.py` to detect your fruits.
Run `.blob` model in your depth camera, and use some printed images or real fruits to test it.<br><br>
<img src="https://github.com/GlowingHorse/depth-yolov4-tiny-tf2-strawberry-git/blob/main/img/detect_strawberry.png" alt="drawing" width="800"/>
