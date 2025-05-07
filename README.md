# Colorize-Grayscale-Images
This project demonstrates how to colorize black-and-white (grayscale) photos using a pre-trained deep learning model in OpenCV's DNN module. It uses a Caffe model trained for image colorization, restoring realistic colors to old or grayscale photos with minimal code.

| Mansion | Grayscale |
|--------|---------|
| ![img1](Pictures/mansion.jpg) | ![img2](Pictures/grayscale.jpg) |

| Dead Colorized | Colorized |
|--------|---------|
| ![img3](Pictures/dead_colorized.jpg) | ![img4](Pictures/colorized.jpg) |

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1200px-Google_Drive_icon_%282020%29.svg.png"  title="Gdown" alt="Gdown" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python
- GDown : Download Resources from Google Drive

## 💻 Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Colorize-Grayscale-Images/blob/main/Code/Colorize_Grayscale_Images.ipynb)

## Models

we have used caffe colorization model.

you can download models here : https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/

## 📝 Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `matplotlib`, `gdown`

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import gdown
```

### Step 2: Download Resources

We need to download the Caffe model and the grayscale image you want to colorize.

We download models from my google drive for protecting the repo in future.

```python
gdown.download(id="1LvgEe5DnG_Vpd6n9laUlfgi1BCrG4dkv", output="pts_in_hull.npy", quiet=False)
gdown.download(id="1yNtqZ0YueocX9TCET2hcK18XRbieR-Jv", output="colorization_release_v2.caffemodel", quiet=False)
gdown.download(id="1hS6a-taPesUwPTqQ8wH32LJqybUov6Vs", output="colorization_deploy_v2.prototxt", quiet=False)
```

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Colorize-Grayscale-Images/main/Pictures/mansion.jpg -O mansion.jpg
```

### Step 3: Load Image

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

Also we normalize the image here.

```python
image = cv.imread('mansion.jpg')
grayscale_image = cv.imread('mansion.jpg', cv.IMREAD_GRAYSCALE)
grayscale_rgb = cv.cvtColor(grayscale_image, cv.COLOR_GRAY2RGB)

scaled = grayscale_rgb.astype("float32")/255.0
lab = cv.cvtColor(scaled, cv.COLOR_RGB2LAB)
```

<div display=flex align=center>
  <img src="/Pictures/grayscale.jpg" width="800px"/>
</div>

### Step 4: Initialize Neural Network

```python
prototxt_src = "colorization_deploy_v2.prototxt"
model_src = "colorization_release_v2.caffemodel"
pts_src = "pts_in_hull.npy"

net = cv.dnn.readNetFromCaffe(prototxt_src, model_src)
kernel = np.load(pts_src)

kernel = kernel.transpose().reshape(2,313,1,1)
class8 = net.getLayerId("class8_ab")
conv8 =  net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [kernel.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype=np.float32)]
```

### Step 5: Resize Image

```python
resized = cv.resize(lab,(224,224))
L_resized = cv.split(resized)[0] # others
L_resized -= 50
```

### Step 6: Forward Pass and colorize the Image

We colorize the image in forward pass it means the image is passing throu the deep neural network.

```python
net.setInput(cv.dnn.blobFromImage(L_resized))
ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_channels = cv.resize(ab_channels, (grayscale_rgb.shape[1], grayscale_rgb.shape[0]))
```

```python
ab_channels *= 1.3
```

### Step 7: Convert LAB to BGR

We convert BGR to LAB here.

```python
L_resized = cv.split(lab)[0]
colorized = np.concatenate((L_resized[:,:,np.newaxis], ab_channels), axis=2)
colorized = cv.cvtColor(colorized,cv.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)
colorized = (255 * colorized).astype("uint8")
```

### Step 8: Boost Saturation

Increasing the saturation with converting it to the HSV color space

```python
hsv = cv.cvtColor(colorized, cv.COLOR_BGR2HSV)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
colorized = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
```

### Step 9: Warm Shift

```python
colorized = colorized.astype(np.float32)
colorized[..., 2] *= 1.03  # Red
colorized[..., 1] *= 1.01  # Green
colorized = np.clip(colorized, 0, 255).astype(np.uint8)
```

### Step 10: Show Image

```python
plt.figure(figsize=[13, 6])
plt.subplot(131),plt.imshow(image[...,::-1]),plt.title('First Image');
plt.subplot(132),plt.imshow(grayscale_rgb[...,::-1]),plt.title('GrayScale');
plt.subplot(133),plt.imshow(colorized[...,::-1]),plt.title('Colorized');
```

<div display=flex align=center>
  <img src="/Pictures/result.jpg"/>
</div>

## 🪪 License

This project is licensed under the MIT License.
