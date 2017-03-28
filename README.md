# Autopilot

## Overview

In this project we have a modified implementation of the
[NVIDIA End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
self-driving car model and [this paper](https://arxiv.org/abs/1604.07316). Our
implementation is based on the python TensorFlow implementation by [Sully Chen](https://github.com/SullyChen/Autopilot-TensorFlow).

The project consists of a python-based training script and an inference
implementation.  A convolutional neural network (CNN) is trained to map raw
pixels to steering commands.  The image frames were captured by a dash-mounted
video camera.  At the end of the training, the model is saved. This saved model
is then loaded by the inference implementation to evaluate what the model
predicts for the steering commands.


## Dependencies

**Python Packages**
* OpenCV 3.1.0
* Pillow 4.0.0
* SciPy 0.19.0
* TensorFlow 1.0.1

The commands below will be useful for those using Anaconda/Miniconda and pip to
manage Python packages.  If you use another method you will have to adapt the
steps.
```
conda create -n autopilot python=3.5
source activate autopilot
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL
pip install pillow==4.0.0
conda install -c menpo opencv3=3.1.0
conda install -c anaconda scipy=0.19.0
```

_Note: Only tested with macOS 10.12.3 and Ubuntu 16.04.2 LTS._

**Dataset**

The driving_dataset.zip file contains 45,568 JPEG images of video frames that
are used to train the model and the observed steering wheel angle readings.  It
is 2.2 GB compressed, so you'll want to be on fairly fast Internet connection
before starting the download.
* Download the [driving dataset](http://bit.ly/autopilot-dataset) and unzip it
into the repository folder.
* The path should be something like
`/path/to/repo/Autopilot-TensorFlow/driving_dataset/`.

## Usage

#### Train the Model Using the Prerecorded Training Dataset
The model needs to be trained with the prerecorded driving data.  Training the
model should create a `save` folder that will contain the saved model and
checkpoint files.  We will use the saved model for inference.
* `python train.py` to train the model
* To visualize the training performance using Tensorboard use `tensorboard
--logdir=./logs`, then open http://0.0.0.0:6006/ in your web browser.

#### Evaluate the Model Using the Prerecorded Training Dataset
The `save` directory contains a model that was already trained using the
JPEG and steering wheel readings dataset that you downloaded earlier.  When run
two windows will be launched displaying the video frames captured during the
drive and the steering wheel angle predicted by the pre-trained model.
* `python run_dataset.py` to run the model on the dataset

#### Evaluate the Model Using a Live Webcam Feed
If you have a dash-mounted video camera that is connected to your computer you
can use `run.py` to evaluate the trained model using the captured frames to see
what steering wheel positions it predicts.  I have not tested this code that
was forked from Sully Chen's repository, so you may need to make some
modifications.
* `python run.py` to run the model on a live webcam feed

## Credits
This repository was forked from [SullyChen/Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow),
modified to support TensorFlow 1.0, and expanded with additional documentation.
Full credit for the original code goes to [Sully Chen](https://github.com/SullyChen).
Additional code and comments from [Tomi Maila](https://github.com/tmaila/autopilot)'s
repository.
