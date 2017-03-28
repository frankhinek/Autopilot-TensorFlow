# Autopilot

## Overview

A modified implementation of the
[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
paper by NVIDIA using TensorFlow. A convolutional neural network (CNN) is
trained to map raw pixels from a camera to steering commands.

## Dependencies

**Python Packages**
* OpenCV 3.1.0
* Pillow 4.0.0
* SciPy 0.19.0
* TensorFlow 0.12.0

The commands below will be useful for those using Anaconda/Miniconda and pip to
manage Python packages.  If you use another method you will have to adapt the
steps.
```
conda create -n autopilot2 python=2.7
source activate autopilot
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py2-none-any.whl
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

#### Evaluate the Model Using the Prerecorded Training Dataset
The `save` directory contains a model that was already trained using the
JPEG and steering wheel readings dataset that you downloaded earlier.  When run
two windows will be launched displaying the video frames captured during the
drive and the steering wheel angle predicted by the pre-trained model.
* `python run_dataset.py` to run the model on the dataset

#### Evaluate the Model Using a Live Webcam Feed
If you have a dash-mounted video camera that is connected to your computer you
can use `run.py` to evaluate the trained model using the captured frames to see
what steering wheel positions it predicts.
* `python run.py` to run the model on a live webcam feed

#### Retrain the Model Using the Prerecorded Training Dataset
* `python train.py` to re-train the model
* To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ in your web browser.

## Credits
This repository was forked from [SullyChen/Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow), modified to support TensorFlow 1.0, and expanded with additional
documentation.  Full credit for the original code goes to [Sully Chen](https://github.com/SullyChen).
