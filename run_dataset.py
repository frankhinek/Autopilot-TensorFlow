"""NVIDIA end-to-end deep learning inference for self-driving cars.

This script loads a pretrained model and performs inference based on that model
using jpeg images as input and produces an output of steering wheel angle as
proportions of a full turn.
"""
import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
from glob import glob1
import config

# Directory and File locations
STEERING_WHEEL_IMAGE = "steering_wheel_image.jpg"

# Create session and restore model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, config.MODELFILE)

# Read in steering wheel image
img = cv2.imread(STEERING_WHEEL_IMAGE ,0)
rows,cols = img.shape

# Count the number of JPEG video frames in the dataset directory
frame_count = len(glob1(config.DATASETDIR,"*.jpg"))
print(frame_count)
# Initialize variables used during the inference loop
smoothed_angle = 0.0
i = 0

# Inference loop. Run through all the images in the dataset and perform
# inference.  Break loop if the 'q' key is pressed.
while((cv2.waitKey(10) & 0xFF) != ord('q') and i < frame_count):

    # Read in the nex 256 x 455 RGB image
    full_image = scipy.misc.imread(config.DATASETDIR + str(i) + ".jpg", mode="RGB")

    # Crop to last 150 rows, resize to 66 x 200, and scale to interval [0,1]
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    # Perform inference
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    # Clear the terminal
    call("clear")

    # Print out the current frame, percent complete, and predicted steering
    # wheel angle
    print("Frame [%5d => %2d%%]: Predicted steering angle: %3.0f degrees" % (i, i*100.0/frame_count, degrees))

    # Display the current frame
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    cv2.moveWindow("frame", 400, 200)

    # Make smooth angle transitions by turning the steering wheel based on the
    # difference of the current angle and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))

    # Display the steering wheel image at the inferred angle
    cv2.imshow("steering wheel", dst)
    cv2.moveWindow("steering wheel", 500, 500)

    # Increment image index
    i += 1

cv2.destroyAllWindows()
