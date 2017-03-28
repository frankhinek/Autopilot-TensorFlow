import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
from glob import glob1

DATASET_DIRECTORY = "driving_dataset/"

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread("steering_wheel_image.jpg",0)
rows,cols = img.shape

# Count the number of JPEG video frames in the dataset directory
frame_count = len(glob1(DATASET_DIRECTORY,"*.jpg"))

# Initialize variables
smoothed_angle = 0.0
i = 0

while(cv2.waitKey(10) != ord('q') and i < frame_count):
    full_image = scipy.misc.imread(DATASET_DIRECTORY + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    call("clear")
    print("Frame [%5d => %2d%%]: Predicted steering angle: %3.0f degrees" % (i, i*100.0/45568, degrees))
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
