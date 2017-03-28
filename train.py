import os
import tensorflow as tf
import driving_data
import model

#
# Initialize parameters
#

# Directory and File locations
MODELDIR = './save'
LOGDIR = './logs'

# Learning parameters
L2NormConst = 0.001
learning_rate = 0.0001
epochs = 30
batch_size = 100

#
# Build a steering model
#

# Initialize TensorFlow session
sess = tf.InteractiveSession()

#
# Loss function
#

# Mean square error loss
mse = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))

# L2 regularization loss
train_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# Total training loss
loss =  mse + l2_loss

#
# Prepare training
#

# Use Adam Optimizer for gradient decent
optimizer = tf.train.AdamOptimizer(learning_rate)

# Training step is to minimize total loss
train_step = optimizer.minimize(loss)

# Initialize parameters
sess.run(tf.global_variables_initializer())

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Create checkpoint saver
saver = tf.train.Saver()

# Op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter(LOGDIR, graph=tf.get_default_graph())

#
# Training
#

# Train over the dataset
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):

    # Load training images and labels
    xs, ys = driving_data.LoadTrainBatch(batch_size)

    # Run gradient decent with training data using dropout
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})

    # Display loss
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # Write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * batch_size + i)

    # Save checkpoint
    if i % batch_size == 0:
      if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)
      checkpoint_path = os.path.join(MODELDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

# Display instructions for TensorBoard
print("Run the command line:\n" \
          "--> tensorboard --logdir=" + LOGDIR + \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
