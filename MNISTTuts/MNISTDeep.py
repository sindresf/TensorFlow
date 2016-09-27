from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.nn import weighted_cross_entropy_with_logits

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

sesh = tf.InteractiveSession()

# vector shape
any_batch_size = None
flattened_image_length = 28 * 28  # 784
num_classes = 10

# x
images = tf.placeholder(tf.float32, shape=[any_batch_size, flattened_image_length])
# y_
output_classes = tf.placeholder(tf.float32, shape=[any_batch_size, num_classes])  # one hot vector

# Model paramaters
# W
weights = tf.Variable(tf.zeros([flattened_image_length, num_classes]))  # [input, output]
# b
biases = tf.Variable(tf.zeros([num_classes]))  # per class bias

# actually assigns the parameters passed to the Variables to them, so that they 'exist'
sesh.run(tf.initialize_all_variables())

weighted_images = tf.matmul(images, weights)
# y
softmax = tf.nn.softmax(weighted_images + biases)

# how the current softmax compares to the actual classes
cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_classes * tf.log(softmax), reduction_indices=[1]))

# defines the whole optimization step,
# with both gradient step, the update step and
# the application of the update steps on the parameters
step_length = 0.5
training_step = tf.train.GradientDescentOptimizer(step_length).minimize(cross_entropy)


def train_for_runs_on_batchsize(runs, batchsize):
    for i in range(runs):
        batch = mnist.train.next_batch(batchsize)
        training_step.run(feed_dict={images: batch[0], output_classes: batch[1]})


training_runs = 1200
batch_size = 140

train_for_runs_on_batchsize(training_runs, batch_size)

correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(output_classes, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(str(round(accuracy.eval(feed_dict={images: mnist.test.images, output_classes: mnist.test.labels})*100,2)) + "%")
