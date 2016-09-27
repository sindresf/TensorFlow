from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.nn import weighted_cross_entropy_with_logits
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

### FIGURE OUT THIS IMPORT OXEN SHITT!!
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def vanilla_conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')

# First convolutional layer
W_Conv1 = weight_variable([5,5,1,32]) #32 features for 5x5 patches in the image
b_conv1 = bias_variable([32])

#making the flattened images into 4D tensors
fourD_images = tf.reshape(images, [-1,28,28,1]) # [batch, height, width, channels]

# applying convolution on the image trough the first layer
h_conv1 = tf.nn.relu(vanilla_conv2D(fourD_images, W_Conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_Conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# not the application on the Pool from layer 1
h_conv2 = tf.nn.relu(vanilla_conv2D(h_pool1, W_Conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# HERE WE GO BACK TO FULLY CONNECTED
# think of this as after having studied the parts, we step back and consider the entirety
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# need to reshape it again, but now it's the "image" after layer 2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#This is "dropout"
#to reduce the chance of outfitting, neurons are stochastically left out of the weight propagation during training
# Tensorflow has it built in and ready! :)
# placeholder stuff, lots of automagics
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

convolution_output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#### WHERE THINGS GET NASTY!! (aka actually run)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_classes*tf.log(convolution_output), reduction_indices=[1]))
training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(convolution_output,1), tf.argmax(output_classes,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sesh.run(tf.initialize_all_variables())

def run_conv(runs, batchsize, neuron_keep_prob, log_step):
    for i in range(runs):
        batch = mnist.train.next_batch(batch_size)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                images:batch[0], output_classes: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g percent" %(i, round(train_accuracy*100,2)))
        training_step.run(feed_dict={images:batch[0], output_classes:batch[1], keep_prob: neuron_keep_prob})

runs = 5000
batchsize = 45
keeps = 0.53
log_step = 100

run_conv(runs,batch_size,keeps,log_step)

print("test accuracy %g percent"%round(accuracy.eval(feed_dict={
    images:mnist.test.images, output_classes: mnist.test.labels, keep_prob: 1.0
})*100,2))


