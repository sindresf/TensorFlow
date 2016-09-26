from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.nn import weighted_cross_entropy_with_logits

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sesh = tf.InteractiveSession()

#vector shape
any_batch_size = None
flattened_image_length = 28*28 #784
num_classes = 10

images = tf.placeholder(tf.float32, shape=[any_batch_size, flattened_image_length])
output_classes = tf.placeholder(tf.float32, shape=[any_batch_size, num_classes]) #one hot vector

#Model paramaters
weights = tf.Variable(tf.zeros([flattened_image_length, num_classes])) #[input, output]
biases = tf.Variable(tf.zeros([num_classes])) #per class bias

#actually assigns the parameters passed to the Variables to them, so that they 'exist'
sesh.run(tf.initialize_all_variables())

weighted_images = tf.matmul(images,weights)
softmax = tf.nn.softmax(weighted_images + biases)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_classes*tf.log(softmax), reduction_indices=[1]))

