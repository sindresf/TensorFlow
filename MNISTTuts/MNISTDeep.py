from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.nn import weighted_cross_entropy_with_logits
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sesh = tf.InteractiveSession()

# vector shape
any_batch_size = None
flattened_image_length = 28 * 28  # 784
num_classes = 10

