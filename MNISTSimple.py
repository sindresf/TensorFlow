import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

images = tf.placeholder(tf.float32, [None, 784])
one_hot_vector_model = tf.placeholder(tf.float32, [None, 10])

Weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

model = tf.nn.softmax(tf.matmul(images, Weights) + bias)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(one_hot_vector_model * tf.log(model), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()


def train_classifier(dataset):
    sesh = tf.Session()
    sesh.run(init)

    for i in range(1000):
        batch_xs, batch_ys = dataset.train.next_batch(100)
        sesh.run(train_step, feed_dict={images: batch_xs, one_hot_vector_model: batch_ys})

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_vector_model, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sesh.run(accuracy, feed_dict={images: dataset.test.images, one_hot_vector_model: dataset.test.labels}))


train_classifier(mnist)
