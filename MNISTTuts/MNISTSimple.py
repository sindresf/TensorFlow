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


def train_classifier(dataset, runs=1000):
    sesh = tf.Session()
    sesh.run(init)

    for i in range(runs):
        batch_xs, batch_ys = dataset.train.next_batch(100)
        sesh.run(train_step, feed_dict={images: batch_xs, one_hot_vector_model: batch_ys})

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_vector_model, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return sesh.run(accuracy, feed_dict={images: dataset.test.images, one_hot_vector_model: dataset.test.labels})


def TC_Stats(tries, per_train_runs, dataset):
    runsum = 0
    #print("starting runs")
    for i in range(1, tries + 1):
        runAcc = train_classifier(dataset, per_train_runs)
        #print(str(i) + ": " + str(round(runAcc * 100, 2)) + "%")
        runsum += runAcc
    avg = round((runsum / tries) * 100, 2)
    #print("over " + str(tries) + " runs the network averaged: " + str(avg) + "%")
    return avg


def compare_run_lengths(tries, runs1, runs2, dataset):
    print("run 1:\n")
    run1avg = TC_Stats(tries, runs1, dataset)
    print("\nrun 2:\n")
    run2avg = TC_Stats(tries, runs2, dataset)

    runDiff = run2avg - run1avg
    print("run 1: " + str(run1avg) + "%, run 2: " + str(run2avg) + "%")
    print("with the set config, run 2 did " + str(runDiff) + "% better")


print("compare training settings.")
tries = int(input("tries:"))
runs1 = int(input("runs 1:"))
runs2 = int(input("runs 2:"))
compare_run_lengths(tries, runs1, runs2, mnist)
