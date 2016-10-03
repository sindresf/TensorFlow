from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
tf.logging.set_verbosity(tf.logging.INFO)


# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST,
                                                   target_dtype=np.int)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="tmp/iris_model",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=0.5
                                            ))
validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data,
                                                                 test_set.target,
                                                                 every_n_steps=50,
                                                                 metrics=validation_metrics,
                                                                 early_stopping_metric="loss",
                                                                 early_stopping_metric_minimize=True,
                                                                 early_stopping_rounds=200)

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1500,
               monitors=[validation_monitor])

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))