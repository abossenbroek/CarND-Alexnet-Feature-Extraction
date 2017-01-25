import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np

# TODO: Load traffic signs data.
training_file = 'train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X, y = train['features'], train['labels']

n_classes = len(np.unique(y_train))

X_mean = np.mean(X_train)
X -= X_mean


# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, [227, 227])


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc9W = tf.Variable(n_classes)
fc9b = tf.Variable(n_classes)
logits = tf.nn.xw_plus_b(fc7, fc9W, fc9b)
probabilities = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.


# TODO: Train and evaluate the feature extraction model.
