   

# coding: utf-8

# Builds the __AttentionVGG__ network.
# ===================================
# Implements the _inference/loss/training pattern_ for model building.
# 1. inference() - Builds the model as far as is required for running the network forward to make predictions.
# 2. loss() - Adds to the inference model the layers required to generate loss.
# 3. training() - Adds to the loss model the Ops required to generate and apply gradients.
#
# This file is used by the various "fully_connected_*.py" files and not meant to be run.

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import math

import tensorflow as tf
import patch_size
from tensorflow.contrib.framework.python.ops import arg_scope


# In[3]:

# The IndianPines dataset has 16 classes, representing different kinds of land-cover.
NUM_CLASSES = 16    # change to 16 in originaldata tell anirban

# We have chopped the IndianPines image into 28x28 pixels patches.
# We will classify each patch
IMAGE_SIZE = patch_size.patch_size
KERNEL_SIZE = 3 #before it was 5 for 37x37
CHANNELS = 220
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS


# Build the IndianPines model up to where it may be used for inference.
# --------------------------------------------------
# Args:
# * images: Images placeholder, from inputs().
# * hidden1_units: Size of the first hidden layer.
# * hidden2_units: Size of the second hidden layer.
#
# Returns:
# * softmax_linear: Output tensor with the computed logits.

# In[5]:
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.00, stddev)
concat_dim = 3

def inception_arg_scope(weight_decay=0.0004,
                        use_batch_norm=False,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=trunc_normal(0.1),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        padding='SAME',
        normalizer_params=normalizer_params) as sc:
      return sc


def inference(images, conv1_channels, conv2_channels, fc1_units, fc2_units):
    """Build the IndianPines model up to where it may be used for inference.
    Args:
    images: Images placeholder, from inputs().
    conv1_channels: Number of filters in the first convolutional layer.
    conv2_channels: Number of filters in the second convolutional layer.
    fc1_units = Number of units in the first fully connected hidden layer
    fc2_units = Number of units in the second fully connected hidden layer

    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    
    # Conv 1
    with tf.variable_scope('conv_pool1') as scope:
        x_image = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
        net = slim.conv2d(x_image, 64, [3,3], padding='SAME')
        net = slim.max_pool2d(net, [2, 2], stride=2)
    
    with tf.variable_scope('inception1') as scope:
    	with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, 128, [1, 1],
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, 128, [1, 1],
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 128, [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 128, [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          #branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              net, 128, [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
    
    with tf.variable_scope('inception2') as scope:
    	with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, 128, [1, 1],
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, 128, [1, 1],
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 128, [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 128, [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          #branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              net, 128, [1, 1],
              scope='Conv2d_0b_1x1')
        nnet = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        net = nnet + net
            
    net = slim.max_pool2d(net, [2, 2], stride=2)
    print(net.shape.as_list())
    h_pool2_flat = tf.reshape(net,[int(net.shape[0]),-1])
    size_after_flatten = int(h_pool2_flat.shape[-1])
    print(fc1_units,size_after_flatten)

    # FC 1
    with tf.variable_scope('h_FC1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([size_after_flatten, fc1_units],
                                stddev=1.0 / math.sqrt(float(size_after_flatten))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]),
                             name='biases')
        h_FC1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name=scope.name)

    # FC 2
    with tf.variable_scope('h_FC2'):
        weights = tf.Variable(
            tf.truncated_normal([fc1_units, fc2_units],
                                stddev=1.0 / math.sqrt(float(fc1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc2_units]),
                             name='biases')
        h_FC2 = tf.nn.relu(tf.matmul(h_FC1, weights) + biases, name=scope.name)

    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([fc2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(h_FC2, weights) + biases


    return logits






# Define the loss function
# ------------------------

# In[6]:

def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


# Define the Training OP
# --------------------

# In[8]:

def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Define the Evaluation OP
# ----------------------

# In[9]:

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

