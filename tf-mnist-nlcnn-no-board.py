### tf-mnist-nlcnn.py #####
# This file implements a convolutional neural network 
# using Tensor Flow to solve the MNIST handwriting
# recognition problem. 
#
# Seth Haney, Ph. D. 5/22/2017
###############################

# The following are useful shortcuts for 
# standard methods in TF

# Initialize weights as normal
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


# Initialize Bias as constant 
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Do 2D convulution with unit strides and padding to equalize input and output size
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#Define grid of max pooling as 2x2 with padding to equailize input and output size
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


################################################
### IMPORT THE DATA ###

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### IMPORT TF and start session

import tensorflow as tf
sess = tf.InteractiveSession()

### Input and output placeholders - NOTE y_ is true output y is predicted

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


## DEFINE Weight and bias variables and sizes
# Weights are for a 5x5 rec. field, 1 color, and 32 features
# a bias is given for each feature.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#Note image sizes are 28 x 28 pixels
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer - NOTE: The input size is now 32 - NFeatures from Conv. Layer 1.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer for classification. NOTE: Input size is smaller due to max pooling. 
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout. This cuts neurons at the densely connected layer based on prob. stored here
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Classifying layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#define loss function - cross_entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#define trianing step with Adam Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#determine accuracy 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initialize variables in backend
sess.run(tf.global_variables_initializer())

#Initialize TensorBoard Graph Writer
#writer = tf.summary.FileWriter('./TB_info', sess.graph)

#Train
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
