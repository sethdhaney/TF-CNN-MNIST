### tf-mnist-nlcnn.py #####
# This file implements a convolutional neural network 
# using Tensor Flow to solve the MNIST handwriting
# recognition problem. 
# 
# We also use tensorboard to visualize the model and training protocol
#
# Seth Haney, Ph. D. 5/22/2017
###############################

################################################
### IMPORT THE DATA ###

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### IMPORT TF and start session

import tensorflow as tf

#tf.reset_default_graph()
sess = tf.InteractiveSession()

### Input and output placeholders - NOTE y_ is true output y is predicted

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


## DEFINE Weight and bias variables and sizes
# Weights are for a 5x5 rec. field, 1 color, and 32 features
# a bias is given for each feature.


#Note image sizes are 28 x 28 pixels
x_image = tf.reshape(x, [-1,28,28,1])
#tf.summary.image('input', x_image,3)

# Convolution Layers
W1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
		+b1)

W2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='SAME')
		+b2)

# Fully Connected Layer
size_in = 7*7*64
Wc1 =  tf.Variable(tf.truncated_normal([size_in, 1024], stddev=0.1))
bc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
fc1 = tf.nn.relu(tf.matmul(tf.reshape(conv2, [-1,size_in]),Wc1)+bc1)
#embedding_input = fc1
#embedding_size = 1024

#Dropout. This cuts neurons at the densely connected layer based on prob. stored here
#keep_prob = tf.placeholder(tf.float32)
#fc1 = tf.nn.dropout(fc1, keep_prob)

# Fully Connected Layer
Wc2 =  tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
bc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.nn.relu(tf.matmul(tf.reshape(fc1, [-1,1024]),Wc2)+bc2)

#define loss function - cross_entropy
with tf.name_scope('Xent'):
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			 logits=y_conv, labels=y_))
	#tf.summary.scalar('xent', cross_entropy)

#define trianing step with Adam Optimizer
with tf.name_scope('Train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#determine accuracy 
with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#tf.summary.scalar('Accuracy', accuracy)

#summ = tf.summary.merge_all()
#embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
#assignment = embedding.assign(embedding_input)
#saver = tf.train.Saver()

#Initialize variables in backend
sess.run(tf.global_variables_initializer())

#Initialize TensorBoard Graph Writer
#writer = tf.summary.FileWriter('./TB_info', sess.graph)

#Train
for i in range(2000):
	batch = mnist.train.next_batch(50)
	#train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
	#if i%5 == 0:
		#[train_accuracy, s] = sess.run([accuracy, summ], 
		#	feed_dict={x: batch[0], y_: batch[1]})
		#writer.add_summary(s,i)
	if i%100 == 0:
		#train_accuracy = sess.run(accuracy, 
		#	feed_dict={x: batch[0], y_: batch[1]})
		train_accuracy = accuracy.eval(
			feed_dict={x: batch[0], y_: batch[1]})
		print("step %d, training accuracy %g"%(i, train_accuracy))


print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
