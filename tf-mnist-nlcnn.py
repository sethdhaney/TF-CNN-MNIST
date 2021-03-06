### tf-mnist-nlcnn.py #####
# This file implements a convolutional neural network 
# using Tensor Flow to solve the MNIST handwriting
# recognition problem. 
# 
# We also use tensorboard to visualize the model and training protocol
#
# Seth Haney, Ph. D. 5/22/2017
###############################

#Create Convolution Layer
def conv_layer(conv_in, size_in, size_out):
	with tf.name_scope('conv'):
		initial = tf.truncated_normal([5,5,size_in,size_out], stddev=0.1)
		W = tf.Variable(initial, name='W')
		initial = tf.constant(0.1, shape=[size_out])
		b = tf.Variable(initial, name='b')
		conv = tf.nn.conv2d(conv_in, W, strides=[1, 1, 1, 1], padding='SAME')
		act = tf.nn.relu(conv + b)
		#tf.summary.histogram('Weights', W)
		#tf.summary.histogram('Biases', b)
		#tf.summary.histogram('Activations', act)
		return tf.nn.max_pool(act, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')

#Create a Fully Connected Layer
def fc_layer(fc_in, size_in, size_out):
	with tf.name_scope('fc'):
		initial = tf.truncated_normal([size_in,size_out], stddev=0.1)
		W = tf.Variable(initial, name='W')
		initial = tf.constant(0.1, shape=[size_out])
		b = tf.Variable(initial, name='b')
		act = tf.nn.relu(tf.matmul(fc_in,W) + b)
		#tf.summary.histogram('Weights', W)
		#tf.summary.histogram('Biases', b)
		#tf.summary.histogram('Activations', act)
		return act
				

################################################
### IMPORT THE DATA ###

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### IMPORT TF and start session

import tensorflow as tf

#tf.reset_default_graph()
sess = tf.InteractiveSession()

### Input and output placeholders - NOTE y_ is true output y is predicted

x = tf.placeholder(tf.float32, shape=[None, 784], name='Features')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Classes')


## DEFINE Weight and bias variables and sizes
# Weights are for a 5x5 rec. field, 1 color, and 32 features
# a bias is given for each feature.


#Note image sizes are 28 x 28 pixels
x_image = tf.reshape(x, [-1,28,28,1])
#tf.summary.image('input', x_image,3)

# Convolution Layers
conv1 = conv_layer(x_image, 1, 32)
conv2 = conv_layer(conv1, 32, 64)

# Fully Connected Layer
fc1 = fc_layer(tf.reshape(conv2, [-1,7*7*64]), 7*7*64, 1024)
#embedding_input = fc1
#embedding_size = 1024

#Dropout. This cuts neurons at the densely connected layer based on prob. stored here
#keep_prob = tf.placeholder(tf.float32)
#fc1 = tf.nn.dropout(fc1, keep_prob)

# Fully Connected Layer
y_conv = fc_layer(tf.reshape(fc1, [-1, 1024]), 1024,10)

#define loss function - cross_entropy
with tf.name_scope('Xent'):
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			labels=y_, logits=y_conv), name='xent')
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
saver = tf.train.Saver()

#Initialize variables in backend
sess.run(tf.global_variables_initializer())

#Initialize TensorBoard Graph Writer
#writer = tf.summary.FileWriter('./TB_info', sess.graph)

#Train
for i in range(2001):
	batch = mnist.train.next_batch(100)
	#if i%5 == 0:
		#[train_accuracy, s] = sess.run([accuracy, summ], 
		#	feed_dict={x: batch[0], y_: batch[1]})
		#writer.add_summary(s,i)
	if i%100 == 0:
		train_accuracy = sess.run(accuracy, 
			feed_dict={x: batch[0], y_: batch[1]})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
