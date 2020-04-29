
# This is an algorithim used in digits recognition which is a basic factor for 
# pattern recognition in machine Learning. 
#Uses the MNIST Data sets with 60,000 training and 10,000 testing Data sets 

#K-Nearest Algorithm
#import TF and numpy\

#STEP 1

import tensorflow as tf
import numpy as np

# Import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

# store the Data
mnist = input_data.read_data_sets("mnist_data/",one_hot=true)

# Training and testing
training_digits, training_labels = mnist.train.next_batch(8000)
testing_digits, teting_labels = mnist.test.next_batch(800)

# PlaceHolders
training_digit_pl = tf.placeholder("float",[None,784])
testing_digit_pl = tf.placeholder("float",784)


#STEP 2

#Nearest Neighbor calculation using L1 Distance
l1_distance = tf.abs(tf.add(training_digit_pl, tf.negative(testing_digit_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

#prediction: Get minimum distance from the closest neighbor
pred = tf.arg_min(distance, 0)


#STEP 3
accuracy = 0.

#initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    #for loop over test data
    for i in range(len(testing_digits))
	nn_index = sess.run(pred, \
     	feed_dict={training_digit_pl:training_digits,testing_digit_pl:testing_digits[i,:]})
	
		#Get the nearest class label and compare it to the true value
		print("Test", i , "Prediction: ",np.argmax(training_labels[nn_index]),\
        	"True label",np.argmax(test_label[i]))

		#Calculation of accuracy
		if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
			accuracy += 1./len(testing_digits)
   
   print("Done Training and Testing")
   print("Accuracy: ", accuracy)
 


