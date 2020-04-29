
# This is an algorithim used in digits recognition which is a basic factor for 
# pattern recognition in machine Learning. 
#Uses the MNIST Data sets with 60,000 training and 10,000 testing Data sets 

#K-Nearest Algorithm
#import TF and numpy
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
