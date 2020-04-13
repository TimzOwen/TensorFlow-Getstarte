#Working and understaning Numpy with Tensorflow
# numpy is a good working array lib for intergration with tensorflow
#we going to look at arrays in both cases
 import tensorflow as tf
 import numpy as np
 
 sess = tf.Session()
 
 zeroD = np.array( 40, dtype=np.int32)
 sess.run(tf.rank(zeroD))
 
 #get the shape of the numpy 
 sess.run(tf.shape(zeroD)) #gives the shape of the array and data type
 
 oneDArray = np.array([3.2, 5.6, 2.5, 4.5, 6.8], dtype=np.float32) 
 sess.run(tf.rank(oneDArray)) #gives you shape and output datatype
 sess.run(tf.shape(oneDArray))
 
 #the three to N-Dimentional array follow the same steps
 
