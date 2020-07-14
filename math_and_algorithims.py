 # import Tensorflow library
 import TensorFlow as tf

#The given  x variable is the independent variable of the function
  x = tf.placeholder(tf.float32)
 
#create a funtion now
    y =  2*x*x
#call the function and pass in the arguements
    var_grad = tf.gradients(y, x)
#Build session to start evaluation
    with tf.Session() as session:

 #evaluate and give x as 1
   var_grad_val = session.run(var_grad,feed_dict={x:1}) 

  #now print the values
  print(var_grad_val)



