Modeling Computation as Graphs
#we are going to cover math operations and run on tensorboard
#install and import tensorflow
import tensorflow as tf
#declare constants for use
a=tf.constant(2.5 , name="constant_a")
b=tf.constant(10.56 , name="constant_b")

c= tf.constant(50 , name="constant_c")
d=tf.constant(100.45, name="constant_d")

#now perform various operations using python math ops rule
square = tf.square(a, name="square_a")
power = tf.pow(b,c, name="power_b")
sqrt = tf.sqrt(d, name="sqroot_d")

#lets get the final sum
final_sum = tf.add_n([square,power,sqrt], name="final_sum")

#start the session and make sure to close it
sess = tf.Session()

#print the output
print("square of a is ", sess.run(square))
print("power of b c is ", sess.run(power))
print("sqrt of d is ", sess.run(sqrt))
