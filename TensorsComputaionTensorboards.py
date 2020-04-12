#CHARACTERISTIC OF TENSORS
#tensors can be rank as:Rank, Shape and Datatype.
import tensorflow as tf
sess = tf.Session()

ZeroDTensor = tf.constant(10)
sess.run(tf.rank(ZeroDTensor)) #gives a 0 Dimension tensor

oneDTensor = tf.constant(("I", "am", "oneD", "Tensor")) #gives 1D Tensor
sess.run(tf.rank(oneDTensor))

twoDTensor= tf.constant([[1,2,3],[4,5,6]]) #gives 2D tensor
sess.run(tf.rank(twoDTensor))

threeDTensor=tf.constant([[[1,2,3],[6,5,4],[7,8,15]]]) #gives 3D Tensor
sess.run(tf.rank(threeDTensor))

#Rem to close the session before exiting or running new session
ses.close()

#COMPUTING TENSORBOARD MATH OPERATIONS WITH TENSORS AND NOT SCALARS
import tensorflow as tf 

#create constants
x = tf.constant([100,200,300], name="x")
y = tf.constant([1,2,3])

#perform operations
sum_x = tf.reduce_sum(x, name='sum_x')
product_y = tf.reduce_prod(y , name='product_y')

#find the mean and division
final_div = tf.div(sum_x, product_y , name='final_div')
final_mean = tf.reduce_mean([sum_x,product_y,name='final_mean'])

#Run , close session, writeFile and plot on tensorboard
sess = tf.Session()

print("X: ",sess.run(x))
print("y: ", sess.run(y))

print("sum (x): ", sess.run(sum_x))
print("product (y): ", sess.run(product_y))

print("final_div", sess.run(final_div))
print("final_mean: " , sess.run(final_mean))
