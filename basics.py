# Tensorflow basics with varibles and constant used to store data and compute graphs
x= tf.constant(40, name="x")
y=tf.Variable(x+20, name="y")
model=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y)) #60
    
#arrays 

x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y)) #40,45,50

#updating varibales within a session using for loop
w = tf.Variable(0,name="w")
model=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)
    for i in range(5):
        w=w+5
        print(sess.run(x))
        
        #gives
 [85 90 95]
[85 90 95]
[85 90 95]
[85 90 95]
[85 90 95]

#graph visualization with tensorboard
x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/basic", session.graph)
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y)) #
    #gives
Tensor("x_4:0", shape=(), dtype=int32)
40

#Loading images 
#install matplotlib pillow
import  matplotlib.image as mpimg
import os
#load image from your directory
#rem to download an image 
dir_path = os.path.dirname(os.path.realpath(_file_))
#give the file name
filename=dir_path + "/filename.png"
#load
image=mpimg.imread(filename)
#print out this shape
print(image.shape) # gives height ,width in pixels and the depth of color of the picture

#plot the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()  # plots the image

#upnext is Geometric manipulation of the image
#Transformation  of the image 90 degrees
import tensorflow as tf
import  matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(_file_))
filename=dir_path + "/filename.png
image=mpimg.imread(filename)

x=tf.Variable(image, name='x')
model=tf.global_variables_initializer()

with tf.Session() as sess:
    x=tf.transpose(x, perm=[1,0,2])
    sess.run(model)
    output=sess.run(x)
    
plt.imshow(image)
plt.show()  # output on readme.ME transpose
