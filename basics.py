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

# Reverse manipulation using (reverse_sequence)
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/picturename.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()

#Placeholders
#This are variables to which we shall assign data to it later
#this allows us to build graphs without having to place the data first but later use feed_data to assign data 
import tensorflow as tf
x=tf.placeholder("float", None)
y=x*3
with tf.Session() as sess:
    result = sess.run(y, feed_dict={[2,3,4]})
    print(result)

#multidimensional arrays with feed_dict
x=tf.placeholder("float", None)
y=x*4
with tf.Session() as sess:
    x_data = [[2,4,6,],
             [8,10,12]]
    result = sess.run(y, feed_dict={x: x_data})
    print(result)

#image slicing using placeholders and pixels
#from the pic on readme,we slice the image into 2D and 3 colors (red green and blue)
# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()


# Upnext Interactive Sessions
#Interactive 
# allows use of variables without object Sessions()
#rem to always close the sessions
sess = tf.InteractiveSession()
x =  tf.constant(list(range(15)))
print(x)
sess.close() # 0-14

# Large matrix example
#first find out size of memory being used
import resource
import numpy as np
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
#create new session and define two matrices
session = tf.InteractiveSession()
x = tf.constant(np.eye(10000))
y = tf.constant(np.random.randn(10000, 300))
z= tf.matmult(x,y)
z.eval()
#print resource but be sure your computer is powerful otherwise don't run (4gb Ram +)

