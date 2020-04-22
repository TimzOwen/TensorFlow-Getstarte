#WORKING WITH IMAGES IN TENSORFLOW AND MATPLOTLIB
#representing and transposing images
import tensorflow as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import os

#file location
file_name = "./directory_image.png"

#read image 
image = mp_image.imread(file_name)

#print array and shape
print("image shape", image.shape)
print("image Array", image)

#show on the plot 
plt.imshow(image)
plt.show()

#NOW RUN TRANSPOSING 
#representing and transposing images
import tensorflow as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import os

#file location
file_name = "./directory_image.png"

#read image 
image = mp_image.imread(file_name)

#print array and shape
print("image shape", image.shape)
print("image Array", image)

#show on the plot 
plt.imshow(image)
plt.show()

#create a variable to hold the image
x = tf.Variable(image, name='x')

#initialize the variables
init = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init)
    #swap axis indexes from 0,1,2 using transpose
    transpose = tf.transpose(x, perm=[1,0,2]) #alternative: transpose = tf.image.transpose_image(x)
    result = sess.run(transpose)
    
    #print shape and display using matplotlib
    print("Transposed image:",result.shape)
    plt.imshow(result)
    plt.show()
#run the above file on terminal and you see th transpose
