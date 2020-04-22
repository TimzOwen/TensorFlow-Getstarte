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
