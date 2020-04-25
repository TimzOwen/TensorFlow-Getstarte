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

#RESIZING IMAGES 
# Image Resizing and loading
import tensorflow as tf
from PIL import Image

#list all the images in your directory
original_image_list = ["./image1",
                        "./image2",
                        "./image3",
                        "./image4"]

#make a queue for all file names and images too
file_queue = tf.train.string_input_producer(original_image_list)

#read the entire image file
image_reader = tf.wholeReader()

# Session, coordinates and iteration
with tf.Session() as sess:
    #coordinate the loading of image files
    coord = tf.train.Coordinator()
    threads = tf.train_start_queue_runners(sess=sess, coord=coord)
    
    image_list = []
    
    # loop for resizing and shaping 
    for i in range(len(original_image_list)):
        #Read files from the queue and the first value returned is tuple
        #ignore file names
        _, image_file = image_reader.read(file_queue)
        
        #now decode the image ad JPEG  which is turned to a tensor ready for training 
        image = tf.image.decode_jpeg(image_file)
        
        #get a resized tensor image
        image = tf.image.resize_images(image,[250,250])
        image.set_shape((250,250,3))
        
        #now print the value after getting image tensor
        image_array = sess.run(image)
        print(image_array.shape) 
        
        Image.formarray(image_array.astype('unit8'),'RGB').show()
        
        #add a new Dimention using expands
        image_list.append(tf.expand_dims(image_array,0))
        
        #finish the file name queues
        coord.request_stop()
        coord.join(threads)
        
        index = 0
         
         #write to tensorbord using file writer to obtain the results
         summary_writer = tf.summary.FileWriter('./directory')
         
         for image_tensor in image_list:
             summary_str = sess.run(tf.summary.image("image-" + str(index),image_tensor))
             summary_writer.add_summary(summary_str)
             index += 1
        #close the summary and print the result to tensorboard
        summary_writer.close()
        
        #now you can view the result from the above code on tensorboard
        
        
        
        
        # REPRESENTING 4-D TENSOR IMAGES IN A LIST 
        # Image Resizing and loading
import tensorflow as tf
from PIL import Image

#list all the images in your directory
original_image_list = ["./image1",
                        "./image2",
                        "./image3",
                        "./image4"]

#make a queue for all file names and images too
file_queue = tf.train.string_input_producer(original_image_list)

#read the entire image file
image_reader = tf.wholeReader()

# Session, coordinates and iteration
with tf.Session() as sess:
    #coordinate the loading of image files
    coord = tf.train.Coordinator()
    threads = tf.train_start_queue_runners(sess=sess, coord=coord)
    
    image_list = []
    
    # loop for resizing and shaping 
    for i in range(len(original_image_list)):
        #Read files from the queue and the first value returned is tuple
        #ignore file names
        _, image_file = image_reader.read(file_queue)
        
        #now decode the image ad JPEG  which is turned to a tensor ready for training 
        image = tf.image.decode_jpeg(image_file)
        
        #get a resized tensor image
        image = tf.image.resize_images(image,[250,250])
        image.set_shape((250,250,3))
        
        #now print the value after getting image tensor
        image_array = sess.run(image)
        print(image_array.shape) 
        
        #convert numpy array to a tensor of type (250,250,3)
        image_tensor = tf.stack(image_array)
        
        print(image_tensor)
        image_list.append(image_tensor)
        
        #add a new Dimention using expands
        image_list.append(tf.expand_dims(image_array,0))
        
        #finish the file name queues
        coord.request_stop()
        coord.join(threads)
        
       #covert all tensors to 4 -D tensor images
       # 4-D tensor can be represented as (0,250,250,3) where 0 is the number of images 
       image_tensor = tf.stack(image_tensor)
       print(image_tensor)

       # print out the summary on a tensorboard and get to see the images,
       #to display the 4th image, add (max_outputs=4)
       summary_writer = tf.summary.FileWriter('./directory', graph = sess.graph)
    
       #write all the images in one go
       summary_str = sess.run(tf.summary.image("images", image_tensor))
       summary_writer.add_summary(summary_str)
        #close the writre and run the images and observe the changes on the tensorboard
       summary_writer.close()
        
        
        
        
        
