
import tensorflow as tf
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# create 4 constants where tensor operations will be done
a=tf.constant(10,name="constant_a")
b=tf.constant(6,name="constant_b")
c=tf.constant(4,name="constant_c")
d=tf.constant(2,name="constant_d")

#perform operations (edges)
mul = tf.multiply(a,b, name="multiplication")
division = tf.add(c,d, name="division")
addition = tf.add_n(mul, division , name="addn")

#print to check shape,dataype and file name
print(addition)

#start session and file writer
sess = tf.Session()
#file writer 
writer = tf.summary.create_file_writer("owen",sess.graph)
#close session and writer
writer.close()
sess.close()
exit()

#now open the file from the terminal using
# tensorboard --logdir="fileName"
#open the tensorboard Ip given and navigate to the graphs section on tensorboard to view the computation on the graph
