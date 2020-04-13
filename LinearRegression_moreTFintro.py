#Linear Regression.
 #Regression is an example of a learning algorithm 
 #involves finding th best line of fit. 
 #check the docs on README.md

#Diggin dipper to more fundamentals
#placeholders: 
#place holders holds space/place where a tensor will be fed with data at runtime
#place holders can hold tensors of any shape
#let's perform some math operation with placeholders and feed_dictionaries 
import tensorflow as tf

#y=Wx+b
w = tf.constant([10,100], name='constant_w')

#create the placeholder
x=tf.constant(tf.int32, name="x")
b=tf.constant(tf.int32, name='b')

Wx=tf.multiply(Wx, name="Wx")
y=tf.add(Wx,b, name='y')
y_=tf.subtract(x,b,name='y_')

#print out the results and instantiate the sessions
with tf.Session() as sess:
    print("Wx result is: ", sess.run(Wx, feed_dict={x:[10,22]}))
    print("addition Wx+b: ", sess.run(y, feed_dict={x:[2,6], b:[12,6]}))
    #you can also use intermediate values 
    print("Wx+b:", sess.run(fetches=y,feed_dict={Wx:[100,1000], b:[7,8]})) #gives [107,1008]
    #multiple session with one node
    print("two results: [Wx+b] [x-b]",sess.run(fetches=[y,y_], feed_dict={x:[10,12], b:[4,2]})) #gives [104,1202][6,10]

#present it on the graph using tensorboard file writer and close the session
writer = tf.summary.FileWriter('./directory', sess.graph)
writer.close()


#Variables,constants,placeholders,sessions and tensorboards
import tensorflow as tf

#y = Wx+b
W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b=tf.Variable([5.0,10.0], tf.float32, name='var_b')

y = W * x + b

#initialize all variables and start session running the initializer also
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print("final Result of W*x+b: ", sess.run(y,feed_dict={x:[10,100]})) #gives [30,410]
    
#initialize new values you only need them
# s = Wx
s = W * x

init = tf.variables_initializer([W])

with tf.Session as sess:
    sess.run(init) #running the prev ops will cast an error of 'b' not initialzed var Wx+b
    print("this now working: Wx + b", sess.run(y,feed_dict={x:[10,200]}))
    
    #now uncomment the above and run this with 's' as the only initialized var
    print("result: s", sess.run(s, feed_dict={[2,8]})) #gives [5,32]

#now use and update variables
number = tf.Variable(6)
multiplier = tf.Variable(2)


#initialize
init = tf.global_variable_initializer()

#reassign with new values and calculate in for loop 
result = number.assign(tf.multiply(number,multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print("result number * multiplier = : ", sess.run(result))
        #increment value by one in each iteration
        print("increament multiplier, new value: ", sess.run(multiplier.assign_add(1)))
