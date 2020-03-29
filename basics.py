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
