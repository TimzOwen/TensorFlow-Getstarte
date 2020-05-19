#DEFINING DATA AS CONSTANTS


# Import constant from TensorFlow
from tensorflow import constant

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('The datatype is:', credit_constant.dtype)

# Print constant shape
print('The shape is:', credit_constant.shape)


#DEFINING VARIABLES
#Values who contents can be modified
# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print(B1)


#UPNEXT OPERATIONS
# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like([2,3,4,5,])
B23 = ones_like([[4,5,6,], [7,8,9]])

# Perform element-wise multiplication
C1 = multiply(A1,B1)
C23 = multiply(A23,B23)

# Print the tensors C1 and C23
print('C1: {}'.format(C1.numpy()))
print('C23: {}'.format(C23.numpy()))


#making prediction with matrix multiplication
# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())

#IMAGE TENSOR AND OPTIMISATION# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352,1))


#working with Gradient Function
def compute_gradient(x0):
      	# Define x as a variable with an initial value of x0
	x = tf.Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = tf.multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))


# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())

#Reading Datasets 
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])

    21608     360000.0
    21609     400000.0
    21610     402101.0
    21611     400000.0
    21612     325000.0
    Name: price, Length: 21613, dtype: float64
			
# Import numpy and tensorflow with their standard aliases
import tensorflow as tf
import numpy as np

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.int32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)


#LOSS FUNCTIONS
#define linear Regression Model
def linear_regression(intercept, slope=slope, features=features):
    return intercept + features*slope

#define a loss function to compute MSE
def loss-fucntion(intercept, slope, target=target, features=features):
    #compute prediction of a linear Model
    prediction = leanear_regression(intercept, slope)
    
    #return loss
    return tf.keras.losses.mse(target, prediction)

#code to compute House price using MSE
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())

#house price using Mae
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.msa(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())


#Model prediction with nested loss function 
# Initialize a variable named scalar
scalar = Variable(1.0,dtype=float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())


#LINEAR REGRESSION
import tensorflow as tf 
#house price and house size prediction
#Define targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

#define intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

#define a linear regression model
def linear_regression(intercept, slope, feature=size):
    return intercept + feature*slope

#compute the predicted values loss
def loss_function(intercept, slope, targets=price, feature=size):
    prediction = linear_regression(intercept, slope)
    return tf.keras.mse(targets, prediction)
#define an optimizer to reduce loss
opt = tf.keras.optimizers.Adam()

#minimize the loss function and print the optimizer
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept,slope))
    var_list=[intercept,slope]
    print(loss_function(intercept,slope))
    
# print the trained parameters
print(intercept.numpy(), slope.numpy())

#setting up a linear Regression Model
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets,predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())


#Train a Linear Model
# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)



#MULTIPLE LINEAR REGRESSION
# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
 
loss: 12.418, intercept: 0.101, slope_1: 0.051, slope_2: 0.021
loss: 12.404, intercept: 0.102, slope_1: 0.052, slope_2: 0.022
loss: 12.391, intercept: 0.103, slope_1: 0.053, slope_2: 0.023
loss: 12.377, intercept: 0.104, slope_1: 0.054, slope_2: 0.024
loss: 12.364, intercept: 0.105, slope_1: 0.055, slope_2: 0.025
loss: 12.351, intercept: 0.106, slope_1: 0.056, slope_2: 0.026
loss: 12.337, intercept: 0.107, slope_1: 0.057, slope_2: 0.027
loss: 12.324, intercept: 0.108, slope_1: 0.058, slope_2: 0.028
loss: 12.311, intercept: 0.109, slope_1: 0.059, slope_2: 0.029
loss: 12.297, intercept: 0.110, slope_1: 0.060, slope_2: 0.030

<script.py> output:
    loss: 12.418, intercept: 0.101, slope_1: 0.051, slope_2: 0.021
    loss: 12.404, intercept: 0.102, slope_1: 0.052, slope_2: 0.022
    loss: 12.391, intercept: 0.103, slope_1: 0.053, slope_2: 0.023
    loss: 12.377, intercept: 0.104, slope_1: 0.054, slope_2: 0.024
    loss: 12.364, intercept: 0.105, slope_1: 0.055, slope_2: 0.025
    loss: 12.351, intercept: 0.106, slope_1: 0.056, slope_2: 0.026
    loss: 12.337, intercept: 0.107, slope_1: 0.057, slope_2: 0.027
    loss: 12.324, intercept: 0.108, slope_1: 0.058, slope_2: 0.028
    loss: 12.311, intercept: 0.109, slope_1: 0.059, slope_2: 0.029
    loss: 12.297, intercept: 0.110, slope_1: 0.060, slope_2: 0.030


				
#BATCH TRAINING 
#splitting data into smaller groups-batches (epoch) to enable
#training on small gadgets such as GPU . 
import pandas as pd
import numpy as np

#load data in batches
for batch in pd.read_csv('kc_housing.csv',chunksize=100):
    #Extract price column
    price = np.array(batch['price'],np.float32)
    #extract size column
    size = np.array(batch['size'],np.float32)


#TRAINING a linear model in batches
import tensorflow as tf
import numpy as np
import pandas as pd

#define trainable variables
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

#define the model
def linear_regression(intercept, slope,features):
    return intercept + features*slope

#compute predicted values and return loss function
def loss_function(intercept, slope, targets, features):
    prediction = linear_regression(intercept, slope, features)
    return tf.keras.losses.mse(targets, prediction)

#define  optimizations operation
opt = tf.keras.optimizers.Adam()

#Train the model in batches
#load in batches 
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    #extract features and target columns
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['lot_size'], np.float32)
    
    #minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch),
    var_list=[intercept, slope])
    
#print the parameter values
print(intercept.numpy(), slope.numpy())

#output
10.21788 , 0.7061



#DENSE LAYERS
#NEURAL NETWORKS 
import tensorflow as tf

#define inputs(features)
inputs = tf.constant([[1,35]])

#define weights
weights = tf.Variable([[-0.05],[-0.01]])

#Define the bias
bias = tf.Variabes([0.5])

#multiply inputes & weights (inputs*weights)
product = tf.matmul(inputs, weights)

#define the dense layer
dense = tf.keras.activations.sigmoid(product + bias)


#Defining a complete model
import tensorflow as tf

#define inputs
inputs = tf.constant(data, tf.float32)

#def first dense layer
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

#define 2nd layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

#define output prediction layer
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)


#EXAMPLE 002 DENSE LAYERS

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))

#output
dense1's output shape: (1, 2)

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2+ bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')

#output
prediction: 0.9525741338729858
 actual: 1

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid( products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)

#output
    shape of weights1:  (3, 2)
    
     shape of bias1:  (1,)
    
     shape of dense1:  (5, 2)


#keras sample 2 activation networks
# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


#ACTIVATION FUNCTION ON LINEAR AND NON-LINEAR STATES
#ACTIVATION FUNCTION ON LINEAR AND NON-LINEAR STATES
#activation on non-linearity
import tensorflow as tf
import numpy as np

#define example borrow feature
young, old = 0.3 ,0.6
low_bill, high_bill = 0.1, 0.5

#apply matrix multiplication for all the features
young_high = 1.0*young + 2.0 * highbill
young_low = 1.0*young + 2.0 * low_bill
old_high = 1.0*old + 2.0 * highbill
old_low = 1.0*young + 2.0 * low_bill

#Difference in default prediction for young
print(young_hight - young-low)
#difference in prediction for the old
print(old_high - old_low)

#SIGMOID ACTIVATION FUNCTION
#difference in default prediction for young
print(tf.keras.activations.sigmoid(young_high).numpy() -
    tf.keras.activations.sigmoid(young_low).numpy())
#difference for the old
print(tf.keras.activations.sigmoid(old_high).numpy() -
    tf.keras.activations.sigmoid(old_low).numpy())
    
    
#summary  of activation functions
import tensorfloe as tf

#Define input layers
inputs = tf.constant(borrow_features, tf.float32)

#define dense layer 1
dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)

#define dense 2
 dense2 = tf.keras.layers.Dense(8, activation='sigmoid')(inputs)
 
 #define layer 3
 output = tf.keras.layers.Dense(4, activation= 'softmax')(inputs)

 #ACTIVATION 002
# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)

# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])

#output
[[0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]
 [0.18890299 0.22651453 0.17812535 0.14867578 0.15733427 0.10044706]
 [0.17524226 0.20379762 0.20934258 0.1681514  0.1354533  0.10801287]
 [0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]
 [0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]]



 
#OPTIMIZERS
#1. GRADIENT DESCENT OPT
#2. ADAM OPT
#3. RMS OPT


import tensorflow as tf
#define the model functions 
def model(bias, weights,features=borrower_features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product + bias)

#compute predicted values and loss
def loss_function(bias, weights, target=default, features=borrower_features):
    prediction=(model, weights)
    return tf.keras.losses.binary_crossentropy(target,predictions)
    
#minimize the loss with RMS propagation
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights),var_list=[bias,weights])

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())




# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())



#VARIABLE INITIALIZATION & Model Training overfitting
#low level

#define a random normal variable
weights = tf.Variable(tf.random.normal([500, 500]))

#truncated normal
weights = tf.Variable(tf.random.truncated_normal([500,500]))

#high-level
#define a dense layer with default initializer
dense = tf.keras.layers.Dense(32, activation='relu')

#dense with zeros initializer
dense = tf.keras.layers.Dense(32, activation='relu',kernel_initializer='zeros')

#OVERFITTING  a neural model
#solved using drop layouts

import tensorflow as tf
import numpy as np

#define input data
inputs = np.array(browser_features, np.float32)

#create dense layer
dense1 = tf.keras.layers.Dense(32, activatino='relu')(inputs)

#dense 2
dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1)

#pass in the drop layouts
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

#define the outputs
outputs = tf.keras.layers.Dense(1, activation='sigmoid') (dropout1)



#initialization Recap

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))

# Define the layer 2 bias
b2 = Variable(0.0)


# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)
 
 #training neural networks
 
 # Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)




#TRAINING AND CLASSIFYING  SIGN LANGUAGE NEURAL NETWORKS

#build a sequential model
from tensorflow import keras

#define a sequential model
model = keras.sequential()

#define first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))

#second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))

#output layer definition
model.add(keras.layers.Dense(4, activation='softmax'))

#model compilations
model.compile('adam',loss='categorical_crossentropy')

#summarize model
print(model.summary())



#FUNCTIONAL API
#allows training of separate models jointly
import tensorflow as tf

#define model1 
model1_inputs = tf.keras.input(shape=(28*28,))

#def model 2 inputs layers
model2_inputs = tf.keras.input(shape=(10,))

#define layer 1 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)

#def layer2 model 1
model1_layer2 = tf.keras.layers.Dense(4,activation='softmax')(model1_layer1)

#def layer1 model 2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)

#def layer2 model 2 
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1)

#merge model 1 and 2 
merged  = tf.keras.layers.add([model1_layer2, model2_layer2])

#def functional model
model = tf.keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)

#compile the model
model.compile('adam',loss='categorical_crossentropy')




#RECAP
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

#output
Total params: 12,732
Trainable params: 12,732
Non-trainable params: 0
__________________________

#Compiling a sequential model
# In this exercise, you will work towards classifying letters 
# from the Sign Language MNIST dataset; however, you will adopt a 
# different network architecture than what you used in the previous
#  exercise. There will be fewer layers, but more nodes. You will 
#  also apply dropout to prevent overfitting. Finally, you will
#   compile the model to use the adam optimizer and the 
# categorical_crossentropy loss. You will also use a method 
# in keras to summarize your model's architecture

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())

#output
Total params: 12,628
Trainable params: 12,628
Non-trainable params: 0



#Defining multiple input layers

# In some cases, the sequential API will not be
# sufficiently flexible to accommodate your
# desired model architecture and you will need
# to use the functional API instead. If, for instance,
# you want to train two models with different architectures
# jointly, you will need to use the functional API to do this. In this exercise,
# we will see how to do this. We will also use the
# .summary() method to examine the joint model's architecture

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m2_inputs)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m1_inputs)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

#output expected 
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4)            3140        input_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            3140        input_1[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 4)            0           dense_1[0][0]                    
                                                                 dense_3[0][0]                    
==================================================================================================
Total params: 6,280
Trainable params: 6,280
Non-trainable params: 0
________________________


# sample2
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())


#TRAINING AND EVALUATING A MODEL 

#how to train a model
import tensorflow as tf

#define sequential model
model = tf.keras.Sequential()

#define hidden layer
model.add(tf.keras.layers.Dense(16,activation='relu', input_shape=(784)))

#def output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

#compile the model
model.compile('adam', loss='categorical_crossentropy')

#Training the model
model.fit(image_features, image_labels)


#Training a model with validation()Data split into training and testing datasets
model.fit(features, labels, epoch=10, validation_split=0.20)

#changing the metric
#recompile the model with accuracy metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)


#The Evaluation() operation
#evaluates the test set



# Training with Keras

# In this exercise, we return to our sign language
#  letter classification problem. We have 2000 images
#  of four letters--A, B, C, and D--and we want to classify
#  them with a high level of accuracy.
#  We will complete all parts of the problem, including the
#  model definition, compilation, and training.


# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

#output
Train on 1000 samples
    Epoch 1/5
    
  32/1000 [..............................] - ETA: 16s - loss: 1.6624
 544/1000 [===============>..............] - ETA: 0s - loss: 1.4221 
1000/1000 [==============================] - 1s 631us/sample - loss: 1.3699
    Epoch 2/5
    
  32/1000 [..............................] - ETA: 0s - loss: 1.2227
 672/1000 [===================>..........] - ETA: 0s - loss: 1.2258
1000/1000 [==============================] - 0s 81us/sample - loss: 1.2104
    Epoch 3/5
    
  32/1000 [..............................] - ETA: 0s - loss: 1.2658
 672/1000 [===================>..........] - ETA: 0s - loss: 1.0857
1000/1000 [==============================] - 0s 84us/sample - loss: 1.0668
    Epoch 4/5
    
  32/1000 [..............................] - ETA: 0s - loss: 0.9158
 640/1000 [==================>...........] - ETA: 0s - loss: 0.9240
1000/1000 [==============================] - 0s 89us/sample - loss: 0.9047
    Epoch 5/5
    
  32/1000 [..............................] - ETA: 0s - loss: 0.8375
 736/1000 [=====================>........] - ETA: 0s - loss: 0.7907
1000/1000 [==============================] - 0s 74us/sample - loss: 0.7763


#Metrics and Validation with Keras
# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.10)

#output 
Epoch 1/10

  32/1799 [..............................] - ETA: 39s - loss: 1.6457 - accuracy: 0.2500
 544/1799 [========>.....................] - ETA: 1s - loss: 1.3519 - accuracy: 0.3934 
 992/1799 [===============>..............] - ETA: 0s - loss: 1.2703 - accuracy: 0.4456
1536/1799 [========================>.....] - ETA: 0s - loss: 1.1882 - accuracy: 0.5143
1799/1799 [==============================] - 1s 641us/sample - loss: 1.1563 - accuracy: 0.5442 - val_loss: 1.0239 - val_accuracy: 0.5050
Epoch 2/10

  32/1799 [..............................] - ETA: 0s - loss: 1.0696 - accuracy: 0.5312
 480/1799 [=======>......................] - ETA: 0s - loss: 0.8834 - accuracy: 0.8313
 960/1799 [===============>..............] - ETA: 0s - loss: 0.8540 - accuracy: 0.7958
1376/1799 [=====================>........] - ETA: 0s - loss: 0.8317 - accuracy: 0.7914
1760/1799 [============================>.] - ETA: 0s - loss: 0.8097 - accuracy: 0.7937
1799/1799 [==============================] - 0s 139us/sample - loss: 0.8082 - accuracy: 0.7927 - val_loss: 0.6956 - val_accuracy: 0.8150
Epoch 3/10

  32/1799 [..............................] - ETA: 0s - loss: 0.7409 - accuracy: 0.8438
 480/1799 [=======>......................] - ETA: 0s - loss: 0.6833 - accuracy: 0.8375
 768/1799 [===========>..................] - ETA: 0s - loss: 0.6714 - accuracy: 0.8398
1152/1799 [==================>...........] - ETA: 0s - loss: 0.6469 - accuracy: 0.8481
1632/1799 [==========================>...] - ETA: 0s - loss: 0.6173 - accuracy: 0.8627
1799/1799 [==============================] - 0s 161us/sample - loss: 0.6119 - accuracy: 0.8660 - val_loss: 0.5418 - val_accuracy: 0.8950
Epoch 4/10

  32/1799 [..............................] - ETA: 0s - loss: 0.5651 - accuracy: 0.8125
 544/1799 [========>.....................] - ETA: 0s - loss: 0.5195 - accuracy: 0.8879
1056/1799 [================>.............] - ETA: 0s - loss: 0.5006 - accuracy: 0.8949
1536/1799 [========================>.....] - ETA: 0s - loss: 0.4805 - accuracy: 0.9062
1799/1799 [==============================] - 0s 149us/sample - loss: 0.4672 - accuracy: 0.9066 - val_loss: 0.4829 - val_accuracy: 0.8250
Epoch 5/10

  32/1799 [..............................] - ETA: 0s - loss: 0.4154 - accuracy: 0.8438
 448/1799 [======>.......................] - ETA: 0s - loss: 0.4048 - accuracy: 0.9397
 928/1799 [==============>...............] - ETA: 0s - loss: 0.3884 - accuracy: 0.9440
1472/1799 [=======================>......] - ETA: 0s - loss: 0.3739 - accuracy: 0.9450
1799/1799 [==============================] - 0s 117us/sample - loss: 0.3682 - accuracy: 0.9439 - val_loss: 0.4221 - val_accuracy: 0.9250
Epoch 6/10

  32/1799 [..............................] - ETA: 0s - loss: 0.3541 - accuracy: 0.9375
 704/1799 [==========>...................] - ETA: 0s - loss: 0.3202 - accuracy: 0.9545
1344/1799 [=====================>........] - ETA: 0s - loss: 0.3009 - accuracy: 0.9635
1799/1799 [==============================] - 0s 91us/sample - loss: 0.2919 - accuracy: 0.9622 - val_loss: 0.3764 - val_accuracy: 0.7850
Epoch 7/10

  32/1799 [..............................] - ETA: 0s - loss: 0.3175 - accuracy: 0.8438
 704/1799 [==========>...................] - ETA: 0s - loss: 0.2567 - accuracy: 0.9645
1376/1799 [=====================>........] - ETA: 0s - loss: 0.2394 - accuracy: 0.9738
1799/1799 [==============================] - 0s 89us/sample - loss: 0.2399 - accuracy: 0.9689 - val_loss: 0.2169 - val_accuracy: 0.9650
Epoch 8/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2129 - accuracy: 1.0000
 576/1799 [========>.....................] - ETA: 0s - loss: 0.2006 - accuracy: 0.9757
1088/1799 [=================>............] - ETA: 0s - loss: 0.2019 - accuracy: 0.9715
1664/1799 [==========================>...] - ETA: 0s - loss: 0.1930 - accuracy: 0.9754
1799/1799 [==============================] - 0s 109us/sample - loss: 0.1911 - accuracy: 0.9767 - val_loss: 0.2745 - val_accuracy: 0.9050
Epoch 9/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2252 - accuracy: 0.9062
 576/1799 [========>.....................] - ETA: 0s - loss: 0.1925 - accuracy: 0.9653
1216/1799 [===================>..........] - ETA: 0s - loss: 0.1710 - accuracy: 0.9794
1792/1799 [============================>.] - ETA: 0s - loss: 0.1583 - accuracy: 0.9827
1799/1799 [==============================] - 0s 106us/sample - loss: 0.1582 - accuracy: 0.9828 - val_loss: 0.2293 - val_accuracy: 0.9150
Epoch 10/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2478 - accuracy: 0.8438
 480/1799 [=======>......................] - ETA: 0s - loss: 0.1520 - accuracy: 0.9646
 896/1799 [=============>................] - ETA: 0s - loss: 0.1495 - accuracy: 0.9676
1376/1799 [=====================>........] - ETA: 0s - loss: 0.1447 - accuracy: 0.9731
1799/1799 [==============================] - 0s 127us/sample - loss: 0.1381 - accuracy: 0.9772 - val_loss: 0.1356 - val_accuracy: 0.9700



#OVERFITTING DETECTION
# You will detect overfitting by checking whether the validation
# sample loss is substantially higher than the training sample
# loss and whether it increases with further training. With a
# small sample and a high learning rate, the model will struggle
# to converge on an optimum.You will set a low learning rate for the optimizer, which will
#   make it easier to identify overfitting.

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=200, validation_split=0.50)




#DEFINING DATA AS CONSTANTS


# Import constant from TensorFlow
from tensorflow import constant

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('The datatype is:', credit_constant.dtype)

# Print constant shape
print('The shape is:', credit_constant.shape)


#DEFINING VARIABLES
#Values who contents can be modified
# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print(B1)


#UPNEXT OPERATIONS
# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like([2,3,4,5,])
B23 = ones_like([[4,5,6,], [7,8,9]])

# Perform element-wise multiplication
C1 = multiply(A1,B1)
C23 = multiply(A23,B23)

# Print the tensors C1 and C23
print('C1: {}'.format(C1.numpy()))
print('C23: {}'.format(C23.numpy()))


#making prediction with matrix multiplication
# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())

#IMAGE TENSOR AND OPTIMISATION# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352,1))


#working with Gradient Function
def compute_gradient(x0):
      	# Define x as a variable with an initial value of x0
	x = tf.Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = tf.multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))


#Reshaping OPs 

# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())


#Reading Datasets 
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])

    21608     360000.0
    21609     400000.0
    21610     402101.0
    21611     400000.0
    21612     325000.0
    Name: price, Length: 21613, dtype: float64
    
# Import numpy and tensorflow with their standard aliases
import tensorflow as tf
import numpy as np

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.int32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)


#LOSS FUNCTIONS
#define linear Regression Model
def linear_regression(intercept, slope=slope, features=features):
    return intercept + features*slope

#define a loss function to compute MSE
def loss-fucntion(intercept, slope, target=target, features=features):
    #compute prediction of a linear Model
    prediction = leanear_regression(intercept, slope)
    
    #return loss
    return tf.keras.losses.mse(target, prediction)

#code to compute House price using MSE
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())

#house price using Mae
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.msa(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())


#Model prediction with nested loss function 
# Initialize a variable named scalar
scalar = Variable(1.0,dtype=float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())


#LINEAR REGRESSION
import tensorflow as tf 
#house price and house size prediction
#Define targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

#define intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

#define a linear regression model
def linear_regression(intercept, slope, feature=size):
    return intercept + feature*slope

#compute the predicted values loss
def loss_function(intercept, slope, targets=price, feature=size):
    prediction = linear_regression(intercept, slope)
    return tf.keras.mse(targets, prediction)
#define an optimizer to reduce loss
opt = tf.keras.optimizers.Adam()

#minimize the loss function and print the optimizer
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept,slope))\
    var_list=[intercept,slope]
    print(loss_function(intercept,slope))
    
# print the trained parameters
print(intercept.numpy(), slope.numpy())


#setting up a linear Regression Model
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets,predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())


#Train a Linear Model
# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)



#MULTIPLE LINEAR REGRESSION
# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
 
loss: 12.418, intercept: 0.101, slope_1: 0.051, slope_2: 0.021
loss: 12.404, intercept: 0.102, slope_1: 0.052, slope_2: 0.022
loss: 12.391, intercept: 0.103, slope_1: 0.053, slope_2: 0.023
loss: 12.377, intercept: 0.104, slope_1: 0.054, slope_2: 0.024
loss: 12.364, intercept: 0.105, slope_1: 0.055, slope_2: 0.025
loss: 12.351, intercept: 0.106, slope_1: 0.056, slope_2: 0.026
loss: 12.337, intercept: 0.107, slope_1: 0.057, slope_2: 0.027
loss: 12.324, intercept: 0.108, slope_1: 0.058, slope_2: 0.028
loss: 12.311, intercept: 0.109, slope_1: 0.059, slope_2: 0.029
loss: 12.297, intercept: 0.110, slope_1: 0.060, slope_2: 0.030

# output:
    loss: 12.418, intercept: 0.101, slope_1: 0.051, slope_2: 0.021
    loss: 12.404, intercept: 0.102, slope_1: 0.052, slope_2: 0.022
    loss: 12.391, intercept: 0.103, slope_1: 0.053, slope_2: 0.023
    loss: 12.377, intercept: 0.104, slope_1: 0.054, slope_2: 0.024
    loss: 12.364, intercept: 0.105, slope_1: 0.055, slope_2: 0.025
    loss: 12.351, intercept: 0.106, slope_1: 0.056, slope_2: 0.026
    loss: 12.337, intercept: 0.107, slope_1: 0.057, slope_2: 0.027
    loss: 12.324, intercept: 0.108, slope_1: 0.058, slope_2: 0.028
    loss: 12.311, intercept: 0.109, slope_1: 0.059, slope_2: 0.029
    loss: 12.297, intercept: 0.110, slope_1: 0.060, slope_2: 0.030
    
    
    
#BATCH TRAINING 
#splitting data into smaller groups-batches (epoch) to enable
#training on small gadgets such as GPU . 
import pandas as pd
import numpy as np

#load data in batches
for batch in pd.read_csv('kc_housing.csv',chunksize=100):
    #Extract price column
    price = np.array(batch['price'],np.float32)
    #extract size column
    size = np.array(batch['size'],np.float32)


#TRAINING a linear model in batches
import tensorflow as tf
import numpy as np
import pandas as pd

#define trainable variables
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

#define the model
def linear_regression(intercept, slope,features):
    return intercept + features*slope

#compute predicted values and return loss function
def loss_function(intercept, slope, targets, features):
    prediction = linear_regression(intercept, slope, features)
    return tf.keras.losses.mse(targets, prediction)

#define  optimizations operation
opt = tf.keras.optimizers.Adam()

#Train the model in batches
#load in batches 
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    #extract features and target columns
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['lot_size'], np.float32)
    
    #minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch),
    var_list=[intercept, slope])
    
#print the parameter values
print(intercept.numpy(), slope.numpy())



#BATCH TRAINING 002
# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + features*slope

# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)

	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)

	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())

10.21788 , 0.7061



#DENSE LAYERS
#NEURAL NETWORKS 
import tensorflow as tf

#define inputs(features)
inputs = tf.constant([[1,35]])

#define weights
weights = tf.Variable([[-0.05],[-0.01]])

#Define the bias
bias = tf.Variabes([0.5])

#multiply inputes & weights (inputs*weights)
product = tf.matmul(inputs, weights)

#define the dense layer
dense = tf.keras.activations.sigmoid(product + bias)


#Defining a complete model
import tensorflow as tf

#define inputs
inputs = tf.constant(data, tf.float32)

#def first dense layer
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

#define 2nd layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

#define output prediction layer
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)


#EXAMPLE 002 DENSE LAYERS

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))

#output
dense1's output shape: (1, 2)

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2+ bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')

#output
prediction: 0.9525741338729858
 actual: 1

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid( products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)

#output
    shape of weights1:  (3, 2)
    
     shape of bias1:  (1,)
    
     shape of dense1:  (5, 2)


#keras sample 2 activation networks
# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


#ACTIVATION FUNCTION ON LINEAR AND NON-LINEAR STATES
#activation on non-linearity
import tensorflow as tf
import numpy as np

#define example borrow feature
young, old = 0.3 ,0.6
low_bill, high_bill = 0.1, 0.5

#apply matrix multiplication for all the features
young_high = 1.0*young + 2.0 * highbill
young_low = 1.0*young + 2.0 * low_bill
old_high = 1.0*old + 2.0 * highbill
old_low = 1.0*young + 2.0 * low_bill

#Difference in default prediction for young
print(young_hight - young-low)
#difference in prediction for the old
print(old_high - old_low)

#SIGMOID ACTIVATION FUNCTION
#difference in default prediction for young
print(tf.keras.activations.sigmoid(young_high).numpy() -
    tf.keras.activations.sigmoid(young_low).numpy())
#difference for the old
print(tf.keras.activations.sigmoid(old_high).numpy() -
    tf.keras.activations.sigmoid(old_low).numpy())
    
    
#summary  of activation functions
import tensorfloe as tf

#Define input layers
inputs = tf.constant(borrow_features, tf.float32)

#define dense layer 1
dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)

#define dense 2
 dense2 = tf.keras.layers.Dense(8, activation='sigmoid')(inputs)
 
 #define layer 3
 output = tf.keras.layers.Dense(4, activation= 'softmax')(inputs)

 #ACTIVATION 002
# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)

# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])

#output
[[0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]
 [0.18890299 0.22651453 0.17812535 0.14867578 0.15733427 0.10044706]
 [0.17524226 0.20379762 0.20934258 0.1681514  0.1354533  0.10801287]
 [0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]
 [0.27273858 0.22793978 0.14991151 0.20651649 0.07163708 0.0712566 ]]
 
 
 
 
#OPTIMIZERS
#1. GRADIENT DESCENT OPT
#2. ADAM OPT
#3. RMS OPT


import tensorflow as tf
#define the model functions 
def model(bias, weights,features=borrower_features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product + bias)

#compute predicted values and loss
def loss_function(bias, weights, target=default, features=borrower_features):
    prediction=(model, weights)
    return tf.keras.losses.binary_crossentropy(target,predictions)
    
#minimize the loss with RMS propagation
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights),var_list=[bias,weights])

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())



#VARIABLE INITIALIZATION
#low level

#define a random normal variable
weights = tf.Variable(tf.random.normal([500, 500]))

#truncated normal
weights = tf.Variable(tf.random.truncated_normal([500,500]))

#high-level
#define a dense layer with default initializer
dense = tf.keras.layers.Dense(32, activation='relu')

#dense with zeros initializer
dense = tf.keras.layers.Dense(32, activation='relu',kernel_initializer='zeros')

#OVERFITTING  a neural model
#solved using drop layouts

import tensorflow as tf
import numpy as np

#define input data
inputs = np.array(browser_features, np.float32)

#create dense layer
dense1 = tf.keras.layers.Dense(32, activatino='relu')(inputs)

#dense 2
dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1)

#pass in the drop layouts
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

#define the outputs
outputs = tf.keras.layers.Dense(1, activation='sigmoid') (dropout1)


#initialization Recap

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))

# Define the layer 2 bias
b2 = Variable(0.0)


# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)
 
 #training neural networks
 
 # Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)




#TRAINING AND CLASSIFYING  SIGN LANGUAGE NEURAL NETWORKS

#build a sequential model
from tensorflow import keras

#define a sequential model
model = keras.sequential()

#define first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))

#second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))

#output layer definition
model.add(keras.layers.Dense(4, activation='softmax'))

#model compilations
model.compile('adam',loss='categorical_crossentropy')

#summarize model
print(model.summary())



#FUNCTIONAL API
#allows training of separate models jointly
import tensorflow as tf

#define model1 
model1_inputs = tf.keras.input(shape=(28*28,))

#def model 2 inputs layers
model2_inputs = tf.keras.input(shape=(10,))

#define layer 1 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)

#def layer2 model 1
model1_layer2 = tf.keras.layers.Dense(4,activation='softmax')(model1_layer1)

#def layer1 model 2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)

#def layer2 model 2 
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1)

#merge model 1 and 2 
merged  = tf.keras.layers.add([model1_layer2, model2_layer2])

#def functional model
model = tf.keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)

#compile the model
model.compile('adam',loss='categorical_crossentropy')


#RECAP
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

#output
Total params: 12,732
Trainable params: 12,732
Non-trainable params: 0
__________________________

#Compiling a sequential model
# In this exercise, you will work towards classifying letters 
# from the Sign Language MNIST dataset; however, you will adopt a 
# different network architecture than what you used in the previous
#  exercise. There will be fewer layers, but more nodes. You will 
#  also apply dropout to prevent overfitting. Finally, you will
#   compile the model to use the adam optimizer and the 
# categorical_crossentropy loss. You will also use a method 
# in keras to summarize your model's architecture

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())

#output
Total params: 12,628
Trainable params: 12,628
Non-trainable params: 0



#Defining multiple input layers

# In some cases, the sequential API will not be
# sufficiently flexible to accommodate your
# desired model architecture and you will need
# to use the functional API instead. If, for instance,
# you want to train two models with different architectures
# jointly, you will need to use the functional API to do this. In this exercise,
# we will see how to do this. We will also use the
# .summary() method to examine the joint model's architecture

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m2_inputs)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m1_inputs)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

#output expected 
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4)            3140        input_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            3140        input_1[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 4)            0           dense_1[0][0]                    
                                                                 dense_3[0][0]                    
==================================================================================================
Total params: 6,280
Trainable params: 6,280
Non-trainable params: 0
________________________


# sample2
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())


#TRAINING AND EVALUATING A MODEL 

#how to train a model
import tensorflow as tf

#define sequential model
model = tf.keras.Sequential()

#define hidden layer
model.add(tf.keras.layers.Dense(16,activation='relu', input_shape=(784)))

#def output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

#compile the model
model.compile('adam', loss='categorical_crossentropy')

#Training the model
model.fit(image_features, image_labels)


#Training a model with validation()Data split into training and testing datasets
model.fit(features, labels, epoch=10, validation_split=0.20)

#changing the metric

#recompile the model with accuracy metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)


#The Evaluation() operation
#evaluates the test set



# Training with Keras

# In this exercise, we return to our sign language
#  letter classification problem. We have 2000 images
#  of four letters--A, B, C, and D--and we want to classify
#  them with a high level of accuracy.
#  We will complete all parts of the problem, including the
#  model definition, compilation, and training.

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

#output
Train on 1000 samples
    Epoch 1/5
    
  32/1000 [..............................] - ETA: 16s - loss: 1.6624
 544/1000 [===============>..............] - ETA: 0s - loss: 1.4221 
1000/1000 [==============================] - 1s 631us/sample - loss: 1.3699
    Epoch 2/5
    
  32/1000 [..............................] - ETA: 0s - loss: 1.2227
 672/1000 [===================>..........] - ETA: 0s - loss: 1.2258
1000/1000 [==============================] - 0s 81us/sample - loss: 1.2104
    Epoch 3/5
    
  32/1000 [..............................] - ETA: 0s - loss: 1.2658
 672/1000 [===================>..........] - ETA: 0s - loss: 1.0857
1000/1000 [==============================] - 0s 84us/sample - loss: 1.0668
    Epoch 4/5
    
  32/1000 [..............................] - ETA: 0s - loss: 0.9158
 640/1000 [==================>...........] - ETA: 0s - loss: 0.9240
1000/1000 [==============================] - 0s 89us/sample - loss: 0.9047
    Epoch 5/5
    
  32/1000 [..............................] - ETA: 0s - loss: 0.8375
 736/1000 [=====================>........] - ETA: 0s - loss: 0.7907
1000/1000 [==============================] - 0s 74us/sample - loss: 0.7763


#Metrics and Validation with Keras

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.10)

#output 
Epoch 1/10

  32/1799 [..............................] - ETA: 39s - loss: 1.6457 - accuracy: 0.2500
 544/1799 [========>.....................] - ETA: 1s - loss: 1.3519 - accuracy: 0.3934 
 992/1799 [===============>..............] - ETA: 0s - loss: 1.2703 - accuracy: 0.4456
1536/1799 [========================>.....] - ETA: 0s - loss: 1.1882 - accuracy: 0.5143
1799/1799 [==============================] - 1s 641us/sample - loss: 1.1563 - accuracy: 0.5442 - val_loss: 1.0239 - val_accuracy: 0.5050
Epoch 2/10

  32/1799 [..............................] - ETA: 0s - loss: 1.0696 - accuracy: 0.5312
 480/1799 [=======>......................] - ETA: 0s - loss: 0.8834 - accuracy: 0.8313
 960/1799 [===============>..............] - ETA: 0s - loss: 0.8540 - accuracy: 0.7958
1376/1799 [=====================>........] - ETA: 0s - loss: 0.8317 - accuracy: 0.7914
1760/1799 [============================>.] - ETA: 0s - loss: 0.8097 - accuracy: 0.7937
1799/1799 [==============================] - 0s 139us/sample - loss: 0.8082 - accuracy: 0.7927 - val_loss: 0.6956 - val_accuracy: 0.8150
Epoch 3/10

  32/1799 [..............................] - ETA: 0s - loss: 0.7409 - accuracy: 0.8438
 480/1799 [=======>......................] - ETA: 0s - loss: 0.6833 - accuracy: 0.8375
 768/1799 [===========>..................] - ETA: 0s - loss: 0.6714 - accuracy: 0.8398
1152/1799 [==================>...........] - ETA: 0s - loss: 0.6469 - accuracy: 0.8481
1632/1799 [==========================>...] - ETA: 0s - loss: 0.6173 - accuracy: 0.8627
1799/1799 [==============================] - 0s 161us/sample - loss: 0.6119 - accuracy: 0.8660 - val_loss: 0.5418 - val_accuracy: 0.8950
Epoch 4/10

  32/1799 [..............................] - ETA: 0s - loss: 0.5651 - accuracy: 0.8125
 544/1799 [========>.....................] - ETA: 0s - loss: 0.5195 - accuracy: 0.8879
1056/1799 [================>.............] - ETA: 0s - loss: 0.5006 - accuracy: 0.8949
1536/1799 [========================>.....] - ETA: 0s - loss: 0.4805 - accuracy: 0.9062
1799/1799 [==============================] - 0s 149us/sample - loss: 0.4672 - accuracy: 0.9066 - val_loss: 0.4829 - val_accuracy: 0.8250
Epoch 5/10

  32/1799 [..............................] - ETA: 0s - loss: 0.4154 - accuracy: 0.8438
 448/1799 [======>.......................] - ETA: 0s - loss: 0.4048 - accuracy: 0.9397
 928/1799 [==============>...............] - ETA: 0s - loss: 0.3884 - accuracy: 0.9440
1472/1799 [=======================>......] - ETA: 0s - loss: 0.3739 - accuracy: 0.9450
1799/1799 [==============================] - 0s 117us/sample - loss: 0.3682 - accuracy: 0.9439 - val_loss: 0.4221 - val_accuracy: 0.9250
Epoch 6/10

  32/1799 [..............................] - ETA: 0s - loss: 0.3541 - accuracy: 0.9375
 704/1799 [==========>...................] - ETA: 0s - loss: 0.3202 - accuracy: 0.9545
1344/1799 [=====================>........] - ETA: 0s - loss: 0.3009 - accuracy: 0.9635
1799/1799 [==============================] - 0s 91us/sample - loss: 0.2919 - accuracy: 0.9622 - val_loss: 0.3764 - val_accuracy: 0.7850
Epoch 7/10

  32/1799 [..............................] - ETA: 0s - loss: 0.3175 - accuracy: 0.8438
 704/1799 [==========>...................] - ETA: 0s - loss: 0.2567 - accuracy: 0.9645
1376/1799 [=====================>........] - ETA: 0s - loss: 0.2394 - accuracy: 0.9738
1799/1799 [==============================] - 0s 89us/sample - loss: 0.2399 - accuracy: 0.9689 - val_loss: 0.2169 - val_accuracy: 0.9650
Epoch 8/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2129 - accuracy: 1.0000
 576/1799 [========>.....................] - ETA: 0s - loss: 0.2006 - accuracy: 0.9757
1088/1799 [=================>............] - ETA: 0s - loss: 0.2019 - accuracy: 0.9715
1664/1799 [==========================>...] - ETA: 0s - loss: 0.1930 - accuracy: 0.9754
1799/1799 [==============================] - 0s 109us/sample - loss: 0.1911 - accuracy: 0.9767 - val_loss: 0.2745 - val_accuracy: 0.9050
Epoch 9/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2252 - accuracy: 0.9062
 576/1799 [========>.....................] - ETA: 0s - loss: 0.1925 - accuracy: 0.9653
1216/1799 [===================>..........] - ETA: 0s - loss: 0.1710 - accuracy: 0.9794
1792/1799 [============================>.] - ETA: 0s - loss: 0.1583 - accuracy: 0.9827
1799/1799 [==============================] - 0s 106us/sample - loss: 0.1582 - accuracy: 0.9828 - val_loss: 0.2293 - val_accuracy: 0.9150
Epoch 10/10

  32/1799 [..............................] - ETA: 0s - loss: 0.2478 - accuracy: 0.8438
 480/1799 [=======>......................] - ETA: 0s - loss: 0.1520 - accuracy: 0.9646
 896/1799 [=============>................] - ETA: 0s - loss: 0.1495 - accuracy: 0.9676
1376/1799 [=====================>........] - ETA: 0s - loss: 0.1447 - accuracy: 0.9731
1799/1799 [==============================] - 0s 127us/sample - loss: 0.1381 - accuracy: 0.9772 - val_loss: 0.1356 - val_accuracy: 0.9700



#OVERFITTING DETECTION


# You will detect overfitting by checking whether the validation
# sample loss is substantially higher than the training sample
# loss and whether it increases with further training. With a
# small sample and a high learning rate, the model will struggle
# to converge on an optimum.You will set a low learning rate for the optimizer, which will
#   make it easier to identify overfitting.

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=200, validation_split=0.50)


#EVALUATING MODELS

"""Two models have been trained and are available: large_model, 
which has many parameters; and small_model, which has fewer 
parameters. Both models have been trained using train_features 
and train_labels, which are available to you. A separate test set, 
which consists of test_features and test_labels, is also available.

Your goal is to evaluate relative model performance and also
 determine whether either model exhibits signs of overfitting. 
 You will do this by evaluating large_model and small_model on 
 both the train and test sets. For each model, you can do this by 
 applying the .evaluate(x, y) method to
 compute the loss for features x and labels y. You will then 
 compare the four losses generated."""

#EVALUATION TEST TO ENSURE NO OVERFITTING

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))

#output
 32/100 [========>.....................] - ETA: 0s - loss: 0.3384
100/100 [==============================] - 0s 1ms/sample - loss: 0.3340

 32/100 [========>.....................] - ETA: 0s - loss: 0.4166
100/100 [==============================] - 0s 82us/sample - loss: 0.4521

 32/100 [========>.....................] - ETA: 0s - loss: 0.0434
100/100 [==============================] - 0s 1ms/sample - loss: 0.0354

 32/100 [========>.....................] - ETA: 0s - loss: 0.0918
100/100 [==============================] - 0s 86us/sample - loss: 0.1651

 Small - Train: 0.3339744055271149, Test: 0.4521311902999878
Large - Train: 0.03536653757095337, Test: 0.16508901357650757

