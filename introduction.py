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

