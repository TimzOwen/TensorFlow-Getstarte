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

