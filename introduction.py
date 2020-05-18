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
