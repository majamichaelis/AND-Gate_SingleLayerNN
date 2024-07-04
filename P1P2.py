import numpy as np

# Constants
N = 3  # Number of Inputs
eta = 0.5  # Learning rate
lambda_ = 1.0  # Parameter for the sigmoid function
desiredError = 0.01

# Data
inputs = np.array([
    [10, 2, -1],
    [2, -5, -1],
    [-5, 5, -1]
])

teacher_signals = np.array([
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

# Activation Functions
#returns discrete values 
def discrete_activation(x):
    return np.where(x >= 0, 1, -1)

# continious values
def sigmoid_activation(x):
    return 2.0 / (1.0 + np.exp(-lambda_ * x)) - 1.0

# set random weights
def initialize_weights(size):
    return np.random.rand(*size) - 0.5


# Neural Network Class
class SingleLayerNN:
    def __init__(self, input_size, output_size, continuous=True):
        self.weights = initialize_weights((output_size, input_size))
        self.continuousBool = continuous

    def predict(self, inputs):
        outputValues = np.dot(self.weights, inputs)
        if self.continuousBool:
            return sigmoid_activation(outputValues)
        else:
            return discrete_activation(outputValues)

    def train(self, training_inputs, teacher_signals):
        epoch = 0
        total_error = float('inf')  # Initial large error
        while total_error > desiredError and epoch < 1000:
            total_error = 0
            for inputs, desired_output in zip(training_inputs, teacher_signals):
                prediction = self.predict(inputs)
                error = desired_output - prediction
                self.weights += eta * np.outer(error, inputs)
                total_error += np.sum(error**2)  # Sum squared errors
            total_error /= len(training_inputs)  # Calculate mean squared error
            print(f"Epoch {epoch}: Total Error = {total_error}")
            print(f"Weights {self.weights}")
            epoch += 1

# Initialize and train networks for both cases
continuous_network = SingleLayerNN(N, 3, continuous=True)
discrete_network = SingleLayerNN(N, 3, continuous=False)

continuous_network.train(inputs, teacher_signals)
discrete_network.train(inputs, teacher_signals)

# Test and display results
print("Continuous Neuron Outputs:")
for input_vector in inputs:
    print(continuous_network.predict(input_vector))

print("\nDiscrete Neuron Outputs:")
for input_vector in inputs:
    print(discrete_network.predict(input_vector))
