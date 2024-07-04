import numpy as np
import math
import random
import matplotlib.pyplot as plt

I = 3  # Two input bits (0 or 1) and one bias (-1)
n_sample = 4  # Number of training examples
eta = 0.5  # Learning rate
_lambda = 1  # Parameter for the sigmoid function
desired_error = 0.01  # Desired error threshold

# discrete Activation Function
def discreteActivation(x):
    return np.where(x >= 0, 1, -1)

# continuous Activation Function 
def sigmoid(x):
    return (2.0 / (1.0 + math.exp(-_lambda * x)) - 1.0)

def frand():
    return random.randint(0, 9999) / 10001.0

# Training examples
input = np.array([
    [0, 0, -1],
    [0, 1, -1],
    [1, 0, -1],
    [1, 1, -1],
])

# Weights initalization
w = np.zeros(I)
# Teaching signals AND (only 1&1 -> 1)
teacher_signals = np.array([-1, -1, -1 , 1])
o = 0.0

# Weights with random numbers 
def Initialization():
    global w
    # W = np.random.rand(I)
    w = np.array([frand() for _ in range(I)])

# Output of a neuron for a given input
def FindOutput(p):
    global o
    temp = np.dot(w, input[p])
    o = sigmoid(temp)
    return o

# Main function for training
def main():
    global w, o
    Initialization()
    Error = float('inf')
    errors = []  # To save the error values

    q = 0  # Counter for epochs
    while Error > desired_error:
        q += 1
        Error = 0
        neuron_outputs = []  # Saves the neuron outputs for each input
        for p in range(n_sample):
            neuron_output = FindOutput(p)
            neuron_outputs.append(neuron_output)  # Saving the output for each input
            Error += 0.5 * (teacher_signals[p] - o) ** 2
            # 4 Error values 
            # print(Error)
            delta = (teacher_signals[p] - o) * (1 - o * o) / 2
            w += eta * delta * input[p]

        print(f"Error {Error}:")
        errors.append(Error)  # Save errors after each cycle
        print(f"Epoch {q}: Neuron output for each input pattern: {neuron_outputs}")

    fig = plt.figure()
    plt.plot(errors)  # plot errors 
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')
    plt.title('Training Progress')
    #plt.show()
    fig.savefig('nameIMG.jpg', format='jpg')
    plt.close()

    print("\nThe connection weights of the neurons:")
    print(w)

# Testing Perzeptron
def test_perceptron(perceptron):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_outputs = [0, 0, 0, 1]  # Erwartete Ausgaben für ein AND-Gate

    print("Testing the perceptron:")
    for i, input_data in enumerate(inputs):
        output = perceptron(input_data)
        expected_output = expected_outputs[i]
        print(f"Input: {input_data}, Expected Output: {expected_output}, Perceptron Output: {output}")

def perceptron(input_data):
    temp = np.dot(w, np.append(input_data, -1))
    output = sigmoid(temp)
    return 1 if output > 0 else 0

if __name__ == "__main__":
    main()
    test_perceptron(perceptron)

    #print("Ausgabe:", perceptron([0.9, 0.9]))
    fig, plot = plt.subplots() 
    colors = []
    
    for i in range (1, 100, 2):
        for teacher_signals in range(1, 100, 2):
            i_= i/100
            d_= teacher_signals/100
            result = perceptron([i_, d_])
            colors.append(result)
            plot.scatter(i_, d_, c='green' if result > 0 else 'red')
    plot.set_xlim(0.1, 1)
    plot.set_ylim(0.1, 1)
    #plt.show()
    fig.savefig('Visualisation.jpg', format='jpg')
    plt.close()



