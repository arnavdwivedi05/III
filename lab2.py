# Implement the learning in neuron using hebbian learning algorithm

import numpy as np

# Initialize inputs
inputs = np.array([0.5, 0.3, 0.2])

# Initialize weights randomly
weights = np.random.rand(len(inputs))

# Learning rate
learning_rate = 0.1

# Number of iterations
num_iterations = 1000

# Hebbian learning rule: weight update
for _ in range(num_iterations):
    activation = np.dot(inputs, weights)  # Compute activation
    weights += learning_rate * activation * inputs  # Update weights

# Print final weights
print("Learned weights:", weights)
