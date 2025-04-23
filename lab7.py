import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        """Train the network using Hebbian learning rule"""
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)  # No self-connections
    
    def recall(self, pattern, steps=5):
        """Recall pattern using asynchronous update"""
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw_input >= 0 else -1
        return pattern

def print_pattern(p, shape):
    """Helper function to print pattern as grid"""
    p = p.reshape(shape)
    for row in p:
        print(''.join(['⬛' if val == 1 else '⬜' for val in row]))
    print()

# Example patterns (2x5 grid)
pattern_A = np.array([ -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1 ])
pattern_B = np.array([ 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1 ])

# Create and train Hopfield network
hopfield = HopfieldNetwork(size=25)  # 5x5 grid
hopfield.train([pattern_A, pattern_B])

# Test the network by recalling a noisy version of pattern_A
noisy_pattern = pattern_A.copy()
noisy_pattern[3] = -noisy_pattern[3]  # Introduce noise to the pattern

# Recall the pattern
recalled_pattern = hopfield.recall(noisy_pattern)

# Print original noisy pattern and recalled pattern
print("Noisy Pattern:")
print(noisy_pattern)

print("Recalled Pattern:")
print(recalled_pattern)

# ###### Universal Approximation Theorem using RNN ###########
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Create a simple sequence dataset (sine wave)
x = np.linspace(0, 100, 1000)
data = np.sin(x)
seq_len = 10

X = np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])
y = np.array([data[i+seq_len] for i in range(len(data)-seq_len)])

X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

# Define a simple RNN model
model = Sequential([
    SimpleRNN(10, activation='tanh', input_shape=(seq_len, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

# Predict and plot
pred = model.predict(X)
plt.plot(y, label='True')
plt.plot(pred, label='Predicted')
plt.legend()
plt.show()
