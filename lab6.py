# A) Implement the features used in Self organizing maps using competitive learning algorithm using python programming 
# import numpy as np
# import matplotlib.pyplot as plt
# from minisom import MiniSom  # Prebuilt SOM library

# # Generate random 2D data (100 samples, 3 features)
# data = np.random.rand(100, 3)

# # Initialize and train SOM
# grid_size = (10, 10)  # 10x10 SOM grid
# som = MiniSom(grid_size[0], grid_size[1], 3, sigma=1.0, learning_rate=0.5)
# som.random_weights_init(data)
# som.train_random(data, 1000)  # Train for 1000 iterations

# # Visualize the SOM weight map
# plt.imshow(som.get_weights().reshape(grid_size[0], grid_size[1], 3))
# plt.title("Self-Organizing Map")
# plt.show()

# B) Implement the back propagation algorithm for training a recurrent network using temporal operation as a parameter into a layer feed forward network using python programming
import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected output layer

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_dim)  # Initial hidden state
        out, _ = self.rnn(x, h0)  # Forward pass through RNN
        out = self.fc(out[:, -1, :])  # Get the last time-step output
        return out

# Hyperparameters
input_dim = 3
hidden_dim = 4
output_dim = 2
learning_rate = 0.01
epochs = 50

# Generate random training data
time_steps = 5
num_samples = 100
data = torch.randn(num_samples, time_steps, input_dim)
labels = torch.randn(num_samples, output_dim)

# Initialize model, loss function, and optimizer
model = SimpleRNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

print("Training complete!")
