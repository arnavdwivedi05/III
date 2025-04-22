# 12a) Implement a single forward step of the RNN-cell

import torch
import torch.nn as nn

# Initialize RNN cell
rnn_cell = nn.RNNCell(input_size=3, hidden_size=5)

# Sample input and previous hidden state
torch.manual_seed(1)
x_t = torch.randn(1, 3)       # shape: (1, input_size)
h_prev = torch.randn(1, 5)    # shape: (1, hidden_size)

# Forward step
h_t = rnn_cell(x_t, h_prev)

print("Next hidden state h_t:\n", h_t.T.detach().numpy())


# 12b) Code the forward propagation of the RNN

import torch
import torch.nn as nn

# Initialize RNN
rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)

# Inputs
torch.manual_seed(1)
X = torch.randn(1, 4, 3)    # (batch=1, time_steps=4, input_size=3)
h0 = torch.zeros(1, 1, 5)   # (num_layers=1, batch=1, hidden_size=5)

# Forward pass
output, hn = rnn(X, h0)

# Match hidden states output format: (time_steps, hidden_size, 1)
hidden_states = output[0].T.unsqueeze(2).permute(1, 0, 2).detach().numpy()
print("Hidden states:\n", hidden_states)

# Simulate output layer
W_y = torch.randn(2, 5)
b_y = torch.randn(2, 1)

# Apply output layer manually for each time step
y = []
for t in range(output.shape[1]):  # time_steps
    y_t = torch.matmul(W_y, output[0, t].view(-1, 1)) + b_y  # (output_size, 1)
    y.append(y_t.unsqueeze(0))  # add time dimension

y = torch.cat(y, dim=0).detach().numpy()  # (time_steps, output_size, 1)

print("\nOutputs:\n", y)


# 12c) Implement the LSTM cell

import torch
import torch.nn as nn

# Initialize LSTM Cell
lstm_cell = nn.LSTMCell(input_size=3, hidden_size=5)

# Input, hidden and cell state
torch.manual_seed(2)
x_t = torch.randn(1, 3)
h_prev = torch.randn(1, 5)
c_prev = torch.randn(1, 5)

# Forward pass
h_next, c_next = lstm_cell(x_t, (h_prev, c_prev))

# Match the print format
print("Next hidden state h_next:\n", h_next.T.detach().numpy())
print("\nNext cell state c_next:\n", c_next.T.detach().numpy())