# A) Implement the Gate Operations (AND, OR, XOR, NAND, NOR, XNOR, NOT) using Single Layer Perceptron.

# from sklearn.linear_model import Perceptron
# import numpy as np

# # Define logic gates with their truth tables
# logic_gates = {
#     "AND": {
#         "inputs": [(0, 0), (0, 1), (1, 0), (1, 1)],
#         "outputs": [0, 0, 0, 1]
#     },
#     "OR": {
#         "inputs": [(0, 0), (0, 1), (1, 0), (1, 1)],
#         "outputs": [0, 1, 1, 1]
#     },
#     "XOR": {  # Perceptron won't learn XOR as it's not linearly separable
#         "inputs": [(0, 0), (0, 1), (1, 0), (1, 1)],
#         "outputs": [0, 1, 1, 0]
#     },
#     "NAND": {
#         "inputs": [(0, 0), (0, 1), (1, 0), (1, 1)],
#         "outputs": [1, 1, 1, 0]
#     },
#     "NOR": {
#         "inputs": [(0, 0), (0, 1), (1, 0), (1, 1)],
#         "outputs": [1, 0, 0, 0]
#     },
#     "NOT": {
#         "inputs": [(0,), (1,)],  # NOT is single input
#         "outputs": [1, 0]
#     }
# }

# # Train and test perceptron for each logic gate
# for gate, data in logic_gates.items():
#     model = Perceptron(max_iter=100, eta0=0.1, random_state=0)
#     model.fit(data["inputs"], data["outputs"])
    
#     print(f"\n{gate} Gate:")
#     for inp in data["inputs"]:
#         output = model.predict([inp])[0]
#         print(f"Input: {inp} Output: {output}")


#  B) 3b) Implement the XOR gate operations using multi-layer perception and show the error propagation by iterating the learning rate

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Print header
print("Output:")

# Try different learning rates
for lr in [0.01, 0.1, 0.5]:
    clf = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                        solver='sgd', learning_rate_init=lr,
                        max_iter=10000, tol=1e-2, random_state=42)

    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]  # Get probabilities
    mse = mean_squared_error(y, y_pred)

    # Print training info
    print(f"\nTraining with Learning Rate: {lr}")
    if clf.n_iter_ < clf.max_iter:
        print(f"Stopped early at epoch {clf.n_iter_} with error: {mse}")
    print("\nXOR Gate Results:")
    for i, x in enumerate(X):
        print(f"Input: {x} Predicted Output: [{y_pred[i]:.2f}]")
    print(f"Final Mean Squared Error: {mse:.15f}")


