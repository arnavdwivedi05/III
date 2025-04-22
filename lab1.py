# implement the Memory-Based Learning Algorithm using K-Nearest Neighbors (KNN) from sklearn, instead of defining a custom Neuron class:

# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

# # Training data
# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train = np.array([0, 0, 1, 1])

# # Test data
# X_test = np.array([[5, 6], [0, 1]])

# # Using KNN with k=3
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, y_train)

# # Predictions
# predictions = model.predict(X_test)

# print("Predictions:", predictions)


#Implement the learning in neuron using error correction learning algorithm

# import numpy as np
# from sklearn.linear_model import LogisticRegression

# # Training data
# X_train = np.array([[0, 0, 1],
#                     [1, 1, 1],
#                     [1, 0, 1],
#                     [0, 1, 1]])

# y_train = np.array([0, 1, 1, 0])  # Target values

# # Create and train logistic regression model
# model = LogisticRegression(solver='lbfgs', max_iter=10000)
# model.fit(X_train, y_train)

# # Test data
# test_data = np.array([[0, 0, 1], 
#                       [1, 1, 1], 
#                       [1, 0, 1], 
#                       [0, 1, 1]])

# # Predict outputs (probabilities instead of 0/1)
# outputs = model.predict_proba(test_data)[:, 1]  # Get probability of class 1

# # Print results
# for inp, out in zip(test_data, outputs):
#     print("Input:", inp, "Output:", out)

