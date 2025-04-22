import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate input data with a multivariate normal distribution
samples = 1000
mean = [0, 0]
cov = [[3, 2], [2, 2]]
data = np.random.multivariate_normal(mean, cov, samples)

# Hebbian learning using matrix operations
weights = np.mean(data.T @ data, axis=1)  # Compute covariance-based weights
weights /= np.linalg.norm(weights)  # Normalize

# PCA for comparison
pca = PCA(n_components=1)
pca.fit(data)
principal_component = pca.components_[0]

# Print results
print("Normalized Neuron Weights (Hebbian):", weights)
print("Normalized Principal Component (PCA):", principal_component)

# Visualization
plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Input Data")
plt.quiver(0, 0, weights[0], weights[1], color='r', scale=3, label="Hebbian Direction")
plt.quiver(0, 0, principal_component[0], principal_component[1], color='g', scale=3, label="PCA Direction")
plt.legend()
plt.title("Hebbian Learning vs PCA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()
plt.axis('equal')
plt.show()
