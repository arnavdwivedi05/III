import numpy as np
import matplotlib.pyplot as plt

# Define inputs and outputs
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,1,1,0])

# Define two centers (like RBF neurons)
w1 = np.array([1,1])
w2 = np.array([0,0])

# Gaussian function
def gaussian(x, w):
    return np.exp(-np.linalg.norm(x - w)**2)

# Display header
print("Output:\n")
print("Input\t\tFirst Function\tSecond Function")

# First part: print RBF values for each input
for x in inputs:
    g1 = gaussian(x, w1)
    g2 = gaussian(x, w2)
    print(f"{x}\t{g1:.4f}\t\t{g2:.4f}")

# Prepare RBF outputs for visualization
inputs = np.array([[0,0],[1,1],[0,1],[1,0]])
outputs = np.array([0,0,1,1])
f1, f2 = [], []

for x in inputs:
    f1.append(gaussian(x, w1))
    f2.append(gaussian(x, w2))

# Plotting
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(f1[:2], f2[:2], marker="x", label="Class 0")
ax.scatter(f1[2:], f2[2:], marker="o", label="Class 1")
ax.plot(np.linspace(0, 1, 10), -np.linspace(0, 1, 10) + 1, label="y = -x + 1")

ax.set_xlim(left=-0.1)
ax.set_ylim(bottom=-0.1)
plt.xlabel("Hidden Function 1")
plt.ylabel("Hidden Function 2")
plt.legend()
plt.show()
