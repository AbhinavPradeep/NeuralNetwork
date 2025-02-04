import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def GenerateData(NumberOfPoints: int):
    X = np.random.uniform(0, 1, NumberOfPoints)
    h = lambda x: 10 - 140 * x + 400 * (x ** 2) - 250 * (x ** 3)
    y = np.random.normal(h(X), np.sqrt(25))
    return (X, y)

def ReLU(z, l: int):
    # if l == 3:
    #     return (z, np.ones_like(z))
    # else:
    #     S = lambda z: 1/(1+np.exp(-z))
    #     Sprime = lambda z: (np.exp(z))/((1+np.exp(z))**2)
    #     return (S(z), Sprime(z))
    if l == 3:
        return (z, np.ones_like(z))
    else:
        return (np.array(z > 0, dtype=float), np.zeros_like(z))
    # if l == 3:  # If last layer, don't apply ReLU
    #     return (z, np.ones_like(z))
    # else:
    #     return (np.maximum(0, z), np.array(z > 0, dtype=float))

def Initialise(p):
    W = [None] * len(p)
    b = [None] * len(p)

    for l in range(1, len(p)):
        W[l] = np.random.randn(p[l], p[l - 1])
        b[l] = np.random.randn(p[l], 1)
    return W, b

def FeedForward(x, W, b):
    a, z, DdS_dz = [None] * 4, [None] * 4, [None] * 4
    a[0] = x.reshape(-1, 1)
    for l in range(1, 4):
        z[l] = W[l] @ a[l - 1] + b[l]
        a[l], DdS_dz[l] = ReLU(z[l], l)
    return a, z, DdS_dz

def Loss(y, g):
    return (g - y) ** 2, 2 * (g - y)

def BackPropagation(W, b, X, y):
    n = len(y)
    Delta = [None] * 4
    dC_db, dC_dW = [None] * 4, [None] * 4
    LossIncurred = 0

    for i in range(n):
        a, z, DdS_dz = FeedForward(X[i, :].T, W, b)
        C, dC_dg = Loss(y[i], a[3])
        LossIncurred += C / n

        Delta[3] = DdS_dz[3] @ dC_dg

        for l in range(3, 0, -1):
            dCi_dbl = Delta[l]
            dCi_dWl = Delta[l] @ a[l - 1].T
            if dC_db[l] is None:
                dC_db[l] = dCi_dbl / n
            else:
                dC_db[l] += dCi_dbl / n

            if dC_dW[l] is None:
                dC_dW[l] = dCi_dWl / n
            else:
                dC_dW[l] += dCi_dWl / n

            if l > 1:
                Delta[l - 1] = DdS_dz[l - 1] * (W[l].T @ Delta[l])

    return dC_dW, dC_db, LossIncurred

def list2vec(W, b):
    b_stack = np.vstack([b[i] for i in range(1, len(b))])
    W_stack = np.vstack([W[i].flatten().reshape(-1, 1) for i in range(1, len(W))])
    return np.vstack([b_stack, W_stack])

def vec2list(vec, p):
    W, b = [None] * len(p), [None] * len(p)
    p_count = 0

    for l in range(1, len(p)):
        b[l] = vec[p_count:p_count + p[l]].reshape(-1, 1)
        p_count += p[l]

    for l in range(1, len(p)):
        W[l] = vec[p_count:p_count + (p[l] * p[l - 1])].reshape(p[l], p[l - 1])
        p_count += p[l] * p[l - 1]

    return W, b

(X, y) = GenerateData(1000)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

batch_size = 50
lr = 0.005
p = [1, 20, 20, 1]

W, b = Initialise(p)
loss_arr = []

num_epochs = 10000
n = len(X)

print("epoch | batch loss")
print("-----------------------")

for epoch in range(1, num_epochs + 1):
    batch_idx = np.random.choice(n, batch_size)
    batch_X = X[batch_idx].reshape(-1, 1)
    batch_y = y[batch_idx].reshape(-1, 1)

    dc_dW, dc_db, batch_loss = BackPropagation(W, b, batch_X, batch_y)
    loss_arr.append(batch_loss.flatten()[0])

    d_beta = list2vec(dc_dW, dc_db)
    beta = list2vec(W, b)

    beta = beta - lr * d_beta

    W, b = vec2list(beta, p)

    if epoch == 1 or epoch % 1000 == 0:
        print(f"{epoch}: {batch_loss.flatten()[0]}")

plt.plot(np.arange(len(loss_arr)), loss_arr, 'b')
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Calculate and print final training loss on entire dataset
_, _, final_loss = BackPropagation(W, b, X, y)
print(f"Entire training set loss = {final_loss.flatten()[0]}")

# Generate a dense set of points between 0 and 1 for plotting a smooth curve
X_dense = np.linspace(0, 1, 500).reshape(-1, 1)

# Generate predictions for the dense set of inputs
predicted_y_dense = np.array([FeedForward(x, W, b)[0][3] for x in X_dense]).flatten()

# Plot the actual data points and the fitted curve
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X_dense, predicted_y_dense, color='red', linewidth=2, label="Fitted Curve")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Actual Data and Fitted Curve")
plt.legend()
plt.savefig("P11.png")
plt.show()