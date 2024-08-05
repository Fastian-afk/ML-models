import numpy as np
import plotly.graph_objects as go

# Training data
x = np.array([[100]])
y = np.array([[100]])

# Weights initialization
w1 = np.random.rand(2, 1) * 0.1
b1 = np.zeros((2, 1))
w2 = np.random.rand(1, 2) * 0.1
b2 = np.zeros((1, 1))


def sigmoid(z):
    S = 1 / (1 + np.exp(-z))
    return S


def derivative_sigmoid(S):
    ds = S * (1 - S)
    return ds


params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def forward(x, params):
    z1 = np.dot(x, params["w1"].T) + params["b1"].T
    a1 = sigmoid(z1)
    z2 = np.dot(a1, params["w2"].T) + params["b2"].T
    a2 = sigmoid(z2)
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache


def backward(cache, x, y, params):
    A1 = cache["a1"]
    A2 = cache["a2"]
    w2 = params["w2"]

    # Compute dz2
    dz2 = A2 - y

    # Compute dw2
    dw2 = np.dot(dz2.T, A1)

    # Compute db2
    db2 = np.sum(dz2, axis=0, keepdims=True).T

    # Compute dz1
    dz1 = np.dot(dz2, w2) * derivative_sigmoid(A1)

    # Compute dw1
    dw1 = np.dot(dz1.T, x)

    # Compute db1
    db1 = np.sum(dz1, axis=0, keepdims=True).T

    derivatives = {"dz1": dz1, "dw1": dw1, "db1": db1, "dz2": dz2, "dw2": dw2, "db2": db2}
    return derivatives


# Perform forward and backward pass
output, cache = forward(x, params)
derivatives = backward(cache, x, y, params)

# Print the output and derivatives
print("Output:", output)
print("Derivatives:", derivatives)

# Visualization of the forward pass output using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=output.flatten(),
    mode='lines+markers',
    name='Output'
))

fig.update_layout(
    title='Forward Pass Output',
    xaxis_title='Neuron',
    yaxis_title='Activation'
)

fig.show()