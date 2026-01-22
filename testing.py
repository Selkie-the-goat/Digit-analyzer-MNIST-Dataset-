import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

testset = "dataset/mnist_test.csv"

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    return A3

def confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

params = np.load("model_params.npz")
W1 = params['weight_layer_1']
W2 = params['weight_layer_2']
W3 = params['weight_layer_3']
b1 = params['bias_layer_1']
b2 = params['bias_layer_2']
b3 = params['bias_layer_3']

df_test = pd.read_csv(testset)
X_test = df_test.iloc[:, 1:].to_numpy()
y_test = df_test.iloc[:, 0].to_numpy()

X_test = X_test / 255.0
X_test = X_test.T

A3_test = forward_pass(X_test, W1, b1, W2, b2, W3, b3)

predictions = np.argmax(A3_test, axis=0)
accuracy = np.mean(predictions == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

for digit in range(10):
    mask = (y_test == digit)
    digit_acc = np.mean(predictions[mask] == digit) * 100
    print(f"Digit {digit}: {digit_acc:.2f}%")
conf_matrix = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(conf_matrix)

