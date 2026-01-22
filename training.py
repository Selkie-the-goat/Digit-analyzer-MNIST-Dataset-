import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

trainset="dataset/mnist_train.csv"
testset="dataset/mnist_test.csv"

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

def initialize_parameters(input_size=784, hidden1=128, hidden2=64, output_size=10):
    np.random.seed(42)
    
    W1 = np.random.randn(hidden1, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden1, 1))
    
    W2 = np.random.randn(hidden2, hidden1) * np.sqrt(2. / hidden1)
    b2 = np.zeros((hidden2, 1))
    
    W3 = np.random.randn(output_size, hidden2) * np.sqrt(2. / hidden2)
    b3 = np.zeros((output_size, 1))
    
    return W1, b1, W2, b2, W3, b3

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)
    
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def compute_loss(A3, Y):
    #cross entroy ko formula
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A3 + 1e-8)) / m
    return loss

def backward_pass(X, Y, cache, W1, W2, W3):
    m = X.shape[1]
    Z1, A1, Z2, A2, Z3, A3 = cache

    #output layer
    dZ3 = A3 - Y              
    dW3 = (dZ3 @ A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    #layer 2
    dA2 = W3.T @ dZ3
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Layer 1
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    return W1, b1, W2, b2, W3, b3


print("Loading data...") #not that deep lil vro
df_train = pd.read_csv(trainset)


X = df_train.iloc[:, 1:].to_numpy()
y = df_train.iloc[:, 0].to_numpy()

# Normalizing input to 0-1 scale(very important)
X = X / 255.0
X = X.T


train_size = 50000
X_train, y_train = X[:, :train_size], y[:train_size]
X_val, y_val = X[:, train_size:], y[train_size:]

Y_train = one_hot_encode(y_train, num_classes=10)
Y_val = one_hot_encode(y_val, num_classes=10)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {Y_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Try to load existing parameters, or initialize fresh ones
try:
    print("\nAttempting to load existing parameters...")
    params = np.load("model_params.npz")
    W1 = params['weight_layer_1']
    W2 = params['weight_layer_2']
    W3 = params['weight_layer_3']
    b1 = params['bias_layer_1']
    b2 = params['bias_layer_2']
    b3 = params['bias_layer_3']
    
    print(f"Loaded parameters:")
    print(f"  W1 shape: {W1.shape}, mean: {W1.mean():.6f}, std: {W1.std():.6f}")
    print(f"  W2 shape: {W2.shape}, mean: {W2.mean():.6f}, std: {W2.std():.6f}")
    print(f"  W3 shape: {W3.shape}, mean: {W3.mean():.6f}, std: {W3.std():.6f}")
    
    # Check if parameters look suspicious (NaN or very low variance) very important step(made me cry)
    if (np.isnan(W1).any() or np.isnan(W2).any() or np.isnan(W3).any() or
        W1.std() < 0.001 or W2.std() < 0.001 or W3.std() < 0.001):
        print("\n⚠️  WARNING: Weights are corrupted (NaN or low variance) - reinitializing!")
        raise ValueError("Bad initialization")
    
    print("✓ Loaded existing parameters successfully")
    
except:
    print("\n⚠️  Could not load parameters or they were corrupted. Initializing fresh weights...")
    W1, b1, W2, b2, W3, b3 = initialize_parameters()
    print(f"Initialized parameters:")
    print(f"  W1 shape: {W1.shape}")
    print(f"  W2 shape: {W2.shape}")
    print(f"  W3 shape: {W3.shape}")



# Training
print("\n" + "="*50)
print("Starting training...")
print("="*50)

learning_rate = 0.01  
epochs = 600

for i in range(epochs):
    
    A3, cache = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
    
    
    loss = compute_loss(A3, Y_train)
    
   
    dW1, db1, dW2, db2, dW3, db3 = backward_pass(X_train, Y_train, cache, W1, W2, W3)
    
    if i == 0:
        print(f"\nFirst iteration gradient magnitudes:")
        print(f"  dW1: {np.abs(dW1).mean():.8f}")
        print(f"  dW2: {np.abs(dW2).mean():.8f}")
        print(f"  dW3: {np.abs(dW3).mean():.8f}")
    
    W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, 
                                                dW1, db1, dW2, db2, dW3, db3, learning_rate)
    
    # Print progress
    if (i + 1) % 50 == 0:
        predictions = np.argmax(A3, axis=0)
        accuracy = np.mean(predictions == y_train) * 100
        
        # Validation
        A3_val, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
        predictions_val = np.argmax(A3_val, axis=0)
        val_accuracy = np.mean(predictions_val == y_val) * 100
        
        print(f"Epoch {i+1:3d}/{epochs} | Loss: {loss:.4f} | Train Acc: {accuracy:5.2f}% | Val Acc: {val_accuracy:5.2f}%")

print("\nSaving parameters...")
np.savez("model_params.npz", weight_layer_1=W1, weight_layer_2=W2, weight_layer_3=W3, bias_layer_1=b1, bias_layer_2=b2, bias_layer_3=b3)

# Final evaluation
print("\n" + "="*50)
print("Final Results:")
print("="*50)
A3_final, _ = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
predictions = np.argmax(A3_final, axis=0)
accuracy = np.mean(predictions == y_train) * 100
print(f"Final Training Accuracy: {accuracy:.2f}%")

A3_val, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
predictions_val = np.argmax(A3_val, axis=0)
val_accuracy = np.mean(predictions_val == y_val) * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")