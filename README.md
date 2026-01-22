# 🧠 Pure NumPy Handwritten Digit Classifier

A professional-grade implementation of a Multilayer Perceptron (MLP) built entirely from scratch using **Python and NumPy**. This project demonstrates the mathematical foundations of deep learning by implementing backpropagation and gradient descent without the aid of high-level frameworks like TensorFlow or PyTorch.



## 🌟 Why This Project is Unique
Most digit classifiers use `model.fit()`. This project uses **pure calculus and linear algebra**. Every weight update is calculated manually through the chain rule, providing full transparency into how the "brain" actually learns.

### Key Technical Achievements:
* **Vectorized Engine:** Optimized matrix multiplications for fast training across 60,000 images.
* **Manual Backpropagation:** Implementation of the derivative of ReLU and Softmax to propagate error backwards.
* **Custom Data Pipeline:** A dedicated image processing layer that mimics the MNIST standard for real-world user drawings.

---

## 📐 Neural Network Architecture
The engine powers a 3-layer dense network:
1.  **Input Layer:** 784 neurons (representing 28x28 grayscale pixels).
2.  **Hidden Layer 1:** 128 neurons utilizing **ReLU** activation.
3.  **Hidden Layer 2:** 64 neurons utilizing **ReLU** activation.
4.  **Output Layer:** 10 neurons with **Softmax** activation for probability distribution.



---

## 🛠️ Built With
* **NumPy:** Linear algebra and matrix operations.
* **Python:** Core logic and training loops.
* **Pillow (PIL):** Image preprocessing for the GUI.
* **Tkinter:** Desktop interface for user interaction.

---

## 🖼️ Application Preview
Here is the GUI in action:
![Application Screenshot](assets/app.png)

## 📊 Training Performance
The model was trained using a custom NumPy-based cross-entropy loss function:
![Training Accuracy Graph](assets/test.png)

---
## 🚀 Getting Started

### 1. Installation
Clone the repo and install the minimal dependencies:
```bash
git clone [https://github.com/Selkie-the-goat/Digit-analyzer-MNIST-Dataset-.git](https://github.com/Selkie-the-goat/Digit-analyzer-MNIST-Dataset-.git)
cd numpy-digit-classifier
pip install numpy pillow pandas