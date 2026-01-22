import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

class NeuralNetwork:
    def __init__(self, param_file="model_params.npz"):
        try:
            params = np.load(param_file)
            self.W1, self.b1 = params['weight_layer_1'], params['bias_layer_1']
            self.W2, self.b2 = params['weight_layer_2'], params['bias_layer_2']
            self.W3, self.b3 = params['weight_layer_3'], params['bias_layer_3']
        except Exception as e:
            raise Exception(f"Failed to load weights: {e}")

    def predict(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = np.maximum(0, z1) 
        z2 = self.W2 @ a1 + self.b2
        a2 = np.maximum(0, z2) 
        z3 = self.W3 @ a2 + self.b3
        
        exp = np.exp(z3 - np.max(z3))
        return exp / np.sum(exp)

class FinalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Digit Scanner")
        self.nn = NeuralNetwork()
        
        self.sz = 280
        self.canvas = tk.Canvas(root, width=self.sz, height=self.sz, bg='white', highlightthickness=1)
        self.canvas.pack(pady=10)
        
        self.setup_buffers()
        
        self.result_text = tk.StringVar(value="Draw a number")
        tk.Label(root, textvariable=self.result_text, font=("Arial", 18, "bold")).pack()
        
        self.debug_text = tk.Label(root, text="Confidence: 0%", font=("Arial", 10))
        self.debug_text.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Identify", command=self.analyze, bg="#4CAF50", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear, bg="#f44336", fg="white", width=10).pack(side=tk.LEFT, padx=5)

        self.canvas.bind("<B1-Motion>", self.draw_line)

    def setup_buffers(self):
        self.pil_img = Image.new("L", (self.sz, self.sz), 255)
        self.draw_handle = ImageDraw.Draw(self.pil_img)

    def draw_line(self, event):
        r = 12 
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw_handle.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.setup_buffers() 
        self.result_text.set("Draw a number")
        self.debug_text.config(text="Confidence: 0%")

    def analyze(self):
        img = ImageOps.invert(self.pil_img)
        bbox = img.getbbox()
        if not bbox:
            self.result_text.set("Empty Canvas!")
            return

        digit = img.crop(bbox)
        w, h = digit.size
        ratio = 18.0 / max(w, h)
        digit = digit.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
        
        final_input = Image.new("L", (28, 28), 0)
        final_input.paste(digit, ((28 - digit.size[0])//2, (28 - digit.size[1])//2))
        
        arr = np.array(final_input).reshape(784, 1) / 255.0
        probs = self.nn.predict(arr)
        
        pred = np.argmax(probs)
        conf = probs[pred][0] * 100
        self.result_text.set(f"I see a {pred}!")
        self.debug_text.config(text=f"Confidence: {conf:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    FinalApp(root)
    root.mainloop()