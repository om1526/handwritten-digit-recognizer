import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# ---------------- Load Model ----------------
model = load_model("saved_model/mnist_cnn_model.h5")

# ---------------- Image Processing ----------------
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    image = np.array(image)

    # Auto invert if background is light
    if np.mean(image) > 127:
        image = 255 - image

    # Normalize
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image

# ---------------- Prediction ----------------
def predict_digit(image_path):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return int(np.argmax(prediction))

# ---------------- UI Setup ----------------
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("500x600")
root.configure(bg="#EAF0F6")
root.resizable(False, False)

# Shadow card using Canvas
canvas = tk.Canvas(root, bg="#EAF0F6", highlightthickness=0)
canvas.place(relwidth=1, relheight=1)
canvas.create_rectangle(40, 40, 460, 560, fill="#BDC3C7", outline="", width=0)

# Main card container
card = tk.Frame(root, bg="white")
card.place(relx=0.5, rely=0.5, anchor="center", width=420, height=520)

# Title
tk.Label(
    card,
    text="Handwritten Digit Recognizer",
    bg="white",
    fg="#2C3E50",
    font=("Segoe UI", 20, "bold")
).pack(pady=(25, 5))

tk.Label(
    card,
    text="Upload an image of a handwritten digit (0–9)",
    bg="white",
    fg="#7F8C8D",
    font=("Segoe UI", 11)
).pack(pady=(0, 20))

# Image display box
image_frame = tk.Frame(card, bg="#F4F6F8", width=200, height=200)
image_frame.pack()
image_frame.pack_propagate(False)

image_label = tk.Label(image_frame, bg="#F4F6F8")
image_label.pack(expand=True)

# Result text
result_label = tk.Label(
    card,
    text="Prediction will appear here",
    bg="white",
    fg="#34495E",
    font=("Segoe UI", 14)
)
result_label.pack(pady=25)

# Upload button logic
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    # Display image
    img = Image.open(file_path)
    img_resized = img.resize((180, 180))
    tk_img = ImageTk.PhotoImage(img_resized)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Predict
    digit = predict_digit(file_path)
    result_label.config(
        text=f"Predicted Digit: {digit}",
        fg="#27AE60"
    )

# Button hover effects
def on_enter(e):
    upload_btn['bg'] = '#2980B9'

def on_leave(e):
    upload_btn['bg'] = '#3498DB'

# Upload button
upload_btn = tk.Button(
    card,
    text="Upload Image",
    command=upload_image,
    font=("Segoe UI", 12, "bold"),
    bg="#3498DB",
    fg="white",
    activebackground="#2980B9",
    bd=0,
    padx=20,
    pady=10,
    cursor="hand2"
)
upload_btn.pack(pady=20)
upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)

# Footer
tk.Label(
    card,
    text="CNN • Image Processing • Python",
    bg="white",
    fg="#95A5A6",
    font=("Segoe UI", 9)
).pack(side="bottom", pady=15)

root.mainloop()

