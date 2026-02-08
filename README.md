# Handwritten Digit Recognizer 

A desktop application that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The application is built using **Python**, **TensorFlow**, and **Tkinter**.

---

##  Features

- Upload an image of a handwritten digit
- Image preprocessing (grayscale, resize, normalization)
- CNN-based digit prediction
- Simple and clean desktop UI using Tkinter

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Tkinter (Desktop UI)

---

## Input Image Guidelines 
   
   The model is trained on the **MNIST dataset**, Images closer to this format give better results :-

- Image should contain one digit (0–9)

- Prefer thick black digit on plain white background

- Supported formats: .png, .jpg, .jpeg

- Digit should be centered in the image

---

## 1️ Clone the Repository
bash:-

git clone https://github.com/om1526/handwritten-digit-recognizer.git

cd handwritten-digit-recognizer

python -m venv venv
venv\Scripts\activate   

pip install -r requirements.txt

python app.py


---


