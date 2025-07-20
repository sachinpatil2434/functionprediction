# 🧠 Function Predictor

**Function Predictor** is a personal project designed to identify the type of mathematical function from a hand-drawn graph using an Artificial Neural Network (ANN).

The application provides a simple interface for users to draw general function curves (like linear, quadratic, sinusoidal, etc.), and the ANN attempts to predict the type of function based on the shape.

---

## 📌 How It Works

- Users draw a function on a canvas using a GUI built with Tkinter.
- The image is passed through a trained ANN model which predicts the type of function.
- If the prediction is incorrect, the user can manually correct the result.
- The corrected data is then saved into a `user_data/` folder for future retraining.

---

## ⚠️ Current Limitations

One of the main challenges in this project is the **lack of a reliable labeled dataset** for hand-drawn functions. Due to this, the ANN struggles to make accurate predictions consistently.

To improve this:
- The system allows user-corrected labels to be saved.
- These saved corrections can later be used to **retrain the ANN**, helping the model learn from its mistakes over time.

---

## 🧪 Future Plan

- Build a more diverse and balanced training dataset using both manual and synthetic data.
- Improve the ANN architecture and training process to boost prediction accuracy.
- Add symbolic output (e.g., "y = sin(x)") in future versions.

---

## 👨‍💻 Author

**Sachin Patil**  
📧 Email: sachindpatil2434@gmail.com

I hope you enjoy using this application and watching it improve as it learns!

