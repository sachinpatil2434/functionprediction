import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random
import numpy as np
import glob
import time # For unique filenames

# --- Part 1: Dataset Generation and Loading ---

# Base functions list - will be extended by user data
BASE_FUNCTIONS = ['sin', 'cos', 'x_squared', 'linear']

class DummyFunctionDataset(Dataset):
    """
    A dummy dataset to simulate hand-drawn function images.
    In a real application, you would load your actual image files.
    """
    def __init__(self, num_samples=5000, img_size=(64, 64), transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        self.functions = BASE_FUNCTIONS # Use base functions
        self.data = self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generates dummy images and labels."""
        data = []
        for i in range(self.num_samples):
            func_type = random.choice(self.functions)
            
            # Create a blank image
            img = Image.new('L', self.img_size, color=255) # White background
            draw = ImageDraw.Draw(img)

            # Draw a simple representation of the function
            if func_type == 'sin':
                for x in range(self.img_size[0]):
                    y = int((np.sin(x / (self.img_size[0] / (2 * np.pi))) * (self.img_size[1] / 4)) + (self.img_size[1] / 2))
                    if 0 <= y < self.img_size[1]:
                        draw.point((x, y), fill=0) # Black line
            elif func_type == 'cos':
                for x in range(self.img_size[0]):
                    y = int((np.cos(x / (self.img_size[0] / (2 * np.pi))) * (self.img_size[1] / 4)) + (self.img_size[1] / 2))
                    if 0 <= y < self.img_size[1]:
                        draw.point((x, y), fill=0)
            elif func_type == 'x_squared':
                for x in range(self.img_size[0]):
                    # Normalize x to [-1, 1] for x^2
                    norm_x = (x / self.img_size[0]) * 2 - 1
                    y = int((1 - norm_x**2) * (self.img_size[1] / 2)) # Invert y for drawing
                    if 0 <= y < self.img_size[1]:
                        draw.point((x, y), fill=0)
            elif func_type == 'linear':
                draw.line([(0, self.img_size[1]), (self.img_size[0], 0)], fill=0, width=1) # Simple diagonal line

            label_idx = self.functions.index(func_type)
            data.append((img, label_idx))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class UserFunctionDataset(Dataset):
    """
    Dataset to load user-drawn images from the 'user_data' directory.
    """
    def __init__(self, root_dir='user_data', img_size=(64, 64), transform=None, functions_map=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.data = []
        self.functions_map = functions_map if functions_map is not None else {}
        self._load_data()

    def _load_data(self):
        """Loads images and labels from the user_data directory."""
        if not os.path.exists(self.root_dir):
            return

        for func_name in os.listdir(self.root_dir):
            func_dir = os.path.join(self.root_dir, func_name)
            if os.path.isdir(func_dir):
                if func_name not in self.functions_map:
                    # This should ideally not happen if functions_map is comprehensive
                    # but as a fallback, add new functions if encountered.
                    # In a real app, you'd manage this map more strictly.
                    print(f"Warning: New function '{func_name}' found in user data not in initial map.")
                    continue # Skip if not in the provided map for consistency

                label_idx = self.functions_map[func_name]
                for img_file in glob.glob(os.path.join(func_dir, '*.png')):
                    try:
                        img = Image.open(img_file).convert('L') # Convert to grayscale
                        self.data.append((img, label_idx))
                    except Exception as e:
                        print(f"Could not load image {img_file}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CombinedFunctionDataset(Dataset):
    """
    Combines dummy and user-generated datasets.
    """
    def __init__(self, dummy_dataset, user_dataset):
        self.dummy_dataset = dummy_dataset
        self.user_dataset = user_dataset
        self.total_len = len(dummy_dataset) + len(user_dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.dummy_dataset):
            return self.dummy_dataset[idx]
        else:
            return self.user_dataset[idx - len(self.dummy_dataset)]

# --- Part 2: CNN + ANN Model Design ---

class FunctionRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(FunctionRecognizer, self).__init__()
        # Convolutional Layers (CNN for feature extraction)
        self.cnn_layers = nn.Sequential(
            # Input: 1 channel (grayscale), e.g., 64x64 image
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Output: 32 channels, 64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 channels, 32x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: 64 channels, 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 channels, 16x16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: 128 channels, 16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128 channels, 8x8
        )

        # Fully Connected Layers (ANN for classification)
        # Calculate the input features for the first linear layer.
        # For a 64x64 input, after 3 MaxPool2d layers (each halving dimensions):
        # 64 -> 32 -> 16 -> 8
        # So, 128 channels * 8 * 8 pixels = 8192 features
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), # Adjust input size based on CNN output
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(512, num_classes) # Output layer with num_classes
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layers
        x = self.fc_layers(x)
        return x

# --- Part 3: Model Training Function ---

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, model_path='function_recognizer.pth'):
    """
    Trains the FunctionRecognizer model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad() # Clear gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# --- Part 4: Tkinter GUI Integration ---

class FunctionDrawingApp:
    def __init__(self, root, model_path='function_recognizer.pth', initial_functions_list=None):
        self.root = root
        self.root.title("Function Recognizer")
        self.model_path = model_path
        self.initial_functions_list = initial_functions_list if initial_functions_list is not None else BASE_FUNCTIONS
        self.img_size = (64, 64) # Image size for model input
        self.user_data_dir = 'user_data'

        # Dynamically update functions list based on all available data
        self.functions_list = self._get_all_function_names()
        self.function_to_idx = {name: i for i, name in enumerate(self.functions_list)}

        # Drawing variables
        self.drawing = False
        self.last_x, self.last_y = None, None

        # Canvas for drawing
        self.canvas = tk.Canvas(root, bg="white", width=400, height=400, bd=2, relief="groove")
        self.canvas.pack(pady=10)

        # Create an in-memory image to draw on
        self.image = Image.new("L", (400, 400), 255) # White background (L for grayscale)
        self.draw = ImageDraw.Draw(self.image)

        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        # Prediction label
        self.prediction_label = tk.Label(root, text="Draw a function and click Predict!", font=("Inter", 14))
        self.prediction_label.pack(pady=5)

        # Label and Entry for saving
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(pady=5)
        tk.Label(self.label_frame, text="Function Name:", font=("Inter", 10)).pack(side=tk.LEFT, padx=5)
        self.label_entry = tk.Entry(self.label_frame, width=20, font=("Inter", 10))
        self.label_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.predict_button = tk.Button(button_frame, text="Predict Function", command=self.predict_function,
                                        bg="#4CAF50", fg="white", font=("Inter", 12), relief="raised", bd=3,
                                        activebackground="#45a049", activeforeground="white", cursor="hand2")
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas,
                                      bg="#f44336", fg="white", font=("Inter", 12), relief="raised", bd=3,
                                      activebackground="#da190b", activeforeground="white", cursor="hand2")
        self.clear_button.pack(side=tk.LEFT, padx=10) # Changed to LEFT for better layout

        self.save_button = tk.Button(button_frame, text="Save Drawing", command=self.save_drawing,
                                     bg="#2196F3", fg="white", font=("Inter", 12), relief="raised", bd=3,
                                     activebackground="#1976D2", activeforeground="white", cursor="hand2")
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.retrain_button = tk.Button(button_frame, text="Retrain Model", command=self.retrain_model_gui,
                                        bg="#FFC107", fg="black", font=("Inter", 12), relief="raised", bd=3,
                                        activebackground="#FFA000", activeforeground="black", cursor="hand2")
        self.retrain_button.pack(side=tk.LEFT, padx=10)

        # Define the transformations for inference
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize grayscale images to [-1, 1]
        ])

        # Load the trained model initially
        self._load_model()

    def _get_all_function_names(self):
        """Collects all unique function names from dummy and user data."""
        all_functions = set(self.initial_functions_list)
        if os.path.exists(self.user_data_dir):
            for func_name in os.listdir(self.user_data_dir):
                if os.path.isdir(os.path.join(self.user_data_dir, func_name)):
                    all_functions.add(func_name)
        return sorted(list(all_functions))

    def _load_model(self):
        """Loads or re-initializes the model."""
        # Re-create function_to_idx map in case new functions were added
        self.functions_list = self._get_all_function_names()
        self.function_to_idx = {name: i for i, name in enumerate(self.functions_list)}
        
        self.model = FunctionRecognizer(len(self.functions_list))
        try:
            # Ensure the model is loaded to CPU if not explicitly using GPU
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval() # Set model to evaluation mode
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            messagebox.showwarning("Model Warning", f"Model file not found at {self.model_path}. Please train the model first.")
            # self.predict_button.config(state=tk.DISABLED) # Don't disable, allow training
        except Exception as e:
            messagebox.showerror("Model Error", f"Error loading model: {e}. Model might need retraining.")
            # self.predict_button.config(state=tk.DISABLED)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line((self.last_x, self.last_y, x, y),
                                     fill="black", width=2, capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on the PIL image as well
            self.draw.line((self.last_x, self.last_y, x, y), fill=0, width=2) # 0 for black
            self.last_x, self.last_y = x, y

    def end_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (400, 400), 255) # Reset PIL image
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a function and click Predict!")
        self.label_entry.delete(0, tk.END) # Clear label entry

    def save_drawing(self):
        func_name = self.label_entry.get().strip().lower()
        if not func_name:
            messagebox.showwarning("Save Error", "Please enter a function name before saving.")
            return

        # Create directory if it doesn't exist
        save_dir = os.path.join(self.user_data_dir, func_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save the image with a unique timestamped filename
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(save_dir, f"{func_name}_{timestamp}.png")
        self.image.save(file_path)
        messagebox.showinfo("Save Success", f"Drawing saved as '{func_name}' to {file_path}")
        
        # Update functions list if new function type added
        if func_name not in self.functions_list:
            self.functions_list = self._get_all_function_names()
            self.function_to_idx = {name: i for i, name in enumerate(self.functions_list)}
            print(f"Updated function list: {self.functions_list}")

        self.clear_canvas()

    def retrain_model_gui(self):
        """Triggers model retraining from the GUI."""
        response = messagebox.askyesno("Retrain Model", "This will retrain the model with all available data (dummy + your saved drawings). This might take a moment. Continue?")
        if not response:
            return

        self.prediction_label.config(text="Retraining model... Please wait.")
        self.root.update_idletasks() # Update GUI immediately

        # Disable buttons during retraining
        for btn in [self.predict_button, self.clear_button, self.save_button, self.retrain_button]:
            btn.config(state=tk.DISABLED)

        try:
            # Re-collect all function names and update mapping
            self.functions_list = self._get_all_function_names()
            self.function_to_idx = {name: i for i, name in enumerate(self.functions_list)}
            num_classes = len(self.functions_list)
            print(f"Retraining with {num_classes} classes: {self.functions_list}")

            # Prepare datasets
            dummy_dataset = DummyFunctionDataset(num_samples=5000, img_size=self.img_size, transform=self.transform)
            # Pass the updated function_to_idx map to UserFunctionDataset
            user_dataset = UserFunctionDataset(root_dir=self.user_data_dir, img_size=self.img_size, 
                                               transform=self.transform, functions_map=self.function_to_idx)
            
            combined_dataset = CombinedFunctionDataset(dummy_dataset, user_dataset)
            train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

            # Re-initialize model with potentially new number of classes
            self.model = FunctionRecognizer(num_classes=num_classes)
            train_model(self.model, train_loader, num_epochs=20, learning_rate=0.001, model_path=self.model_path)
            
            # Reload the newly trained model
            self._load_model()
            messagebox.showinfo("Retraining Complete", "Model has been successfully retrained!")
            self.prediction_label.config(text="Model Retrained! Draw and Predict.")

        except Exception as e:
            messagebox.showerror("Retraining Error", f"An error occurred during retraining: {e}")
            self.prediction_label.config(text="Retraining failed.")
        finally:
            # Re-enable buttons
            for btn in [self.predict_button, self.clear_button, self.save_button, self.retrain_button]:
                btn.config(state=tk.NORMAL)

    def predict_function(self):
        if not hasattr(self, 'model') or self.model.training:
            messagebox.showwarning("Prediction Warning", "Model is not loaded or not in evaluation mode.")
            return
        
        if not self.functions_list:
            messagebox.showwarning("Prediction Warning", "No functions defined. Please train the model first.")
            return

        # Get the image from the canvas
        pil_image = self.image

        # Preprocess the image for the model
        input_tensor = self.transform(pil_image).unsqueeze(0) # Add batch dimension

        # Perform inference
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            
            if predicted_index >= len(self.functions_list):
                # This can happen if the model was trained with fewer classes than current list
                messagebox.showwarning("Prediction Error", "Model's output classes do not match current function list. Please retrain.")
                self.prediction_label.config(text="Prediction Error: Retrain model.")
                return

            predicted_function = self.functions_list[predicted_index]
            confidence = probabilities[0, predicted_index].item() * 100

        self.prediction_label.config(text=f"Predicted Function: {predicted_function} (Confidence: {confidence:.2f}%)")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define image size and transformations for training
    IMG_SIZE = (64, 64)
    transform_common = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

    # Initial setup of functions list for the first run
    # This will be dynamically updated by the app
    all_current_functions = sorted(list(set(BASE_FUNCTIONS)))
    current_function_to_idx = {name: i for i, name in enumerate(all_current_functions)}

    # --- Training Phase (Initial or if model not found) ---
    MODEL_FILE = 'function_recognizer.pth'

    if not os.path.exists(MODEL_FILE):
        print(f"Model '{MODEL_FILE}' not found. Performing initial training...")
        
        dummy_dataset_initial = DummyFunctionDataset(num_samples=5000, img_size=IMG_SIZE, transform=transform_common)
        # User dataset for initial training (might be empty)
        user_dataset_initial = UserFunctionDataset(root_dir='user_data', img_size=IMG_SIZE, 
                                                   transform=transform_common, functions_map=current_function_to_idx)
        
        combined_dataset_initial = CombinedFunctionDataset(dummy_dataset_initial, user_dataset_initial)
        train_loader_initial = DataLoader(combined_dataset_initial, batch_size=32, shuffle=True)

        model_initial = FunctionRecognizer(num_classes=len(all_current_functions))
        train_model(model_initial, train_loader_initial, num_epochs=20, learning_rate=0.001, model_path=MODEL_FILE)
    else:
        print(f"Model '{MODEL_FILE}' found. Skipping initial training.")

    # --- Tkinter App Phase ---
    root = tk.Tk()
    # Pass initial_functions_list, which will be expanded by the app itself
    app = FunctionDrawingApp(root, model_path=MODEL_FILE, initial_functions_list=BASE_FUNCTIONS)
    root.mainloop()

