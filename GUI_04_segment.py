import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Step 1: Dataset Preparation (No Retraining)
def prepare_dataset(data_dir, batch_size=32, img_size=(128, 128)):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )

    # Extract class names before normalization
    class_names = train_ds.class_names

    # Normalize pixel values to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache and prefetch to improve performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# Step 2: Classify and Label Image
def classify_and_label_image(model, img_path, class_names, font_path=None, font_size=50, text_color="black"):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]

    # Predict the class
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]

    # Open the image for labeling
    img_with_label = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img_with_label)

    # Set the font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()

    # Add the classification label to the image
    text = f"Classified: {class_name}"
    draw.text((25, 25), text, fill=text_color, font=font)

    return img_with_label, class_name



# Step 3: Evaluate Model and Display Classification Report
def evaluate_model(model, val_ds, class_names):
    y_test = []
    y_pred = []

    for img_batch, label_batch in val_ds:
        predictions = model.predict(img_batch)
        y_test.extend(label_batch.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(y_test, y_pred, target_names=class_names)
    accuracy = accuracy_score(y_test, y_pred)
    return report, accuracy


# Step 4: Plot Training History
def plot_training_history(history):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(history['accuracy'], label='Training Accuracy', marker='o')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history['loss'], label='Training Loss', marker='o')
    ax[1].plot(history['val_loss'], label='Validation Loss', marker='o')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    return fig


# Step 5: GUI for displaying images, classification report, and training history
class ImageClassifierApp(tk.Tk):
    def __init__(self, model, class_names, history=None):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.history = history
        self.title("Image Classifier")
        self.geometry("900x800")
        self.configure(bg="#f0f8ff")  # Light background color

        # Title Label
        title_label = tk.Label(
            self, text="Image Classification Application",
            font=("Arial", 24, "bold"), bg="#f0f8ff", fg="#333"
        )
        title_label.pack(pady=20)

        # Frame for displaying images
        self.image_frame = tk.Frame(self, width=600, height=400, bg="#ffffff", relief="ridge", bd=2)
        self.image_frame.pack(pady=20)

        # Labels for classification report and accuracy
        self.report_label = tk.Label(
            self, text="Classification Report:\n", justify="left",
            font=("Courier", 10), bg="#f0f8ff", fg="#000080"
        )
        self.report_label.pack(pady=10)

        self.accuracy_label = tk.Label(
            self, text="Accuracy: \n", font=("Arial", 12, "bold"), bg="#f0f8ff", fg="#006400"
        )
        self.accuracy_label.pack()

        # Button for file selection and classification
        self.classify_button = tk.Button(
            self, text="Classify Image", command=self.classify_image,
            font=("Arial", 12), bg="#4682b4", fg="#ffffff", relief="raised"
        )
        self.classify_button.pack(pady=10)

        # Button to open training history window
        self.history_button = tk.Button(
            self, text="Training History", command=self.show_training_history,
            font=("Arial", 12), bg="#4682b4", fg="#ffffff", relief="raised"
        )
        self.history_button.pack(pady=10)

    def classify_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img_with_label, class_name = classify_and_label_image(self.model, file_path, self.class_names)

            # Clear the image frame before displaying new images
            for widget in self.image_frame.winfo_children():
                widget.destroy()

            # Display the original and classified images side by side
            original_image = Image.open(file_path)
            original_image.thumbnail((300, 300))
            img_with_label.thumbnail((300, 300))

            original_image_tk = ImageTk.PhotoImage(original_image)
            classified_image_tk = ImageTk.PhotoImage(img_with_label)

            original_label = tk.Label(self.image_frame, image=original_image_tk)
            classified_label = tk.Label(self.image_frame, image=classified_image_tk)

            original_label.image = original_image_tk
            classified_label.image = classified_image_tk

            original_label.pack(side="left", padx=10)
            classified_label.pack(side="right", padx=10)

            # Update Classification Report and Accuracy
            report, accuracy = evaluate_model(self.model, val_ds, self.class_names)
            self.report_label.config(text=f"Classification Report:\n{report}")
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.4f}")

    def show_training_history(self):
        if self.history is None:
            tk.messagebox.showinfo("Training History", "No training history available.")
            return

        # Create a new window for the training history plot
        history_window = tk.Toplevel(self)
        history_window.title("Training History")
        history_window.geometry("800x600")
        history_window.configure(bg="#f0f8ff")

        fig = plot_training_history(self.history)
        canvas = FigureCanvasTkAgg(fig, master=history_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        

if __name__ == "__main__":
    data_dir = "D:/New_downloads/Activity_01/dataset"  # Replace with your dataset path
    saved_model_path = "D:/New_downloads/Activity_01/saved_model/my_model.keras"  # Path to the saved model
    history_path = "D:/New_downloads/Activity_01/saved_model/history.npy"  # Path to the saved training history

    batch_size = 32
    img_size = (128, 128)

    # Step 1: Prepare dataset for evaluation (No retraining)
    train_ds, val_ds, class_names = prepare_dataset(data_dir, batch_size=batch_size, img_size=img_size)
    print("Classes:", class_names)

    # Step 2: Load the saved model
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path}...")
        model = load_model(saved_model_path)
    else:
        print("Saved model not found! Please ensure the model exists.")
        exit()

    # Step 3: Load training history if available
    history = None
    if os.path.exists(history_path):
        print(f"Loading training history from {history_path}...")
        history = np.load(history_path, allow_pickle=True).item()

    # Step 4: Launch the GUI
    app = ImageClassifierApp(model, class_names, history)
    app.mainloop()
