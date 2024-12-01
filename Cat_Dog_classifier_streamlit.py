import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

# Step 1: Load the Saved Model and Training History
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_history(history_path):
    return np.load(history_path, allow_pickle=True).item()

# Step 2: Display Training History
def plot_training_history(history):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(history['accuracy'], label='Training Accuracy', marker='o', color='blue')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='o', color='green')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history['loss'], label='Training Loss', marker='o', color='red')
    ax[1].plot(history['val_loss'], label='Validation Loss', marker='o', color='orange')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

# Step 3: Classify Image
def classify_image(model, img_path, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]
    confidence = predictions[0][class_idx]

    return class_name, confidence, img

# Step 4: Overlay Label on Image
def overlay_label_on_image(img, label, confidence):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{label} ({confidence * 100:.2f}%)"

    text_position = (10, 10)
    text_color = (0, 255, 255)
    draw.text(text_position, text, font=font, fill=text_color)

    return img

# Streamlit App Interface
def main():
    # Apply background color and custom styles
    st.markdown(
        """
        <style>
        body {
            background-color: #e6f7ff;
        }
        .title {
            color: #007acc;
            font-size: 32px;
            font-weight: bold;
        }
        .header {
            color: #005c99;
            font-size: 24px;
            font-weight: bold;
        }
        .sidebar-title {
            color: #003d66;
            font-size: 20px;
            font-weight: bold;
        }
        .info {
            color: #004d80;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title'>Cat vs Dog Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='info'>This application uses a MobileNetV2 model to classify images of cats and dogs.</div>", unsafe_allow_html=True)

    # Sidebar: Load Model and Training History
    model_path = "saved_model/my_model.keras"
    history_path = "saved_model/history.npy"

    st.sidebar.markdown("<div class='sidebar-title'>Model Information</div>", unsafe_allow_html=True)
    st.sidebar.write("Loading model and training history...")
    model = load_model(model_path)
    history = load_history(history_path)
    st.sidebar.success("Model loaded successfully!")

    # Display Training History
    st.markdown("<div class='header'>Model Training History</div>", unsafe_allow_html=True)
    plot_training_history(history)

    # Upload or Select Image for Classification
    st.markdown("<div class='header'>Upload or Select an Image for Classification</div>", unsafe_allow_html=True)

    # Option 1: Upload an Image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Option 2: Select an Image from Unseen Folder
    unseen_folder = "C:/Users/Acer/Downloads/Activity_01/unseen"
    if os.path.exists(unseen_folder):
        subfolders = [f for f in os.listdir(unseen_folder) if os.path.isdir(os.path.join(unseen_folder, f))]
        selected_folder = st.selectbox("Select a folder", subfolders)
        selected_folder_path = os.path.join(unseen_folder, selected_folder)

        if os.path.exists(selected_folder_path):
            images = [f for f in os.listdir(selected_folder_path) if f.endswith(("jpg", "jpeg", "png"))]
            selected_image = st.selectbox("Select an image", images)
            selected_image_path = os.path.join(selected_folder_path, selected_image)
        else:
            selected_image_path = None
    else:
        selected_image_path = None
        st.error("Unseen folder not found!")

    # Classify Image
    if uploaded_file is not None or selected_image_path is not None:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            image_path = uploaded_file
        else:
            img = Image.open(selected_image_path)
            image_path = selected_image_path

        st.image(img, caption="Selected Image", use_container_width=True)

        with st.spinner("Classifying..."):
            class_names = ["Cat", "Dog"]
            class_name, confidence, img = classify_image(model, image_path, class_names)

            labeled_img = overlay_label_on_image(img, class_name, confidence)

        st.image(labeled_img, caption=f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)", use_container_width=True)

    # Download Model and History
    st.markdown("<div class='header'>Download Resources</div>", unsafe_allow_html=True)
    with open(model_path, "rb") as f:
        st.download_button("Download Model (.keras)", f, file_name="my_model.keras")
    with open(history_path, "rb") as f:
        st.download_button("Download Training History (.npy)", f, file_name="history.npy")

# Run the app
if __name__ == "__main__":
    main()
