# detector.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('traffic_light_detector.h5')

# Define the class labels based on your model's output
class_labels = ['green', 'red', 'yellow']

def predict_traffic_light(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))  # Resize to match the input shape
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

def test_on_folder(folder_path):
    total_images = 0
    correct_predictions = 0
    # Iterate through each class folder (green, red, yellow)
    for class_label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_label)

        # Check if it's a directory
        if os.path.isdir(class_folder):
            print(f"\nTesting images in '{class_label}' folder:")

            # Iterate through each image in the class folder
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                # Predict the traffic light color
                predicted_color = predict_traffic_light(img_path)

                # Print the result
                print(f"Image: {img_name} | Actual: {class_label} | Predicted: {predicted_color}")

                # Update accuracy calculation
                total_images += 1
                if predicted_color == class_label:
                    correct_predictions += 1

    # Calculate and print accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"\nAccuracy on test dataset: {accuracy:.2f}% ({correct_predictions}/{total_images})")
if __name__ == "__main__":
    # Path to test folder
    test_folder_path = 'data/test'
    test_on_folder(test_folder_path)

