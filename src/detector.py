import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('traffic_light_detector.keras')

class_labels = ['green', 'red', 'yellow']

def predict_traffic_light(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

def add_noise(image, noise_level=0.2):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image


def adjust_brightness(image, factor=1.0):
    return np.clip(image * factor, 0, 1)

def test_on_folder(folder_path, modify_func=None, modification_name=""):
    total_images = 0
    correct_predictions = 0
    for class_label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_label)

        if os.path.isdir(class_folder):

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0

                if modify_func:
                    img_array = modify_func(img_array)

                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)
                predicted_color = class_labels[np.argmax(predictions)]

                total_images += 1
                if predicted_color == class_label:
                    correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100
    print(f"\nAccuracy on test dataset with {modification_name}: {accuracy:.2f}% ({correct_predictions}/{total_images})")

if __name__ == "__main__":
    test_folder_path = 'data/test'
    test_on_folder(test_folder_path)
    test_on_folder(test_folder_path, add_noise, "Noise Addition")
    test_on_folder(test_folder_path, lambda img: adjust_brightness(img, 0.5), "Dimmed")
    test_on_folder(test_folder_path, lambda img: adjust_brightness(img, 1.5), "Brightened")