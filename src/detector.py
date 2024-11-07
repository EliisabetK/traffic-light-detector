import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

model_paths = {
    'brightness_robust_detector': 'brightness_robust_detector.keras',
    'default_detector': 'default_detector.keras',
    'low_light_detector': 'low-light_detector.keras',
    'noise_robust_detector': 'noise_robust_detector.keras'
}

class_labels = ['green', 'red', 'yellow']

def predict_traffic_light(model, img_array):
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)] # gets the index of the highest probability value
    return predicted_class

def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape) #adds the noise to the image (values between 0 and 1 to keep pixel values valid)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def adjust_brightness(image, factor=1.0): #1.0 is no change (default), >1.0 is brighter, <1.0 is darker
    return np.clip(image * factor, 0, 1)

def test_on_folder(model, folder_path, modify_func=None): # looks though all images in the specified folder and predicts the traffic light color
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

                predicted_color = predict_traffic_light(model, img_array)
                total_images += 1
                if predicted_color == class_label:
                    correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100
    return accuracy, correct_predictions, total_images

if __name__ == "__main__":
    test_folder_path = 'data/test'
    modifications = [
        (None, "Default"),
        (add_noise, "Noise Addition"),
        (lambda img: adjust_brightness(img, 0.4), "Dimmed"),
        (lambda img: adjust_brightness(img, 1.8), "Brightened")
    ]

    results_summary = {}

    for model_name, model_path in model_paths.items():
        model = tf.keras.models.load_model(model_path)
        results_summary[model_name] = {}

        for modify_func, modification_name in modifications:
            accuracy, correct, total = test_on_folder(model, test_folder_path, modify_func)
            results_summary[model_name][modification_name] = accuracy
            print(f"Accuracy on test dataset with {modification_name} ({model_name}): {accuracy:.2f}% ({correct}/{total})")

    print("\nSummary of Results:")
    print(f"{'Model':<30} {'Original data':<15} {'Noise Addition':<15} {'Dimmed':<10} {'Brightened':<12}")
    print("="*100)
    for model_name, modifications in results_summary.items():
        default_acc = modifications.get("Default", 0)
        noise_acc = modifications.get("Noise Addition", 0)
        dimmed_acc = modifications.get("Dimmed", 0)
        brightened_acc = modifications.get("Brightened", 0)
        print(f"{model_name:<30} {default_acc:<15.2f} {noise_acc:<15.2f} {dimmed_acc:<10.2f} {brightened_acc:<12.2f}")
