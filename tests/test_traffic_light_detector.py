import unittest
import cv2
import numpy as np
import os
from main import detect 

class TestDetectFunction(unittest.TestCase):

    def setUp(self):
        self.dataset_path = "../light"
        self.image_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

    def add_noise(self, image):
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def dim_image(self, image, factor=0.5):
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def occlude_image(self, image):
        occluded_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskr = cv2.add(mask1, mask2)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)

        combined_mask = maskr | maskg | masky
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(occluded_image, (x, y + h // 2), (x + w, y + h), (0, 0, 0), -1)

        return occluded_image

    def test_initial_accuracy(self):
        total_tests = 0
        correct_detections = 0

        for file in self.image_files:
            image_path = os.path.join(self.dataset_path, file)
            image = cv2.imread(image_path)
            true_label = file.split('_')[0]

            detected_color = detect(image)  
            total_tests += 1
            if detected_color == true_label:
                correct_detections += 1
            else:
                print(f"Initial test failed for {file}: expected {true_label}, got {detected_color}")

        accuracy = (correct_detections / total_tests) * 100
        print(f"Initial accuracy with unmodified images: {accuracy:.2f}% ({correct_detections}/{total_tests})")

    def run_modification_test(self, modification_name, modify_image_func):
        total_tests = 0
        correct_detections = 0
        for file in self.image_files:
            image_path = os.path.join(self.dataset_path, file)
            image = cv2.imread(image_path)
            true_label = file.split('_')[0]

            modified_image = modify_image_func(image)
            detected_color = detect(modified_image)
            total_tests += 1
            if detected_color == true_label:
                correct_detections += 1
            else:
                print(f"{modification_name} test failed for {file}: expected {true_label}, got {detected_color}")

        accuracy = (correct_detections / total_tests) * 100
        print(f"Accuracy after {modification_name}: {accuracy:.2f}% ({correct_detections}/{total_tests})")

    def test_images(self):
        print("\nRunning Initial Accuracy Test...")
        self.test_initial_accuracy()

        print("\nRunning Noise Robustness Test...")
        self.run_modification_test("Noise_Robustness", self.add_noise)

        print("\nRunning Occlusion Handling Test...")
        self.run_modification_test("Occlusion_Handling", self.occlude_image)

        print("\nRunning Low-light Performance Test...")
        self.run_modification_test("Low_Light_Performance", lambda image: self.dim_image(image, factor=0.5))

        print("\nRunning Lighting Variability Test...")
        self.run_modification_test("Lighting_Variability", lambda image: self.dim_image(image, factor=1.5))

        print("\nRunning Feature Scaling Test...")
        self.run_modification_test("Feature_Scaling", lambda image: cv2.resize(image, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR))

if __name__ == '__main__':
    unittest.main()