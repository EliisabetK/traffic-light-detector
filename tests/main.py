import cv2
import numpy as np

def detect(image):
  
    if image is None:
        print("Error: Unable to read the image.")
        return None

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

    size = image.shape

    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=30)
    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=30)
    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=5, minRadius=0, maxRadius=30)

    detected_color = 'unknown'

    r = 5  # radius of the detected region for intensity check
    bound = 4.0 / 10  # limit detection to upper part of the image

    def is_in_bounds(i, m, n, size):
        return (0 <= i[1] + m < size[0]) and (0 <= i[0] + n < size[1])

    def check_and_print_circles(circles, mask, color_name):
        nonlocal detected_color
        if circles is not None and detected_color == 'unknown':
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if i[0] < size[1] and i[1] < size[0] and i[1] < size[0] * bound:
                    h, s = 0.0, 0.0
                    for m in range(-r, r):
                        for n in range(-r, r):
                            if is_in_bounds(i, m, n, size):
                                h += np.clip(mask[i[1] + m, i[0] + n], 0, 255)
                                s += 1

                    if s > 0 and h / s > 50:
                        detected_color = color_name

    check_and_print_circles(r_circles, maskr, 'red')
    check_and_print_circles(g_circles, maskg, 'green')
    check_and_print_circles(y_circles, masky, 'yellow')

    return detected_color