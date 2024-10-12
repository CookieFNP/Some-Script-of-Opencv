import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    center_xx = width // 2
    center_yy = height // 2

    _, binary_image = cv2.threshold(frame, 240, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    large_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    # 绘制框
    cv2.rectangle(frame, (x_min, y_min), (x_max+5, y_max+5), (0, 255, 0), 2)
    cv2.imshow('Image with Large Bounding Box', frame)

    center_x = x_min + (x_max - x_min) // 2
    center_y = y_min + (y_max - y_min) // 2
    print(f"Center of the large bounding box: ({center_x}, {center_y})")

    roi = frame[y_min:y_max+5, x_min:x_max+5]
    cv2.imshow('roi',roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    num_pixels_red = cv2.countNonZero(mask_red)
    num_pixels_green = cv2.countNonZero(mask_green)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if num_pixels_green > num_pixels_red:
        print("green")
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    else :
        print("red")
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)


    cv2.imshow('binarry',binary)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()