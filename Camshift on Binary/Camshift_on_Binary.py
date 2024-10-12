import cv2
import numpy as np

def binarize_image(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def apply_pseudo_color(binary_image, color=(0, 255, 0)):
    h, w = binary_image.shape
    pseudo_color_image = np.zeros((h, w, 3), dtype=np.uint8)
    mask = binary_image == 255
    pseudo_color_image[mask] = color

    return pseudo_color_image


cap = cv2.VideoCapture("video.mp4")

ret, frame = cap.read()
if not ret:
    print("Can't open this video")
    cap.release()
    exit()

ori_frame = frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)
binary_inverted = cv2.bitwise_not(binary)
pseudo_color_image = apply_pseudo_color(binary_inverted)
frame = pseudo_color_image
cv2.imshow('ee',frame)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


lower_color = np.array([35, 100, 100])
upper_color = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower_color, upper_color)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    c = max(contours, key=cv2.contourArea)

    r = cv2.minEnclosingCircle(c)
    (x, y), radius = r
    center = (int(x), int(y))

    x, y, w, h = [int(v) for v in cv2.boundingRect(c)]

    track_window = (x, y, w, h)

    roi_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
else:
    print("Can't find target.")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ori_frame = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    binary_inverted = cv2.bitwise_not(binary)
    pseudo_color_image = apply_pseudo_color(binary_inverted)
    frame = pseudo_color_image

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    (x, y, w, h) = track_window

    cv2.rectangle(ori_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center_x =int (x + w / 2)
    center_y = int (y + h / 2)
    cv2.circle(ori_frame,(center_x,center_y),3,(255,0,0))
    print(center_x,center_y)
    cv2.imshow('CamShift Demo', ori_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()