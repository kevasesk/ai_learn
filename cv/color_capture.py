import cv2
import numpy as np

def get_limits(color):
    color = np.uint8([[color]])
    hsvC = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    color_range = 10

    hue = hsvC[0][0][0]
    lower_hue = hue - color_range
    upper_hue = hue + color_range

    if lower_hue < 0:
        lowerLimit1 = np.array([0, 50, 50], dtype=np.uint8)
        upperLimit1 = np.array([upper_hue, 255, 255], dtype=np.uint8)
        lowerLimit2 = np.array([179 + lower_hue + 1, 50, 50], dtype=np.uint8)
        upperLimit2 = np.array([179, 255, 255], dtype=np.uint8)
        return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)
    elif upper_hue > 179:
        lowerLimit1 = np.array([lower_hue, 50, 50], dtype=np.uint8)
        upperLimit1 = np.array([179, 255, 255], dtype=np.uint8)
        lowerLimit2 = np.array([0, 50, 50], dtype=np.uint8)
        upperLimit2 = np.array([upper_hue - 180, 255, 255], dtype=np.uint8)
        return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)
    else:
        lowerLimit = np.array([max(lower_hue, 0), 50, 50], dtype=np.uint8)
        upperLimit = np.array([min(upper_hue, 179), 255, 255], dtype=np.uint8)
        return (lowerLimit, upperLimit), None

detect_color = [0, 0, 255]
cap = cv2.VideoCapture(0)

while True:
    status, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    limits = get_limits(detect_color)

    if limits[1] is None:
        mask = cv2.inRange(hsvImage, limits[0][0], limits[0][1])
    else:
        mask1 = cv2.inRange(hsvImage, limits[0][0], limits[0][1])
        mask2 = cv2.inRange(hsvImage, limits[1][0], limits[1][1])
        mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()