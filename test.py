import cv2

from dna import Frame
from dna.detect.ultralytics.ultralytics_detector import load, UltralyticsDetector

detector = load('model=yolov8l&type=v8&score=0.1&classes=car,bus,truck&agnostic_nms=True')

img = cv2.imread("data/etri_04.jpg")
frame = Frame(img, 1, 1)

detections = detector.detect(frame)
print(detections)