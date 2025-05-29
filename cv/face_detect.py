import cv2
import numpy as np
import mediapipe as mp


def detect_face(img):
    detection_model = mp.solutions.face_detection
    faceHeight, faceWidth , _ = img.shape

    with detection_model.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_detection.process(img_cvt)

        if not result.detections:
            return
        
        for detection in result.detections:
            box = detection.location_data.relative_bounding_box
    
            xmin, ymin, width, height = box.xmin, box.ymin, box.width, box.height
            xmin, ymin, width, height = int(xmin * faceWidth), int(ymin * faceHeight), int(width * faceWidth), int(height * faceHeight)
            
            cv2.rectangle(img, (xmin, ymin), (xmin + width, ymin + height), (255, 0, 0))


def detect_by_image(image_index=0):
    someFaces = [
        cv2.imread('cv/images/faces/face_1.png'),
        cv2.imread('cv/images/faces/face_2.jpeg'),
        cv2.imread('cv/images/faces/face_3.jpg')
    ]
    assert image_index < 0 or image_index < len(someFaces), f'No such image with index {image_index}'
    testFace = someFaces[image_index]
   
    detect_face(testFace)

    cv2.imshow('some face', testFace)
    cv2.waitKey(0)


def detect_online():
    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()
        assert success, 'Error getting frame'

        detect_face(frame)
        cv2.imshow('some face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    capture.release()
    cv2.destroyAllWindows()

#detect_by_image(1)
detect_online()
        
