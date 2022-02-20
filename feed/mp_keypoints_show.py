from wsgiref.handlers import format_date_time
from matplotlib.pyplot import contour
from draw_detect import *
from extract_mp_keypoints import extract_keypoints
from tensorflow.python.keras import models
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

from cvzone.HandTrackingModule import HandDetector
import time

trained_model = models.load_model("../trained_models/trained_vggface.h5", compile=False)
trained_model.summary()
cv2.ocl.setUseOpenCL(False)
pTime = 0
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
detector = MTCNN()

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
black = np.zeros((96, 96))

# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(extract_keypoints(results))

        # Draw landmarks
        draw_styled_landmarks(image, results)

        hands, image = detector.findHands(image)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1["center"]
            handType1 = hand1["type"]
            fingers1 = detector.fingersUp(hand1)
            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2["center"]
                handType2 = hand2["type"]
                fingers2 = detector.fingersUp(hand2)
                length, info, image = detector.findDistance(
                    centerPoint1, centerPoint2, image
                )

        if not ret:
            break
        results = detector.detect_faces(frame)
        if len(results) == 1:
            try:
                x1, y1, width, height = results[0]["box"]
                x2, y2 = x1 + width, y1 + height
                face = frame[y1:y2, x1:x2]
                cv2.rectangle(
                    frame, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2
                )
                cropped_img = cv2.resize(face, (96, 96))
                cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
                cropped_img_float = cropped_img_expanded.astype(float)
                prediction = trained_model.predict(cropped_img_float)
                print(prediction)
                maxindex = int(np.argmax(prediction))
                cv2.putText(
                    frame,
                    emotion_dict[maxindex],
                    (x1 + 20, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except:
                pass

        # print fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            image,
            "Current FPS: " + str(int(fps)),
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            2,
        )

        # Show to screen
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
