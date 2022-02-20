from wsgiref.handlers import format_date_time
from matplotlib.pyplot import contour
from draw_detect import *
from extract_mp_keypoints import extract_keypoints
import numpy as np

from cvzone.HandTrackingModule import HandDetector
import time

pTime = 0

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

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
