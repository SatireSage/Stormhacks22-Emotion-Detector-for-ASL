import cv2
import mediapipe as mp
import math
import pyautogui

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    # COLOR CONVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
    )  # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )  # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw left-hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw right-hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    """
    # Draw left-hand connections
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right-hand connections
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )
    """


"""
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        cv_frame = frame[100:300, 100:300]

        blur = cv2.GaussianBlur(cv_frame, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

        kernel = np.ones((5, 5), np.uint8)

        dillation = cv2.dilate(mask, kernel, iterations=1)
        erosion = cv2.erode(dillation, kernel, iterations=1)

        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        contours, hierachy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        try:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cv_frame, (x, y), (x + w, y + h), (0, 0, 255), 0)

            hull = cv2.convexHull(contour)

            drawing = np.zeros(cv_frame.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            def_count = 0

            for num in range(defects.shape[0]):
                s, e, f, d = defects[num, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b**2 + c**2 - a**2) / 2 * b * c) * 180) / 3.14

                if angle <= 90:
                    def_count += 1
                    cv2.circle(cv_frame, far, 1, [0, 0, 255], -1)

                cv2.line(cv_frame, start, end, [0, 255, 0], 2)

            if def_count >= 4:
                pyautogui.press("space")
                cv2.putText(
                    cv_frame,
                    "JUMP",
                    (115, 00),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    2,
                    2,
                )
        except:
            pass
        cv2.imshow("OpenCV Feed", frame)
"""
