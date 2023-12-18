import cv2
import mediapipe as mp


def draw_rectangle(image, bbox, color):
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def draw_line(image, point1, point2, color):
    cv2.line(image, point1, point2, color, 2)


def main():
    cap = cv2.VideoCapture(0)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    detected_person = None

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            h, w, c = frame.shape
            detected_person = (int(x_min * w), int(y_min * h), int((x_max - x_min) * w), int((y_max - y_min) * h))

        right_hand_landmarks = []
        if results.left_hand_landmarks:  # Используем left_hand_landmarks для левой руки
            right_hand_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.left_hand_landmarks.landmark]

        if detected_person is not None:
            draw_rectangle(frame, detected_person, (0, 255, 0))  # Зеленый прямоугольник для человека

        if right_hand_landmarks:
            min_x = min([x for x, _ in right_hand_landmarks])
            max_x = max([x for x, _ in right_hand_landmarks])
            min_y = min([y for _, y in right_hand_landmarks])
            max_y = max([y for _, y in right_hand_landmarks])
            right_hand_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            draw_rectangle(frame, right_hand_bbox, (0, 0, 255))  # Красный прямоугольник для правой руки

            point_12 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * w),
                        int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * h))
            point_14 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x * w),
                       int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y * h))
            point_16 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x * w),
                       int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y * h))

            draw_line(frame, point_12, point_14, (255, 0, 0))  # Линия от плеча до точки 12 на правой руке
            draw_line(frame, point_12, point_14, (255, 0, 0))  # Линия от точки 12 до точки 14 на правой руке
            draw_line(frame, point_14, point_16, (255, 0, 0))  # Линия от точки 14 до точки 16 на правой руке

        cv2.imshow('Human Detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
