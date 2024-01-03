import os
from typing import List, Optional, Tuple

import cv2
import dlib
import numpy as np


class FaceDetector:
    def __init__(self, predictor_path: str) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, image: np.ndarray) -> dlib.rectangles:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces

    def get_landmarks(self, image: np.ndarray, faces: dlib.rectangles) -> List[List[Tuple[int, int]]]:
        landmarks = []
        for face in faces:
            shape = self.predictor(image, face)
            landmarks.append([(point.x, point.y) for point in shape.parts()])
        return landmarks

    def draw_faces(self, image: np.ndarray, faces: dlib.rectangles) -> np.ndarray:
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return image

    def draw_landmarks(self, image: np.ndarray, landmarks: List[List[Tuple[int, int]]]) -> np.ndarray:
        for face_landmarks in landmarks:
            for x, y in face_landmarks:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        return image

    def estimate_head_pose(self, landmarks, camera_matrix, dist_coeffs):
        # Define 3D model points.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

        # Assuming no lens distortion
        image_points = np.array([landmarks[i] for i in [30, 8, 36, 45, 48, 54]], dtype="double")

        # Solve PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        return rotation_vector, translation_vector

    def draw_head_orientation(self, image, landmarks, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
        # Define points to draw the axes
        axis_points = np.float32([[500, 0, 0], [0, 500, 0], [0, 0, 500]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw the axes on the image
        nose_tip = tuple(landmarks[30])
        nose_tip_2d = tuple(int(i) for i in nose_tip)  # Ensure nose_tip coordinates are integer tuple
        imgpts = np.int32(imgpts).reshape(-1, 2)  # Convert imgpts to integer

        image = cv2.line(image, nose_tip_2d, tuple(imgpts[0]), (255, 0, 0), 5)  # X Axis in red
        image = cv2.line(image, nose_tip_2d, tuple(imgpts[1]), (0, 255, 0), 5)  # Y Axis in green
        image = cv2.line(image, nose_tip_2d, tuple(imgpts[2]), (0, 0, 255), 5)  # Z Axis in blue
        # image = cv2.line(image, nose_tip, tuple(imgpts[0].ravel()), (255, 0, 0), 5) # X Axis in red
        # image = cv2.line(image, nose_tip, tuple(imgpts[1].ravel()), (0, 255, 0), 5) # Y Axis in green
        # image = cv2.line(image, nose_tip, tuple(imgpts[2].ravel()), (0, 0, 255), 5) # Z Axis in blue
        return image


def get_dummy_camera_informations(image: np.ndarray):
    height, width = image.shape[:2]
    focal_length = width
    image_center_x = width / 2
    image_center_y = height / 2
    camera_matrix = np.array(
        [[focal_length, 0, image_center_x], [0, focal_length, image_center_y], [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    return camera_matrix, dist_coeffs


class CameraHandler:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        detector_path = self.look_for_detector_path()
        self.face_detector = FaceDetector(detector_path)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

    def __del__(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def look_for_detector_path(self) -> str:
        this_code_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(this_code_dir)
        return os.path.join(upper_dir, "model", "shape_predictor_68_face_landmarks.dat")

    def run(self) -> None:
        while True:
            frame = self.capture_frame()
            if frame is None:
                break

            faces = self.face_detector.detect_faces(frame)
            processed_frame = self.face_detector.draw_faces(frame, faces)
            landmarks = self.face_detector.get_landmarks(frame, faces)
            processed_frame = self.face_detector.draw_landmarks(processed_frame, landmarks)
            camera_matrix, dist_coeffs = get_dummy_camera_informations(frame)
            if len(landmarks) > 0:
                rotation_vector, translation_vector = self.face_detector.estimate_head_pose(
                    landmarks[0], camera_matrix, dist_coeffs
                )
                processed_frame = self.face_detector.draw_head_orientation(
                    processed_frame, landmarks[0], rotation_vector, translation_vector, camera_matrix, dist_coeffs
                )

            self.display_frame(processed_frame)
            if cv2.waitKey(1) == ord("q"):
                break

    def capture_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def display_frame(self, frame: np.ndarray) -> None:
        cv2.imshow("Frame", frame)


if __name__ == "__main__":
    camera_handler = CameraHandler()
    camera_handler.run()
