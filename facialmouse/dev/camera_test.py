import os

import cv2
import dlib


class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces

    def get_landmarks(self, image, faces):
        landmarks = []
        for face in faces:
            shape = self.predictor(image, face)
            landmarks.append([(point.x, point.y) for point in shape.parts()])
        return landmarks

    def draw_faces(self, image, faces):
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return image

    def draw_landmarks(self, image, landmarks):
        for face_landmarks in landmarks:
            for x, y in face_landmarks:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        return image


class CameraHandler:
    def __init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        detector_path = self.look_for_detector_path()
        self.face_detector = FaceDetector(detector_path)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

    def __del__(self):
        # Release the camera resources
        self.cap.release()
        cv2.destroyAllWindows()

    def look_for_detector_path(self):
        detector_path = None
        # get directory of this file
        this_code_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(this_code_dir)
        # get this_code_dir/../model/shape_predictor_68_face_landmarks.dat
        detector_path = os.path.join(
            upper_dir, "model", "shape_predictor_68_face_landmarks.dat"
        )
        return detector_path

    def run(self):
        while True:
            frame = self.capture_frame()
            if frame is None:
                break

            # faces = self.detect_face(frame)
            # processed_frame = self.extract_facial_features(frame, faces)
            faces = self.face_detector.detect_faces(frame)
            processed_frame = self.face_detector.draw_faces(frame, faces)
            landmarks = self.face_detector.get_landmarks(frame, faces)
            processed_frame = self.face_detector.draw_landmarks(
                processed_frame, landmarks
            )

            # self.face_detector.get_landmarks(frame, faces)
            self.display_frame(processed_frame)

            if cv2.waitKey(1) == ord("q"):  # Exit on pressing 'q'
                break

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def detect_face(self, frame):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def extract_facial_features(self, frame, faces):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def display_frame(self, frame):
        cv2.imshow("Frame", frame)


# Usage
if __name__ == "__main__":
    camera_handler = CameraHandler()
    camera_handler.run()
