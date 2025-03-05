import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading

from mainMediapipe_samples import FaceMeshDetector


# Shared camera capture object to be used by both classes
shared_cap = cv2.VideoCapture(0)
original_init = FaceMeshDetector.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if hasattr(self, 'cap'):
        try:
            self.cap.release()
        except Exception:
            pass
    self.cap = shared_cap
FaceMeshDetector.__init__ = patched_init


class DrowsinessDetector:
    def __init__(self):
        self.face_mesh_detector = FaceMeshDetector()

        self.count = 0
        self.alarm_threshold = 30  # frames
        self.alarm_playing = False  # flag to check if alarm has been played

        self.cap = shared_cap

        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm2.mp3")
        self.alarm_thread = threading.Thread(target=self.alarm_loop)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()

    def calc_ear(self, eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def alarm_loop(self):
        while True:
            if self.alarm_playing:
                if not pygame.mixer.get_busy():
                    self.alarm_sound.play(loops=-1)
            else:
                self.alarm_sound.stop()

    def process_frame(self, image):
        results = self.face_mesh_detector.face_mesh.process(image)
        eye_keypoints = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_keypoints = self.face_mesh_detector.get_eye_mouth_keypoints(face_landmarks, image.shape)
                left_eye = eye_keypoints["left_eye"]
                right_eye = eye_keypoints["right_eye"]

                left_ear = self.calc_ear(left_eye)
                right_ear = self.calc_ear(right_eye)

                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < 0.25:
                    self.count += 1
                else:
                    self.count = 0
                    self.alarm_playing = False

                for (x, y) in left_eye + right_eye:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                if self.count >= self.alarm_threshold:
                    cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    self.alarm_playing = True

        return image

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            processed_image = self.process_frame(image)
            cv2.imshow("Drowsiness Detection", processed_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()