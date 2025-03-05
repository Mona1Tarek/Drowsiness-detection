import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading

class DrowsinessDetector:
    LEFT_EYE_INDICES = [133, 160, 158, 33, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 291, 39, 181, 17, 405]

    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.cap = cv2.VideoCapture(0)
        self.drowsy_frame_count = 0
        self.alarm_threshold = 30
        self.alarm_playing = False  # Track if the alarm is currently playing

        # Initialize pygame mixer for sound playback
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm2.mp3")

    def get_eye_mouth_keypoints(self, face_landmarks, image_shape):
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        h, w, _ = image_shape


        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["left_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["right_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        for idx in self.MOUTH_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["mouth"].append((int(landmark.x * w), int(landmark.y * h)))

        return eye_mouth_keypoints

    def calculate_ear(self, eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def play_alarm(self):
        """Play the alarm sound in a separate thread."""
        if not self.alarm_playing:
            self.alarm_playing = True
            self.alarm_sound.play(loops=-1)  # Loop indefinitely

    def stop_alarm(self):
        """Stop the alarm when the eyes are opened."""
        if self.alarm_playing:
            self.alarm_playing = False
            self.alarm_sound.stop()

    def process_frame(self, image):
        results = self.face_mesh.process(image)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_mouth_keypoints = self.get_eye_mouth_keypoints(face_landmarks, image.shape)
                left_eye = eye_mouth_keypoints["left_eye"]
                right_eye = eye_mouth_keypoints["right_eye"]
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Dynamically adjust the EAR threshold based on head movement
                left_eye_z = [face_landmarks.landmark[idx].z for idx in self.LEFT_EYE_INDICES]
                right_eye_z = [face_landmarks.landmark[idx].z for idx in self.RIGHT_EYE_INDICES]
                left_eye_z_diff = np.max(left_eye_z) - np.min(left_eye_z)
                right_eye_z_diff = np.max(right_eye_z) - np.min(right_eye_z)

                # If significant head movement is detected, adjust the EAR threshold
                if left_eye_z_diff > 0.1 or right_eye_z_diff > 0.1:
                    ear_threshold = 0.35  # Increase the threshold when head movement is detected
                else:
                    ear_threshold = 0.25  # Normal threshold for steady head position

                if avg_ear < ear_threshold:
                    self.drowsy_frame_count += 1
                else:
                    self.drowsy_frame_count = 0  # Reset if eyes are open
                    self.stop_alarm()  # Stop alarm when eyes open
                
                for (x, y) in left_eye + right_eye:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                
                if self.drowsy_frame_count >= self.alarm_threshold:
                    cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    threading.Thread(target=self.play_alarm).start()  # Play alarm in a separate thread
        
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
        self.stop_alarm()  # Ensure the alarm stops when the program closes

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
