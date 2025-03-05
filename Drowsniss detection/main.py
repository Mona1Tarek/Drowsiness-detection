import cv2
import mediapipe as mp
import math
from typing import Tuple, Union, Dict, List

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)
KEYPOINT_COLOR = (255, 255, 255)

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_conf)
        self.cap = cv2.VideoCapture(0)
        self.keypoints = {}
    
    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def visualize(self, image, detection_result):
        annotated_image = image.copy()
        height, width, _ = image.shape
        bbox_data = []
        for detection in detection_result.detections:
            bbox = detection.location_data.relative_bounding_box
            start_point = int(bbox.xmin * width), int(bbox.ymin * height)
            end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
            bbox_data.append({
                "start_point": start_point,
                "end_point": end_point,
                "width": bbox.width * width,
                "height": bbox.height * height
            })
        return annotated_image, bbox_data
    
    def get_eye_mouth_keypoints(self, face_landmarks, image_shape) -> Dict[str, List[Tuple[int, int]]]:
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        h, w, _ = image_shape
        
        LEFT_EYE_INDICES = [33, 133, 160, 144, 158, 153]
        RIGHT_EYE_INDICES = [362, 263, 387, 373, 380, 374]
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]
        
        for idx, landmark in enumerate(face_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if idx in LEFT_EYE_INDICES:
                eye_mouth_keypoints["left_eye"].append((cx, cy))
            if idx in RIGHT_EYE_INDICES:
                eye_mouth_keypoints["right_eye"].append((cx, cy))
            if idx in MOUTH_INDICES:
                eye_mouth_keypoints["mouth"].append((cx, cy))
        return eye_mouth_keypoints
    
    def process_frame(self, image):
        detection_results = self.face_detection.process(image)
        self.keypoints = {}
        bbox_data = []
        eye_mouth_keypoints = {}
        
        if detection_results.detections:
            image_with_bbox, bbox_data = self.visualize(image, detection_results)
        else:
            image_with_bbox = image
        
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image_with_bbox,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=image_with_bbox,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec
                )
                eye_mouth_keypoints = self.get_eye_mouth_keypoints(face_landmarks, image.shape)
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    self.keypoints[idx] = (cx, cy)
                    cv2.circle(image_with_bbox, (cx, cy), 1, KEYPOINT_COLOR, -1)
                    cv2.putText(image_with_bbox, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, KEYPOINT_COLOR, 1, cv2.LINE_AA)
        print("Detected Keypoints:", self.keypoints)
        return image_with_bbox, bbox_data, eye_mouth_keypoints
    
    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break
            image_with_bbox, bbox_data, eye_mouth_keypoints = self.process_frame(image)
            cv2.imshow("Face Mesh Detection", cv2.flip(image_with_bbox, 1))
            
            print("Bounding Box Data:", bbox_data)
            print("Eye and Mouth Keypoints:", eye_mouth_keypoints)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceMeshDetector()
    detector.run()