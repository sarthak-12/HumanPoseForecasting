import cv2
import mediapipe as mp
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import time


class MediaPipePoseDetector:
    """
    Real-time pose detection using MediaPipe Pose.
    
    This class provides:
    - Real-time pose detection from webcam/video
    - 33 pose landmarks extraction
    - Conversion to 17 keypoint format for forecasting
    - Smoothing and filtering for better stability
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose detector
        
        Args:
            static_image_mode: Whether to treat input as static image
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
            enable_segmentation: Whether to enable segmentation
            smooth_segmentation: Whether to smooth segmentation
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # COCO format keypoint mapping (17 keypoints)
        self.coco_keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # MediaPipe to COCO mapping
        self.mp_to_coco = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 1, 'left_eye_outer': 1,
            'right_eye_inner': 2, 'right_eye': 2, 'right_eye_outer': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # Smoothing parameters
        self.smoothing_window = 5
        self.pose_history = []
        
    def detect_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose in the given image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Pose keypoints in COCO format (17 keypoints, 2 coordinates each)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Convert to COCO format
            coco_pose = self._convert_to_coco_format(landmarks, image.shape)
            
            # Apply smoothing
            coco_pose = self._apply_smoothing(coco_pose)
            
            return coco_pose
        
        return None
    
    def _convert_to_coco_format(self, landmarks: List, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert MediaPipe landmarks to COCO format
        
        Args:
            landmarks: MediaPipe landmarks
            image_shape: Image shape (height, width, channels)
            
        Returns:
            COCO format keypoints (17, 2)
        """
        h, w, _ = image_shape
        coco_pose = np.zeros((17, 2))
        
        # Map MediaPipe landmarks to COCO format
        for mp_idx, landmark in enumerate(landmarks):
            mp_name = self.mp_pose.PoseLandmark(mp_idx).name
            
            if mp_name in self.mp_to_coco:
                coco_idx = self.mp_to_coco[mp_name]
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * w
                y = landmark.y * h
                
                # Use the first occurrence of each COCO keypoint
                if coco_pose[coco_idx, 0] == 0 and coco_pose[coco_idx, 1] == 0:
                    coco_pose[coco_idx] = [x, y]
        
        return coco_pose
    
    def _apply_smoothing(self, pose: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to pose keypoints
        
        Args:
            pose: Current pose keypoints
            
        Returns:
            Smoothed pose keypoints
        """
        self.pose_history.append(pose)
        
        # Keep only the last N frames
        if len(self.pose_history) > self.smoothing_window:
            self.pose_history.pop(0)
        
        # Apply exponential moving average
        if len(self.pose_history) > 1:
            alpha = 0.7  # Smoothing factor
            smoothed_pose = pose.copy()
            
            for i in range(len(self.pose_history) - 1):
                weight = alpha * (1 - alpha) ** i
                smoothed_pose += weight * self.pose_history[-(i + 2)]
            
            return smoothed_pose
        
        return pose
    
    def draw_pose(self, image: np.ndarray, pose: np.ndarray, 
                  draw_connections: bool = True, 
                  draw_keypoints: bool = True) -> np.ndarray:
        """
        Draw pose keypoints and connections on the image
        
        Args:
            image: Input image
            pose: Pose keypoints in COCO format
            draw_connections: Whether to draw connections between keypoints
            draw_keypoints: Whether to draw keypoints
            
        Returns:
            Image with pose visualization
        """
        image_copy = image.copy()
        
        if draw_keypoints:
            for i, (x, y) in enumerate(pose):
                if x > 0 and y > 0:  # Valid keypoint
                    cv2.circle(image_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(image_copy, str(i), (int(x) + 10, int(y) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if draw_connections:
            # Define connections (COCO format)
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12),  # Torso
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            for connection in connections:
                start_idx, end_idx = connection
                start_point = pose[start_idx]
                end_point = pose[end_idx]
                
                if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                    cv2.line(image_copy, 
                            (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])),
                            (255, 0, 0), 2)
        
        return image_copy
    
    def get_pose_sequence(self, num_frames: int = 10) -> Optional[torch.Tensor]:
        """
        Get the current pose sequence for forecasting
        
        Args:
            num_frames: Number of frames to return
            
        Returns:
            Pose sequence tensor (num_frames, 34) or None if insufficient data
        """
        if len(self.pose_history) >= num_frames:
            # Convert to tensor format (num_frames, 34)
            sequence = []
            for pose in self.pose_history[-num_frames:]:
                # Flatten 17 keypoints * 2 coordinates = 34
                sequence.append(pose.flatten())
            
            return torch.tensor(sequence, dtype=torch.float32)
        
        return None
    
    def reset_history(self):
        """Reset pose history"""
        self.pose_history.clear()
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


class PoseForecastingPipeline:
    """
    Complete pipeline for real-time pose detection and forecasting
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pose forecasting pipeline
        
        Args:
            model_path: Path to trained forecasting model
        """
        self.detector = MediaPipePoseDetector()
        self.forecasting_model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the forecasting model"""
        try:
            from models.modern_pose_forecaster import ModernPoseForecaster
            self.forecasting_model = ModernPoseForecaster.load_from_checkpoint(model_path)
            self.forecasting_model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.forecasting_model = None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for pose detection and forecasting
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing detection results and predictions
        """
        results = {
            'pose_detected': False,
            'current_pose': None,
            'predicted_poses': None,
            'visualization': frame.copy()
        }
        
        # Detect pose
        pose = self.detector.detect_pose(frame)
        
        if pose is not None:
            results['pose_detected'] = True
            results['current_pose'] = pose
            
            # Get pose sequence for forecasting
            pose_sequence = self.detector.get_pose_sequence(num_frames=10)
            
            if pose_sequence is not None and self.forecasting_model is not None:
                # Make prediction
                with torch.no_grad():
                    # Add batch dimension
                    input_seq = pose_sequence.unsqueeze(0)
                    predictions = self.forecasting_model.predict_sequence(input_seq, num_frames=10)
                    results['predicted_poses'] = predictions.squeeze().numpy()
            
            # Draw visualization
            results['visualization'] = self.detector.draw_pose(frame, pose)
        
        return results
    
    def run_realtime(self, camera_id: int = 0, max_frames: int = None):
        """
        Run real-time pose detection and forecasting
        
        Args:
            camera_id: Camera device ID
            max_frames: Maximum number of frames to process (None for infinite)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Display results
                cv2.imshow('Pose Detection & Forecasting', results['visualization'])
                
                # Print FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check max frames
                if max_frames and frame_count >= max_frames:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'detector'):
            del self.detector 