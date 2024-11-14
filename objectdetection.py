from ultralytics import YOLO
from enum import Enum
import cv2
import numpy as np
import time
import os
from datetime import datetime
import threading
from collections import defaultdict

class InputSource:
    def __init__(self, source):
        self.source = source
        self.is_video = isinstance(source, str) and os.path.isfile(source)
        self.is_webcam = isinstance(source, (int, str)) and not self.is_video

    def get_source(self):
        return int(self.source) if self.is_webcam else self.source

class ModelType(Enum):
    YOLOV8N = 'yolov8n.pt'
    YOLOV8S = 'yolov8s.pt'
    YOLOV8M = 'yolov8m.pt'
    YOLOV8L = 'yolov8l.pt'
    YOLOV8X = 'yolov8x.pt'

class DetectionConfig:
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.45, max_det=50,
                 save_output=False, output_path="output"):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.save_output = save_output
        self.output_path = output_path
        if save_output:
            os.makedirs(output_path, exist_ok=True)

class VideoWriter:
    def __init__(self, output_path, fps, frame_size):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_path, f"detection_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_file, fourcc, fps, frame_size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

class PerformanceMetrics:
    def __init__(self):
        self.frame_times = []
        self.fps_history = []
        self.current_fps = 0

    def update_frame_time(self, frame_time):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        self.current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        self.fps_history.append(self.current_fps)

    def get_average_fps(self):
        return np.mean(self.fps_history[-100:]) if self.fps_history else 0

class AnomalyDetector:
    def __init__(self, speed_threshold=20):
        """
        Initialize anomaly detector.
        :param speed_threshold: Speed in pixels/frame that, if exceeded, flags an anomaly.
        """
        self.previous_positions = defaultdict(lambda: None)
        self.speed_threshold = speed_threshold

    def detect_anomalies(self, detections, frame_count):
        """
        Detect anomalies based on object movement.
        :param detections: List of detected objects with their bounding boxes.
        :param frame_count: Current frame number.
        :return: List of anomalies.
        """
        anomalies = []
        current_positions = {}

        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
            obj_id = f"{class_id}_{i}"  # Unique identifier using class and detection index
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_positions[obj_id] = center

            if self.previous_positions[obj_id] is not None:
                prev_center = self.previous_positions[obj_id]
                distance = np.linalg.norm(np.array(center) - np.array(prev_center))
                if distance > self.speed_threshold:
                    anomalies.append({
                        'frame': frame_count,
                        'id': obj_id,
                        'center': center,
                        'distance': distance
                    })

        self.previous_positions.update(current_positions)
        return anomalies

class ObjectDetectionSystem:
    def __init__(self, model_type, config=DetectionConfig()):
        self.model = YOLO(model_type.value)
        self.config = config
        self.metrics = PerformanceMetrics()
        self.is_running = False
        self.video_writer = None
        self.anomaly_detector = AnomalyDetector(speed_threshold=30)

    def start_detection(self, input_source):
        self.is_running = True
        cap = cv2.VideoCapture(input_source.get_source())
        if not cap.isOpened():
            raise ValueError(f"Failed to open input source: {input_source.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.config.save_output:
            self.video_writer = VideoWriter(self.config.output_path, fps, (frame_width, frame_height))

        frame_count = 0

        try:
            while self.is_running:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    if input_source.is_video:
                        print("Reached end of video file")
                    break

                results = self.model.predict(
                    source=frame,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_det
                )[0]

                detections = [
                    (x1, y1, x2, y2, conf, class_id)
                    for (x1, y1, x2, y2), conf, class_id in zip(
                        results.boxes.xyxy.cpu().numpy(),
                        results.boxes.conf.cpu().numpy(),
                        results.boxes.cls.cpu().numpy()
                    )
                ]

                anomalies = self.anomaly_detector.detect_anomalies(detections, frame_count)
                if anomalies:
                    for anomaly in anomalies:
                        print(f"Anomaly detected in frame {anomaly['frame']}: Object {anomaly['id']} moved {anomaly['distance']:.2f} pixels.")

                processed_frame = self._process_results(frame, results)

                frame_time = time.time() - start_time
                self.metrics.update_frame_time(frame_time)

                if self.video_writer:
                    self.video_writer.write(processed_frame)

                self._display_frame(processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

    def _process_results(self, frame, results):
        processed_frame = frame.copy()
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        fps_text = f"FPS: {self.metrics.get_average_fps():.2f}"
        cv2.putText(processed_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return processed_frame

    def _display_frame(self, frame):
        cv2.imshow('Optimized Object Detection', frame)

    def stop_detection(self):
        self.is_running = False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Object Detection System')
    parser.add_argument('--source', '-s', default='0', help='Input source (video file path or webcam index)')
    parser.add_argument('--model', '-m', choices=[m.name for m in ModelType], default='YOLOV8X', help='YOLO model type')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', '-o', action='store_true', help='Save output video')
    parser.add_argument('--output-path', default='output', help='Output directory for saved videos')

    args = parser.parse_args()
    config = DetectionConfig(
        confidence_threshold=args.conf,
        iou_threshold=0.45,
        max_det=50,
        save_output=args.save,
        output_path=args.output_path
    )

    try:
        input_source = InputSource(args.source)
        detection_system = ObjectDetectionSystem(ModelType[args.model], config)
        detection_thread = threading.Thread(target=detection_system.start_detection, args=(input_source,))
        detection_thread.start()
        input("Press Enter to stop detection...\n")
        detection_system.stop_detection()
        detection_thread.join()

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main()
