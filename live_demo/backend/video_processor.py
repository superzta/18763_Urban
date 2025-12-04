import cv2
import numpy as np
import os
from pathlib import Path
from .model_wrapper import ModelWrapper

class VideoProcessor:
    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        
    def process_video(self, input_path, output_path):
        """
        Process a video file and save the result.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path} ({total_frames} frames)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            boxes, labels, scores = self.model.predict(frame_rgb)
            
            # Draw detections on frame (BGR)
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"Class {label}: {score:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
                
        cap.release()
        out.release()
        print(f"Video processing complete. Saved to {output_path}")
        return output_path
