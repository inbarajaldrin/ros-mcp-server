#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
from datetime import datetime

class SimplePhotoCapture(Node):
    def __init__(self):
        super().__init__('simple_photo_capture')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Processing control
        self.image_captured = False
        
        # Create screenshots directory
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Subscribe to camera topic
        self.image_subscription = self.create_subscription(
            Image, 
            '/intel_camera_rgb_raw',  # Adjust this topic name as needed
            self.image_callback, 
            10
        )
        
        self.get_logger().info("Simple Photo Capture started")
        self.get_logger().info("Waiting for image from camera topic...")
    
    def image_callback(self, msg):
        """Process incoming image and capture annotated photo"""
        if self.image_captured:
            return
            
        self.get_logger().info("Capturing image for annotation...")
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f"Image captured: {cv_image.shape}")
            
            # Save the original image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = f"original_photo_{timestamp}.jpg"
            original_output_path = os.path.join(self.screenshots_dir, original_filename)
            cv2.imwrite(original_output_path, cv_image)
            self.get_logger().info(f"Saved original image to {original_output_path}")
            
            # Run detection and annotation
            annotated_image = self.detect_and_annotate(cv_image)
            
            # Save the annotated image with timestamp
            annotated_filename = f"annotated_photo_{timestamp}.jpg"
            annotated_output_path = os.path.join(self.screenshots_dir, annotated_filename)
            cv2.imwrite(annotated_output_path, annotated_image)
            self.get_logger().info(f"Saved annotated image to {annotated_output_path}")
            
            self.image_captured = True
            
            # Shutdown after capturing
            self.get_logger().info("Photo capture completed! Shutting down...")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            self.image_captured = True  # Set flag to exit
    
    def detect_and_annotate(self, image):
        """Detect objects and annotate the image using prompt-free model"""
        try:
            # Use the prompt-free model in the same directory
            model_path = os.path.join(os.path.dirname(__file__), 'yoloe-11s-seg-pf.pt')
            
            if not os.path.exists(model_path):
                self.get_logger().warn(f"YOLOE prompt-free model not found at {model_path}")
                return image
            
            try:
                from ultralytics import YOLO
                self.get_logger().info("YOLO imported successfully")
            except ImportError:
                self.get_logger().warn("YOLO not available in ultralytics")
                return image
            
            # Load YOLOE prompt-free model
            model = YOLO(model_path)
            self.get_logger().info(f"Loaded YOLOE prompt-free model")
            
            # Run prediction (no prompts needed for prompt-free model)
            results = model(image, verbose=False, conf=0.45)
            
            # Check results and annotate
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                detections = len(results[0].boxes)
                self.get_logger().info(f"Found {detections} detections!")
                
                for j, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if cls in model.names else f"class_{cls}"
                    self.get_logger().info(f"  {class_name} (conf: {conf:.3f})")
                
                # Get annotated image
                annotated = results[0].plot()
                return annotated
            else:
                self.get_logger().info("No detections found")
                return image
                
        except Exception as e:
            self.get_logger().error(f"Detection failed: {str(e)}")
            return image

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimplePhotoCapture()
        node.get_logger().info("Starting Simple Photo Capture...")
        
        # Spin until image is captured
        while rclpy.ok() and not node.image_captured:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        node.get_logger().info("Exiting...")
        
    except KeyboardInterrupt:
        print("\nShutting down Simple Photo Capture...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
