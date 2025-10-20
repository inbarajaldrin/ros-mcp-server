#!/usr/bin/env python3
"""
Calibrated RGB-to-Depth Converter
Uses learned calibration parameters to match Isaac Sim depth values
"""

import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import json
import os


class CalibratedDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('calibrated_depth_estimation_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load depth model
        self.get_logger().info("Loading depth model...")
        self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model.eval()
        self.get_logger().info("Depth model loaded!")
        
        # Load calibration parameters
        self.calibration_params = self.load_calibration_params()
        
        # ROS2 subscribers and publishers
        self.image_sub = self.create_subscription(ROSImage, '/rgb', self.image_callback, 10)
        self.depth_pub = self.create_publisher(ROSImage, '/calibrated_rgb_depth_lego', 10)
        
        self.get_logger().info("Calibrated depth estimation node ready")
        self.get_logger().info(f"Using calibration: scale={self.calibration_params['scale_factor']:.3f}, "
                              f"offset={self.calibration_params['offset']:.3f}")
    
    def load_calibration_params(self):
        """Load calibration parameters from file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calibration_file = os.path.join(script_dir, 'calibration_data', 'calibration_params.json')
        
        default_params = {
            'scale_factor': 1.0,
            'offset': 0.0,
            'min_depth': 0.1,
            'max_depth': 10.0
        }
        
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    params = json.load(f)
                self.get_logger().info(f"✅ Loaded calibration parameters from {calibration_file}")
                return params
            except Exception as e:
                self.get_logger().warn(f"Failed to load calibration parameters: {e}")
                self.get_logger().info("Using default calibration parameters")
        else:
            self.get_logger().warn(f"Calibration file not found: {calibration_file}")
            self.get_logger().info("Using default calibration parameters")
        
        return default_params
    
    def image_callback(self, msg):
        """Process RGB image and publish calibrated depth"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Generate depth map
            inputs = self.depth_processor(images=pil_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            
            depth = prediction.squeeze().cpu().numpy()
            
            # Apply calibration
            calibrated_depth = self.apply_calibration(depth)
            
            # Create depth message in 32FC1 format (meters, like Isaac Sim)
            depth_msg = self.bridge.cv2_to_imgmsg(calibrated_depth, "32FC1")
            depth_msg.header = msg.header
            
            # Publish calibrated depth
            self.depth_pub.publish(depth_msg)
            self.get_logger().info("Published calibrated depth image")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def apply_calibration(self, raw_depth):
        """Apply calibration parameters to raw depth prediction"""
        # Normalize depth to reasonable range (0.1m to 10m) - same as original
        depth_normalized = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
        depth_meters = depth_normalized * 9.9 + 0.1  # Scale to 0.1-10 meters
        
        # Apply learned calibration parameters
        scale_factor = self.calibration_params['scale_factor']
        offset = self.calibration_params['offset']
        min_depth = self.calibration_params['min_depth']
        max_depth = self.calibration_params['max_depth']
        
        # Apply scale and offset: calibrated = scale * depth + offset
        calibrated_depth = depth_meters * scale_factor + offset
        
        # Clamp to reasonable depth range
        calibrated_depth = np.clip(calibrated_depth, min_depth, max_depth)
        
        # Keep as float32 in meters (same format as Isaac Sim)
        return calibrated_depth.astype(np.float32)


def main():
    rclpy.init()
    node = CalibratedDepthEstimationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
