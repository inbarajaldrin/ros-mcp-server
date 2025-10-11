#!/usr/bin/env python3
"""
Simple ROS2 Depth Estimation Node
Subscribes to RGB images and publishes depth maps using Depth Anything V2
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


class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load depth model
        self.get_logger().info("Loading depth model...")
        self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model.eval()
        self.get_logger().info("Depth model loaded!")
        
        # ROS2 subscribers and publishers
        self.image_sub = self.create_subscription(ROSImage, '/rgb_lego', self.image_callback, 10)
        self.depth_pub = self.create_publisher(ROSImage, '/rgb_depth_lego', 10)
        
        self.get_logger().info("Depth estimation node ready")
    
    def image_callback(self, msg):
        """Process RGB image and publish depth"""
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
            
            # Normalize depth to reasonable range (0.1m to 10m)
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_meters = depth_normalized * 9.9 + 0.1  # Scale to 0.1-10 meters
            
            # Keep as float32 in meters (same format as Isaac Sim)
            depth_meters_float = depth_meters.astype(np.float32)
            
            # Create depth message in 32FC1 format (meters, like Isaac Sim)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_meters_float, "32FC1")
            depth_msg.header = msg.header
            
            # Publish depth
            self.depth_pub.publish(depth_msg)
            self.get_logger().info("Published depth image")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()
    node = DepthEstimationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

