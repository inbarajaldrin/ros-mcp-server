#!/usr/bin/env python3
"""
Combined YOLOE (Prompt Set) + Depth Detection Node
Detects objects with YOLOE prompted model and annotates them with depth values
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLOE
import os
from message_filters import ApproximateTimeSynchronizer, Subscriber
import argparse


class YOLOEDepthDetectorNode(Node):
    def __init__(self, prompts=None):
        super().__init__('yoloe_depth_detector_ps')
        
        # Default prompts if none provided
        self.prompts = prompts if prompts else ["red object", "blue object"]
        
        # Initialize YOLOE prompted model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'yoloe-11s-seg.pt')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"YOLOE model not found: {model_path}")
        
        # Change to script directory for model loading
        original_cwd = os.getcwd()
        os.chdir(script_dir)
        
        self.model = YOLOE(model_path)
        self.get_logger().info(f"✅ Loaded YOLOE prompted model from {model_path}")
        
        # Pre-compute text embeddings for all prompts together
        self.get_logger().info("Pre-computing text embeddings...")
        self.text_embeddings = self.model.get_text_pe(self.prompts)
        
        # Set prompts using the combined embeddings
        self.model.set_classes(self.prompts, self.text_embeddings)
        self.get_logger().info(f"✅ Using prompts: {self.prompts}")
        
        # Restore working directory
        os.chdir(original_cwd)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # OpenCV window
        cv2.namedWindow("YOLOE (Prompt Set) + Depth Detection", cv2.WINDOW_AUTOSIZE)
        
        # Create subscribers for RGB and depth images
        self.rgb_sub = Subscriber(self, Image, '/rgb')
        self.depth_sub = Subscriber(self, Image, '/depth')
        # self.depth_sub = Subscriber(self, Image, '/rgb_depth_lego')

        # Synchronize RGB and depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # Publisher for annotated image with depth
        self.annotated_publisher = self.create_publisher(
            Image, 
            '/yoloe_depth_annotated', 
            10
        )
        
        self.get_logger().info("🤖 YOLOE (Prompt Set) + Depth Detector started")
        self.get_logger().info("Subscribing to: /rgb")
        self.get_logger().info("Subscribing to: /rgb_depth_lego")
        self.get_logger().info("Publishing to: /yoloe_depth_annotated")
        self.get_logger().info("Press 'q' to quit")
    
    def synchronized_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB and depth images"""
        try:
            # Convert ROS images to OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Try different depth encodings (Isaac Sim might use different formats)
            try:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            except Exception as e:
                self.get_logger().warn(f"Failed passthrough, trying 16UC1: {e}")
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Debug: Log depth image info once
            if not hasattr(self, '_depth_info_logged'):
                self._depth_info_logged = True
                self.get_logger().info(f"\n{'='*60}")
                self.get_logger().info(f"DEPTH IMAGE INFO:")
                self.get_logger().info(f"  └─ Encoding: {depth_msg.encoding}")
                self.get_logger().info(f"  └─ Shape: {depth_image.shape}")
                self.get_logger().info(f"  └─ Dtype: {depth_image.dtype}")
                self.get_logger().info(f"  └─ Min: {depth_image.min()}")
                self.get_logger().info(f"  └─ Max: {depth_image.max()}")
                self.get_logger().info(f"  └─ Mean: {depth_image.mean():.2f}")
                self.get_logger().info(f"  └─ Non-zero pixels: {np.count_nonzero(depth_image)}")
                self.get_logger().info(f"{'='*60}\n")
            
            # Run YOLOE prompted detection with optimizations
            results = self.model.predict(rgb_image, verbose=False, conf=0.3)
            
            # Get inference time and calculate FPS
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time
            fps_text = f'FPS: {fps:.1f}'
            
            # Use YOLOE's built-in plotting for base annotations
            annotated_image = results[0].plot(boxes=True, masks=False)
            
            # Add depth information on top of YOLOE annotations
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                self.get_logger().info(f"\n{'='*60}")
                self.get_logger().info(f"Frame detections: {len(boxes)}")
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name
                    class_name = self.model.names[int(cls)]
                    
                    # Get depth value at center
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    depth_value = 0.0
                    depth_raw = 0
                    if (0 <= center_x < depth_image.shape[1] and 
                        0 <= center_y < depth_image.shape[0]):
                        depth_raw = depth_image[center_y, center_x]
                        
                        # Handle different depth formats
                        if depth_raw > 0:
                            # Isaac Sim typically uses meters (float32), regular depth uses mm (uint16)
                            if depth_image.dtype == np.float32:
                                depth_value = depth_raw * 100.0  # Convert meters to cm
                            else:
                                depth_value = depth_raw / 10.0  # Convert mm to cm
                    
                    # Print depth info for this detection
                    self.get_logger().info(f"Detection {i+1}: {class_name} (conf: {conf:.2f})")
                    self.get_logger().info(f"  └─ BBox: [{x1}, {y1}, {x2}, {y2}]")
                    self.get_logger().info(f"  └─ Center: ({center_x}, {center_y})")
                    self.get_logger().info(f"  └─ Depth: {depth_value:.1f}cm (raw: {depth_raw})")
                    
                    # Draw depth label below the YOLOE annotation
                    if depth_value > 0:
                        depth_label = f"{depth_value:.1f}cm"
                        
                        # Position depth label at bottom of bounding box
                        label_size = cv2.getTextSize(depth_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Draw background for depth label (yellow)
                        cv2.rectangle(annotated_image, 
                                    (x1, y2), 
                                    (x1 + label_size[0] + 10, y2 + label_size[1] + 10), 
                                    (0, 255, 255), -1)
                        
                        # Draw depth text
                        cv2.putText(annotated_image, depth_label, 
                                  (x1 + 5, y2 + label_size[1] + 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                self.get_logger().info(f"{'='*60}\n")
            
            # Add FPS text to frame (top-right corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
            text_x = annotated_image.shape[1] - text_size[0] - 10  # 10 pixels from the right
            text_y = text_size[1] + 10  # 10 pixels from the top
            cv2.putText(annotated_image, fps_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add prompts text to frame (top-left corner)
            prompts_text = f"Prompts: {', '.join(self.prompts)}"
            prompts_size = cv2.getTextSize(prompts_text, font, 0.6, 1)[0]
            cv2.putText(annotated_image, prompts_text, (10, prompts_size[1] + 10), 
                       font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Publish annotated image
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                annotated_msg.header = rgb_msg.header
                self.annotated_publisher.publish(annotated_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish annotated image: {str(e)}")
            
            # Display image
            cv2.imshow("YOLOE (Prompt Set) + Depth Detection", annotated_image)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("🛑 'q' pressed, shutting down...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"❌ Error processing synchronized images: {str(e)}")


def main(args=None):
    # Parse command-line arguments BEFORE rclpy.init
    parser = argparse.ArgumentParser(description='YOLOE Prompt Set + Depth Detector')
    parser.add_argument('--prompts', nargs='+', 
                       help='List of prompts to detect (e.g., --prompts "red object" "blue object")', 
                       default=None)
    
    # Only parse known args to avoid conflicts with ROS args
    parsed_args, unknown = parser.parse_known_args()
    
    # Get prompts from command line or use defaults
    prompts_to_use = parsed_args.prompts if parsed_args.prompts else None
    
    # Initialize ROS with remaining args
    rclpy.init(args=unknown if unknown else args)
    
    try:
        node = YOLOEDepthDetectorNode(prompts=prompts_to_use)
        node.get_logger().info("🚀 Starting YOLOE (Prompt Set) + Depth detection...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down YOLOE + Depth Detector Node...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

