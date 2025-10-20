#!/usr/bin/env python3
"""
Depth Calibration and Debugging System
Compares Isaac Sim ground truth depth with RGB-to-depth model estimates
and provides calibration parameters to match them.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber
import json
import os
from datetime import datetime
import argparse


class DepthCalibrationDebugger(Node):
    def __init__(self, save_data=False, calibration_mode=False):
        super().__init__('depth_calibration_debugger')
        
        # Configuration
        self.save_data = save_data
        self.calibration_mode = calibration_mode
        self.data_dir = os.path.join(os.path.dirname(__file__), 'calibration_data')
        
        # Create data directory if saving
        if self.save_data:
            os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Data collection
        self.frame_count = 0
        self.depth_comparisons = []
        self.calibration_data = []
        
        # Calibration parameters (will be learned)
        self.scale_factor = 1.0
        self.offset = 0.0
        self.min_depth = 0.1
        self.max_depth = 10.0
        
        # Create subscribers for synchronized data
        self.rgb_sub = Subscriber(self, Image, '/rgb')
        self.isaac_depth_sub = Subscriber(self, Image, '/depth')  # Ground truth from Isaac Sim
        self.rgb_depth_sub = Subscriber(self, Image, '/rgb_depth_lego')  # RGB-to-depth model
        
        # Synchronize all three streams
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.isaac_depth_sub, self.rgb_depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # Publishers for visualization
        self.comparison_pub = self.create_publisher(Image, '/depth_comparison', 10)
        self.calibrated_depth_pub = self.create_publisher(Image, '/calibrated_depth', 10)
        
        # OpenCV windows
        cv2.namedWindow("Depth Comparison", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Calibrated Depth", cv2.WINDOW_AUTOSIZE)
        
        # Load existing calibration if available
        self.load_calibration()
        
        self.get_logger().info("🔧 Depth Calibration Debugger started")
        self.get_logger().info("Subscribing to: /rgb, /depth, /rgb_depth_lego")
        self.get_logger().info("Publishing to: /depth_comparison, /calibrated_depth")
        self.get_logger().info(f"Calibration mode: {self.calibration_mode}")
        self.get_logger().info(f"Save data: {self.save_data}")
        self.get_logger().info("Press 'c' to calibrate, 's' to save data, 'q' to quit")
    
    def load_calibration(self):
        """Load existing calibration parameters"""
        calibration_file = os.path.join(self.data_dir, 'calibration_params.json')
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    params = json.load(f)
                self.scale_factor = params.get('scale_factor', 1.0)
                self.offset = params.get('offset', 0.0)
                self.min_depth = params.get('min_depth', 0.1)
                self.max_depth = params.get('max_depth', 10.0)
                self.get_logger().info(f"✅ Loaded calibration: scale={self.scale_factor:.3f}, offset={self.offset:.3f}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load calibration: {e}")
    
    def save_calibration(self):
        """Save calibration parameters"""
        calibration_file = os.path.join(self.data_dir, 'calibration_params.json')
        params = {
            'scale_factor': self.scale_factor,
            'offset': self.offset,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(calibration_file, 'w') as f:
                json.dump(params, f, indent=2)
            self.get_logger().info(f"💾 Saved calibration parameters to {calibration_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration: {e}")
    
    def synchronized_callback(self, rgb_msg, isaac_depth_msg, rgb_depth_msg):
        """Process synchronized RGB, Isaac depth, and RGB-to-depth images"""
        try:
            # Convert ROS images to OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Convert depth images
            try:
                isaac_depth = self.bridge.imgmsg_to_cv2(isaac_depth_msg, "passthrough")
            except:
                isaac_depth = self.bridge.imgmsg_to_cv2(isaac_depth_msg, "32FC1")
            
            try:
                rgb_depth = self.bridge.imgmsg_to_cv2(rgb_depth_msg, "passthrough")
            except:
                rgb_depth = self.bridge.imgmsg_to_cv2(rgb_depth_msg, "32FC1")
            
            # Ensure both depth images are float32
            if isaac_depth.dtype != np.float32:
                isaac_depth = isaac_depth.astype(np.float32)
            if rgb_depth.dtype != np.float32:
                rgb_depth = rgb_depth.astype(np.float32)
            
            # Apply calibration to RGB-to-depth
            calibrated_depth = self.apply_calibration(rgb_depth)
            
            # Collect data for analysis
            if self.calibration_mode or self.save_data:
                self.collect_calibration_data(isaac_depth, rgb_depth, calibrated_depth)
            
            # Create comparison visualization
            comparison_image = self.create_comparison_visualization(
                rgb_image, isaac_depth, rgb_depth, calibrated_depth
            )
            
            # Publish comparison image
            try:
                comparison_msg = self.bridge.cv2_to_imgmsg(comparison_image, "bgr8")
                comparison_msg.header = rgb_msg.header
                self.comparison_pub.publish(comparison_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish comparison: {e}")
            
            # Publish calibrated depth
            try:
                calibrated_msg = self.bridge.cv2_to_imgmsg(calibrated_depth, "32FC1")
                calibrated_msg.header = rgb_msg.header
                self.calibrated_depth_pub.publish(calibrated_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish calibrated depth: {e}")
            
            # Display images
            cv2.imshow("Depth Comparison", comparison_image)
            cv2.imshow("Calibrated Depth", self.visualize_depth(calibrated_depth))
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("🛑 'q' pressed, shutting down...")
                rclpy.shutdown()
            elif key == ord('c'):
                self.perform_calibration()
            elif key == ord('s'):
                self.save_calibration()
            elif key == ord('r'):
                self.reset_calibration()
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing synchronized images: {str(e)}")
    
    def apply_calibration(self, rgb_depth):
        """Apply calibration parameters to RGB-to-depth model output"""
        # Apply scale and offset
        calibrated = rgb_depth * self.scale_factor + self.offset
        
        # Clamp to reasonable depth range
        calibrated = np.clip(calibrated, self.min_depth, self.max_depth)
        
        return calibrated.astype(np.float32)
    
    def collect_calibration_data(self, isaac_depth, rgb_depth, calibrated_depth):
        """Collect data for calibration analysis"""
        # Sample points for analysis (avoid edges and invalid regions)
        h, w = isaac_depth.shape
        margin = 50
        
        # Create mask for valid depth values
        isaac_mask = (isaac_depth > 0) & (isaac_depth < 100)  # Reasonable depth range
        rgb_mask = (rgb_depth > 0) & (rgb_depth < 100)
        valid_mask = isaac_mask & rgb_mask
        
        # Sample points in center region
        center_h, center_w = h // 2, w // 2
        sample_size = min(1000, np.sum(valid_mask))
        
        if sample_size > 0:
            # Get valid points
            valid_points = np.where(valid_mask)
            if len(valid_points[0]) > sample_size:
                # Randomly sample points
                indices = np.random.choice(len(valid_points[0]), sample_size, replace=False)
                y_coords = valid_points[0][indices]
                x_coords = valid_points[1][indices]
            else:
                y_coords = valid_points[0]
                x_coords = valid_points[1]
            
            # Collect depth values
            isaac_values = isaac_depth[y_coords, x_coords]
            rgb_values = rgb_depth[y_coords, x_coords]
            calibrated_values = calibrated_depth[y_coords, x_coords]
            
            # Store comparison data
            self.depth_comparisons.append({
                'isaac': isaac_values,
                'rgb': rgb_values,
                'calibrated': calibrated_values,
                'frame': self.frame_count
            })
            
            # Keep only recent data (last 100 frames)
            if len(self.depth_comparisons) > 100:
                self.depth_comparisons.pop(0)
    
    def perform_calibration(self):
        """Perform calibration using collected data"""
        if len(self.depth_comparisons) < 10:
            self.get_logger().warn("Not enough data for calibration. Need at least 10 frames.")
            return
        
        self.get_logger().info("🔧 Performing calibration...")
        
        # Collect all depth values
        all_isaac = []
        all_rgb = []
        
        for comparison in self.depth_comparisons:
            all_isaac.extend(comparison['isaac'])
            all_rgb.extend(comparison['rgb'])
        
        all_isaac = np.array(all_isaac)
        all_rgb = np.array(all_rgb)
        
        # Remove outliers
        isaac_q1, isaac_q3 = np.percentile(all_isaac, [25, 75])
        rgb_q1, rgb_q3 = np.percentile(all_rgb, [25, 75])
        
        isaac_iqr = isaac_q3 - isaac_q1
        rgb_iqr = rgb_q3 - rgb_q1
        
        isaac_mask = (all_isaac >= isaac_q1 - 1.5 * isaac_iqr) & (all_isaac <= isaac_q3 + 1.5 * isaac_iqr)
        rgb_mask = (all_rgb >= rgb_q1 - 1.5 * rgb_iqr) & (all_rgb <= rgb_q3 + 1.5 * rgb_iqr)
        
        valid_mask = isaac_mask & rgb_mask
        
        if np.sum(valid_mask) < 100:
            self.get_logger().warn("Not enough valid data points after outlier removal.")
            return
        
        isaac_clean = all_isaac[valid_mask]
        rgb_clean = all_rgb[valid_mask]
        
        # Linear regression: isaac = scale * rgb + offset
        # Using least squares: A * [scale, offset] = isaac
        A = np.column_stack([rgb_clean, np.ones(len(rgb_clean))])
        params, residuals, rank, s = np.linalg.lstsq(A, isaac_clean, rcond=None)
        
        self.scale_factor = params[0]
        self.offset = params[1]
        
        # Calculate statistics
        predicted = self.scale_factor * rgb_clean + self.offset
        mae = np.mean(np.abs(predicted - isaac_clean))
        rmse = np.sqrt(np.mean((predicted - isaac_clean) ** 2))
        r2 = 1 - np.sum((isaac_clean - predicted) ** 2) / np.sum((isaac_clean - np.mean(isaac_clean)) ** 2)
        
        self.get_logger().info(f"✅ Calibration complete!")
        self.get_logger().info(f"  └─ Scale factor: {self.scale_factor:.4f}")
        self.get_logger().info(f"  └─ Offset: {self.offset:.4f}")
        self.get_logger().info(f"  └─ MAE: {mae:.4f}m")
        self.get_logger().info(f"  └─ RMSE: {rmse:.4f}m")
        self.get_logger().info(f"  └─ R²: {r2:.4f}")
        
        # Save calibration
        self.save_calibration()
        
        # Clear old data
        self.depth_comparisons.clear()
    
    def reset_calibration(self):
        """Reset calibration to default values"""
        self.scale_factor = 1.0
        self.offset = 0.0
        self.depth_comparisons.clear()
        self.get_logger().info("🔄 Calibration reset to defaults")
    
    def create_comparison_visualization(self, rgb_image, isaac_depth, rgb_depth, calibrated_depth):
        """Create side-by-side comparison visualization"""
        h, w = rgb_image.shape[:2]
        
        # Resize images to fit in comparison
        display_h = h // 2
        display_w = w // 2
        
        # Resize all images
        rgb_small = cv2.resize(rgb_image, (display_w, display_h))
        isaac_viz = cv2.resize(self.visualize_depth(isaac_depth), (display_w, display_h))
        rgb_viz = cv2.resize(self.visualize_depth(rgb_depth), (display_w, display_h))
        calibrated_viz = cv2.resize(self.visualize_depth(calibrated_depth), (display_w, display_h))
        
        # Create comparison grid
        top_row = np.hstack([rgb_small, isaac_viz])
        bottom_row = np.hstack([rgb_viz, calibrated_viz])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        labels = [
            ("RGB", (10, 30)),
            ("Isaac Sim Depth", (display_w + 10, 30)),
            ("RGB-to-Depth", (10, display_h + 30)),
            ("Calibrated Depth", (display_w + 10, display_h + 30))
        ]
        
        for label, (x, y) in labels:
            cv2.putText(comparison, label, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Add calibration info
        calib_text = f"Scale: {self.scale_factor:.3f}, Offset: {self.offset:.3f}"
        cv2.putText(comparison, calib_text, (10, comparison.shape[0] - 20), 
                   font, 0.5, (0, 255, 255), 1)
        
        return comparison
    
    def visualize_depth(self, depth_image):
        """Convert depth image to color visualization"""
        # Normalize depth to 0-255 range
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def save_analysis_data(self):
        """Save collected data for offline analysis"""
        if not self.save_data or len(self.depth_comparisons) == 0:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw comparison data
        data_file = os.path.join(self.data_dir, f'depth_comparison_{timestamp}.json')
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = []
            for comparison in self.depth_comparisons:
                json_data.append({
                    'isaac': comparison['isaac'].tolist(),
                    'rgb': comparison['rgb'].tolist(),
                    'calibrated': comparison['calibrated'].tolist(),
                    'frame': comparison['frame']
                })
            
            with open(data_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.get_logger().info(f"💾 Saved analysis data to {data_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save analysis data: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(description='Depth Calibration Debugger')
    parser.add_argument('--save-data', action='store_true', 
                       help='Save calibration data for offline analysis')
    parser.add_argument('--calibration-mode', action='store_true',
                       help='Start in calibration mode (collects more data)')
    
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=unknown if unknown else args)
    
    try:
        node = DepthCalibrationDebugger(
            save_data=parsed_args.save_data,
            calibration_mode=parsed_args.calibration_mode
        )
        node.get_logger().info("🚀 Starting Depth Calibration Debugger...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Depth Calibration Debugger...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.save_analysis_data()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
