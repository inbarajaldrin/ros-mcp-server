#!/usr/bin/env python3
"""
YOLOE Prompted Model Tester

This script tests multiple text prompts with a YOLOE prompted model:
1. Captures a single image from the camera
2. Tests all configured prompts and saves results
3. Identifies the best-performing prompt
4. Exits

Configuration: Edit PROMPTS_TO_TEST below to customize the prompts to test.
"""

import os
import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse

# ============================================================
# CONFIGURE YOUR PROMPTS HERE
# ============================================================
DEFAULT_PROMPTS = [
    "red object",
    "blue object"
]

# Configuration
IMAGE_TOPIC = '/rgb_lego'
MODEL_NAME = 'yoloe-11s-seg.pt'
CONFIDENCE_THRESHOLD = 0.3
# ============================================================


class PromptTester(Node):
    """ROS2 node for testing YOLOE prompts"""
    
    def __init__(self, prompts_to_test=None):
        super().__init__('prompt_tester')
        
        # State management
        self.image_captured = False
        self.test_image = None
        
        # Model and prompt data
        self.model = None
        self.prompts = prompts_to_test if prompts_to_test else DEFAULT_PROMPTS
        self.text_embeddings = {}
        self.test_results = {}
        self.best_prompt = None
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera topic
        self.image_subscription = self.create_subscription(
            Image, 
            IMAGE_TOPIC,
            self.image_callback, 
            10
        )
        
        self.get_logger().info("Prompt Tester started")
        self.get_logger().info(f"Will test {len(self.prompts)} prompts: {self.prompts}")
        self.get_logger().info("Capturing test image...")
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        if self.image_captured:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.test_image = cv_image.copy()
            self.image_captured = True
            self.get_logger().info("Image captured, starting prompt tests...")
            self.run_prompt_tests()
                
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {str(e)}")
    
    def load_model(self):
        """Load YOLOE model and pre-compute text embeddings"""
        try:
            from ultralytics import YOLOE
            
            # Change to script directory to keep model downloads local
            script_dir = os.path.dirname(os.path.abspath(__file__))
            original_cwd = os.getcwd()
            os.chdir(script_dir)
            
            # Load model
            model_path = os.path.join(script_dir, MODEL_NAME)
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model not found at {model_path}")
                os.chdir(original_cwd)
                return False
            
            self.model = YOLOE(model_path)
            self.get_logger().info(f"Loaded YOLOE model: {MODEL_NAME}\n")
            
            # Pre-compute text embeddings to avoid re-downloading mobileclip
            self.get_logger().info("Pre-computing text embeddings...")
            for prompt in self.prompts:
                self.text_embeddings[prompt] = self.model.get_text_pe([prompt])
            
            # Restore working directory
            os.chdir(original_cwd)
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False
    
    def run_prompt_tests(self):
        """Test all prompts and save results"""
        if not self.load_model():
            rclpy.shutdown()
            return
        
        # Create screenshots directory
        screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Test each prompt
        for i, prompt in enumerate(self.prompts):
            self.get_logger().info(f"Testing prompt {i+1}/{len(self.prompts)}: '{prompt}'")
            
            try:
                # Set prompt using pre-computed embeddings
                self.model.set_classes([prompt], self.text_embeddings[prompt])
                
                # Run detection
                results = self.model.predict(
                    self.test_image, 
                    verbose=False, 
                    conf=CONFIDENCE_THRESHOLD
                )
                
                # Count and log detections
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                if num_detections > 0:
                    self.get_logger().info(f"  Found {num_detections} detections!")
                    
                    # Save annotated result
                    annotated = results[0].plot()
                    output_path = os.path.join(
                        screenshots_dir, 
                        f'prompt_{i+1}_{prompt.replace(" ", "_")}.jpg'
                    )
                    cv2.imwrite(output_path, annotated)
                    
                    self.test_results[prompt] = {
                        'detections': num_detections,
                        'image_path': output_path
                    }
                else:
                    self.get_logger().info(f"  No detections")
                    self.test_results[prompt] = {'detections': 0}
                    
            except Exception as e:
                self.get_logger().error(f"  Error testing prompt: {e}")
                self.test_results[prompt] = {'detections': 0}
        
        # Analyze results and exit
        self.analyze_results_and_exit()
    
    def analyze_results_and_exit(self):
        """Analyze test results"""
        self.print_test_results()
        
        # Find successful prompts
        successful = [
            (prompt, data['detections']) 
            for prompt, data in self.test_results.items() 
            if data['detections'] > 0
        ]
        
        if successful:
            # Sort by detection count (descending)
            successful.sort(key=lambda x: x[1], reverse=True)
            
            # Use the best prompt
            self.best_prompt = successful[0][0]
            best_detections = successful[0][1]
            
            self.get_logger().info(f"\nBEST PROMPT: '{self.best_prompt}' ({best_detections} detections)")
            self.get_logger().info("="*60)
            self.get_logger().info("Prompt testing completed! Shutting down...")
        else:
            self.get_logger().warn("No prompts found any detections!")
            self.get_logger().info("="*60)
            self.get_logger().info("Shutting down...")
    
    def print_test_results(self):
        """Print formatted test results"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("TEST RESULTS")
        self.get_logger().info("="*60)
        
        successful = [
            (prompt, data['detections']) 
            for prompt, data in self.test_results.items() 
            if data['detections'] > 0
        ]
        
        if successful:
            self.get_logger().info("SUCCESSFUL PROMPTS:")
            for prompt, detections in successful:
                self.get_logger().info(f"  - {prompt}: {detections} detections")
        else:
            self.get_logger().info("No successful prompts found")


def main(args=None):
    """Main function"""
    # Parse command-line arguments BEFORE rclpy.init
    parser = argparse.ArgumentParser(description='Test YOLOE with multiple prompts')
    parser.add_argument('--prompts', nargs='+', help='List of prompts to test', default=None)
    
    # Only parse known args to avoid conflicts with ROS args
    parsed_args, unknown = parser.parse_known_args()
    
    # Get prompts from command line or use defaults
    prompts_to_test = parsed_args.prompts if parsed_args.prompts else None
    
    # Initialize ROS with remaining args
    rclpy.init(args=unknown if unknown else args)
    
    try:
        node = PromptTester(prompts_to_test=prompts_to_test)
        node.get_logger().info("Starting Prompt Tester...")
        
        # Spin until image is captured and testing is done
        while rclpy.ok() and not node.image_captured:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        node.get_logger().info("Exiting...")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
