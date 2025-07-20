import base64
import json
from typing import Optional, Dict, Any
from pathlib import Path
from typing import Protocol
from datetime import datetime
import numpy as np
import cv2

class Subscriber(Protocol):
    def receive_binary(self) -> bytes:
        ...
    
    def send(self, message: dict) -> None:
        ...

class Image:
    def __init__(self, subscriber: Subscriber, topic: str = "/camera/image_raw"):
        self.subscriber = subscriber
        self.topic = topic

    def subscribe(self, save_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Subscribe to image topic and return data in format compatible with MCP server.
        Returns dict with 'data' key containing RGB numpy array.
        """
        try:
            subscribe_msg = {
                "op": "subscribe",
                "topic": self.topic,
                "type": "sensor_msgs/Image"
            }
            self.subscriber.send(subscribe_msg)
            
            raw = self.subscriber.receive_binary()
            if not raw:
                print(f"[Image] No data received from subscriber for topic {self.topic}")
                return None

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            
            msg = json.loads(raw)
            msg = msg["msg"]

            # Extract metadata
            height = msg["height"]
            width = msg["width"]
            encoding = msg["encoding"]
            data_b64 = msg["data"]

            # Decode base64 to raw bytes
            image_bytes = base64.b64decode(data_b64)
            img_np = np.frombuffer(image_bytes, dtype=np.uint8)

            # Handle different encodings and convert to RGB for MCP
            if encoding == "rgb8":
                img_rgb = img_np.reshape((height, width, 3))
                img_cv_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # For saving
            elif encoding == "bgr8":
                img_bgr = img_np.reshape((height, width, 3))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for MCP
                img_cv_bgr = img_bgr  # Already BGR for saving
            elif encoding == "mono8":
                img_gray = img_np.reshape((height, width))
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)  # Convert to RGB for MCP
                img_cv_bgr = img_gray  # Keep gray for saving
            else:
                print(f"[Image] Unsupported encoding: {encoding}")
                return None

            # Save image to screenshots folder (like your working system)
            if save_path is None:
                downloads_dir = Path(__file__).resolve().parents[2] / "screenshots"
                if not downloads_dir.exists():
                    downloads_dir.mkdir(parents=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = downloads_dir / f"{timestamp}.png"
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_cv_bgr)
            print(f"[Image] Saved to {save_path}")

            # Return in format expected by MCP server
            return {
                "data": img_rgb,  # RGB numpy array for MCP processing
                "metadata": {
                    "topic": self.topic,
                    "height": height,
                    "width": width,
                    "encoding": encoding,
                    "timestamp": datetime.now().isoformat(),
                    "saved_to": str(save_path)
                }
            }

        except Exception as e:
            print(f"[Image] Failed to receive or decode from {self.topic}: {e}")
            return None

    def subscribe_raw(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Legacy method that returns raw OpenCV image (BGR format).
        Kept for backward compatibility.
        """
        result = self.subscribe(save_path)
        if result and 'data' in result:
            # Convert RGB back to BGR for legacy compatibility
            rgb_data = result['data']
            return cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        return None

    def get_latest_screenshot(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest screenshot from the screenshots folder.
        Useful when live camera data isn't available.
        """
        try:
            screenshots_dir = Path(__file__).resolve().parents[2] / "screenshots"
            if not screenshots_dir.exists():
                print("[Image] Screenshots directory does not exist")
                return None
            
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(screenshots_dir.glob(ext))
            
            if not image_files:
                print("[Image] No screenshot files found")
                return None
            
            # Get the most recent file
            latest_file = max(image_files, key=lambda x: x.stat().st_mtime)
            
            # Load the image
            img_bgr = cv2.imread(str(latest_file))
            if img_bgr is None:
                print(f"[Image] Failed to load image from {latest_file}")
                return None
            
            # Convert to RGB for MCP
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            return {
                "data": img_rgb,
                "metadata": {
                    "topic": "screenshot",
                    "height": img_rgb.shape[0],
                    "width": img_rgb.shape[1],
                    "encoding": "rgb8",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": str(latest_file),
                    "file_size": latest_file.stat().st_size
                }
            }
            
        except Exception as e:
            print(f"[Image] Failed to get latest screenshot: {e}")
            return None
            