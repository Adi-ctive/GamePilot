#!/usr/bin/env python3
"""
Forza Traffic Detection & Speed Signs (TDFS) System
Uses only best_tdfs.pt model for traffic light and speed sign detection
Based on the proven detection logic from real.py
"""

import cv2
import numpy as np
import torch
import time
import mss
import win32gui
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
from collections import deque

class ForzaTDFSSystem:
    """Traffic Detection and Speed Signs system for Forza using best_tdfs.pt"""
    
    def __init__(self):
        try:
            print("🚦📏 Initializing Forza TDFS Detection System...")
            
            # Model setup
            self.model_path = "best_tdfs.pt"
            self.model = self._load_model()
            
            # Detection settings
            self.conf_threshold = 0.45
            
            # Screen capture setup
            self.sct = mss.mss()
            self.monitor = self._find_forza_window()
            
            # Detection history for stability
            self.speed_detections = deque(maxlen=10)
            self.traffic_light_detections = deque(maxlen=5)
            
            # Performance tracking
            self.fps_counter = deque(maxlen=30)
            self.last_fps_time = time.time()
            self.frame_count = 0
            
            # Current detections
            self.current_speed_limit = None
            self.current_traffic_light = "Unknown"
            self.last_detection_time = time.time()
            
            print("✅ Forza TDFS System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def _load_model(self):
        """Load the TDFS YOLO model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"📦 Loading TDFS model: {self.model_path}")
            model = YOLO(self.model_path)
            print("✅ TDFS model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load TDFS model: {e}")
            raise
    
    def _find_forza_window(self):
        """Find Forza window automatically"""
        try:
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if any(keyword in window_title.lower() for keyword in ['forza', 'horizon']):
                        rect = win32gui.GetWindowRect(hwnd)
                        windows.append({
                            'title': window_title,
                            'hwnd': hwnd,
                            'rect': rect
                        })
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if windows:
                rect = windows[0]['rect']
                monitor = {
                    "top": rect[1],
                    "left": rect[0], 
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
                print(f"🎮 Found Forza window: {monitor}")
                return monitor
            else:
                print("⚠️  Forza window not found, using full screen")
                return {"top": 0, "left": 0, "width": 1920, "height": 1080}
                
        except Exception as e:
            print(f"❌ Error finding Forza window: {e}")
            return {"top": 0, "left": 0, "width": 1920, "height": 1080}
    
    def detect_traffic_light_color(self, cropped_img):
        """Detect traffic light color using HSV analysis - from real.py"""
        try:
            if cropped_img is None or cropped_img.size == 0:
                return "Unknown"
            
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            
            # Color ranges from real.py
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([179, 255, 255])
            yellow_lower = np.array([15, 100, 100])
            yellow_upper = np.array([35, 255, 255])
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([90, 255, 255])
            
            # Create masks
            red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count pixels
            red_pixels = np.sum(red_mask > 0)
            yellow_pixels = np.sum(yellow_mask > 0)
            green_pixels = np.sum(green_mask > 0)
            
            max_val = max(red_pixels, yellow_pixels, green_pixels)
            
            if max_val < 10:  # Minimum threshold
                return "Unknown"
            elif max_val == red_pixels:
                return "Red"
            elif max_val == yellow_pixels:
                return "Yellow"
            elif max_val == green_pixels:
                return "Green"
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"Traffic light color detection error: {e}")
            return "Unknown"
    
    def extract_speed_from_sign(self, cropped_img):
        """Extract speed value using enhanced OCR - from real.py"""
        try:
            if cropped_img is None or cropped_img.size == 0:
                return None
            
            # Enhanced preprocessing from real.py
            scale_factor = 4
            width = int(cropped_img.shape[1] * scale_factor)
            height = int(cropped_img.shape[0] * scale_factor)
            resized_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
            )
            
            # Morphological opening to remove noise and separate digits
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Invert back to black text on white background
            processed_img = 255 - opening
            
            # Enhanced Tesseract config from real.py
            custom_config = r"--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Clean and validate
            cleaned_text = re.sub(r"\D", "", text).strip()
            
            if cleaned_text and cleaned_text.isdigit():
                speed_value = int(cleaned_text)
                # Reasonable speed limit range
                if 5 <= speed_value <= 200:
                    return speed_value
            
            return None
            
        except Exception as e:
            print(f"Speed extraction error: {e}")
            return None
    
    def process_detections(self, frame):
        """Process frame for traffic and speed sign detections"""
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = {
                'speed_signs': [],
                'traffic_lights': [],
                'timestamp': time.time()
            }
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return detections
            
            # Process each detection
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.model.names[class_id].lower()
                    
                    if confidence < self.conf_threshold:
                        continue
                    
                    # Crop region for analysis
                    cropped_region = frame[y1:y2, x1:x2]
                    
                    # Process speed signs
                    if any(keyword in label for keyword in ["speed sign", "speed limit", "speed"]):
                        speed_value = self.extract_speed_from_sign(cropped_region)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'label': label,
                            'speed_value': speed_value
                        }
                        detections['speed_signs'].append(detection)
                        
                        if speed_value:
                            self.speed_detections.append(speed_value)
                            print(f"🚗 Detected Speed Limit: {speed_value} km/h")
                    
                    # Process traffic lights
                    elif "traffic light" in label:
                        color = self.detect_traffic_light_color(cropped_region)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'label': label,
                            'color': color
                        }
                        detections['traffic_lights'].append(detection)
                        
                        self.traffic_light_detections.append(color)
                        print(f"🚦 Detected Traffic Light: {color}")
                
                except Exception as e:
                    print(f"Detection processing error: {e}")
                    continue
            
            # Update stable detections
            self._update_stable_detections()
            
            return detections
            
        except Exception as e:
            print(f"Process detections error: {e}")
            return {'speed_signs': [], 'traffic_lights': [], 'timestamp': time.time()}
    
    def _update_stable_detections(self):
        """Update stable detection values based on history"""
        # Update stable speed limit
        if self.speed_detections:
            # Use most common speed in recent detections
            speed_counts = {}
            for speed in list(self.speed_detections)[-5:]:  # Last 5 detections
                speed_counts[speed] = speed_counts.get(speed, 0) + 1
            
            if speed_counts:
                most_common_speed = max(speed_counts, key=speed_counts.get)
                if speed_counts[most_common_speed] >= 2:  # At least 2 confirmations
                    self.current_speed_limit = most_common_speed
        
        # Update stable traffic light
        if self.traffic_light_detections:
            # Use most recent non-unknown detection
            recent_lights = [light for light in list(self.traffic_light_detections)[-3:] if light != "Unknown"]
            if recent_lights:
                self.current_traffic_light = recent_lights[-1]
    
    def create_visualization(self, frame, detections):
        """Create visualization with detections"""
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            
            # Draw speed sign detections
            for detection in detections['speed_signs']:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                speed_value = detection['speed_value']
                
                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label
                if speed_value:
                    label = f"Speed: {speed_value} km/h ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Speed Sign ({confidence:.2f})"
                    color = (0, 255, 255)
                
                # Background for text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(vis, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(vis, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw traffic light detections
            for detection in detections['traffic_lights']:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                color = detection['color']
                
                # Color mapping
                color_map = {
                    "Red": (0, 0, 255),
                    "Yellow": (0, 255, 255),
                    "Green": (0, 255, 0),
                    "Unknown": (128, 128, 128)
                }
                
                box_color = color_map.get(color, (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw label
                label = f"Traffic Light: {color} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(vis, (x1, y2 + 5), (x1 + label_size[0] + 10, y2 + 35), box_color, -1)
                cv2.putText(vis, label, (x1 + 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw info panel
            self._draw_info_panel(vis)
            
            return vis
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return frame
    
    def _draw_info_panel(self, vis):
        """Draw information panel"""
        try:
            h, w = vis.shape[:2]
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = len(self.fps_counter)
                self.fps_counter.clear()
                self.last_fps_time = current_time
            else:
                fps = len(self.fps_counter)
            
            # Panel background
            panel_height = 140
            cv2.rectangle(vis, (10, 10), (450, panel_height), (0, 0, 0), -1)
            cv2.rectangle(vis, (10, 10), (450, panel_height), (255, 255, 255), 2)
            
            # Info lines
            info_lines = [
                f"🚦📏 Forza TDFS Detection - FPS: {fps}",
                f"Current Speed Limit: {self.current_speed_limit if self.current_speed_limit else 'Unknown'} km/h",
                f"Current Traffic Light: {self.current_traffic_light}",
                f"Model: best_tdfs.pt",
                f"Confidence Threshold: {self.conf_threshold}",
                f"Frame: {self.frame_count}"
            ]
            
            for i, line in enumerate(info_lines):
                if "Speed Limit" in line and self.current_speed_limit:
                    color = (0, 255, 0)  # Green if speed detected
                elif "Traffic Light" in line:
                    color_map = {
                        "Red": (0, 0, 255),
                        "Yellow": (0, 255, 255), 
                        "Green": (0, 255, 0),
                        "Unknown": (128, 128, 128)
                    }
                    color = color_map.get(self.current_traffic_light, (255, 255, 255))
                else:
                    color = (255, 255, 255)
                
                cv2.putText(vis, line, (15, 35 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Detection statistics
            if self.speed_detections or self.traffic_light_detections:
                stats_y = panel_height + 30
                cv2.putText(vis, "Recent Detections:", (15, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if self.speed_detections:
                    recent_speeds = list(self.speed_detections)[-5:]
                    speeds_text = f"Speeds: {recent_speeds}"
                    cv2.putText(vis, speeds_text, (15, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if self.traffic_light_detections:
                    recent_lights = list(self.traffic_light_detections)[-3:]
                    lights_text = f"Lights: {recent_lights}"
                    cv2.putText(vis, lights_text, (15, stats_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
        except Exception as e:
            print(f"Info panel error: {e}")
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Forza TDFS Detection System...")
        print("Controls:")
        print("  Q or ESC - Quit")
        print("  R - Reset detection history")
        
        # Setup visualization
        cv2.namedWindow('Forza TDFS Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Forza TDFS Detection', 1200, 700)
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture screen
                img = np.array(self.sct.grab(self.monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Process detections
                detections = self.process_detections(frame)
                
                # Create visualization
                vis_frame = self.create_visualization(frame, detections)
                
                # Display
                cv2.imshow('Forza TDFS Detection', vis_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('r'):
                    # Reset detection history
                    self.speed_detections.clear()
                    self.traffic_light_detections.clear()
                    self.current_speed_limit = None
                    self.current_traffic_light = "Unknown"
                    print("🔄 Detection history RESET")
                
                # Update counters
                self.frame_count += 1
                self.fps_counter.append(time.time())
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                if elapsed < 1/30:
                    time.sleep(1/30 - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping TDFS System...")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("✅ TDFS System stopped")

def main():
    """Main function"""
    try:
        print("🚦📏 Forza Traffic Detection & Speed Signs System")
        print("=" * 55)
        print("Features:")
        print("  ✅ Traffic Light Color Detection")
        print("  ✅ Speed Sign Recognition with OCR")
        print("  ✅ Real-time Detection History")
        print("  ✅ Stable Detection Filtering")
        print("  ✅ Enhanced Preprocessing (from real.py)")
        print("=" * 55)
        
        # Check required model
        model_path = "best_tdfs.pt"
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("Please place best_tdfs.pt in the same directory as this script")
            return
        
        print("✅ TDFS model found")
        
        # Initialize and run
        system = ForzaTDFSSystem()
        system.run()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()