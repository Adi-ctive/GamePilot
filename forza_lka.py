#!/usr/bin/env python3
"""
Forza Horizon 4 Lane Keep Assist Controller - INSTANT REACTIVE STEERING
Short bursts with lockout periods - no input queues, instant frame reactions
"""

import cv2
import numpy as np
import torch
import time
import math
import mss
import win32api
import win32con
import win32gui
from pathlib import Path
from collections import deque
from scipy import ndimage

class ForzaController:
    """Instant reactive steering with short bursts and lockout"""
    
    def __init__(self):
        self.keys = {
            'throttle': ord('W'),
            'brake': ord('S'), 
            'steer_left': ord('A'),
            'steer_right': ord('D')
        }
        
        self.current_keys_pressed = set()
        
        # Throttle settings
        self.throttle_pulse_duration = 0.05
        self.throttle_rest_duration = 0.5
        self.last_throttle_time = time.time()
        self.throttle_state = False
        
        # INSTANT REACTIVE STEERING - NO QUEUES
        self.steering_burst_duration = 0.15  # Max 0.2s as requested (using 0.15 for safety)
        self.steering_lockout_duration = 1.0  # 1 second lockout as requested
        
        self.last_steering_time = 0.0
        self.steering_active = False
        self.steering_start_time = 0.0
        self.current_steering_key = None
        
        # Threshold for steering activation
        self.steering_threshold = 0.008  # Small threshold for instant reaction
    
    def press_key(self, key):
        """Press a key"""
        try:
            if key not in self.current_keys_pressed:
                win32api.keybd_event(key, 0, 0, 0)
                self.current_keys_pressed.add(key)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
    
    def release_key(self, key):
        """Release a key"""
        try:
            if key in self.current_keys_pressed:
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)
                self.current_keys_pressed.discard(key)
        except Exception as e:
            print(f"Error releasing key {key}: {e}")
    
    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in list(self.current_keys_pressed):
            self.release_key(key)
    
    def apply_controls(self, steering_command, throttle_enabled):
        """INSTANT REACTIVE steering - no Dark Souls input queues"""
        try:
            current_time = time.time()
            
            # Handle active steering burst
            if self.steering_active:
                # Check if burst duration is over (max 0.2s)
                if current_time - self.steering_start_time >= self.steering_burst_duration:
                    # End the burst
                    if self.current_steering_key:
                        self.release_key(self.current_steering_key)
                        print(f"🛑 Burst ended")
                    
                    self.steering_active = False
                    self.current_steering_key = None
                    self.last_steering_time = current_time  # Start lockout timer
                
                # Continue throttle during steering
                self._handle_throttle(throttle_enabled)
                return
            
            # Check lockout period (1 second)
            time_since_last_steering = current_time - self.last_steering_time
            if time_since_last_steering < self.steering_lockout_duration:
                # Still locked out - no steering allowed
                self._handle_throttle(throttle_enabled)
                return
            
            # INSTANT REACTION - check current frame steering command
            if abs(steering_command) > self.steering_threshold:
                # Immediate steering response for this frame
                if steering_command > 0:
                    key_to_press = self.keys['steer_right']
                    direction = "RIGHT"
                else:
                    key_to_press = self.keys['steer_left']
                    direction = "LEFT"
                
                # Start immediate burst
                self.press_key(key_to_press)
                self.steering_active = True
                self.steering_start_time = current_time
                self.current_steering_key = key_to_press
                
                print(f"⚡ INSTANT {direction}: {steering_command:.4f}")
            
            # Handle throttle
            self._handle_throttle(throttle_enabled)
            
        except Exception as e:
            print(f"Error in apply_controls: {e}")
    
    def _handle_throttle(self, throttle_enabled):
        """Handle throttle independently"""
        if throttle_enabled:
            current_time = time.time()
            
            if self.throttle_state:
                if current_time - self.last_throttle_time >= self.throttle_pulse_duration:
                    self.release_key(self.keys['throttle'])
                    self.throttle_state = False
                    self.last_throttle_time = current_time
                else:
                    self.press_key(self.keys['throttle'])
            else:
                if current_time - self.last_throttle_time >= self.throttle_rest_duration:
                    self.press_key(self.keys['throttle'])
                    self.throttle_state = True
                    self.last_throttle_time = current_time
                else:
                    self.release_key(self.keys['throttle'])
        else:
            self.release_key(self.keys['throttle'])

class LaneDetector:
    """Lane detection using YOLOPv2 model"""
    
    def __init__(self, model_path='yolopv2.pt', img_size=640):
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.vehicle_center_x = None
        
    def _load_model(self, model_path):
        """Load YOLOPv2 model"""
        try:
            print(f"Loading model from {model_path} on {self.device}")
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            
            if self.device.type == 'cuda':
                model.half()
            
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model inference"""
        try:
            h, w = frame.shape[:2]
            
            scale = min(self.img_size / h, self.img_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            y_offset = (self.img_size - new_h) // 2
            x_offset = (self.img_size - new_w) // 2
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.to(self.device).unsqueeze(0)
            
            if self.device.type == 'cuda':
                img_tensor = img_tensor.half()
            
            return img_tensor, scale, (x_offset, y_offset)
        
        except Exception as e:
            print(f"Error in preprocess_frame: {e}")
            return None, None, None
    
    def extract_lanes(self, lane_output, original_shape, scale, offsets):
        """Extract lane mask from model output"""
        try:
            if isinstance(lane_output, torch.Tensor):
                lane_np = lane_output.detach().cpu().numpy()
            else:
                lane_np = lane_output
            
            if lane_np.ndim == 4:
                lane_np = lane_np[0, 0]
            elif lane_np.ndim == 3:
                lane_np = lane_np[0]
            
            if lane_np.max() > 1.0 or lane_np.min() < 0.0:
                lane_np = 1 / (1 + np.exp(-lane_np))
            
            binary_mask = (lane_np > 0.3).astype(np.uint8)
            
            x_offset, y_offset = offsets
            scaled_h = int(original_shape[0] * scale)
            scaled_w = int(original_shape[1] * scale)
            relevant = binary_mask[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w]
            final = cv2.resize(relevant, (original_shape[1], original_shape[0]))
            
            return self._process_lane_mask(final)
            
        except Exception as e:
            print(f"Lane extraction error: {e}")
            return np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    
    def _process_lane_mask(self, lane_mask):
        """Process lane mask"""
        try:
            h, w = lane_mask.shape[:2]
            
            thresh = int(h * 0.45)
            limited = np.zeros_like(lane_mask)
            limited[thresh:, :] = lane_mask[thresh:, :]
            
            labeled, count = ndimage.label(limited)
            if count == 0:
                return np.zeros_like(lane_mask)
            
            center_x = w // 2
            components = []
            
            for i in range(1, count + 1):
                ys, xs = np.where(labeled == i)
                if ys.size == 0:
                    continue
                
                cx, cy = xs.mean(), ys.mean()
                area = ys.size
                
                center_distance = abs(cx - center_x)
                max_center_distance = w * 0.4
                
                x_span = xs.max() - xs.min()
                y_span = ys.max() - ys.min()
                aspect_ratio = x_span / max(y_span, 1)
                
                if (cy >= thresh and 
                    area > 100 and 
                    center_distance < max_center_distance and 
                    aspect_ratio < 3.0):
                    
                    components.append((i, cx, center_distance, area, y_span))
            
            components.sort(key=lambda x: (x[2], -x[4]))
            
            ego = np.zeros_like(lane_mask)
            left_comps = [c for c in components if c[1] < center_x]
            right_comps = [c for c in components if c[1] >= center_x]
            
            if left_comps:
                ego[labeled == left_comps[0][0]] = 1
            if right_comps:
                ego[labeled == right_comps[0][0]] = 1
            
            if np.any(ego):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
                ego = cv2.morphologyEx(ego, cv2.MORPH_CLOSE, kernel)
            
            return ego
        
        except Exception as e:
            print(f"Error in _process_lane_mask: {e}")
            return np.zeros_like(lane_mask)
    
    def extract_lane_center_points(self, lane_mask, frame_shape):
        """Extract lane center points from mask"""
        try:
            if lane_mask is None or not lane_mask.any():
                return []
            
            h, w = frame_shape[:2]
            self.vehicle_center_x = w // 2
            
            y_levels = np.linspace(h - 20, int(h * 0.25), num=12, dtype=int)
            center_points = []
            
            contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for y in y_levels:
                lx = rx = None
                tolerance = 25
                
                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue
                    
                    points = contour.reshape(-1, 2)
                    nearby = points[np.abs(points[:, 1] - y) < tolerance]
                    
                    if nearby.size > 0:
                        for px, _ in nearby:
                            if px < self.vehicle_center_x:
                                if lx is None or px > lx:
                                    lx = px
                            else:
                                if rx is None or px < rx:
                                    rx = px
                
                if lx is not None and rx is not None:
                    center_x = (lx + rx) / 2
                    center_points.append((center_x, y))
            
            return center_points
        
        except Exception as e:
            print(f"Error in extract_lane_center_points: {e}")
            return []
    
    def detect_lanes(self, frame):
        """Main lane detection function"""
        try:
            input_tensor, scale, offsets = self.preprocess_frame(frame)
            if input_tensor is None:
                return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8), []
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                lane_output = outputs[2] if isinstance(outputs, (list, tuple)) and len(outputs) >= 3 else outputs
            
            lane_mask = self.extract_lanes(lane_output, frame.shape, scale, offsets)
            center_points = self.extract_lane_center_points(lane_mask, frame.shape)
            
            return lane_mask, center_points
            
        except Exception as e:
            print(f"Lane detection error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8), []

class PIDController:
    """Simple PID for steering calculation"""
    
    def __init__(self, kp=0.15, ki=0.001, kd=0.04, integral_limit=0.06):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.reset()
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
    
    def update(self, error, dt):
        try:
            proportional = self.kp * error
            
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
            integral = self.ki * self.integral
            
            derivative = self.kd * (error - self.prev_error) / max(dt, 1e-6)
            self.prev_error = error
            
            return proportional + integral + derivative
        
        except Exception as e:
            print(f"Error in PID update: {e}")
            return 0.0

class WindowSelector:
    """Helper class to find Forza window"""
    
    @staticmethod
    def find_forza_window():
        """Automatically find Forza window"""
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
                return {
                    "top": rect[1],
                    "left": rect[0], 
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
            
            return None
        
        except Exception as e:
            print(f"Error finding Forza window: {e}")
            return None

class ForzaLaneKeepAssist:
    """INSTANT REACTIVE LKA - no input queues"""
    
    def __init__(self, model_path='yolopv2.pt'):
        try:
            self.detector = LaneDetector(model_path)
            self.controller = ForzaController()
            self.pid = PIDController()
            
            self.sct = mss.mss()
            
            self.monitor = WindowSelector.find_forza_window()
            if self.monitor is None:
                print("Could not auto-detect Forza window. Using full screen.")
                self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            else:
                print(f"Found Forza window: {self.monitor}")
            
            self.enable_throttle = True
            self.last_time = time.time()
            self.frame_count = 0
            
            # Minimal filtering for instant reactions
            self.steering_filter = deque(maxlen=3)  # Very small filter
            
            print("Forza LKA initialized with INSTANT REACTIVE steering!")
            print("Max burst: 0.15s, Lockout: 1.0s")
            print("Controls:")
            print("  Q - Quit")
            print("  T - Toggle throttle")
            print("  S - Manual brake")
        
        except Exception as e:
            print(f"Error initializing ForzaLaneKeepAssist: {e}")
            raise
    
    def calculate_steering_command(self, lane_center_points, frame_shape):
        """Calculate steering with majority-based outlier filtering"""
        try:
            if len(lane_center_points) < 3:
                return 0.0
            
            h, w = frame_shape[:2]
            vehicle_center = w / 2
            
            # Quick outlier filtering
            x_coords = [p[0] for p in lane_center_points]
            
            if len(x_coords) >= 5:
                median_x = np.median(x_coords)
                std_x = np.std(x_coords)
                
                filtered_points = []
                for i, (x, y) in enumerate(lane_center_points):
                    is_extreme = (i == 0 or i == len(lane_center_points) - 1)
                    is_outlier = abs(x - median_x) > 1.5 * std_x
                    
                    if is_extreme or not is_outlier:
                        filtered_points.append((x, y))
                
                if len(filtered_points) >= 2:
                    lane_center_points = filtered_points
            
            # Weighted error calculation
            total_error = 0.0
            total_weight = 0.0
            
            for i, (x, y) in enumerate(lane_center_points):
                error = (x - vehicle_center) / (w / 2)
                
                distance_from_bottom = (h - y) / h
                weight = 1.0 / (1.0 + distance_from_bottom * 2)
                
                total_error += error * weight
                total_weight += weight
            
            combined_error = total_error / max(total_weight, 1e-6)
            
            # PID control
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            steering_cmd = self.pid.update(combined_error, dt)
            
            # Limits
            steering_cmd = np.clip(steering_cmd, -0.2, 0.2)
            
            # Minimal filtering
            self.steering_filter.append(steering_cmd)
            smoothed_steering = np.mean(self.steering_filter)
            
            return smoothed_steering
        
        except Exception as e:
            print(f"Error in calculate_steering_command: {e}")
            return 0.0
    
    def create_visualization(self, frame, lane_mask, lane_center_points, steering_cmd):
        """Create visualization overlay"""
        try:
            vis = frame.copy()
            h, w = frame.shape[:2]
            
            # Lane mask overlay
            if lane_mask is not None and lane_mask.any():
                colored_mask = np.zeros_like(frame)
                colored_mask[lane_mask > 0] = [0, 255, 255]
                vis = cv2.addWeighted(vis, 0.8, colored_mask, 0.2, 0)
            
            # Vehicle center line
            vehicle_center = w // 2
            cv2.line(vis, (vehicle_center, h), (vehicle_center, int(h * 0.25)), (255, 0, 0), 2)
            
            # Lane center points
            for i, (x, y) in enumerate(lane_center_points):
                color = (0, 255, 0) if i == 0 else (0, 200, 0)
                cv2.circle(vis, (int(x), int(y)), 5, color, -1)
            
            # Lane center line
            if len(lane_center_points) > 1:
                pts = np.array(lane_center_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, (0, 255, 0), 3)
            
            # Steering status
            current_time = time.time()
            if self.controller.steering_active:
                status = "STEERING ACTIVE"
                color = (0, 0, 255)
            elif (current_time - self.controller.last_steering_time) < self.controller.steering_lockout_duration:
                remaining = self.controller.steering_lockout_duration - (current_time - self.controller.last_steering_time)
                status = f"LOCKOUT: {remaining:.1f}s"
                color = (0, 255, 255)
            else:
                status = "READY"
                color = (0, 255, 0)
            
            # Info panel
            throttle_status = "PULSE" if self.controller.throttle_state else "REST"
            info_text = [
                f"Status: {status}",
                f"Steering: {steering_cmd:.4f}",
                f"Throttle: {'ON' if self.enable_throttle else 'OFF'} ({throttle_status})",
                f"Lane Points: {len(lane_center_points)}"
            ]
            
            # Background for text
            cv2.rectangle(vis, (10, 10), (300, 120), (0, 0, 0), -1)
            
            for i, text in enumerate(info_text):
                text_color = color if i == 0 else (255, 255, 255)
                cv2.putText(vis, text, (15, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, text_color, 2)
            
            return vis
        
        except Exception as e:
            print(f"Error in create_visualization: {e}")
            return frame
    
    def run(self):
        """Main execution loop"""
        print("Starting INSTANT REACTIVE Forza LKA...")
        print("No Dark Souls input queues - instant frame reactions!")
        time.sleep(2)
        
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    # Capture screen
                    img = np.array(self.sct.grab(self.monitor))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # Lane detection
                    lane_mask, lane_center_points = self.detector.detect_lanes(frame)
                    
                    # Calculate steering for THIS FRAME
                    steering_cmd = self.calculate_steering_command(lane_center_points, frame.shape)
                    
                    # INSTANT reaction to this frame's steering command
                    self.controller.apply_controls(steering_cmd, self.enable_throttle)
                    
                    # Visualization
                    vis_frame = self.create_visualization(frame, lane_mask, lane_center_points, steering_cmd)
                    cv2.imshow('Forza LKA - INSTANT REACTIVE', vis_frame)
                    
                    # Handle user input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        self.enable_throttle = not self.enable_throttle
                        print(f"Throttle: {'ENABLED' if self.enable_throttle else 'DISABLED'}")
                    elif key == ord('r'):
                        # Reset
                        self.pid.reset()
                        self.controller.last_steering_time = 0.0
                        self.controller.steering_active = False
                        print("RESET - Ready for immediate steering")
                    
                    self.frame_count += 1
                    
                    # Target 30 FPS
                    elapsed = time.time() - start_time
                    if elapsed < 1/30:
                        time.sleep(1/30 - elapsed)
                
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.controller.release_all_keys()
            cv2.destroyAllWindows()
            print("LKA stopped")

def main():
    """Main function"""
    try:
        MODEL_PATH = 'yolopv2.pt'
        
        if not Path(MODEL_PATH).exists():
            print(f"Model file '{MODEL_PATH}' not found!")
            print("Please place your YOLOPv2 model weights in the same directory as this script.")
            return
        
        lka = ForzaLaneKeepAssist(model_path=MODEL_PATH)
        lka.run()
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()