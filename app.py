import cv2
import numpy as np
import pygame
from scipy.ndimage import gaussian_filter
import os
import json
from datetime import datetime

class SmartCrowdManagementSystem:
    def __init__(self, video_path, base_threshold=20):
        self.video_path = video_path
        self.base_threshold = base_threshold
        self.alert_triggered = False
        self.alarm_enabled = True
        self.total_alerts = 0
        self.people_history = []
        self.frames_processed = 0
        
        # Sound setup - FIXED
        pygame.mixer.init()
        self.alert_sound = self.load_alert_sound()
        
        # Load YOLO model
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"❌ YOLO loading error: {e}")
            self.net = None
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        
        # Heatmap setup
        self.heatmap = None
        self.heatmap_decay = 0.95
        
        # Dynamic threshold settings
        self.area_divisions = 4
        
        # Initialize dashboard data file
        self.initialize_dashboard_data()
        
        print(f"🚀 Smart Crowd Management System Started!")
        print(f"🔊 Alarm will trigger based on area density")
        print(f"🎯 Base threshold: {base_threshold} people")
        print("🎮 Controls: Q=Quit | P=Pause | A=Toggle Alarm | R=Reset Heatmap | +/-=Adjust Threshold")
    
    def initialize_dashboard_data(self):
        """Initialize dashboard data file"""
        default_data = {
            'current_count': 0,
            'alert_level': 'LOW',
            'total_alerts': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'people_history': [],
            'status_message': 'Waiting for data from detection system...',
            'frames_processed': 0,
            'average_crowd': 0.0,
            'maximum_crowd': 0,
            'alarm_active': False,
            'system_status': 'INACTIVE',
            'video_file': os.path.basename(self.video_path)
        }
        
        try:
            with open("crowd_data.json", "w") as f:
                json.dump(default_data, f, indent=4)
            print("✅ Dashboard data file initialized!")
        except Exception as e:
            print(f"❌ Error initializing dashboard data: {e}")
    
    def update_dashboard_data(self, people_count, alert_triggered):
        """Update dashboard data file with current information"""
        # Update people history (keep last 100 entries)
        self.people_history.append(people_count)
        if len(self.people_history) > 100:
            self.people_history = self.people_history[-100:]
        
        # Determine alert level
        if people_count <= 10:
            alert_level = 'LOW'
        elif people_count <= 20:
            alert_level = 'NORMAL'
        elif people_count <= 30:
            alert_level = 'HIGH'
        else:
            alert_level = 'CRITICAL'
        
        # Calculate statistics
        avg_crowd = np.mean(self.people_history) if self.people_history else 0.0
        max_crowd = max(self.people_history) if self.people_history else 0
        
        dashboard_data = {
            'current_count': people_count,
            'alert_level': alert_level,
            'total_alerts': self.total_alerts,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'people_history': self.people_history,
            'status_message': f'Monitoring crowd - {people_count} people detected',
            'frames_processed': self.frames_processed,
            'average_crowd': round(avg_crowd, 1),
            'maximum_crowd': max_crowd,
            'alarm_active': alert_triggered,
            'system_status': 'ACTIVE',
            'video_file': os.path.basename(self.video_path)
        }
        
        try:
            with open("crowd_data.json", "w") as f:
                json.dump(dashboard_data, f, indent=4)
        except Exception as e:
            print(f"❌ Error updating dashboard data: {e}")
    
    def load_alert_sound(self):
        """FIXED: Better sound generation"""
        try:
            # Initialize pygame mixer properly
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Create a simple beep sound using pygame directly
            duration = 1000  # milliseconds
            frequency = 800  # Hz
            
            # Generate sound using pygame's built-in method
            sound = pygame.mixer.Sound(buffer=bytes([128] * 22050))  # Simple beep
            
            # Alternative: Use a WAV file if available
            # You can also download a beep sound and use it
            print("✅ Alarm sound system ready!")
            return sound
        except Exception as e:
            print(f"❌ Sound error: {e}")
            print("🔇 Continuing without sound...")
            return None
    
    def play_alarm_sound(self):
        """FIXED: Play alarm sound"""
        try:
            if self.alert_sound:
                # Stop any currently playing sound
                pygame.mixer.stop()
                # Play the alarm sound
                self.alert_sound.play()
                print("🔊 ALARM SOUND PLAYING!")
            else:
                # Fallback: Use system beep
                print('\a')  # System beep
        except Exception as e:
            print(f"❌ Could not play sound: {e}")
            # Fallback methods
            try:
                # Try alternative sound method
                import winsound
                winsound.Beep(1000, 1000)  # Frequency, Duration
            except:
                print("🔇 No sound available")
    
    def detect_people_yolo(self, frame):
        height, width = frame.shape[:2]
        people_centers = []
        people_boxes = []
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == 0 and confidence > 0.5:  # Person class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, min(x, width-1))
                    y = max(0, min(y, height-1))
                    w = max(1, min(w, width-x))
                    h = max(1, min(h, height-y))
                    
                    people_centers.append((center_x, center_y))
                    people_boxes.append((x, y, w, h))
        
        return people_centers, people_boxes
    
    def detect_people_motion(self, frame):
        fgMask = self.backSub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        people_centers = []
        people_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                people_centers.append((center_x, center_y))
                people_boxes.append((x, y, w, h))
        
        return people_centers, people_boxes
    
    def calculate_area_based_threshold(self, frame, people_centers):
        """Calculate dynamic threshold based on area density"""
        height, width = frame.shape[:2]
        
        area_width = width // self.area_divisions
        area_height = height // self.area_divisions
        
        max_area_density = 0
        critical_areas = []
        
        for i in range(self.area_divisions):
            for j in range(self.area_divisions):
                area_x1 = i * area_width
                area_y1 = j * area_height
                area_x2 = (i + 1) * area_width
                area_y2 = (j + 1) * area_height
                
                # Count people in this area
                area_people = 0
                for center in people_centers:
                    x, y = center
                    if area_x1 <= x < area_x2 and area_y1 <= y < area_y2:
                        area_people += 1
                
                # Calculate area density
                area_size = area_width * area_height
                area_density = area_people / (area_size / 10000)  # People per 100x100 pixel area
                
                if area_density > max_area_density:
                    max_area_density = area_density
                
                # Mark critical areas
                if area_density > 2:
                    critical_areas.append((area_x1, area_y1, area_x2, area_y2, area_density))
        
        # Dynamic threshold based on maximum density
        if max_area_density > 8:  # Very crowded
            dynamic_threshold = max(5, self.base_threshold // 4)
        elif max_area_density > 5:  # Crowded
            dynamic_threshold = max(8, self.base_threshold // 2)
        elif max_area_density > 3:  # Moderate
            dynamic_threshold = max(12, self.base_threshold - 5)
        elif max_area_density > 1:  # Light
            dynamic_threshold = max(15, self.base_threshold - 3)
        else:  # Normal
            dynamic_threshold = self.base_threshold
        
        return dynamic_threshold, critical_areas, max_area_density
    
    def create_heatmap(self, frame, people_centers):
        height, width = frame.shape[:2]
        
        if self.heatmap is None:
            self.heatmap = np.zeros((height, width), dtype=np.float32)
        
        density_map = np.zeros((height, width), dtype=np.float32)
        
        for center in people_centers:
            x, y = center
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(density_map, (x, y), 50, 1, -1)
        
        density_map = gaussian_filter(density_map, sigma=25)
        self.heatmap = self.heatmap * self.heatmap_decay + density_map
        self.heatmap = np.clip(self.heatmap, 0, 1)
        
        return self.heatmap
    
    def apply_heatmap_overlay(self, frame, heatmap):
        """Apply beautiful heatmap overlay"""
        heatmap_vis = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        # Better blending
        blended = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        return blended
    
    def trigger_alarm(self, frame, people_count, dynamic_threshold, max_density):
        """Trigger visual and audio alarm"""
        # Flashing red border
        border_thickness = 20
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), border_thickness)
        
        # Alert background
        alert_bg = np.zeros((120, 600, 3), dtype=np.uint8)
        alert_bg[:] = (0, 0, 100)  # Dark red background
        
        # Add alert background to frame
        frame[10:130, 10:610] = cv2.addWeighted(frame[10:130, 10:610], 0.3, alert_bg, 0.7, 0)
        
        # Alert text
        cv2.putText(frame, "🚨 HIGH CROWD DENSITY ALARM! 🚨", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"People: {people_count} | Threshold: {dynamic_threshold} | Density: {max_density:.1f}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sound alarm - FIXED
        if self.alarm_enabled and not self.alert_triggered:
            self.play_alarm_sound()
            self.alert_triggered = True
            self.total_alerts += 1
            print(f"🔊 ALARM TRIGGERED! {people_count} people detected")
            print(f"📊 Area density: {max_density:.1f} | Dynamic threshold: {dynamic_threshold}")
    
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            print(f"📁 Check if this file exists: {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video loaded: {fps} FPS, {total_frames} frames")
        print("🎮 Controls: Q=Quit | P=Pause | A=Toggle Alarm | R=Reset Heatmap | +/-=Adjust Threshold")
        print("⏳ Starting video...")
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ Video finished playing")
                    break
                
                frame_count += 1
                self.frames_processed = frame_count
                
                # Detect people
                if self.net:
                    people_centers, people_boxes = self.detect_people_yolo(frame)
                else:
                    people_centers, people_boxes = self.detect_people_motion(frame)
                
                people_count = len(people_centers)
                
                # Calculate dynamic threshold based on area density
                dynamic_threshold, critical_areas, max_density = self.calculate_area_based_threshold(frame, people_centers)
                
                # Create heatmap
                heatmap = self.create_heatmap(frame, people_centers)
                frame_with_heatmap = self.apply_heatmap_overlay(frame, heatmap)
                
                # Draw critical areas (high density zones)
                for area in critical_areas:
                    x1, y1, x2, y2, density = area
                    cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_with_heatmap, f"Density: {density:.1f}", 
                               (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw bounding boxes around people
                for (x, y, w, h) in people_boxes:
                    cv2.rectangle(frame_with_heatmap, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw person centers
                for center in people_centers:
                    cv2.circle(frame_with_heatmap, center, 6, (255, 255, 255), -1)
                    cv2.circle(frame_with_heatmap, center, 8, (0, 255, 0), 2)
                
                # Display information panel
                status_color = (0, 255, 0)  # Green
                alarm_status = "ON" if self.alarm_enabled else "OFF"
                
                if people_count > dynamic_threshold:
                    status_color = (0, 0, 255)  # Red
                
                # Information background
                info_bg = np.zeros((160, 400, 3), dtype=np.uint8)
                info_bg[:] = (0, 0, 0)
                frame_with_heatmap[10:170, 10:410] = cv2.addWeighted(
                    frame_with_heatmap[10:170, 10:410], 0.3, info_bg, 0.7, 0
                )
                
                # Display stats
                cv2.putText(frame_with_heatmap, f"PEOPLE COUNT: {people_count}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame_with_heatmap, f"DYNAMIC THRESHOLD: {dynamic_threshold}", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_with_heatmap, f"MAX DENSITY: {max_density:.1f}", 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_with_heatmap, f"BASE THRESHOLD: {self.base_threshold}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame_with_heatmap, f"ALARM: {alarm_status}", 
                           (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.alarm_enabled else (0, 0, 255), 2)
                
                # Update dashboard data
                self.update_dashboard_data(people_count, self.alert_triggered)
                
                # Trigger alarm if crowd exceeds dynamic threshold
                if people_count > dynamic_threshold:
                    self.trigger_alarm(frame_with_heatmap, people_count, dynamic_threshold, max_density)
                else:
                    self.alert_triggered = False
                
                # Show frame
                cv2.imshow('🎥 AI Crowd Management - Heatmap & Smart Alarm', frame_with_heatmap)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("⏹️ Stopping video...")
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord('a'):
                self.alarm_enabled = not self.alarm_enabled
                status = "ENABLED" if self.alarm_enabled else "DISABLED"
                print(f"🔊 Alarm {status}")
            elif key == ord('r'):
                self.heatmap = None
                print("🔄 Heatmap reset")
            elif key == ord('+'):
                self.base_threshold += 1
                print(f"📈 Base threshold increased to: {self.base_threshold}")
            elif key == ord('-'):
                self.base_threshold = max(1, self.base_threshold - 1)
                print(f"📉 Base threshold decreased to: {self.base_threshold}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ System stopped successfully!")

# Run the system
if __name__ == "__main__":
    # Change this to your video path
    video_path = r"C:\Users\DELL\Downloads\crowd\archive\production_id_3687560 (2160p).mp4"
    
    print("🚀 Starting AI Crowd Management System...")
    system = SmartCrowdManagementSystem(
        video_path=video_path,
        base_threshold=20  # Alarm base threshold
    )
    system.run()