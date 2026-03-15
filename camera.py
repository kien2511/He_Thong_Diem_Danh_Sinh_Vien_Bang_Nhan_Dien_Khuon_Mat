import cv2
import threading
import sqlite3
import pandas as pd
from datetime import datetime
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2_putText_utf8(img, text, position, color_bgr, font_size=24):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, str(text), font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class Camera(object):
    def __init__(self):
        self.video = None
        self.is_running = False
        self.mode = "OFF" # Modes: OFF, RECOGNIZE, CAPTURE
        self.current_student_id = None
        self.capture_count = 0
        self.max_capture = 50
        
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_model()
        
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        
        self.attendance_logs = [] # In-memory recent logs for real-time frontend update

    def load_model(self):
        if os.path.exists('trainer/trainer.yml'):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read('trainer/trainer.yml')
            self.model_loaded = True
        else:
            self.model_loaded = False

    def start(self, mode="RECOGNIZE", student_id=None):
        with self.lock:
            if not self.is_running:
                self.video = cv2.VideoCapture(0)
                # Set resolution to 640x480 to reduce processing lag
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
            
            self.mode = mode
            self.current_student_id = student_id
            self.capture_count = 0
            
            if mode == "RECOGNIZE":
                self.load_model()
            
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            self.mode = "OFF"
            if self.video:
                self.video.release()
                self.video = None
        return True

    def _get_student(self, sid):
        # Quick sqlite fetch without keeping connection open
        import database
        return database.get_student_by_id(sid)
        
    def _record_attendance(self, sid):
        import database
        # This will return True if newly recorded, False if already present today
        return database.record_attendance(sid)

    def _capture_loop(self):
        while self.is_running:
            if not self.video or not self.video.isOpened():
                time.sleep(0.1)
                continue
                
            ret, img = self.video.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if self.mode == "CAPTURE" and self.current_student_id is not None:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    self.capture_count += 1
                    
                    if not os.path.exists("dataset"):
                        os.makedirs("dataset")
                        
                    cv2.imwrite(f"dataset/User.{self.current_student_id}.{self.capture_count}.jpg", gray[y:y+h, x:x+w])
                    
                    img = cv2_putText_utf8(img, f"Capturing: {self.capture_count}/{self.max_capture}", (x, y-10), (255, 0, 0), 24)
                
                # Check outside the loop to stop when 50 images are captured
                if self.capture_count >= self.max_capture:
                    self.mode = "OFF"
                    self.stop()
                    
            elif self.mode == "RECOGNIZE":
                if self.model_loaded:
                    for (x, y, w, h) in faces:
                        id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                        
                        if confidence < 70:
                            student = self._get_student(id)
                            if student:
                                name = str(student['name'])
                                mssv = str(student['mssv'])
                                conf_percent = round(100 - confidence)
                                display_text = f"{name} ({conf_percent}%)"
                                img = cv2_putText_utf8(img, display_text, (x, y-10), (0, 255, 0), 24)
                                
                                # Log attendance
                                recorded = self._record_attendance(id)
                                if recorded:
                                    t_str = datetime.now().strftime('%H:%M:%S')
                                    self.attendance_logs.insert(0, {"name": name, "mssv": mssv, "time": t_str})
                                    if len(self.attendance_logs) > 20:
                                        self.attendance_logs.pop() # Keep list small
                            else:
                                img = cv2_putText_utf8(img, f"Unknown ID {id}", (x, y-10), (0, 0, 255), 24)
                        else:
                            conf_percent = round(100 - confidence)
                            img = cv2_putText_utf8(img, f"Unknown ({conf_percent}%)", (x, y-10), (0, 0, 255), 24)
                        
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                else:
                    img = cv2_putText_utf8(img, "Model not trained", (50, 50), (0, 0, 255), 24)

            ret, jpeg = cv2.imencode('.jpg', img)
            if ret:
                with self.lock:
                    self.frame = jpeg.tobytes()


    def get_frame(self):
        with self.lock:
            return self.frame

    def get_logs(self):
        return self.attendance_logs
    
    def get_status(self):
        return {
            "mode": self.mode,
            "running": self.is_running,
            "capture_progress": self.capture_count if self.mode == "CAPTURE" else 0
        }

camera = Camera()
