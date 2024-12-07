import subprocess
from flask import Flask, jsonify, render_template, request, Response
import sys
import os
import time
import speech_recognition as sr  
import cv2
import threading
import pytesseract
from ultralytics import YOLO
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

app = Flask(__name__)

# Global variables for video streaming and detection
global_frame_text = None
global_frame_object = None
text_detection_active = False
object_detection_active = False

# Initialize models
text_model = pytesseract
object_model = YOLO("yolov8n.pt")

# Video capture
cap = cv2.VideoCapture(0)

def generate_text_frames():
    global global_frame_text, text_detection_active
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if text_detection_active:
            # Perform text detection
            results = text_model.image_to_string(frame)
            
            # If text is detected, speak it out
            if len(results) > 1:
                language = 'en'
                myobj = gTTS(text=results, lang=language, slow=False)
                myobj.save('output.mp3')
                sound = AudioSegment.from_file("output.mp3")
                play(sound)
                os.remove('output.mp3')
            
            # Draw rectangles around detected text
            boxes = text_model.image_to_boxes(frame)
            for box in boxes.splitlines():
                box = box.split(' ')
                x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                cv2.rectangle(frame, (x, frame.shape[0]-y), (w, frame.shape[0]-h), (0, 0, 255), 3)
            
            cv2.putText(frame, f'Text: {results}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_object_frames():
    global global_frame_object, object_detection_active
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if object_detection_active:
            # Perform object detection
            results = object_model.predict(frame, imgsz=512, stream=True)
            
            for r in results:
                for box in r.boxes:
                    # Get class name
                    class_name = object_model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    # Only speak if confidence is high
                    if confidence > 0.5:
                        # Text to speech for object
                        language = 'en'
                        mytext = f"{class_name} detected with {confidence:.2f} confidence"
                        myobj = gTTS(text=mytext, lang=language, slow=False)
                        myobj.save('output.mp3')
                        sound = AudioSegment.from_file("output.mp3")
                        play(sound)
                        os.remove('output.mp3')
                    
                    # Draw bounding boxes
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name} {confidence:.2f}', 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_text')
def video_text():
    return Response(generate_text_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_object')
def video_object():
    return Response(generate_object_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_text')
def start_text():
    global text_detection_active
    text_detection_active = True
    return render_template('index.html')

@app.route('/stop_text')
def stop_text():
    global text_detection_active
    text_detection_active = False
    return render_template('index.html')

@app.route('/start_object')
def start_object():
    global object_detection_active
    object_detection_active = True
    return render_template('index.html')

@app.route('/stop_object')
def stop_object():
    global object_detection_active
    object_detection_active = False
    return render_template('index.html')

@app.route('/listen')
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
        try:
            voice_input = r.recognize_google(audio).lower()
            print(f"You said: {voice_input}")
            
            if "text" in voice_input:
                start_text()
            elif "object" in voice_input:
                start_object()
            elif "stop" in voice_input:
                stop_text()
                stop_object()
            
            return jsonify({"command": voice_input})
        
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"})
        except sr.RequestError as e:
            return jsonify({"error": f"Could not request results: {e}"})

if __name__ == "__main__":
    app.run(debug=True)