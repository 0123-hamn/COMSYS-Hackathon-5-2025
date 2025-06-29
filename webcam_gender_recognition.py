import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import deque

class GenderRecognitionWebcam:
    def __init__(self, model_path='best_gender_model.h5'):
        # Load the trained model
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except:
            print(f"Could not load model from {model_path}")
            print("Please make sure you have trained the model first using train_gender_model.py")
            return
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize variables
        self.prediction_history = deque(maxlen=10)  # Store last 10 predictions
        self.confidence_threshold = 0.7
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Colors for visualization
        self.colors = {
            'female': (255, 0, 255),  # Pink
            'male': (255, 0, 0),      # Blue
            'unknown': (0, 255, 255), # Yellow
            'text': (255, 255, 255)   # White
        }
        
        print("Gender Recognition Webcam initialized!")
        print("Press 'q' to quit, 's' to save screenshot")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        # Resize to model input size
        face_resized = cv2.resize(face_img, (224, 224))
        
        # Convert to RGB (model expects RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_gender(self, face_img):
        """Predict gender from face image"""
        try:
            # Preprocess the face
            face_processed = self.preprocess_face(face_img)
            
            # Make prediction
            prediction = self.model.predict(face_processed, verbose=0)[0][0]
            
            # Convert to gender label
            gender = 'male' if prediction > 0.5 else 'female'
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 'unknown', 0.0
    
    def get_stable_prediction(self, current_prediction):
        """Get stable prediction using history"""
        self.prediction_history.append(current_prediction)
        
        if len(self.prediction_history) < 5:
            return current_prediction
        
        # Count predictions in history
        female_count = sum(1 for pred in self.prediction_history if pred[0] == 'female')
        male_count = sum(1 for pred in self.prediction_history if pred[0] == 'male')
        
        # Return majority prediction
        if female_count > male_count:
            return ('female', np.mean([pred[1] for pred in self.prediction_history if pred[0] == 'female']))
        elif male_count > female_count:
            return ('male', np.mean([pred[1] for pred in self.prediction_history if pred[0] == 'male']))
        else:
            return current_prediction
    
    def draw_info_panel(self, frame, fps, total_faces, detected_genders):
        """Draw information panel on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Faces Detected: {total_faces}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Female: {detected_genders.get('female', 0)}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['female'], self.thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Male: {detected_genders.get('male', 0)}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['male'], self.thickness)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                   self.font, 0.5, self.colors['text'], 1)
    
    def run(self):
        """Main webcam loop"""
        prev_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            detected_genders = {'female': 0, 'male': 0}
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Predict gender
                gender, confidence = self.predict_gender(face_img)
                
                # Get stable prediction
                stable_gender, stable_confidence = self.get_stable_prediction((gender, confidence))
                
                # Only show prediction if confidence is high enough
                if stable_confidence > self.confidence_threshold:
                    detected_genders[stable_gender] += 1
                    
                    # Draw rectangle around face
                    color = self.colors[stable_gender]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw gender label
                    label = f"{stable_gender.upper()}: {stable_confidence:.2f}"
                    label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x, y-label_size[1]-10), 
                                (x+label_size[0], y), color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x, y-5), 
                               self.font, self.font_scale, (255, 255, 255), self.thickness)
                else:
                    # Draw rectangle for low confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['unknown'], 2)
                    cv2.putText(frame, "Unknown", (x, y-5), 
                               self.font, self.font_scale, self.colors['text'], self.thickness)
            
            # Draw information panel
            self.draw_info_panel(frame, fps, len(faces), detected_genders)
            
            # Display the frame
            cv2.imshow('Gender Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gender_recognition_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam session ended")

def main():
    print("Starting Gender Recognition Webcam...")
    print("Make sure you have trained the model first!")
    
    # Try to load the best model, fallback to regular model
    model_paths = ['best_gender_model.h5', 'gender_recognition_model.h5']
    model_loaded = False
    
    for model_path in model_paths:
        try:
            webcam = GenderRecognitionWebcam(model_path)
            model_loaded = True
            break
        except:
            continue
    
    if not model_loaded:
        print("Error: No trained model found!")
        print("Please run train_gender_model.py first to train the model.")
        return
    
    # Start webcam
    webcam.run()

if __name__ == "__main__":
    main() 