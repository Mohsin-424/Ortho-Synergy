import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pandas as pd
import os
import datetime

# Load the pre-trained MobileNetV2 model for feature extraction
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load the pain recognition model
model_path = r'D:\OrthoSynergy\ortho_project\models\nn_700_4fps_79.keras'
pain_model = load_model(model_path)

# Define the face detector using Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_features(img):
    img = cv2.resize(img, (224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor.predict(img_array)
    return features.flatten()

def process_video(patient_folder):
    #st.title("Pain Recognition")
    st.write("Press 'Start Stream' to begin.")

    # Initialize session state variables
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
        st.session_state.pain_levels = []
        st.session_state.timestamps = []
        st.session_state.show_save_button = False

    # Get patient information from session state
    patient_name = st.session_state.get('patient_name', 'Unknown')
    patient_age = st.session_state.get('patient_age', 'Unknown')

    # Display buttons based on streaming state
    if st.session_state.streaming:
        st.button('Stop Stream', key='stop_stream', on_click=stop_stream)
    else:
        st.button('Start Stream', key='start_stream', on_click=start_stream)
        st.button('Reset State', key='reset_state', on_click=reset_state)
    
    # Create placeholders for the video feed and graph
    video_placeholder = st.empty()
    graph_placeholder = st.empty()

    if st.session_state.streaming:
        cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
        resize_width = 400
        resize_height = 380

        while st.session_state.streaming:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (resize_width, resize_height))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                x *= (frame.shape[1] / resize_width)
                y *= (frame.shape[0] / resize_height)
                w *= (frame.shape[1] / resize_width)
                h *= (frame.shape[0] / resize_height)
                
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                face = frame[y:y+h, x:x+w]
                features = extract_features(face)
                features = np.expand_dims(features, axis=0)
                
                # Predict pain level
                prediction = pain_model.predict(features)
                pain_level = np.argmax(prediction, axis=1)[0]
                
                # Collect pain level data
                st.session_state.pain_levels.append(pain_level)
                st.session_state.timestamps.append(datetime.datetime.now())

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'Pain Level: {pain_level}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the video feed
            video_placeholder.image(frame, channels='BGR', use_column_width=True)

            # Update and display the graph
            if st.session_state.pain_levels:
                df = pd.DataFrame({'Time': st.session_state.timestamps, 'Pain Level': st.session_state.pain_levels})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Time'], y=df['Pain Level'], mode='lines+markers', name='Pain Level'))
                fig.update_layout(title=f'Pain Level Over Time for {patient_name}, Age {patient_age}',
                                  xaxis_title='Time',
                                  yaxis_title='Pain Level')
                graph_placeholder.plotly_chart(fig)
        
        cap.release()
        cv2.destroyAllWindows()

    # Show Save Report button after streaming stops
    if st.session_state.show_save_button:
        if st.button('Save Report', key='save_report'):
            if st.session_state.pain_levels:
                df = pd.DataFrame({'Time': st.session_state.timestamps, 'Pain Level': st.session_state.pain_levels})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Time'], y=df['Pain Level'], mode='lines+markers', name='Pain Level'))
                fig.update_layout(title=f'Pain Level Over Time for {patient_name}, Age {patient_age}',
                                  xaxis_title='Time',
                                  yaxis_title='Pain Level')
                
                # Ensure the patient folder exists
                if not os.path.exists(patient_folder):
                    os.makedirs(patient_folder)
                
                report_path = os.path.join(patient_folder, f'pain_level_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                fig.write_image(report_path)
                st.success(f'Report saved to {report_path}')
            else:
                st.warning('No data available to save.')

def start_stream():
    st.session_state.streaming = True

def stop_stream():
    st.session_state.streaming = False
    st.session_state.show_save_button = True  # Show save button when streaming stops

def reset_state():
    st.session_state.pain_levels = []
    st.session_state.timestamps = []
    st.session_state.streaming = False
    st.session_state.show_save_button = False