import base64
from datetime import datetime
import io
from django.shortcuts import redirect, render

from app.models import Categories, Course, Emotion, Level, User
from django.template.loader import render_to_string
from django.http import HttpResponse, JsonResponse
from django.utils.text import slugify
from django.db.models.signals import pre_save
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.contrib.auth.models import User
from app.models import Emotion
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
import cv2
from deepface import DeepFace


import webbrowser

import numpy as np
from django.http import StreamingHttpResponse
from django.contrib.auth.decorators import login_required

from django.views.decorators import gzip

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model


def BASE(request):
    return render(request, 'base.html')

def camera(request):
    return render(request,'registration/camera.html')






 ##face_cascade = cv2.CascadeClassifier('C:/Users/ACER/Desktop/Learning Management System/LMS/LMS/haarcascade_frontalface_default.xml')

@login_required
def detect_faces_view(request):
    user = request.user
    emotions=emotions.objects.all()
    

    if request.method=="POST":
        singer=request.POST.get('singer')
        lang=request.POST.get('lang')
        emotions = []

        # Load pre-trained models for face detection and emotion recognition
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Start the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Read the frame from the webcam
            success, frame = cap.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for idx, face in enumerate(faces):

                # Draw a rectangle around each face and detect emotions
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = analysis['dominant_emotion']
                    emotions.append(dominant_emotion)
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

            # Display the frame
            cv2.imshow('Face Detection and Emotion Recognition', frame)

            # Exit the loop if the 'ESC (27)' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

        if len(faces) == 0:
            print("Error in capturing emotion. Please Try Again.")
        elif len(faces) != 0:
            max_emotion = max(emotions)
            print('Dominant Emotion: ', max_emotion)

            # Create a new instance of the Emotion model and save it to the database
            emotion = Emotion(user=request.user, dominant_emotion=max_emotion, created_at=datetime.now())
            emotion.save()

            if max_emotion == 'neutral':
                webbrowser.open(f"https://www.youtube.com/results?search_query={singer}+{lang}+songs")
            elif max_emotion != 'neutral':
                webbrowser.open(f"https://www.youtube.com/results?search_query={singer}+{lang}+{max_emotion}")
            

def visualize(request):
     
    emotions = Emotion.objects.filter(user=request.user)

    # Create a dictionary to store emotion counts
    emotion_counts = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }

        # Count the occurrences of each emotion
    for emotion in emotions:
            
            emotion_counts[emotion.dominant_emotion] += 1

        # Plot a bar chart of the emotion counts
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title('Emotion Chart')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')

        # Save the chart to a BytesIO object
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png')
    chart_buffer.seek(0)

        # Render the chart as a base64-encoded string
    chart_data = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
    chart_html = f'<img src="data:image/png;base64,{chart_data}">'

        # Pass the chart HTML to the template
    context = {'chart_html': chart_html}
    return render(request, 'registration/dashboard.html', context)
    
    

            


        

        
        


    
    
    
 


    
    


    







    


def HOME(request):
    category = Categories.objects.all().order_by('id')[0:5]
    course = Course.objects.filter(status='PUBLISH').order_by('-id')
    context = {
        'category': category,
        'course': course,
    }

    return render(request, 'Main/home.html', context)










def ABOUT_US(request):
    return render(request, 'Main/about_us.html')


