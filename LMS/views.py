import base64
from collections import Counter
from django.db.models import Count
from datetime import datetime
import io
from django.shortcuts import render
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from app.models import  Emotion, User
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
    emotions=Emotion.objects.all()
    

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
                webbrowser.open(f"https://www.youtube.com/results?search_query={singer}+{lang}+{max_emotion}+songs")

        return HttpResponse("Detection Completed")
        
@login_required
def visualize(request):
    current_user = request.user

    # Filter emotions by the current user and count the number of instances of each emotion
    emotions_data = Emotion.objects.filter(user=current_user).values('dominant_emotion').annotate(total=Count('dominant_emotion'))
    

    # Create a dictionary of emotions and their corresponding values
    emotions_dict = {d['dominant_emotion']: d['total'] for d in emotions_data}

    # Extract the emotions and values as lists
    emotions = list(emotions_dict.keys())
    values = list(emotions_dict.values())

    # Create a pie chart using the values and labels
    fig, ax = plt.subplots()
    ax.pie(values, labels=emotions, autopct='%1.1f%%')
    ax.set_title('Emotions Distribution')

    # Convert the pie chart to an image
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)

    # Return the image as an HTTP response
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
    
   
   
            


     
    

            


        

        
        


    
    
    
 


    
    


    







    


def HOME(request):


    return render(request, 'Main/home.html')










def ABOUT_US(request):
    return render(request, 'Main/about_us.html')


