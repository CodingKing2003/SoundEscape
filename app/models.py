import datetime
from django.db import models
from django.utils.text import slugify
from django.db.models.signals import pre_save
from django.contrib.auth import get_user_model


# Create your models here.




User = get_user_model()

class Emotion(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dominant_emotion = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.dominant_emotion}"










