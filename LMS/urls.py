

from django.contrib import admin
from django.urls import path, include
from . import views, user_login
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    path('admin/', admin.site.urls),

    path('base', views.BASE, name='base'),
    path('', views.HOME, name='home'),
   
    path('about', views.ABOUT_US, name='about_us'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/register', user_login.REGISTER, name='register'),
    path('doLogin', user_login.DOLOGIN, name='doLogin'),
    path('accounts/profile', user_login.PROFILE, name='profile'),
    path('accounts/profile/update',user_login.PROFILE_UPDATE,name='profile_update'),
    path('product/filter-data',views.filter_data,name="filter-data"),
  path('search',views.SEARCH_COURSE,name='search_course'),
  path('detect_faces', views.detect_faces_view, name='detect_faces'),
  path('camera',views.camera,name='camera'),
  path('visualize',views.visualize,name='visualize')
  
  
  
 
 


    
] +static(settings.MEDIA_URL,document_root = settings.MEDIA_ROOT)
