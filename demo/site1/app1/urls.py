from django.conf.urls import url
from django.urls import path
from .views import my_view
from . import views

urlpatterns = [
    # /app1
    path('', my_view, name='my-view'),
    url(r'^plot1d/$', views.Plot1DView.as_view(), name='plot1d'),
]