"""site1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.urls import path
from django.views.generic.base import RedirectView

from app1.views import my_view

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    path('', my_view, name='my-view'),
    url(r'^$', RedirectView.as_view(url='/app1/'), name='index'),
    url(r'^app1/',  include(('app1.urls', 'app1'), namespace='app1', )),
]
