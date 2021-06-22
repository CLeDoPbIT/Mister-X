from __future__ import unicode_literals

from django.db import models

class Document(models.Model):
    docfile = models.FileField(upload_to='uploads')
# Create your models here.
