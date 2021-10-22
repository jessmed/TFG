# mi_aplicacion/urls.py

from django.urls import path
from . import views

urlpatterns = [
  path('', views.index, name='index'),
  path('document_list/', views.doclist, name='doclist'),
  path('statistics/', views.statistics, name='statistics'),
  path('document/', views.document, name='document'),

]