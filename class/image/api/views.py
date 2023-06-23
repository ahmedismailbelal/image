import datetime
import json
from rest_framework.views import APIView
from .serializer import *
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from PIL import ImageFile
import face_recognition
import os
import numpy as np
import joblib
from django.core.files import File
from datetime import datetime


def getFace(img):
    image = face_recognition.load_image_file(img)
    return image

def findEcoding(img):
    encodings = []
    names = []

    image = getFace(img)

    face_locations = face_recognition.face_locations(image, model="hog")
    face_detection = face_recognition.face_encodings(image,known_face_locations=face_locations)
   

    for faceing in face_detection:
        faceing = np.array(faceing).ravel()
        encodings.append(faceing)
        names.append(os.path.basename(img).split('.')[0])

    return encodings,names

def load_data(path):
    Xdata = []
    yLabel = []
    data , name = findEcoding(f'{path}') 
    for da,na in zip(data,name) :
        Xdata.append(da) 
        yLabel.append(na)

    return Xdata,yLabel
from rest_framework.parsers import MultiPartParser
import pandas as pd
class predict_image(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser]
    def post(self,request):
        if request.method == 'POST':
            image_data = request.FILES.get('image')
            image_instance = image(image=image_data)
            image_instance.save()
            test_data,test_label = load_data(image_instance.image.path)
            # test_data_list = test_data.tolist()
            test = pd.Series(test_data).to_json(orient='values')
            print (test_data)
            return JsonResponse({'test data' : test})
        else:
            return JsonResponse({'error': 'Invalid request method'})
