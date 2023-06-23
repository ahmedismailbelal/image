from rest_framework.views import APIView
from .serializer import *
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
import face_recognition
import os
import numpy as np
import joblib
import numpy as np
class KNN:
    distances = [[]*2]
    final_label = []
    def getDistance(self,test_vector,train_feature_vectors,train_labels): 
        for i in range(len(train_feature_vectors)):
            distance = np.sqrt(np.sum((test_vector-train_feature_vectors[i])**2)) #calculate the distance between test vector and train vectors
            self.distances.append([distance,train_labels[i]]) #append the distance and label to the distances array
        self.distances = sorted(self.distances, key=lambda x:x[0]) #sort the distances array by the distance
        return self.distances
    def getLabel(self,k):
        labels = []
        for i in range(k):
            if self.distances[i][0] < 0.55:
                labels.append(self.distances[i][1])
            else:
                labels.append("UNKNOWN PERSON!")
        return labels

    def getNearestNeighbor(self,k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)

    def Classifier(self,k,train_features, test_features, Trainlabels):

        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(test_features[i],train_features,Trainlabels)
            self.final_label.append(self.getNearestNeighbor(k))
        return self.final_label



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
import csv
class predict_image(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser]
    def post(self,request):
        if request.method == 'POST':
            image_data = request.FILES.get('image')
            image_instance = image(image=image_data)
            image_instance.save()
            model = KNN()
            train_data_loaded = joblib.load('D:\\college\\final project\\image\\class\\image\\savedModels\\train_data.joblib')
            train_label_loaded = joblib.load('D:\\college\\final project\\image\\class\\image\\savedModels\\train_labels.joblib')
            test_data,test_label = load_data(image_instance.image.path)
            y_pred = model.Classifier(9,train_data_loaded,test_data,train_label_loaded)
            num = y_pred.count('UNKNOWN PERSON!')
            merged_list = list(set(y_pred) | set(y_pred))
            for i in range(num-1):
                self.merged_list.append('UNKNOWN PERSON!')
            image_instance.classification_results = merged_list
            image_instance.save()
            y_pred.clear()
            return JsonResponse({'prediction': merged_list})
        else:
            return JsonResponse({'error': 'Invalid request method'})


