from django.db import models




class image (models.Model):
    
    image = models.ImageField(upload_to='uploads/', null = True , blank= True)
    classification_results = models.CharField(max_length=100, null=True, blank=True)
    # csv_file = models.FileField (upload_to='files/', null = True , blank= True)
    # feature_vectors = models.CharField(max_length=100, null=True, blank=True)
    # date_of_student = models.DateTimeField(auto_now_add=True)
    # date_of_student = models.CharField(max_length=100, null=True, blank=True)
    # history_of_lecture = models.CharField(max_length=100, null=True, blank=True)
    def __str__(self):
        
        return str(self.id)
    
    