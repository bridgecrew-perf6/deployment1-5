# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:40:29 2018

@author: Kaushik
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.preprocessing import image
import time
from tensorflow.keras import applications 

vgg16 = applications.VGG16(include_top=False, weights='imagenet')
#model = load_model('models/testModel.h5')

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')


@app.route('/upload_ct.html')
def upload_ct():
   return render_template('upload_ct.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   #resnet_chest = load_model('models/resnet_chest.h5')
   vgg_chest = load_model('models/vgg_chest.h5')
   inception_chest = load_model('models/inceptionv3_chest.h5')
   xception_chest = load_model('models/xception_chest.h5')
   model = load_model('models/testModel.h5')

   # image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   # image = cv2.resize(image,(224,224))
   # image = np.array(image) / 255
   # image = np.expand_dims(image, axis=0)
   file_path='./flask app/assets/images/upload_chest.jpg'
   print("[INFO] loading and preprocessing image…") 
   image = load_img(file_path, target_size=(224, 224)) 
   image = img_to_array(image) 
   image = np.expand_dims(image, axis=0)
   image /= 255. 
   
   # resnet_pred = resnet_chest.predict(image)
   # probability = resnet_pred[0]
   # print("Resnet Predictions:")
   # if probability[0] > 0.5:
   #    resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   # else:
   #    resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   # print(resnet_chest_pred)

   vgg_pred = vgg_chest.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(vgg_chest_pred)

   inception_pred = inception_chest.predict(image)
   probability = inception_pred[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_chest_pred)

   xception_pred = xception_chest.predict(image)
   probability = xception_pred[0]
   print("Xception Predictions:")
   if probability[0] > 0.5:
      xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(xception_chest_pred)
   
   bt_prediction = vgg16.predict(image) 
   preds = model.predict(bt_prediction)
   c1=preds[0][0]
   c2=preds[0][1]
   c3=preds[0][2]
   c4=preds[0][3]
   c5=preds[0][4]
   c6=preds[0][5]
   c7=preds[0][6]
   c8=preds[0][7]
   c9=preds[0][8]
   print(c1,c2,c3,c4,c5,c6,c7,c8,c9)
   # print("testModel Predictions:")
   # 
   # print(testModel_pred)

   return render_template('results_chest.html',vgg_chest_pred=vgg_chest_pred,inception_chest_pred=inception_chest_pred,xception_chest_pred=xception_chest_pred, c1=c1)

# @app.route('/uploaded_chest', methods = ['POST', 'GET'])
# def uploaded_chest():
#    if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file:
#             # filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))
#    return
# uploaded_chest()
# def read_image(file_path):
#    print("[INFO] loading and preprocessing image…") 
#    image = load_img(file_path, target_size=(224, 224)) 
#    image = img_to_array(image) 
#    image = np.expand_dims(image, axis=0)
#    image /= 255. 
#    return image

# def test_single_image(path):
  
#   diseases= ['Atelectasis',
# 'Cardiomegaly',
# 'Consolidation',
# 'EnlargedCardiomediastinum',
# 'Fracture',
# 'LungLesion',
# 'NoFinding',
# 'Pneumonia',
# 'Pneumothorax']
#   images = read_image(path)
#   time.sleep(.5)
#   bt_prediction = vgg16.predict(images) 
#   preds = model.predict(bt_prediction)
#   c1=preds[0][0]
#   c2=preds[0][1]
#   c3=preds[0][2]
#   c4=preds[0][3]
#   c5=preds[0][4]
#   c6=preds[0][5]
#   c7=preds[0][6]
#   c8=preds[0][7]
#   c9=preds[0][8]
#   print(c1,c2,c3,c4,c5,c6,c7,c8,c9)
#   return load_img(path)
  

# path = './flask app/assets/images/upload_chest.jpg'
# test_single_image(path)







if __name__ == '__main__':
   app.secret_key = ".."
   app.run()