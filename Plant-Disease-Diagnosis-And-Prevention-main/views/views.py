from flask import make_response,Flask, flash, redirect, render_template, request, url_for, session
from app import *

import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

# from tensorflow.keras.utils import load_img,img_to_array
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from PIL import Image


import numpy as np
labels =["Apple Apple scab","Apple Black rot","Apple Cedar apple rust","Apple healthy","Bacterial leaf blight in rice leaf","Blight in corn Leaf","Blueberry healthy","Brown spot in rice leaf","Cercospora leaf spot","Cherry (including sour) Powdery mildew   Cause of disease Podosphaera clandestina, a fungus that most commonly infects young, expanding leaves but can also be found on buds, fruit and fruit stems. It overwinters as small, round, black bodies (chasmothecia) on dead leaves, on the orchard floor, or in tree crotches. Colonies produce more (asexual) spores generally around shuck fall and continue the disease cycle. How to prevent/cure the disease 1. Remove and destroy sucker shoots.2. Keep irrigation water off developing fruit and leaves by using irrigation that does not wet the leaves. Also, keep irrigation sets as short as possible.3. Follow cultural practices that promote good air circulation, such as pruning, and moderate shoot growth through judicious nitrogen management.","Cherry (including_sour) healthy","Common Rust in corn Leaf","Corn (maize) healthy","Garlic","Grape Black rot","Grape Esca Black Measles","Grape Leaf blight Isariopsis Leaf Spot","Grape healthy","Gray Leaf Spot in corn Leaf","Leaf smut in rice leaf","Orange Haunglongbing Citrus greening","Peach healthy","Pepper bell Bacterial spot","Pepper bell healthy","Potato Early blight","Potato Late blight","Potato healthy","Raspberry healthy","Sogatella rice","Soybean healthy","Strawberry Leaf scorch","Strawberry healthy","Tomato Bacterial spot","Tomato Early blight","Tomato Late blight","Tomato Leaf Mold","Tomato Septoria leaf spot","Tomato Spider mites Two spotted spider mite","Tomato Target Spot","Tomato Tomato mosaic virus Cause of disease 1. Tomato mosaic virus and tobacco mosaic virus can exist for two years in dry soil or leaf debris, but will only persist one month if soil is moist. The viruses can also survive in infected root debris in the soil for up to two years.2. Seed can be infected and pass the virus to the plant but the disease is usually introduced and spread primarily through human activity. The virus can easily spread between plants on workers hands, tools, and clothes with normal activities such as plant tying, removing of suckers, and harvest  3. The virus can even survive the tobacco curing process, and can spread from cigarettes and other tobacco products to plant material handled by workers after a cigarette How to prevent/cure the disease  1.Purchase transplants only from reputable sources. Ask about the sanitation procedures they use to prevent disease.  2. Inspect transplants prior to purchase. Choose only transplants showing no clear symptoms. 3. Avoid planting in fields where tomato root debris is present, as the virus can survive long-term in roots. 4. Wash hands with soap and water before and during the handling of plants to reduce potential spread between plants","Tomato healthy","algal leaf in tea","anthracnose in tea","bird eye spot in tea","brown blight in tea","cabbage looper","corn crop","ginger","healthy tea leaf","lemon canker","onion","potassium deficiency in plant","potato crop","potato hollow heart","red leaf spot in tea","tomato canker"
]
Model = keras.models.load_model('./Model/plant disease_99.80.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def model():
    return render_template('prediction.html')

@app.route('/upload', methods=['GET'])
def UploadGet():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def UploadPost():
    print('!!')

    file = request.files['file']
    if file.filename == '' :
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        file.save(file.filename)
        img = image.load_img(file.filename, target_size=(200, 200))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Expand the dimensions to match the model's input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess the image (normalize and prepare for prediction)
        img_array = preprocess_input(img_array)
        
        # Make a prediction using the loaded model
        predictions = Model.predict(img_array)
        
        # Get the class label with the highest probability
        predicted_class = np.argmax(predictions)
        # image = load_img(file.filename, target_size=(200,200))
        # image = img_to_array(image)
        # image = image.reshape((None,200,200,3))
        # pred = Model.predict(image)
        # p = np.argmax(pred)
        
    return render_template('upload.html',disease=labels[predicted_class])
