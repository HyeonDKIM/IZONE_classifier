import os, re, glob
import cv2
import numpy as np
from keras.models import model_from_json
import json

result_path = 'C:/Users/USER/Desktop/hyeonji/MachineLearning/result/'
categories = ["Eunbi","Minju", "Wonyoung", "Sakura", "Yuri", "Yena", "Chaewon", "Chaeyeon","Nako", "Hitomi", "Yujin", "Hyewon"]

with open(result_path+"tst.json",'r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(result_path+"tst.h5")
print("loaded model from disk")

def Dataization(img_path):
    image_w = 64
    image_h = 64
    img=cv2.imread(img_path)
    img=cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

src=[]
name=[]
test=[]

image_dir= "C:/Users/USER/Desktop/hyeonji/MachineLearning/test/"

for file in os.listdir(image_dir):
    if (file.find('.png') is not -1):
        src.append(image_dir+file)
        name.append(file)
        test.append(Dataization(image_dir+file))

test = np.array(test)
predict = model.predict_classes(test)

for i in range(len(test)):
    print(name[i]+":, predict: "+str(categories[predict[i]]))
