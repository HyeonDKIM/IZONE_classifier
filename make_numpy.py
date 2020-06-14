import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import json
import matplotlib.pyplot as plt

groups_folder_path = 'C:/Users/USER/Desktop/hyeonji/MachineLearning/Data/'
categories = ["Eunbi", "Minju", "Wonyoung", "Sakura", "Yuri", "Yena",
              "Chaewon", "Chaeyeon", "Nako", "Hitomi", "Yujin", "Hyewon"]
num_classes = len(categories)

image_w = 64
image_h = 64

X = []
Y = []

# 전처리
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + "/"

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img,None,fx=image_w/img.shape[1],fy=image_h/img.shape[0])

            X.append(img/256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=1)

xy = (X_train, Y_train, X_validation, Y_validation)
print(xy[0].shape)
print(xy[1].shape)
print(xy[2].shape)
print(xy[3].shape)
np.save("C:/Users/USER/Desktop/hyeonji/MachineLearning/result/tst.npy", xy)

print()
print("======================== Train Start ==========================")
print()

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
numpy_path = 'C:/Users/USER/Desktop/hyeonji/MachineLearning/result/tst.npy'
result_path = 'C:/Users/USER/Desktop/hyeonji/MachineLearning/result/'

X_train, Y_train, X_validation, Y_validation = np.load(numpy_path, allow_pickle=True)
num_classes = len(categories)
accuracy = []
skf = KFold(n_splits=5, shuffle=True)

for train, validation in skf.split(X,Y):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X[train], Y[train], batch_size=100, nb_epoch=500, callbacks=[early_stopping])

    k_accuracy = '%.4f'%(model.evaluate(X[validation], Y[validation])[1])
    accuracy.append(k_accuracy)

'''
    fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                    figsize=(10,10),
                                    colorbar=True,
                                    class_names=categories)
    plt.show()
'''
#print(confmat)
#print(classification_report(Y_test.argmax(axis=1), test_predictions.argmax(axis=1),
#                                target_names=["Eunbi","Minju", "Wonyoung", "Sakura", "Yuri", "Yena", "Chaewon", "Chaeyeon", "Nako", "Hitomi", "Yujin", "Hyewon"]))
'''
    f = open(result_path + 'base_'+str(kfold_num)+'fold.txt', 'a')
    f.write('------------------' + str(foldnum+1) + 'fold result------------------\n')
    f.write(str(confmat) + '\n')
    f.write(str(classification_report(Y_test.argmax(axis=1), test_predictions.argmax(axis=1),
                                      target_names=["bacteria", "healthy", "lateblight", "targetspot",
                                                    "yellowleafcurl"]) + '\n'))
                 
                                           
for i in range(len(categories)):
    axis_sum = 0
    for j in range(len(categories)):
        axis_sum = axis_sum + confmat[i, j]
    answer = confmat[i, i] / axis_sum
    answer = str(answer)
    print(categories[i] + " accuracy : ", end='')
    print(answer)
'''
print('\nK-fold cross validation Accuracy: {}'.format(accuracy))

model_json = model.to_json()
with open(result_path + 'tst.json',
            "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights(result_path + 'tst.h5')
print("saved model to disk")


