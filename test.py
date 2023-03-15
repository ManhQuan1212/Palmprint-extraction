import os
import numpy as np
import cv2
import LMTRP
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# def preprocess(img):
#     # top, right, bottom, left
#     # (t, r, b, l) = face_recognition.face_locations(img)[0]
#     # crop image
#     # face_img = img[t:b, l:r]
#     # resize 
#     # face_img = cv2.resize(face_img, (128, 128))
#     # encode
#     encode = LMTRP.LMTRP_process(face_img)[0]

#     return encode

labels = os.listdir('Dataset/')

model = pickle.load(open('svm-63.model', 'rb'))
X_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
#print(X_test)

pred = model.predict(X_test)
print(accuracy_score(pred, y_test)*100)

img = cv2.imread('random/003_001.jpg')
encode= LMTRP.LMTRP_process(img)
print(encode)
# encode = np.reshape(encode, (1, -1))
pred = model.predict(encode)
print(pred)
person = labels[pred[0]]
print(person)
print("Is it {} ?".format(person))

