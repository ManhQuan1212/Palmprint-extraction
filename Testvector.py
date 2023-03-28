import LMTRP
import cv2
import landmark
import os
import numpy as np
import RandomImg
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
option=1
cap=cv2.VideoCapture(0)

def final():
    landmark.roiImageFromHand('./Output/',option,cap)
    img = RandomImg.chooseRandomImage()
    filename = os.path.join('Output',img)
    labels = os.listdir('Dataset/')
    model = pickle.load(open('svm-70.model', 'rb'))
    X_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    pred = model.predict(X_test)
    print(accuracy_score(pred, y_test)*100)
    img = cv2.imread(filename)
    encode= LMTRP.LMTRP_process(img)
    print(encode)
    pred = model.predict(encode)
    print(pred)
    person = labels[pred[0]]
    print(person)
    print("Is it {} ?".format(person))
    RandomImg.Removefile()
final()