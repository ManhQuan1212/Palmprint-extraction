import LMTRP
import numpy as np
import cv2
import matplotlib.pyplot as plt
import landmark
import ROI
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
option=1
cap=cv2.VideoCapture(0)
def final():
    img = cv2.imread('Input\Anh.jpg')
    #plt.imshow(img)
    #plt.show()
    #cv2.imwrite('Output\', img)
    #PictureofROI = landmark.roiImageFromHand('./Output/',option,cap)
    #img = cv2.imread('./Output/14.bmp')

    labels = os.listdir('Dataset/')

    model = pickle.load(open('svm-70.model', 'rb'))
    X_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    #print(X_test)

    pred = model.predict(X_test)
    print(accuracy_score(pred, y_test)*100)
    img = cv2.imread('')
    encode = LMTRP.LMTRP_process(img)
    print(type(encode))
    print(encode)
    print("Kiểu dữ liệu của phần tử trong mảng:", encode.dtype)
    print("Kích thước của mảng:", encode.shape)
    print("Số phần tử trong mảng:", encode.size)
    print("Số chiều của mảng:", encode.ndim)
    # encode = np.reshape(encode, (1, -1))
    pred = model.predict(encode)
    print(pred)
    #person = labels[pred[0]]
    #print(person)
    #print("Is it {} ?".format(person))

final()