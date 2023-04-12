import Database
import os
import landmark
import cv2

new_people = input('Nhap ten nguoi moi: \n')
dir = os.path.join("KLTN", "Dataset", "train", new_people)
if not os.path.exists(dir):
    os.mkdir(dir)
path =  os.path.join("KLTN", "Dataset", "train", )
label = Database.getlabel(path)
Database.Update_database(label, new_people)
landmark.roiImageFromHand(dir, option=2, cap = cv2.VideoCapture(0))