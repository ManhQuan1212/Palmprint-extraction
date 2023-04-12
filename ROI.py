from ctypes import sizeof
from tkinter import W
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import os
import torch
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
import numpy as np

from torchvision import models

import torch.nn as nn
import time


def check_and_convert_to_rgb(img):
    # Check if image is in RGB format, if not convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


transform1 = transforms.Compose([                         
                                 transforms.ToTensor(),                               
                                 transforms.Resize((128, 128)),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                                 
])
model_path = "/model_10.checkpointsckpt"

model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=20,
                                kernel_size=1)#
model.num_classes = 20 #

model.load_state_dict(torch.load(model_path)) 



def IncreaseContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #result = np.hstack((img, enhanced_img))
    return enhanced_img
time_now=time.time()
fram_id=0
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        start_time = time.time() # Lấy thời điểm bắt đầu xử lý khung hình
        OK,frame=cap.read()
        
        if not OK:
            print("Ignoring empty camera frame.")
            continue

        
        
        imgaeResize = IncreaseContrast(frame)
        imgaeRGB = imgaeResize
        imgaeResize.flags.writeable = False
        imgaeRGB.flags.writeable = False
        imgaeRGB = imgaeResize
        results = hands.process(imgaeResize)
        # cv2.imshow("RESIZE ", imgaeResize)
        cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)
        h = cropped_image.shape[0]
        w = cropped_image.shape[1]
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50
                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 
                R = cv2.getRotationMatrix2D(
                    (int(x2), int(y2)), theta, 1)
                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 
        results = hands.process(imgaeRGB)
        cropped_image = cv2.cvtColor(imgaeRGB, cv2.COLOR_BGR2GRAY)
        h = cropped_image.shape[0]
        w = cropped_image.shape[1]
        print("Đưa tay vào đi bạn....!!!!!!!!!")
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 
                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 
                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 
                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 
                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 
                R = cv2.getRotationMatrix2D(
                    (int(x2), int(y2)), theta, 1)
                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)
                point_1 = [x1, y1]
                point_2 = [x2, y2]
                point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int)
                point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int)
                landmarks_selected_align = {
                    "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}
                point_1 = np.array([landmarks_selected_align["x"]
                                    [0], landmarks_selected_align["y"][0]])
                point_2 = np.array([landmarks_selected_align["x"]
                                    [1], landmarks_selected_align["y"][1]])
                uxROI = pixelCoordinatesLandmarkPoint17[0]
                uyROI = pixelCoordinatesLandmarkPoint17[1]
                lxROI = pixelCoordinatesLandmarkPoint5[0]
                lyROI = point_2[1] + 4*(point_2-point_1)[0]//3 
                
                roi_img = align_img[uyROI:lyROI, uxROI:lxROI]
                roi_img = cv2.resize(roi_img, (128,128))
                # roi_img = check_and_convert_to_rgb(roi_img)
                roi_img=cv2.cvtColor(roi_img,cv2.COLOR_RGB2BGR)
                roi_img = transform1(roi_img)
                roi_img = roi_img.unsqueeze(0) # Thêm chiều batch (batch size = 1)
                with torch.no_grad():
                    model.eval()  # Chuyển sang chế độ infereqnce
                    output = model(roi_img) # Dự đoán kết quả
                    probs = nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)
                    class_dict = predicted.item() # Lấy chỉ số của lớp dự đoán được
                cv2.rectangle(imgaeResize, (uxROI, uyROI),
                    (lxROI, lyROI), (10, 255, 15), 2)
                # cv2.putText(roi_img,"nhan",(w,h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,25,255),2)
                # cv2.putText(imgaeResize, str(class_dict[predicted.item()]), (uxROI, uyROI), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 255), 2)
                # cv2.putText(imgaeResize, str(class_dict[(predicted.item())]), (uxROI, uyROI), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 255), 2)
                # text=class_dict[predicted.item()]
                # # 
                # # class_dict =['010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
                #               '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', 
                #               '034', '035', '036', '037', '038', '039', '040', 'DU', 'DUY', 'DUYKHANG', 'HAN', 'HIEU', 'KHOA', 
                #              'NHAN', 'NHI', 'QUAN', 'QUANG', 'TAI', 'THAI', 'THONG', 'TIN', 'TRINH', 'T_QUAN', 'VIET', 'VINH', 'VY']
                
                class_dict=['DU', 'DUY', 'HAN', 'HIEU', 'HUNG', 'KHANG', 'KHOA', 'NHAN', 'NHI', 'QUAN', 'QUANG', 'TAI', 'THAI', 'THONG', 'THUC QUAN', 'TIN', 'TRINH', 'VIET', 'VINH', 'VY']
                confidence = probs[0][predicted].item()
                if confidence >= 0.5:
                    confidence = probs[0][predicted].item() * 100
                    label = "{} {:.2f}%".format(class_dict[predicted.item()], confidence)
                    cv2.putText(imgaeResize, label, (uxROI, uyROI), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 25, 255), 2)
                    
                    print("Kết quả nhận dạng là:",class_dict[(predicted.item())])
                    
                else:
                    cv2.putText(imgaeResize, "UNKNOW", (uxROI, uyROI), cv2.FONT_HERSHEY_DUPLEX ,0.8, (255, 25, 255), 2)
                    print("Không xác định được nhãn") 
            fps=1/(time.time()-start_time)
            cv2.putText(imgaeResize, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Hiển thị FPS trên khung hình
            cv2.imshow('Frame',imgaeResize)
            
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()