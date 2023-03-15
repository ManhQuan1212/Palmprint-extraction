from ctypes import sizeof
from tkinter import W
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import os
option=1
cap=cv2.VideoCapture(0)



def is_valid(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])
    ##### Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr

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

def roiImageFromHand(path_out_img, option, cap):
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    with mp_hands.Hands(
            #model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            if (option == 1): # option 1 is data collection
                valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                # print("self.valueOfImage", self.valueOfImage)
                if (valueOfImage < 31):
                    success, image = cap.read()
                    print(image.shape)
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    try:

                        imgaeResize = IncreaseContrast(image)

                        imageOutput = imgaeResize

                        #cv2.imshow("DEFAULT ", image)
                        cv2.imshow("RESIZE ", imgaeResize)

                        imgaeRGB = imgaeResize
                        imgaeResize.flags.writeable = False
                        imgaeRGB.flags.writeable = False
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(imgaeResize)
                        # print(results)

                        cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)

                        h = cropped_image.shape[0]
                        w = cropped_image.shape[1]
                        if results.multi_hand_landmarks:

                            # loop for get poin 5 9 13 15
                            for hand_landmark in results.multi_hand_landmarks:
                                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)

                                # pixelCoordinatesLandmarkPoint0 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, w, h)
                                # pixelCoordinatesLandmarkPoint3 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[3].x, hand_landmark.landmark[3].y, w, h)
                                # pixelCoordinatesLandmarkPoint7 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[7].x, hand_landmark.landmark[7].y, w, h)
                                # pixelCoordinatesLandmarkPoint20 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[20].x, hand_landmark.landmark[20].y, w, h)

                                # center5 = np.array(
                                #     [np.mean(hand_landmark.landmark[5].x)*w, np.mean(hand_landmark.landmark[5].y)*h]).astype('int32')
                                # center9 = np.array(
                                #     [np.mean(hand_landmark.landmark[9].x)*w, np.mean(hand_landmark.landmark[9].y)*h]).astype('int32')
                                # center13 = np.array(
                                #     [np.mean(hand_landmark.landmark[13].x)*w, np.mean(hand_landmark.landmark[13].y)*h]).astype('int32')
                                # center17 = np.array(
                                #     [np.mean(hand_landmark.landmark[17].x)*w, np.mean(hand_landmark.landmark[17].y)*h]).astype('int32')


                                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50
                                #sau khi cos 4 diem
                                #h, w = cropped_image.shape
                                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 


                                R = cv2.getRotationMatrix2D(
                                    (int(x2), int(y2)), theta, 1)

                                #print(int(x2), int(y2))
                                #print("R", R)
                                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                                imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 

                                #cv2.imshow("imgaeRGB", imgaeRGB)

                        results = hands.process(imgaeRGB)
                        # print(results)

                        cropped_image = cv2.cvtColor(imgaeRGB, cv2.COLOR_BGR2GRAY)

                        h = cropped_image.shape[0]
                        w = cropped_image.shape[1]

                         # print("toi day co duoc khong vay ?")
                        if results.multi_hand_landmarks:

                            # loop for get poin 5 9 13 15
                            for hand_landmark in results.multi_hand_landmarks:
                                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                                # x = [landmark.x for landmark in hand_landmark.landmark]
                                # y = [landmark.y for landmark in hand_landmark.landmark]

                                #----------------add

                                # print("toi day")
                                # pixelCoordinatesLandmarkPoint0 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, w, h)
                                # pixelCoordinatesLandmarkPoint3 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[3].x, hand_landmark.landmark[3].y, w, h)
                                # pixelCoordinatesLandmarkPoint7 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[7].x, hand_landmark.landmark[7].y, w, h)
                                # pixelCoordinatesLandmarkPoint20 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[20].x, hand_landmark.landmark[20].y, w, h)
                                # print("toi day 2")

                                # center5 = np.array(
                                #     [np.mean(hand_landmark.landmark[5].x)*w, np.mean(hand_landmark.landmark[5].y)*h]).astype('int32')
                                # center9 = np.array(
                                #     [np.mean(hand_landmark.landmark[9].x)*w, np.mean(hand_landmark.landmark[9].y)*h]).astype('int32')
                                # center13 = np.array(
                                #     [np.mean(hand_landmark.landmark[13].x)*w, np.mean(hand_landmark.landmark[13].y)*h]).astype('int32')
                                # center17 = np.array(
                                #     [np.mean(hand_landmark.landmark[17].x)*w, np.mean(hand_landmark.landmark[17].y)*h]).astype('int32')

                                #cropped_image = cropped_image[0:pixelCoordinatesLandmarkPoint0[1] + 50, 0:pixelCoordinatesLandmarkPoint5[0] + 100]

                                # distance = abs(pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])


                                # yCrop1 = pixelCoordinatesLandmarkPoint9[1] - distance*6
                                # yCrop2 = pixelCoordinatesLandmarkPoint0[1] + distance
                                # xCrop1 = pixelCoordinatesLandmarkPoint17[0] - distance*3
                                # xCrop2 = pixelCoordinatesLandmarkPoint9[0] + distance*5

                                # if (yCrop1 >= 0):
                                #     yCrop1 = pixelCoordinatesLandmarkPoint9[1] - distance*6
                                # else:
                                #     yCrop1 = 0

                                # if (yCrop2 <= h):
                                #     yCrop2 = pixelCoordinatesLandmarkPoint0[1] + distance
                                # else:
                                #     yCrop2 = h

                                # if (xCrop1 >= 0):
                                #     xCrop1 = pixelCoordinatesLandmarkPoint17[0] - distance*3
                                # else:
                                #     xCrop1 = 0

                                # if (xCrop2 <= w):
                                #     xCrop2 = pixelCoordinatesLandmarkPoint9[0] + distance*5
                                # else:
                                #     xCrop2 = w
   

                                # imgaeRGB = imgaeRGB[yCrop1:yCrop2, xCrop1:xCrop2]
                                

                                # cv2.imshow("imgaeRGB CROP", imgaeRGB)


                                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 
                                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 
                                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 
                                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 
                                #sau khi cos 4 diem
                                #h, w = cropped_image.shape
                                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 

                                R = cv2.getRotationMatrix2D(
                                    (int(x2), int(y2)), theta, 1)

                                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 

                                

                                point_1 = [x1, y1]
                                point_2 = [x2, y2]


                                point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int)
                                point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int)
                                
                                # bien doi
                                landmarks_selected_align = {
                                    "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}

                                point_1 = np.array([landmarks_selected_align["x"]
                                            [0], landmarks_selected_align["y"][0]])
                                point_2 = np.array([landmarks_selected_align["x"]
                                                    [1], landmarks_selected_align["y"][1]])
                                #print(point_1, point_2)
                                # ux = point_1[0]
                                # uy = point_1[1] + (point_2-point_1)[0]//3
                                # lx = point_2[0]
                                # ly = point_2[1] + 4*(point_2-point_1)[0]//3


                                # roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)

                                uxROI = pixelCoordinatesLandmarkPoint17[0]
                                uyROI = pixelCoordinatesLandmarkPoint17[1]
                                lxROI = pixelCoordinatesLandmarkPoint5[0]
                                lyROI = point_2[1] + 4*(point_2-point_1)[0]//3 
                                # print("lyROI", lyROI)

                                # cv2.rectangle(roi_zone_img, (lxROI, lyROI),
                                #     (uxROI, uyROI), (0, 255, 0), 2)

                                # cv2.imshow("roi_zone_img", roi_zone_img)

                                # valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                                path = path_out_img + "/005_" + str(valueOfImage) + ".bmp"


                                roi_img = align_img[uyROI:lyROI, uxROI:lxROI]


                                roi_img = cv2.resize(roi_img, (256,256))


                                #roi_img = LMTrP.LMTRP_processWithImage(roi_img)
                                cv2.imwrite(path, roi_img)
                                
                                
                        if cv2.waitKey(5) & 0xFF == 27:
                            break

                    except:
                        print("loi ROI")
                else:
                    cap.release()
                    break
            else:
                valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                if (valueOfImage <= 10):
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    try:

                        imgaeResize = IncreaseContrast(image)

                        imageOutput = imgaeResize

                        #cv2.imshow("DEFAULT ", image)
                        # cv2.imshow("RESIZE ", imgaeResize)
                        imgaeResize.flags.writeable = False
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(imgaeResize)
                        # print(results)

                        cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)

                        h = cropped_image.shape[0]
                        w = cropped_image.shape[1]
                        if results.multi_hand_landmarks:

                            # loop for get poin 5 9 13 15
                            for hand_landmark in results.multi_hand_landmarks:
                                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                                pixelCoordinatesLandmarkPoint0 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, w, h)
                                # x = [landmark.x for landmark in hand_landmark.landmark]
                                # y = [landmark.y for landmark in hand_landmark.landmark]
                                print(pixelCoordinatesLandmarkPoint5)
                                print(pixelCoordinatesLandmarkPoint17)
                                #print(hand_landmark.INDEX_FINGER_MCP)

                                center5 = np.array(
                                    [np.mean(hand_landmark.landmark[5].x)*w, np.mean(hand_landmark.landmark[5].y)*h]).astype('int32')
                                center9 = np.array(
                                    [np.mean(hand_landmark.landmark[9].x)*w, np.mean(hand_landmark.landmark[9].y)*h]).astype('int32')
                                center13 = np.array(
                                    [np.mean(hand_landmark.landmark[13].x)*w, np.mean(hand_landmark.landmark[13].y)*h]).astype('int32')
                                center17 = np.array(
                                    [np.mean(hand_landmark.landmark[17].x)*w, np.mean(hand_landmark.landmark[17].y)*h]).astype('int32')
                                # # for checking the center
                                cv2.circle(imgaeResize, tuple(center5), 10, (255, 0, 0), 1)
                                cv2.circle(imgaeResize, tuple(center9), 10, (255, 0, 0), 1)
                                cv2.circle(imgaeResize, tuple(center13), 10, (255, 0, 0), 1)
                                cv2.circle(imgaeResize, tuple(center17), 10, (255, 0, 0), 1)

                                cropped_image = cropped_image[0:pixelCoordinatesLandmarkPoint0[1] + 50, 0:pixelCoordinatesLandmarkPoint5[0] + 100]

                                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50
                                #sau khi cos 4 diem
                                #h, w = cropped_image.shape
                                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 

                                if (theta >= -15 and theta < 0):
                                    print("theta", theta)
                                    R = cv2.getRotationMatrix2D(
                                        (int(x2), int(y2)), theta, 1)

                                    #print(int(x2), int(y2))
                                    #print("R", R)
                                    align_img = cv2.warpAffine(cropped_image, R, (w, h))
                                    #cv2.imshow("a",align_img)

                                    point_1 = [x1, y1]
                                    point_2 = [x2, y2]

                                    
                                    #co 2 diem dau vao roi
                                    #print(point_1, point_2)


                                    point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int)
                                    point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int)
                                    
                                    # bien doi
                                    landmarks_selected_align = {
                                        "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}

                                    point_1 = np.array([landmarks_selected_align["x"]
                                                [0], landmarks_selected_align["y"][0]])
                                    point_2 = np.array([landmarks_selected_align["x"]
                                                        [1], landmarks_selected_align["y"][1]])
                                    #print(point_1, point_2)
                                    ux = point_1[0]
                                    uy = point_1[1] + (point_2-point_1)[0]//3
                                    lx = point_2[0]
                                    ly = point_2[1] + 4*(point_2-point_1)[0]//3


                                    roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)
                                    

                                    # self.valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                                    path = path_out_img + "/0001_000" + str(valueOfImage) + ".bmp"

                                    cv2.rectangle(roi_zone_img, (lx, ly),
                                                (ux, uy), (0, 255, 0), 2)

                                    print(uy, ly, ux, lx)

                                    roi_img = align_img[uy:ly + 85, ux:lx + 85]
                                    roi_img = cv2.resize(roi_img, (128, 128))
                                    cv2.imwrite(path, roi_img)
                                    
                    except:
                        print("loi ROI")
                else:
                    cap.release()
                    return 1
            
    #cap.release()
    return 1
def ROI_Extration(img):
    #$=%04d
    img = roiImageFromHand(path_out_img, option, cap)
    return img

path_out_img = "./ROI/"

