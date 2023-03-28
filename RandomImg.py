import os
import random
imgExtension = ["png", "jpeg", "bmp"] 
allImages = list()  
def chooseRandomImage(directory="Output"):
    for img in os.listdir(directory): #Lists all files
        ext = img.split(".")[len(img.split(".")) - 1]
        if (ext in imgExtension):
            allImages.append(img)
    choice = random.randint(0, len(allImages) - 1)
    chosenImage = allImages[choice] #Do Whatever you want with the image file
    return chosenImage

def Removefile():
    path = './Output/'
    for file_name in os.listdir(path):
        file = path + file_name
        if os.path.isfile(file):
            print('Deleting file:', file)
            os.remove(file)
Removefile()