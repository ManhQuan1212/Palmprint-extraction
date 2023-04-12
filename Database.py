import firebase_admin
import os

from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate('KLTN_final\plamprint-database-firebase-adminsdk-a2ub7-43d3061c7e.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://plamprint-database-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
import os

def getlabel(path):
    #directory = os.path.join("KLTN_final", "DATA3", "train", )
    label = {}
    i = 1
    for file in os.listdir(path):
        label[file] = i
        i = i + 1
def Update_database(label,new_people):
    directory = os.path.join("KLTN_final", "DATA3", "train", )
    label = {}
    i = 1
    for file in os.listdir(directory):
        label[file] = i
        i = i + 1
    label[new_people] = len(label) + 1
    ref = db.reference('/')
    ref.set(label)

