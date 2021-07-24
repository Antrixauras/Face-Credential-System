import cv2
import csv
from tkinter import *
import os
import numpy as np
from PIL import Image

root=Tk()
root.title('Attendance Management')
root.geometry('450x250')
dict={}

def new_entry():
    faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    cnt=0
    try:
        Id = int(roll_no_entry.get())
        name=name_entry.get()
        dict[Id]=name
        print(Id)
        if name=='':
            res = 'Please enter name!!!'
            Notification.configure(text=res, bg="SpringGreen3", font=('times', 18, 'bold'))
            Notification.place(x=100, y=100)
        else:
            if not os.path.exists('./dataset_attend'):
                os.makedirs('./dataset_attend')
            while True:
                success,img=cap.read()
                if success:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces=faceCascade.detectMultiScale(gray,1.1,4)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                        cnt=cnt+1
                        cv2.imwrite("dataset_attend/ " + str(Id) + "." +name + "."+str(cnt)+".jpg",gray[y:y + h, x:x + w])
                    cv2.imshow("Face Detect",img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    elif cnt>20:
                        break

            # with open('data_csv.csv', 'a') as data_csv:
            #     writer = csv.writer(data_csv)
            #     writer.writerow((Id,cnt))
            # data_csv.close()

            print('Done')
            res = 'Successful !!!'
            Notification.configure(text=res, bg="SpringGreen3", font=('times', 18, 'bold'))
            Notification.place(x=100, y=100)
            cap.release()
            cv2.destroyAllWindows()

            if not os.path.exists('./trainner'):
                os.makedirs('./trainner')
    except Exception as e:
        print('Error')

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[0])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
        return faceSamples, Ids

    faces, Ids = getImagesAndLabels('dataset_attend')
    recognizer.train(faces, np.array(Ids))
    recognizer.save('trainner/trainner.yml')
    res='Successfully trained !!!'
    Notification.configure(text=res, bg="SpringGreen3", font=('times', 18, 'bold'))
    Notification.place(x=100, y=80)

def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainner/trainner.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = cv2.VideoCapture(0)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf<50:
                nm=dict[Id]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
                cv2.putText(im,nm, (x, y - 40), font, 2, (255, 255, 255), 3)
        cv2.imshow('recognize', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

name=Label(root,text='Enter name :',bg='black',fg='white')
name.place(x=50,y=30)
name_entry=Entry(root,width=30)
name_entry.place(x=150,y=30)

roll_no=Label(root,text='Enter roll no :',bg='black',fg='white')
roll_no.place(x=50,y=60)
roll_no_entry=Entry(root,width=30)
roll_no_entry.place(x=150,y=60)

new_reg=Button(root,text='New Registration',bg='orange',fg='white',command=new_entry)
new_reg.place(x=50,y=150)


Notification = Label(root, text="notify", bg="Green", fg="white", width=15,height=1, font=('times', 17, 'bold'))

train_reg=Button(root,text='Train Images',bg='orange',fg='white',command=train_images)
train_reg.place(x=190,y=150)

fill_attend=Button(root,text='Fill attendance',bg='orange',fg='white',command=recognize)
fill_attend.place(x=300,y=150)

root.mainloop()