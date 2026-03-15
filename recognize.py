import cv2
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

students = pd.read_csv("students.csv")

cam = cv2.VideoCapture(0)

attendance = []

while True:

    ret,img = cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray,1.2,5)

    for(x,y,w,h) in faces:

        id,confidence=recognizer.predict(gray[y:y+h,x:x+w])

        if confidence<70:

            student_record = students.loc[students['id']==id]
            if not student_record.empty:
                name=str(student_record['name'].values[0])
                mssv=str(student_record['mssv'].values[0])
                time=datetime.now().strftime('%H:%M:%S')
                attendance.append([name,mssv,time])
                cv2.putText(img,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(img,f"Unknown ID {id}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        else:

            cv2.putText(img,"Unknown",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("camera",img)

    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()

df=pd.DataFrame(attendance,columns=["Name","MSSV","Time"])

df.to_excel("attendance.xlsx",index=False)

print("Da xuat Excel")