import cv2
import pandas as pd
import datetime
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

cam = cv2.VideoCapture(0)

attendance = pd.DataFrame(columns=["ID", "Date", "Time"])

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            date = datetime.date.today().strftime("%d-%m-%Y")
            time = datetime.datetime.now().strftime("%H:%M:%S")

            attendance.loc[len(attendance)] = [id_, date, time]

            cv2.putText(frame, f"ID: {id_}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cam.release()
cv2.destroyAllWindows()

attendance.drop_duplicates(subset=["ID"], keep="first", inplace=True)

if os.path.exists("Attendance.csv"):
    old = pd.read_csv("Attendance.csv")
    attendance = pd.concat([old, attendance])

attendance.to_csv("Attendance.csv", index=False)

print("Attendance marked successfully!")
