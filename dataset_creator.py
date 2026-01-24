import cv2
import os

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

user_id = input("Enter User ID: ")
name = input("Enter Name: ")

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Creating Dataset", frame)

    if cv2.waitKey(1) == 13 or count == 30:  # Enter key or 30 images
        break

cam.release()
cv2.destroyAllWindows()

print("Dataset creation completed!")
