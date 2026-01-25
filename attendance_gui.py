import cv2
import pandas as pd
import datetime
import os
import tkinter as tk
from tkinter import messagebox

# ---------------- DATA ----------------
names = {
    1: "Rida"
    # add more later: 2: "Ali", 3: "Bilal"
}

attendance = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])

running = False
cam = None

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# ---------------- FUNCTIONS ----------------
def start_attendance():
    global cam, running
    cam = cv2.VideoCapture(0)
    running = True
    process_frame()

def process_frame():
    global cam, running, attendance

    if not running:
        return

    ret, frame = cam.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        date = datetime.date.today().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H:%M:%S")

        if confidence < 70 and id_ in names:
            name = names[id_]

            # Prevent duplicate per day
            if not ((attendance["ID"] == id_) & (attendance["Date"] == date)).any():
                attendance.loc[len(attendance)] = [id_, name, date, time]

            label = f"{name}"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Attendance", frame)
    cv2.waitKey(1)

    root.after(10, process_frame)  # 🔥 keeps GUI responsive

def stop_and_save():
    global cam, running, attendance
    running = False

    if cam:
        cam.release()
    cv2.destroyAllWindows()

    if os.path.exists("Attendance.csv") and os.path.getsize("Attendance.csv") > 0:
        old = pd.read_csv("Attendance.csv")
        attendance = pd.concat([old, attendance], ignore_index=True)
        attendance.drop_duplicates(subset=["ID", "Date"], inplace=True)

    attendance.to_csv("Attendance.csv", index=False)
    messagebox.showinfo("Saved", "Attendance saved successfully!")

def exit_app():
    stop_and_save()
    root.destroy()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("420x300")
root.configure(bg="#1e1e1e")

tk.Label(root, text="Face Recognition Attendance",
         font=("Arial", 16, "bold"),
         fg="white", bg="#1e1e1e").pack(pady=20)

tk.Button(root, text="Start Attendance",
          font=("Arial", 12),
          width=20,
          bg="green", fg="white",
          command=start_attendance).pack(pady=10)

tk.Button(root, text="Stop & Save",
          font=("Arial", 12),
          width=20,
          bg="red", fg="white",
          command=stop_and_save).pack(pady=10)

tk.Button(root, text="Exit",
          font=("Arial", 12),
          width=20,
          command=exit_app).pack(pady=10)

root.mainloop()
