import cv2
import pandas as pd
import datetime
import os
import tkinter as tk
from tkinter import messagebox

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    messagebox.showerror("Error", "Haar Cascade not loaded")
    exit()

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Attendance DataFrame
attendance = pd.DataFrame(columns=["ID", "Date", "Time"])

# Camera variable
cam = None
running = False


def start_attendance():
    global cam, running, attendance
    cam = cv2.VideoCapture(0)
    running = True

    while running:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                date = datetime.date.today().strftime("%d-%m-%Y")
                time = datetime.datetime.now().strftime("%H:%M:%S")

                attendance.loc[len(attendance)] = [id_, date, time]

                cv2.putText(
                    frame,
                    f"ID: {id_}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            stop_attendance()
            break


def stop_attendance():
    global cam, running, attendance
    running = False

    if cam:
        cam.release()

    cv2.destroyAllWindows()

    # Remove duplicate IDs
    attendance.drop_duplicates(subset=["ID"], keep="first", inplace=True)

    # Save to CSV safely
    if os.path.exists("Attendance.csv") and os.path.getsize("Attendance.csv") > 0:
        old = pd.read_csv("Attendance.csv")
        attendance = pd.concat([old, attendance], ignore_index=True)

    attendance.to_csv("Attendance.csv", index=False)

    messagebox.showinfo("Success", "Attendance Marked Successfully!")


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x300")
root.configure(bg="#1e1e1e")

title = tk.Label(
    root,
    text="Face Recognition Attendance",
    font=("Arial", 16, "bold"),
    fg="white",
    bg="#1e1e1e"
)
title.pack(pady=20)

start_btn = tk.Button(
    root,
    text="Start Attendance",
    font=("Arial", 12),
    width=20,
    bg="green",
    fg="white",
    command=start_attendance
)
start_btn.pack(pady=15)

stop_btn = tk.Button(
    root,
    text="Stop & Save",
    font=("Arial", 12),
    width=20,
    bg="red",
    fg="white",
    command=stop_attendance
)
stop_btn.pack(pady=10)

exit_btn = tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    width=20,
    command=root.quit
)
exit_btn.pack(pady=10)

root.mainloop()
