import cv2
import pandas as pd
import datetime
import os
import json
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess

# ================= CONFIG =================
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "Attendance.csv"
ADMIN_FILE = "admin.json"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINER_FILE = os.path.join("trainer", "trainer.yml")
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"

CONF_THRESHOLD = 70
CAPTURE_COUNT = 30

# ================= GLOBALS =================
students = {}
attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
running = False
cam = None
recognizer = None
face_cascade = None
root = None


# ================= FILE SETUP =================
def ensure_files():
    if not os.path.exists(STUDENTS_FILE):
        pd.DataFrame(columns=["ID", "Name"]).to_csv(STUDENTS_FILE, index=False)
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(TRAINER_DIR):
        os.makedirs(TRAINER_DIR)
    if not os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, "w") as f:
            json.dump({"username": "admin", "password": "admin123"}, f, indent=2)


def load_students():
    global students
    df = pd.read_csv(STUDENTS_FILE)
    if len(df) == 0:
        students = {}
        return
    df["ID"] = df["ID"].astype(int)
    students = dict(zip(df["ID"], df["Name"]))


def save_students(df):
    df.to_csv(STUDENTS_FILE, index=False)
    load_students()


# ================= MODEL =================
def load_models():
    global face_cascade, recognizer
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists(TRAINER_FILE):
        messagebox.showwarning("Model Missing", "Train model first.")
        return False
    recognizer.read(TRAINER_FILE)
    return True


# ================= STUDENT =================
def add_student_window():
    win = tk.Toplevel(root)
    win.title("Add Student")
    win.geometry("300x220")

    tk.Label(win, text="Student ID").pack(pady=6)
    id_entry = tk.Entry(win)
    id_entry.pack()

    tk.Label(win, text="Student Name").pack(pady=6)
    name_entry = tk.Entry(win)
    name_entry.pack()

    def save():
        try:
            sid = int(id_entry.get())
            name = name_entry.get().strip()
            df = pd.read_csv(STUDENTS_FILE)
            if sid in df["ID"].values:
                messagebox.showerror("Error", "ID already exists")
                return
            df.loc[len(df)] = [sid, name]
            save_students(df)
            messagebox.showinfo("Success", "Student added")
            win.destroy()
        except:
            messagebox.showerror("Error", "Invalid input")

    tk.Button(win, text="Save Student", command=save).pack(pady=15)


def view_students_window():
    win = tk.Toplevel(root)
    win.title("Students")
    win.geometry("480x360")

    tree = ttk.Treeview(win, columns=("ID", "Name"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Name")
    tree.pack(fill="both", expand=True)

    def refresh():
        tree.delete(*tree.get_children())
        df = pd.read_csv(STUDENTS_FILE)
        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["ID"], r["Name"]))

    refresh()


# ================= DATASET =================
def capture_dataset_window():
    win = tk.Toplevel(root)
    win.title("Capture Dataset")
    win.geometry("300x200")

    tk.Label(win, text="Student ID").pack(pady=10)
    id_entry = tk.Entry(win)
    id_entry.pack()

    def start():
        sid = int(id_entry.get())
        cam = cv2.VideoCapture(0)
        face = cv2.CascadeClassifier(CASCADE_FILE)
        count = 0

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"{DATASET_DIR}/User.{sid}.{count}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Capturing Dataset", frame)
            if cv2.waitKey(1) == 27 or count >= CAPTURE_COUNT:
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Dataset captured")
        win.destroy()

    tk.Button(win, text="Start Capture", command=start).pack(pady=20)


# ================= TRAIN =================
def train_model_gui():
    subprocess.run(["python", "trainer.py"])
    messagebox.showinfo("Done", "Model trained successfully")


# ================= ATTENDANCE =================
def start_attendance():
    global cam, running
    if not load_models():
        return
    cam = cv2.VideoCapture(0)
    running = True
    process_frame()


def process_frame():
    global running
    if not running:
        return

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if id_ in students and conf < CONF_THRESHOLD:
            name = students[id_]
            date = datetime.date.today().strftime("%d-%m-%Y")
            time = datetime.datetime.now().strftime("%H:%M:%S")

            attendance_run_df.loc[len(attendance_run_df)] = [id_, name, date, time]
            label = name
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) == 27:
        stop_and_save()
        return

    root.after(10, process_frame)


def stop_and_save():
    global running, cam
    running = False
    cam.release()
    cv2.destroyAllWindows()

    if len(attendance_run_df) == 0:
        messagebox.showinfo("Saved", "No attendance detected")
        return

    if os.path.exists(ATTENDANCE_FILE):
        old = pd.read_csv(ATTENDANCE_FILE)
        combined = pd.concat([old, attendance_run_df])
        combined.drop_duplicates(subset=["ID", "Date"], inplace=True)
        combined.to_csv(ATTENDANCE_FILE, index=False)
    else:
        attendance_run_df.to_csv(ATTENDANCE_FILE, index=False)

    attendance_run_df.drop(attendance_run_df.index, inplace=True)
    messagebox.showinfo("Saved", "Attendance saved successfully")


# ================= ANALYTICS DASHBOARD =================
def attendance_analytics_window():
    win = tk.Toplevel(root)
    win.title("Attendance Analytics")
    win.geometry("500x360")

    tk.Label(win, text="Attendance Analytics Dashboard",
             font=("Segoe UI", 16, "bold")).pack(pady=15)

    total_students = len(students)
    total_records = 0
    today_present = 0
    attendance_percent = 0
    top_student = "N/A"

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        total_records = len(df)
        today = datetime.date.today().strftime("%d-%m-%Y")
        today_present = df[df["Date"] == today]["ID"].nunique()

        if total_students > 0:
            attendance_percent = (today_present / total_students) * 100

        if len(df) > 0:
            top_id = df["ID"].value_counts().idxmax()
            top_student = students.get(top_id, f"ID {top_id}")

    stats = [
        ("Total Students", total_students),
        ("Total Attendance Records", total_records),
        ("Present Today", today_present),
        ("Attendance % Today", f"{attendance_percent:.2f}%"),
        ("Most Frequent Student", top_student),
    ]

    for label, value in stats:
        frame = tk.Frame(win)
        frame.pack(pady=6)
        tk.Label(frame, text=f"{label}:", width=25, anchor="w").pack(side="left")
        tk.Label(frame, text=value, font=("Segoe UI", 11, "bold")).pack(side="left")


def attendance_analytics_window():
    win = tk.Toplevel(root)
    win.title("Attendance Analytics Dashboard")
    win.geometry("520x380")
    win.configure(bg="#1e1e2e")

    tk.Label(
        win,
        text="Attendance Analytics Dashboard",
        font=("Segoe UI", 16, "bold"),
        fg="white",
        bg="#1e1e2e"
    ).pack(pady=15)

    # ---------------- DATA CALCULATION ----------------
    total_students = len(students)
    total_records = 0
    today_present = 0
    attendance_percent = 0
    top_student = "N/A"

    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE)

        total_records = len(df)

        today = datetime.date.today().strftime("%d-%m-%Y")
        today_present = df[df["Date"] == today]["ID"].nunique()

        if total_students > 0:
            attendance_percent = (today_present / total_students) * 100

        if len(df) > 0:
            top_id = df["ID"].value_counts().idxmax()
            top_student = students.get(int(top_id), f"ID {top_id}")

    # ---------------- UI CARDS ----------------
    stats = [
        ("Total Students", total_students),
        ("Total Attendance Records", total_records),
        ("Present Today", today_present),
        ("Attendance % Today", f"{attendance_percent:.2f}%"),
        ("Most Frequent Student", top_student)
    ]

    for label, value in stats:
        card = tk.Frame(win, bg="#27293d", padx=15, pady=10)
        card.pack(fill="x", padx=20, pady=6)

        tk.Label(
            card, text=label,
            font=("Segoe UI", 11),
            fg="#cdd6f4",
            bg="#27293d"
        ).pack(side="left")

        tk.Label(
            card, text=value,
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#27293d"
        ).pack(side="right")

# ================= MAIN APP =================
def main_app():
    global root
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("480x580")

    tk.Label(root, text="Face Recognition Attendance",
             font=("Segoe UI", 16, "bold")).pack(pady=15)

    tk.Button(root, text="Add Student", width=30, command=add_student_window).pack(pady=6)
    tk.Button(root, text="View Students", width=30, command=view_students_window).pack(pady=6)
    tk.Button(root, text="Capture Dataset", width=30, command=capture_dataset_window).pack(pady=6)
    tk.Button(root, text="Train Model", width=30, command=train_model_gui).pack(pady=6)
    tk.Button(root, text="Start Attendance", width=30, command=start_attendance).pack(pady=6)
    tk.Button(root, text="Stop & Save Attendance", width=30, command=stop_and_save).pack(pady=6)

    # ⭐ NEW FEATURE BUTTON
    tk.Button(root, text="Attendance Analytics Dashboard", width=30,
              command=attendance_analytics_window).pack(pady=6)

    tk.Button(root, text="Exit", width=30, command=root.destroy).pack(pady=15)
    tk.Button(
    root,
    text="Attendance Analytics Dashboard",
    width=30,
    font=("Segoe UI", 12),
    command=attendance_analytics_window).pack(pady=8)


    root.mainloop()


# ================= LOGIN =================
def login_window():
    win = tk.Tk()
    win.title("Admin Login")
    win.geometry("320x220")

    tk.Label(win, text="Admin Login",
             font=("Segoe UI", 14, "bold")).pack(pady=15)

    tk.Label(win, text="Username").pack()
    user = tk.Entry(win)
    user.pack()

    tk.Label(win, text="Password").pack(pady=6)
    pwd = tk.Entry(win, show="*")
    pwd.pack()

    def login():
        admin = json.load(open(ADMIN_FILE))
        if user.get() == admin["username"] and pwd.get() == admin["password"]:
            win.destroy()
            main_app()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    tk.Button(win, text="Login", command=login).pack(pady=15)
    win.mainloop()


# ================= RUN =================
if __name__ == "__main__":
    ensure_files()
    load_students()
    login_window()
