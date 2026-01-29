import cv2
import pandas as pd
import datetime
import os
import json
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= CONFIG =================
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "Attendance.csv"
ADMIN_FILE = "admin.json"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINER_FILE = os.path.join("trainer", "trainer.yml")
DATASET_DIR = "dataset"

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

    if not os.path.exists("trainer"):
        os.makedirs("trainer")

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


# ================= STUDENTS =================
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
            df.to_csv(STUDENTS_FILE, index=False)
            load_students()

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

    df = pd.read_csv(STUDENTS_FILE)
    for _, r in df.iterrows():
        tree.insert("", "end", values=(r["ID"], r["Name"]))


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
                cv2.imwrite(f"{DATASET_DIR}/User.{sid}.{count}.jpg",
                            gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Dataset Capture", frame)
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
        date = datetime.date.today().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H:%M:%S")

        if id_ in students and conf < CONF_THRESHOLD:
            if not ((attendance_run_df["ID"] == id_) &
                    (attendance_run_df["Date"] == date)).any():
                attendance_run_df.loc[len(attendance_run_df)] = [
                    id_, students[id_], date, time
                ]
            label = students[id_]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) == 27:
        stop_and_save()
        return

    root.after(10, process_frame)


def stop_and_save():
    global running
    running = False
    cam.release()
    cv2.destroyAllWindows()

    if len(attendance_run_df) == 0:
        messagebox.showinfo("Saved", "No attendance detected")
        return

    if os.path.exists(ATTENDANCE_FILE):
        old = pd.read_csv(ATTENDANCE_FILE)
        df = pd.concat([old, attendance_run_df])
        df.drop_duplicates(subset=["ID", "Date"], inplace=True)
    else:
        df = attendance_run_df

    df.to_csv(ATTENDANCE_FILE, index=False)
    attendance_run_df.drop(attendance_run_df.index, inplace=True)
    messagebox.showinfo("Saved", "Attendance saved")


# ================= ANALYTICS DASHBOARD =================
def attendance_analytics_window():
    win = tk.Toplevel(root)
    win.title("Attendance Analytics")
    win.geometry("420x260")

    tk.Label(win, text="Attendance Analytics Dashboard",
             font=("Segoe UI", 16, "bold")).pack(pady=15)

    total_students = len(students)
    today = datetime.date.today().strftime("%d-%m-%Y")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        present_today = df[df["Date"] == today]["ID"].nunique()
    else:
        present_today = 0

    percent = (present_today / total_students * 100) if total_students else 0

    stats = [
        ("Total Students", total_students),
        ("Present Today", present_today),
        ("Attendance %", f"{percent:.2f}%")
    ]

    for k, v in stats:
        frame = tk.Frame(win)
        frame.pack(pady=6)
        tk.Label(frame, text=f"{k}:", width=18, anchor="w").pack(side="left")
        tk.Label(frame, text=v,
                 font=("Segoe UI", 11, "bold")).pack(side="left")

    tk.Button(
        win,
        text="📊 View Attendance Charts",
        command=attendance_charts_window
    ).pack(pady=15)


# ================= CHARTS WINDOW =================
def attendance_charts_window():
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showwarning("No Data", "No attendance data found")
        return

    df = pd.read_csv(ATTENDANCE_FILE)
    if len(df) == 0:
        messagebox.showwarning("No Data", "Attendance file is empty")
        return

    win = tk.Toplevel(root)
    win.title("Attendance Charts")
    win.geometry("720x520")

    # -------- BAR CHART --------
    date_counts = df.groupby("Date")["ID"].nunique()

    fig, ax = plt.subplots(figsize=(6, 4))
    date_counts.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Attendance Count by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Students Present")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=15)

    # -------- PIE CHART --------
    today = datetime.date.today().strftime("%d-%m-%Y")
    present_today = df[df["Date"] == today]["ID"].nunique()
    absent_today = max(len(students) - present_today, 0)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.pie(
        [present_today, absent_today],
        labels=["Present", "Absent"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.set_title("Today's Attendance")

    canvas2 = FigureCanvasTkAgg(fig2, master=win)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)


# ================= MAIN APP =================
def main_app():
    global root
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("480x600")

    tk.Label(root, text="Face Recognition Attendance",
             font=("Segoe UI", 16, "bold")).pack(pady=15)

    buttons = [
        ("Add Student", add_student_window),
        ("View Students", view_students_window),
        ("Capture Dataset", capture_dataset_window),
        ("Train Model", train_model_gui),
        ("Start Attendance", start_attendance),
        ("Stop & Save Attendance", stop_and_save),
        ("Attendance Analytics Dashboard", attendance_analytics_window),
        ("Exit", root.destroy)
    ]

    for text, cmd in buttons:
        tk.Button(root, text=text, width=30,
                  command=cmd).pack(pady=6)

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
