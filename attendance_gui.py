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

CONF_THRESHOLD = 65     # lower = stricter (usually improves accuracy)
CAPTURE_COUNT = 40      # more images per person -> better model

# ================= GLOBALS =================
students = {}
attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
running = False
cam = None
recognizer = None
face_cascade = None
root = None
current_role = None   # kept for login display only (not restricting)

# ================= UI STYLES =================
BG = "#1e1e2e"
BTN = "#313244"
BTN_HOVER = "#45475a"
TXT = "#cdd6f4"
ACCENT = "#89b4fa"
DANGER = "#f38ba8"
SUCCESS = "#a6e3a1"


# ================= FILE SETUP =================
def ensure_files():
    if not os.path.exists(STUDENTS_FILE):
        pd.DataFrame(columns=["ID", "Name"]).to_csv(STUDENTS_FILE, index=False)

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs("trainer", exist_ok=True)

    if not os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, "w") as f:
            json.dump({
                "users": [
                    {"username": "admin", "password": "admin123", "role": "admin"},
                    {"username": "teacher", "password": "teacher123", "role": "teacher"}
                ]
            }, f, indent=2)


def load_students():
    global students
    if not os.path.exists(STUDENTS_FILE):
        students = {}
        return

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
    if face_cascade.empty():
        messagebox.showerror("Error", f"Could not load cascade file: {CASCADE_FILE}")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists(TRAINER_FILE):
        messagebox.showwarning("Model Missing", "Train model first.")
        return False

    recognizer.read(TRAINER_FILE)
    return True


# ================= BUTTON HELPER =================
def styled_button(parent, text, cmd, color=BTN):
    btn = tk.Button(
        parent,
        text=text,
        command=cmd,
        bg=color,
        fg=TXT,
        font=("Segoe UI", 11),
        relief="flat",
        width=32,
        pady=8,
        activebackground=BTN_HOVER,
        activeforeground=TXT,
        cursor="hand2"
    )

    # hover effect
    def on_enter(_):
        btn.configure(bg=BTN_HOVER)

    def on_leave(_):
        btn.configure(bg=color)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    btn.pack(pady=7)
    return btn


# ================= STUDENTS =================
def add_student_window():
    win = tk.Toplevel(root)
    win.title("Add Student")
    win.geometry("340x260")
    win.configure(bg=BG)

    tk.Label(win, text="Add Student", bg=BG, fg=ACCENT,
             font=("Segoe UI", 14, "bold")).pack(pady=12)

    form = tk.Frame(win, bg=BG)
    form.pack(pady=5)

    tk.Label(form, text="Student ID", bg=BG, fg=TXT).grid(row=0, column=0, sticky="w", pady=6)
    id_entry = tk.Entry(form, width=25)
    id_entry.grid(row=0, column=1, pady=6)

    tk.Label(form, text="Student Name", bg=BG, fg=TXT).grid(row=1, column=0, sticky="w", pady=6)
    name_entry = tk.Entry(form, width=25)
    name_entry.grid(row=1, column=1, pady=6)

    def save():
        try:
            sid = int(id_entry.get().strip())
            name = name_entry.get().strip()

            if name == "":
                messagebox.showerror("Error", "Name cannot be empty.")
                return

            df = pd.read_csv(STUDENTS_FILE)

            if "ID" in df.columns and len(df) > 0:
                df["ID"] = df["ID"].astype(int)

            if len(df) > 0 and sid in df["ID"].values:
                messagebox.showerror("Error", "ID already exists")
                return

            df.loc[len(df)] = [sid, name]
            df.to_csv(STUDENTS_FILE, index=False)
            load_students()

            messagebox.showinfo("Success", "Student added successfully!")
            win.destroy()
        except:
            messagebox.showerror("Error", "Enter valid numeric ID and name.")

    styled_button(win, "Save Student", save, color=SUCCESS)


def view_students_window():
    win = tk.Toplevel(root)
    win.title("Students")
    win.geometry("520x380")
    win.configure(bg=BG)

    tk.Label(win, text="Students List", bg=BG, fg=ACCENT,
             font=("Segoe UI", 14, "bold")).pack(pady=10)

    tree = ttk.Treeview(win, columns=("ID", "Name"), show="headings", height=12)
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Name")
    tree.column("ID", width=120)
    tree.column("Name", width=350)
    tree.pack(fill="both", expand=True, padx=12, pady=12)

    df = pd.read_csv(STUDENTS_FILE) if os.path.exists(STUDENTS_FILE) else pd.DataFrame(columns=["ID", "Name"])
    if len(df) > 0:
        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["ID"], r["Name"]))


# ================= DATASET =================
def capture_dataset_window():
    win = tk.Toplevel(root)
    win.title("Capture Dataset")
    win.geometry("360x250")
    win.configure(bg=BG)

    tk.Label(win, text="Capture Dataset", bg=BG, fg=ACCENT,
             font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Label(win, text="Student ID", bg=BG, fg=TXT).pack(pady=6)
    id_entry = tk.Entry(win, width=28)
    id_entry.pack()

    def start():
        try:
            sid = int(id_entry.get().strip())
        except:
            messagebox.showerror("Error", "Enter valid numeric Student ID.")
            return

        cam_local = cv2.VideoCapture(0)
        face = cv2.CascadeClassifier(CASCADE_FILE)

        if face.empty():
            messagebox.showerror("Error", "Cascade file not loaded.")
            return

        count = 0

        while True:
            ret, frame = cam_local.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                count += 1
                roi = cv2.GaussianBlur(gray[y:y+h, x:x+w], (5, 5), 0)
                cv2.imwrite(os.path.join(DATASET_DIR, f"User.{sid}.{count}.jpg"), roi)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, f"{count}/{CAPTURE_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Dataset Capture (ESC to stop)", frame)

            if cv2.waitKey(1) == 27 or count >= CAPTURE_COUNT:
                break

        cam_local.release()
        cv2.destroyAllWindows()

        messagebox.showinfo("Done", f"Captured {count} images for ID {sid}")
        win.destroy()

    styled_button(win, "Start Capture", start, color=SUCCESS)


# ================= TRAIN =================
def train_model_gui():
    if not os.path.exists("trainer.py"):
        messagebox.showerror("Error", "trainer.py not found in project folder.")
        return

    try:
        subprocess.run(["python", "trainer.py"], check=True)
        messagebox.showinfo("Done", "Model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Training failed:\n\n{e}")


# ================= ATTENDANCE =================
def start_attendance():
    global cam, running
    if not load_models():
        return

    if len(students) == 0:
        messagebox.showwarning("No Students", "Add students first.")
        return

    cam = cv2.VideoCapture(0)
    running = True
    process_frame()


def process_frame():
    global running

    if not running:
        return

    ret, frame = cam.read()
    if not ret:
        stop_and_save()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        try:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        except:
            continue

        date = datetime.date.today().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H:%M:%S")

        if id_ in students and conf < CONF_THRESHOLD:
            if not ((attendance_run_df["ID"] == id_) & (attendance_run_df["Date"] == date)).any():
                attendance_run_df.loc[len(attendance_run_df)] = [id_, students[id_], date, time]
            label = f"{students[id_]} ({int(conf)})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({int(conf)})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Attendance (ESC to stop)", frame)

    if cv2.waitKey(1) == 27:
        stop_and_save()
        return

    root.after(10, process_frame)


def stop_and_save():
    global running, attendance_run_df

    running = False
    try:
        cam.release()
    except:
        pass
    cv2.destroyAllWindows()

    if len(attendance_run_df) == 0:
        messagebox.showinfo("Saved", "No attendance detected.")
        return

    if os.path.exists(ATTENDANCE_FILE):
        old = pd.read_csv(ATTENDANCE_FILE)
        df = pd.concat([old, attendance_run_df], ignore_index=True)
        df.drop_duplicates(subset=["ID", "Date"], inplace=True)
    else:
        df = attendance_run_df

    df.to_csv(ATTENDANCE_FILE, index=False)
    attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    messagebox.showinfo("Saved", "Attendance saved successfully!")


# ================= ANALYTICS =================
def attendance_analytics_window():
    win = tk.Toplevel(root)
    win.title("Attendance Analytics")
    win.geometry("420x280")
    win.configure(bg=BG)

    tk.Label(win, text="Attendance Analytics Dashboard",
             font=("Segoe UI", 14, "bold"), fg=ACCENT, bg=BG).pack(pady=14)

    total_students = len(students)
    today = datetime.date.today().strftime("%d-%m-%Y")

    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    present_today = df[df["Date"] == today]["ID"].nunique() if len(df) else 0
    percent = (present_today / total_students * 100) if total_students else 0

    stats = [
        ("Total Students", total_students),
        ("Present Today", present_today),
        ("Attendance %", f"{percent:.2f}%")
    ]

    for k, v in stats:
        card = tk.Frame(win, bg="#27293d", padx=14, pady=10)
        card.pack(fill="x", padx=16, pady=6)
        tk.Label(card, text=k, bg="#27293d", fg=TXT, font=("Segoe UI", 11)).pack(side="left")
        tk.Label(card, text=v, bg="#27293d", fg="white", font=("Segoe UI", 11, "bold")).pack(side="right")


# ================= MAIN APP =================
def main_app():
    global root

    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("520x640")
    root.configure(bg=BG)

    tk.Label(root, text="Face Recognition Attendance",
             font=("Segoe UI", 18, "bold"),
             fg=ACCENT, bg=BG).pack(pady=18)

    # NOTE: no buttons disabled now ✅
    styled_button(root, "Add Student", add_student_window)
    styled_button(root, "View Students", view_students_window)
    styled_button(root, "Capture Dataset", capture_dataset_window)
    styled_button(root, "Train Model", train_model_gui)
    styled_button(root, "Start Attendance", start_attendance, color=SUCCESS)
    styled_button(root, "Stop & Save Attendance", stop_and_save, color=DANGER)
    styled_button(root, "Attendance Analytics Dashboard", attendance_analytics_window)
    styled_button(root, "Exit", root.destroy)

    root.mainloop()


# ================= LOGIN =================
def login_window():
    global current_role

    win = tk.Tk()
    win.title("Login")
    win.geometry("340x280")
    win.configure(bg=BG)

    tk.Label(win, text="System Login",
             font=("Segoe UI", 15, "bold"),
             fg=ACCENT, bg=BG).pack(pady=15)

    tk.Label(win, text="Username", fg=TXT, bg=BG).pack()
    user = tk.Entry(win, width=28)
    user.pack(pady=4)

    tk.Label(win, text="Password", fg=TXT, bg=BG).pack(pady=6)
    pwd = tk.Entry(win, show="*", width=28)
    pwd.pack(pady=4)

    def login():
        data = json.load(open(ADMIN_FILE))
        for u in data["users"]:
            if user.get().strip() == u["username"] and pwd.get().strip() == u["password"]:
                current_role = u["role"]  # kept only for record
                win.destroy()
                main_app()
                return
        messagebox.showerror("Error", "Invalid credentials")

    styled_button(win, "Login", login, color=SUCCESS)
    styled_button(win, "Exit", win.destroy, color=DANGER)

    win.mainloop()


# ================= RUN =================
if __name__ == "__main__":
    ensure_files()
    load_students()
    login_window()
