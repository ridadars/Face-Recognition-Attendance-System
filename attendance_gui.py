import cv2
import pandas as pd
import datetime
import os
import json
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess

# ---------------- CONFIG ----------------
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "Attendance.csv"
ADMIN_FILE = "admin.json"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINER_FILE = os.path.join("trainer", "trainer.yml")
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"

CONF_THRESHOLD = 70      # lower = stricter matching
CAPTURE_COUNT = 30       # images per student for dataset capture

# ---------------- GLOBALS ----------------
students = {}            # {id:int -> name:str}
attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
running = False
cam = None
recognizer = None
face_cascade = None
root = None


# ---------------- UTIL: FILE SETUP ----------------
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

    # If file exists but empty/corrupt, recreate with headers
    try:
        df = pd.read_csv(STUDENTS_FILE)
    except:
        df = pd.DataFrame(columns=["ID", "Name"])
        df.to_csv(STUDENTS_FILE, index=False)

    if "ID" not in df.columns or "Name" not in df.columns:
        df = pd.DataFrame(columns=["ID", "Name"])
        df.to_csv(STUDENTS_FILE, index=False)

    if len(df) == 0:
        students = {}
        return

    df["ID"] = df["ID"].astype(int)
    students = dict(zip(df["ID"], df["Name"]))


def save_students(df: pd.DataFrame):
    df.to_csv(STUDENTS_FILE, index=False)
    load_students()


# ---------------- UTIL: MODEL SETUP ----------------
def load_models():
    global face_cascade, recognizer

    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    if face_cascade.empty():
        messagebox.showerror("Error", f"Could not load Haar cascade: {CASCADE_FILE}")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists(TRAINER_FILE):
        messagebox.showwarning("Model Missing", "trainer/trainer.yml not found. Train model first.")
        return False

    recognizer.read(TRAINER_FILE)
    return True


# ---------------- STUDENT CRUD ----------------
def add_student_window():
    win = tk.Toplevel(root)
    win.title("Add Student")
    win.geometry("320x220")
    win.configure(bg="#1e1e1e")

    tk.Label(win, text="Student ID (number)", fg="white", bg="#1e1e1e").pack(pady=8)
    id_entry = tk.Entry(win)
    id_entry.pack()

    tk.Label(win, text="Student Name", fg="white", bg="#1e1e1e").pack(pady=8)
    name_entry = tk.Entry(win)
    name_entry.pack()

    def save_new():
        try:
            sid = int(id_entry.get().strip())
            name = name_entry.get().strip()
            if name == "":
                raise ValueError

            df = pd.read_csv(STUDENTS_FILE)
            if len(df) > 0:
                df["ID"] = df["ID"].astype(int)

            if sid in df["ID"].values:
                messagebox.showerror("Error", "Student ID already exists.")
                return

            df.loc[len(df)] = [sid, name]
            save_students(df)
            messagebox.showinfo("Success", f"Student added: {sid} - {name}")
            win.destroy()
        except:
            messagebox.showerror("Error", "Please enter a valid numeric ID and name.")

    tk.Button(win, text="Save Student", command=save_new, bg="#007acc", fg="white", width=18).pack(pady=15)


def view_students_window():
    win = tk.Toplevel(root)
    win.title("Students List")
    win.geometry("520x380")

    tree = ttk.Treeview(win, columns=("ID", "Name"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Name")
    tree.column("ID", width=120)
    tree.column("Name", width=360)
    tree.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_table():
        for item in tree.get_children():
            tree.delete(item)

        df2 = pd.read_csv(STUDENTS_FILE)
        if len(df2) > 0:
            df2["ID"] = df2["ID"].astype(int)
        for _, row in df2.iterrows():
            tree.insert("", "end", values=(row["ID"], row["Name"]))

    refresh_table()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=8)

    def get_selected_id():
        sel = tree.selection()
        if not sel:
            return None
        values = tree.item(sel[0], "values")
        return int(values[0])

    def delete_selected():
        sid = get_selected_id()
        if sid is None:
            messagebox.showwarning("Select", "Please select a student first.")
            return

        if not messagebox.askyesno("Confirm", f"Delete student ID {sid}?"):
            return

        df2 = pd.read_csv(STUDENTS_FILE)
        if len(df2) > 0:
            df2["ID"] = df2["ID"].astype(int)
        df2 = df2[df2["ID"] != sid]
        save_students(df2)
        refresh_table()
        messagebox.showinfo("Deleted", f"Student {sid} deleted.")

    def edit_selected():
        sid = get_selected_id()
        if sid is None:
            messagebox.showwarning("Select", "Please select a student first.")
            return

        df2 = pd.read_csv(STUDENTS_FILE)
        if len(df2) > 0:
            df2["ID"] = df2["ID"].astype(int)

        current_name = df2.loc[df2["ID"] == sid, "Name"].values[0]

        edit_win = tk.Toplevel(win)
        edit_win.title("Edit Student")
        edit_win.geometry("300x180")

        tk.Label(edit_win, text=f"Editing ID: {sid}").pack(pady=10)
        name_entry = tk.Entry(edit_win)
        name_entry.insert(0, current_name)
        name_entry.pack()

        def save_edit():
            new_name = name_entry.get().strip()
            if new_name == "":
                messagebox.showerror("Error", "Name cannot be empty.")
                return
            df2.loc[df2["ID"] == sid, "Name"] = new_name
            save_students(df2)
            refresh_table()
            messagebox.showinfo("Saved", "Student updated.")
            edit_win.destroy()

        tk.Button(edit_win, text="Save", command=save_edit).pack(pady=12)

    ttk.Button(btn_frame, text="Edit Selected", command=edit_selected).grid(row=0, column=0, padx=8)
    ttk.Button(btn_frame, text="Delete Selected", command=delete_selected).grid(row=0, column=1, padx=8)
    ttk.Button(btn_frame, text="Refresh", command=refresh_table).grid(row=0, column=2, padx=8)


# ---------------- ATTENDANCE VIEWER (GUI) ----------------
def view_attendance_window():
    win = tk.Toplevel(root)
    win.title("Attendance Records")
    win.geometry("700x420")

    # Top: Filter controls
    top = tk.Frame(win)
    top.pack(pady=8)

    tk.Label(top, text="Filter by Date (DD-MM-YYYY):").pack(side="left", padx=5)
    date_entry = tk.Entry(top)
    date_entry.pack(side="left", padx=5)

    # Table
    columns = ("ID", "Name", "Date", "Time")
    tree = ttk.Treeview(win, columns=columns, show="headings")

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=160)

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    def load_data(filter_date=None):
        for row in tree.get_children():
            tree.delete(row)

        if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
            return

        df = pd.read_csv(ATTENDANCE_FILE)

        # Safety if file missing expected columns
        needed = {"ID", "Name", "Date", "Time"}
        if not needed.issubset(set(df.columns)):
            return

        if filter_date:
            df = df[df["Date"] == filter_date]

        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["ID"], r["Name"], r["Date"], r["Time"]))

    def apply_filter():
        d = date_entry.get().strip()
        load_data(d)

    def refresh():
        date_entry.delete(0, tk.END)
        load_data()

    load_data()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=8)

    ttk.Button(btn_frame, text="Apply Filter", command=apply_filter).grid(row=0, column=0, padx=10)
    ttk.Button(btn_frame, text="Refresh", command=refresh).grid(row=0, column=1, padx=10)


# ---------------- DATASET CAPTURE (GUI) ----------------
def capture_dataset_window():
    win = tk.Toplevel(root)
    win.title("Capture Dataset")
    win.geometry("340x240")
    win.configure(bg="#1e1e1e")

    tk.Label(win, text="Capture Dataset (30 images)", fg="white", bg="#1e1e1e",
             font=("Arial", 12, "bold")).pack(pady=10)

    tk.Label(win, text="Student ID (must exist in students list)", fg="white", bg="#1e1e1e").pack(pady=6)
    id_entry = tk.Entry(win)
    id_entry.pack()

    def start_capture():
        try:
            sid = int(id_entry.get().strip())
        except:
            messagebox.showerror("Error", "Enter a valid numeric student ID.")
            return

        if sid not in students:
            messagebox.showerror("Error", "Student ID not found. Add student first.")
            return

        face = cv2.CascadeClassifier(CASCADE_FILE)
        if face.empty():
            messagebox.showerror("Error", "Haar cascade not loaded for capture.")
            return

        cam_local = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cam_local.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(os.path.join(DATASET_DIR, f"User.{sid}.{count}.jpg"), gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, f"Capturing: {count}/{CAPTURE_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Dataset Capture (ESC to stop)", frame)

            key = cv2.waitKey(1)
            if key == 27 or count >= CAPTURE_COUNT:
                break

        cam_local.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", f"Captured {count} images for ID {sid}")
        win.destroy()

    tk.Button(win, text="Start Capture", command=start_capture, bg="green", fg="white", width=18).pack(pady=15)


# ---------------- TRAIN MODEL (GUI) ----------------
def train_model_gui():
    if not os.path.exists("trainer.py"):
        messagebox.showerror("Error", "trainer.py not found in project folder.")
        return

    try:
        subprocess.run(["python", "trainer.py"], check=True)
        messagebox.showinfo("Success", "Model training completed! (trainer/trainer.yml updated)")
    except Exception as e:
        messagebox.showerror("Error", f"Training failed.\n\n{e}")


# ---------------- ATTENDANCE (GUI) ----------------
def start_attendance():
    global cam, running, attendance_run_df

    if len(students) == 0:
        messagebox.showwarning("No Students", "Please add students first.")
        return

    if not load_models():
        return

    cam = cv2.VideoCapture(0)
    running = True
    process_frame()


def process_frame():
    global cam, running, attendance_run_df

    if not running:
        return

    ret, frame = cam.read()
    if not ret:
        stop_and_save()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        try:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        except:
            continue

        date = datetime.date.today().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H:%M:%S")

        if confidence < CONF_THRESHOLD and id_ in students:
            name = students[id_]

            # prevent duplicate per day in current run
            if not ((attendance_run_df["ID"] == id_) & (attendance_run_df["Date"] == date)).any():
                attendance_run_df.loc[len(attendance_run_df)] = [id_, name, date, time]

            label = f"{name}"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Attendance (Press ESC to stop)", frame)

    if cv2.waitKey(1) == 27:  # ESC
        stop_and_save()
        return

    root.after(10, process_frame)


def stop_and_save():
    global cam, running, attendance_run_df

    running = False
    if cam:
        cam.release()
    cv2.destroyAllWindows()

    if len(attendance_run_df) == 0:
        messagebox.showinfo("Saved", "No recognized students detected this run.")
        return

    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        old = pd.read_csv(ATTENDANCE_FILE)
        combined = pd.concat([old, attendance_run_df], ignore_index=True)
        combined.drop_duplicates(subset=["ID", "Date"], keep="first", inplace=True)
        combined.to_csv(ATTENDANCE_FILE, index=False)
    else:
        attendance_run_df.drop_duplicates(subset=["ID", "Date"], keep="first", inplace=True)
        attendance_run_df.to_csv(ATTENDANCE_FILE, index=False)

    attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    messagebox.showinfo("Saved", "Attendance saved successfully!")


# ---------------- EXPORT EXCEL ----------------
def export_excel():
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        messagebox.showwarning("No Data", "Attendance.csv not found or empty.")
        return

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        df.to_excel("Attendance.xlsx", index=False)
        messagebox.showinfo("Exported", "Exported to Attendance.xlsx successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Excel export failed.\n\n{e}")


# ---------------- MAIN DASHBOARD ----------------
def main_app():
    global root
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("460x560")
    root.configure(bg="#1e1e1e")

    tk.Label(root, text="Face Recognition Attendance",
             font=("Arial", 16, "bold"), fg="white", bg="#1e1e1e").pack(pady=16)

    tk.Button(root, text="Add Student", width=28, font=("Arial", 12),
              bg="#007acc", fg="white", command=add_student_window).pack(pady=8)

    tk.Button(root, text="View / Edit / Delete Students", width=28, font=("Arial", 12),
              command=view_students_window).pack(pady=8)

    tk.Button(root, text="Capture Dataset (GUI)", width=28, font=("Arial", 12),
              bg="green", fg="white", command=capture_dataset_window).pack(pady=8)

    tk.Button(root, text="Train Model", width=28, font=("Arial", 12),
              command=train_model_gui).pack(pady=8)

    tk.Button(root, text="Start Attendance", width=28, font=("Arial", 12),
              bg="orange", fg="black", command=start_attendance).pack(pady=8)

    tk.Button(root, text="Stop & Save Attendance", width=28, font=("Arial", 12),
              bg="red", fg="white", command=stop_and_save).pack(pady=8)

    # ✅ NEW BUTTON
    tk.Button(root, text="View Attendance", width=28, font=("Arial", 12),
              command=view_attendance_window).pack(pady=8)

    tk.Button(root, text="Export Attendance to Excel", width=28, font=("Arial", 12),
              command=export_excel).pack(pady=8)

    tk.Button(root, text="Exit", width=28, font=("Arial", 12),
              command=lambda: root.destroy()).pack(pady=12)

    root.mainloop()


# ---------------- LOGIN WINDOW ----------------
def login_window():
    login_win = tk.Tk()
    login_win.title("Admin Login")
    login_win.geometry("320x220")
    login_win.configure(bg="#1e1e1e")

    tk.Label(login_win, text="Admin Login", font=("Arial", 14, "bold"),
             fg="white", bg="#1e1e1e").pack(pady=12)

    tk.Label(login_win, text="Username", fg="white", bg="#1e1e1e").pack()
    user_entry = tk.Entry(login_win)
    user_entry.pack()

    tk.Label(login_win, text="Password", fg="white", bg="#1e1e1e").pack(pady=6)
    pass_entry = tk.Entry(login_win, show="*")
    pass_entry.pack()

    def login():
        username = user_entry.get().strip()
        password = pass_entry.get().strip()

        with open(ADMIN_FILE, "r") as f:
            admin = json.load(f)

        if username == admin["username"] and password == admin["password"]:
            login_win.destroy()
            main_app()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    tk.Button(login_win, text="Login", command=login, width=16,
              bg="#007acc", fg="white").pack(pady=14)

    login_win.mainloop()


# ---------------- RUN ----------------
if __name__ == "__main__":
    ensure_files()
    load_students()
    login_window()
