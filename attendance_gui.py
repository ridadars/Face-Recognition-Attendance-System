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

CONF_THRESHOLD = 65
CAPTURE_COUNT = 40

# ================= GLOBALS =================
students = {}   # {id: (name, class)}
attendance_run_df = pd.DataFrame(columns=["ID", "Name", "Class", "Date", "Time"])
running = False
cam = None
recognizer = None
face_cascade = None
root = None

# ================= UI PALETTE =================
BG_MAIN = "#0F172A"
BG_CARD = "#1E293B"
BTN_NEUTRAL = "#334155"
BTN_HOVER = "#475569"
ACCENT = "#38BDF8"
SUCCESS = "#4ADE80"
DANGER = "#FB7185"
TEXT_PRIMARY = "#E5E7EB"
TEXT_SECONDARY = "#94A3B8"

# ================= FILE SETUP =================
def ensure_files():
    if not os.path.exists(STUDENTS_FILE):
        pd.DataFrame(columns=["ID", "Name", "Class"]).to_csv(STUDENTS_FILE, index=False)

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=["ID", "Name", "Class", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs("trainer", exist_ok=True)

    if not os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, "w") as f:
            json.dump({"users":[{"username":"admin","password":"admin123"}]}, f)

def load_students():
    global students
    df = pd.read_csv(STUDENTS_FILE)
    if "Class" not in df.columns:
        df["Class"] = "General"
        df.to_csv(STUDENTS_FILE, index=False)

    students = {int(r["ID"]):(r["Name"], r["Class"]) for _,r in df.iterrows()}

# ================= MODEL =================
def load_models():
    global face_cascade, recognizer
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists(TRAINER_FILE):
        messagebox.showwarning("Missing", "Train the model first.")
        return False

    recognizer.read(TRAINER_FILE)
    return True

# ================= BUTTON =================
def modern_button(parent, text, cmd, bg):
    btn = tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=TEXT_PRIMARY,
        font=("Segoe UI", 11),
        relief="flat", pady=10, cursor="hand2"
    )

    def on_enter(e): btn.configure(bg=BTN_HOVER)
    def on_leave(e): btn.configure(bg=bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    btn.pack(fill="x", pady=6)
    return btn

# ================= ADD STUDENT =================
def add_student_window():
    win = tk.Toplevel(root)
    win.title("Add Student")
    win.geometry("360x300")
    win.configure(bg=BG_CARD)

    entries = {}
    for lbl in ["Student ID", "Name", "Class"]:
        tk.Label(win, text=lbl, bg=BG_CARD, fg=TEXT_PRIMARY).pack(pady=4)
        e = tk.Entry(win, width=30)
        e.pack()
        entries[lbl] = e

    def save():
        try:
            sid = int(entries["Student ID"].get())
            name = entries["Name"].get()
            cls = entries["Class"].get() or "General"

            df = pd.read_csv(STUDENTS_FILE)
            if sid in df["ID"].values:
                messagebox.showerror("Error","ID exists")
                return

            df.loc[len(df)] = [sid,name,cls]
            df.to_csv(STUDENTS_FILE,index=False)
            load_students()
            messagebox.showinfo("Success","Student added")
            win.destroy()
        except:
            messagebox.showerror("Error","Invalid input")

    modern_button(win,"Save Student",save,SUCCESS)

# ================= VIEW STUDENTS =================
def view_students_window():
    win = tk.Toplevel(root)
    win.title("Students")
    win.geometry("600x400")

    tree = ttk.Treeview(win, columns=("ID","Name","Class"), show="headings")
    for c in ("ID","Name","Class"):
        tree.heading(c,text=c)
        tree.column(c,width=180)
    tree.pack(fill="both",expand=True)

    for sid,(name,cls) in students.items():
        tree.insert("", "end", values=(sid,name,cls))

# ================= DATASET =================
def capture_dataset_window():
    win = tk.Toplevel(root)
    win.title("Capture Dataset")
    win.geometry("300x200")

    tk.Label(win,text="Student ID").pack(pady=10)
    id_e = tk.Entry(win)
    id_e.pack()

    def start():
        sid = int(id_e.get())
        cam = cv2.VideoCapture(0)
        face = cv2.CascadeClassifier(CASCADE_FILE)
        count = 0

        while True:
            ret,frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                count+=1
                cv2.imwrite(f"{DATASET_DIR}/User.{sid}.{count}.jpg", gray[y:y+h,x:x+w])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.imshow("Capture",frame)
            if cv2.waitKey(1)==27 or count>=CAPTURE_COUNT:
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done","Dataset captured")
        win.destroy()

    modern_button(win,"Start Capture",start,SUCCESS)

# ================= TRAIN =================
def train_model_gui():
    subprocess.run(["python","trainer.py"])
    messagebox.showinfo("Done","Model trained")

# ================= ATTENDANCE =================
def start_attendance():
    global cam,running
    if not load_models(): return
    cam = cv2.VideoCapture(0)
    running = True
    process_frame()

def process_frame():
    global running
    if not running: return

    ret,frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2,5)

    for (x,y,w,h) in faces:
        id_,conf = recognizer.predict(gray[y:y+h,x:x+w])
        date = datetime.date.today().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H:%M:%S")

        if id_ in students and conf<CONF_THRESHOLD:
            name,cls = students[id_]
            if not ((attendance_run_df["ID"]==id_) & (attendance_run_df["Date"]==date)).any():
                attendance_run_df.loc[len(attendance_run_df)] = [id_,name,cls,date,time]
            label=f"{name} [{cls}]"
            color=(0,255,0)
        else:
            label="Unknown"
            color=(0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1)==27:
        stop_and_save()
        return

    root.after(10,process_frame)

def stop_and_save():
    global running,attendance_run_df
    running=False
    cam.release()
    cv2.destroyAllWindows()

    if len(attendance_run_df):
        old=pd.read_csv(ATTENDANCE_FILE)
        df=pd.concat([old,attendance_run_df])
        df.drop_duplicates(subset=["ID","Date"],inplace=True)
        df.to_csv(ATTENDANCE_FILE,index=False)

    attendance_run_df=attendance_run_df.iloc[0:0]
    messagebox.showinfo("Saved","Attendance saved")

# ================= HISTORY =================
def attendance_history_window():
    win=tk.Toplevel(root)
    win.title("Attendance History")
    win.geometry("800x420")

    df=pd.read_csv(ATTENDANCE_FILE)
    search=tk.Entry(win,width=30)
    search.pack(pady=6)

    tree=ttk.Treeview(win,columns=df.columns.tolist(),show="headings")
    for c in df.columns:
        tree.heading(c,text=c)
        tree.column(c,width=150)
    tree.pack(fill="both",expand=True)

    def refresh(d):
        tree.delete(*tree.get_children())
        for _,r in d.iterrows():
            tree.insert("", "end", values=list(r))

    refresh(df)

    search.bind("<KeyRelease>", lambda e:
        refresh(df[df.apply(lambda r: search.get().lower() in str(r).lower(), axis=1)])
    )

# ================= MAIN =================
def main_app():
    global root
    root=tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("720x720")
    root.configure(bg=BG_MAIN)

    card=tk.Frame(root,bg=BG_CARD)
    card.place(relx=0.5,rely=0.5,anchor="center",width=480,height=620)

    tk.Label(card,text="Face Recognition Attendance",
             bg=BG_CARD,fg=ACCENT,font=("Segoe UI",18,"bold")).pack(pady=20)

    modern_button(card,"Add Student",add_student_window,BTN_NEUTRAL)
    modern_button(card,"View Students",view_students_window,BTN_NEUTRAL)
    modern_button(card,"Capture Dataset",capture_dataset_window,BTN_NEUTRAL)
    modern_button(card,"Train Model",train_model_gui,BTN_NEUTRAL)
    modern_button(card,"▶ Start Attendance",start_attendance,SUCCESS)
    modern_button(card,"⏹ Stop & Save Attendance",stop_and_save,DANGER)
    modern_button(card,"📊 Attendance History Viewer",attendance_history_window,BTN_NEUTRAL)
    modern_button(card,"Exit",root.destroy,BTN_NEUTRAL)

    root.mainloop()

# ================= LOGIN =================
def login_window():
    win=tk.Tk()
    win.title("Login")
    win.geometry("420x300")
    win.configure(bg=BG_MAIN)

    card=tk.Frame(win,bg=BG_CARD)
    card.place(relx=0.5,rely=0.5,anchor="center",width=360,height=220)

    tk.Label(card,text="Login",bg=BG_CARD,fg=ACCENT,
             font=("Segoe UI",16,"bold")).pack(pady=15)

    user=tk.Entry(card,width=30)
    pwd=tk.Entry(card,show="*",width=30)
    user.pack(pady=5)
    pwd.pack(pady=5)

    def login():
        data=json.load(open(ADMIN_FILE))
        for u in data["users"]:
            if user.get()==u["username"] and pwd.get()==u["password"]:
                win.destroy()
                main_app()
                return
        messagebox.showerror("Error","Invalid credentials")

    modern_button(card,"Login",login,SUCCESS)
    win.mainloop()

# ================= RUN =================
if __name__=="__main__":
    ensure_files()
    load_students()
    login_window()
