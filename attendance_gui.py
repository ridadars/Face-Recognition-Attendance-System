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

# ================= THEMES (ONLY ADDITION) =================
# Dark theme (your current palette)
DARK_THEME = {
    "BG_MAIN": "#0F172A",
    "BG_CARD": "#1E293B",
    "BTN_NEUTRAL": "#334155",
    "BTN_HOVER": "#475569",
    "ACCENT": "#38BDF8",
    "SUCCESS": "#4ADE80",
    "DANGER": "#FB7185",
    "TEXT_PRIMARY": "#E5E7EB",
    "TEXT_SECONDARY": "#94A3B8",
}

# Light theme (neutral elegance vibe you asked)
LIGHT_THEME = {
    "BG_MAIN": "#FFF7F2",
    "BG_CARD": "#FFD8BB",
    "BTN_NEUTRAL": "#CCBEB1",
    "BTN_HOVER": "#D6CABF",
    "ACCENT": "#997E67",
    "SUCCESS": "#86EFAC",
    "DANGER": "#FCA5A5",
    "TEXT_PRIMARY": "#664930",
    "TEXT_SECONDARY": "#997E67",
}

# Active theme (start with dark)
THEME = DARK_THEME.copy()

# ================= UI PALETTE (kept as-is, but now set from THEME) =================
BG_MAIN = THEME["BG_MAIN"]
BG_CARD = THEME["BG_CARD"]
BTN_NEUTRAL = THEME["BTN_NEUTRAL"]
BTN_HOVER = THEME["BTN_HOVER"]
ACCENT = THEME["ACCENT"]
SUCCESS = THEME["SUCCESS"]
DANGER = THEME["DANGER"]
TEXT_PRIMARY = THEME["TEXT_PRIMARY"]
TEXT_SECONDARY = THEME["TEXT_SECONDARY"]

# ================= THEME UI REFERENCES (ONLY ADDITION) =================
_main_card = None
_main_title = None
_theme_btn = None
_registered_buttons = []  # list of (button, base_bg)
_login_card = None
_login_title = None
_login_theme_btn = None
_login_registered_buttons = []  # list of (button, base_bg)

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

# ================= THEME APPLY (ONLY ADDITION) =================
def _sync_globals_from_theme():
    global BG_MAIN, BG_CARD, BTN_NEUTRAL, BTN_HOVER, ACCENT, SUCCESS, DANGER, TEXT_PRIMARY, TEXT_SECONDARY
    BG_MAIN = THEME["BG_MAIN"]
    BG_CARD = THEME["BG_CARD"]
    BTN_NEUTRAL = THEME["BTN_NEUTRAL"]
    BTN_HOVER = THEME["BTN_HOVER"]
    ACCENT = THEME["ACCENT"]
    SUCCESS = THEME["SUCCESS"]
    DANGER = THEME["DANGER"]
    TEXT_PRIMARY = THEME["TEXT_PRIMARY"]
    TEXT_SECONDARY = THEME["TEXT_SECONDARY"]

def _apply_theme_to_widget_tree(widget):
    """Safely recolor known widgets without changing logic."""
    try:
        # For frames/windows
        if isinstance(widget, (tk.Tk, tk.Toplevel, tk.Frame)):
            # Only update if it was using old theme colors
            widget.configure(bg=BG_MAIN if widget == root else widget.cget("bg"))
    except:
        pass

def _refresh_main_ui_colors():
    # main window + main card
    if root is not None:
        try:
            root.configure(bg=BG_MAIN)
        except:
            pass

    if _main_card is not None:
        try:
            _main_card.configure(bg=BG_CARD)
        except:
            pass

    if _main_title is not None:
        try:
            _main_title.configure(bg=BG_CARD, fg=ACCENT)
        except:
            pass

    # update theme button icon/text
    if _theme_btn is not None:
        try:
            _theme_btn.configure(
                text=("☀ Light" if THEME is DARK_THEME else "🌙 Dark"),
                bg=BTN_NEUTRAL,
                fg=TEXT_PRIMARY,
                activebackground=BTN_HOVER,
                activeforeground=TEXT_PRIMARY
            )
        except:
            pass

    # recolor registered buttons (keep their "meaning" — neutral/success/danger)
    for btn, base in _registered_buttons:
        try:
            # map old base roles to new theme roles
            if base == "NEUTRAL":
                new_bg = BTN_NEUTRAL
            elif base == "SUCCESS":
                new_bg = SUCCESS
            elif base == "DANGER":
                new_bg = DANGER
            else:
                new_bg = BTN_NEUTRAL

            btn.configure(bg=new_bg, fg=TEXT_PRIMARY, activebackground=BTN_HOVER)

            # rebind hover to use current theme hover color
            def on_enter(e, b=btn):
                b.configure(bg=BTN_HOVER)
            def on_leave(e, b=btn, bg_keep=new_bg):
                b.configure(bg=bg_keep)

            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
        except:
            pass

def _refresh_login_ui_colors(win):
    try:
        win.configure(bg=BG_MAIN)
    except:
        pass

    if _login_card is not None:
        try:
            _login_card.configure(bg=BG_CARD)
        except:
            pass

    if _login_title is not None:
        try:
            _login_title.configure(bg=BG_CARD, fg=ACCENT)
        except:
            pass

    if _login_theme_btn is not None:
        try:
            _login_theme_btn.configure(
                text=("☀ Light" if THEME is DARK_THEME else "🌙 Dark"),
                bg=BTN_NEUTRAL,
                fg=TEXT_PRIMARY,
                activebackground=BTN_HOVER,
                activeforeground=TEXT_PRIMARY
            )
        except:
            pass

    for btn, base in _login_registered_buttons:
        try:
            if base == "NEUTRAL":
                new_bg = BTN_NEUTRAL
            elif base == "SUCCESS":
                new_bg = SUCCESS
            elif base == "DANGER":
                new_bg = DANGER
            else:
                new_bg = BTN_NEUTRAL

            btn.configure(bg=new_bg, fg=TEXT_PRIMARY, activebackground=BTN_HOVER)

            def on_enter(e, b=btn):
                b.configure(bg=BTN_HOVER)
            def on_leave(e, b=btn, bg_keep=new_bg):
                b.configure(bg=bg_keep)

            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
        except:
            pass

def toggle_theme(for_login_win=None):
    """Switch between DARK_THEME and LIGHT_THEME and repaint UI."""
    global THEME
    THEME = LIGHT_THEME if THEME is DARK_THEME else DARK_THEME
    _sync_globals_from_theme()

    # repaint whichever screen is alive
    _refresh_main_ui_colors()
    if for_login_win is not None:
        _refresh_login_ui_colors(for_login_win)

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
    global root, _main_card, _main_title, _theme_btn, _registered_buttons
    root=tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("720x720")
    root.configure(bg=BG_MAIN)

    card=tk.Frame(root,bg=BG_CARD)
    card.place(relx=0.5,rely=0.5,anchor="center",width=480,height=620)

    title = tk.Label(card,text="Face Recognition Attendance",
             bg=BG_CARD,fg=ACCENT,font=("Segoe UI",18,"bold"))
    title.pack(pady=20)

    # ===== Theme Toggle (ONLY ADDITION) =====
    # small toggle button top-right inside the card
    toggle_btn = tk.Button(
        card,
        text="☀ Light" if THEME is DARK_THEME else "🌙 Dark",
        command=lambda: toggle_theme(),
        bg=BTN_NEUTRAL,
        fg=TEXT_PRIMARY,
        relief="flat",
        cursor="hand2",
        font=("Segoe UI", 10),
        padx=10, pady=6,
        activebackground=BTN_HOVER,
        activeforeground=TEXT_PRIMARY
    )
    toggle_btn.place(relx=0.92, rely=0.04, anchor="ne")

    # store refs for recolor
    _main_card = card
    _main_title = title
    _theme_btn = toggle_btn
    _registered_buttons = []

    # Buttons (exact same commands, only register for theme recolor)
    b = modern_button(card,"Add Student",add_student_window,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))
    b = modern_button(card,"View Students",view_students_window,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))
    b = modern_button(card,"Capture Dataset",capture_dataset_window,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))
    b = modern_button(card,"Train Model",train_model_gui,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))
    b = modern_button(card,"▶ Start Attendance",start_attendance,SUCCESS); _registered_buttons.append((b, "SUCCESS"))
    b = modern_button(card,"⏹ Stop & Save Attendance",stop_and_save,DANGER); _registered_buttons.append((b, "DANGER"))
    b = modern_button(card,"📊 Attendance History Viewer",attendance_history_window,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))
    b = modern_button(card,"Exit",root.destroy,BTN_NEUTRAL); _registered_buttons.append((b, "NEUTRAL"))

    root.mainloop()

# ================= LOGIN =================
def login_window():
    win=tk.Tk()
    win.title("Login")
    win.geometry("420x300")
    win.configure(bg=BG_MAIN)

    card=tk.Frame(win,bg=BG_CARD)
    card.place(relx=0.5,rely=0.5,anchor="center",width=360,height=220)

    title = tk.Label(card,text="Login",bg=BG_CARD,fg=ACCENT,
             font=("Segoe UI",16,"bold"))
    title.pack(pady=15)

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

    # ===== Theme Toggle on Login (ONLY ADDITION) =====
    global _login_card, _login_title, _login_theme_btn, _login_registered_buttons
    _login_card = card
    _login_title = title
    _login_registered_buttons = []

    login_toggle_btn = tk.Button(
        card,
        text="☀ Light" if THEME is DARK_THEME else "🌙 Dark",
        command=lambda: toggle_theme(for_login_win=win),
        bg=BTN_NEUTRAL,
        fg=TEXT_PRIMARY,
        relief="flat",
        cursor="hand2",
        font=("Segoe UI", 9),
        padx=8, pady=5,
        activebackground=BTN_HOVER,
        activeforeground=TEXT_PRIMARY
    )
    login_toggle_btn.place(relx=0.92, rely=0.06, anchor="ne")
    _login_theme_btn = login_toggle_btn

    b = modern_button(card,"Login",login,SUCCESS); _login_registered_buttons.append((b, "SUCCESS"))

    win.mainloop()

# ================= RUN =================
if __name__=="__main__":
    ensure_files()
    load_students()
    login_window()
