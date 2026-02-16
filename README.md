# Face Attendance System

A real-time face recognition-based attendance management system that automatically marks attendance by recognizing faces using OpenCV and LBPH (Local Binary Patterns Histograms) face recognizer.

## Features

- 📷 **Face Detection** - Detects faces in real-time using Haar Cascade classifiers
- 👤 **Face Recognition** - Recognizes registered users using LBPH face recognizer
- 📊 **Automatic Attendance** - Automatically marks attendance with date and time
- 🔐 **User Authentication** - Admin and Teacher login system
- 📁 **CSV Export** - Exports attendance records to CSV format
- 🖼️ **GUI Interface** - User-friendly graphical interface (optional)

## Project Structure

```
Face_Attendance/
├── attendance.py          # Main attendance marking system
├── attendance_gui.py      # GUI-based attendance system
├── dataset_creator.py     # Script to capture face images
├── dataset_gui.py         # GUI for dataset creation
├── trainer.py             # Trains the face recognition model
├── haarcascade_frontalface_default.xml  # Haar Cascade classifier
├── admin.json             # User authentication data
├── students.csv           # Student information
├── Attendance.csv         # Attendance records
├── dataset/               # Directory for captured face images
└── trainer/               # Directory for trained model
    └── trainer.yml        # Trained face recognition model
```

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Pandas
- PIL (Pillow)

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```
bash
pip install opencv-python numpy pandas pillow
```

3. Ensure all project files are in the same directory

## Usage

### Step 1: Create Dataset (Register New Users)

Run the dataset creator to capture face images:
```
bash
python dataset_creator.py
```

Enter the User ID and Name when prompted. The system will capture 30 face images per user.

### Step 2: Train the Model

After capturing images for all users, train the face recognition model:
```
bash
python trainer.py
```

This will create `trainer/trainer.yml` with the trained model.

### Step 3: Mark Attendance

Run the attendance system:
```
bash
python attendance.py
```

The system will:
- Open the webcam
- Detect and recognize faces in real-time
- Automatically mark attendance when a recognized face is detected
- Save attendance to `Attendance.csv`

Press **Enter** to exit the system.

### Alternative: GUI Version

You can also use the GUI versions:
```
bash
python dataset_gui.py   # GUI for dataset creation
python attendance_gui.py # GUI for attendance
```

## How It Works

1. **Face Detection**: Uses Haar Cascade classifiers to detect faces in video frames
2. **Feature Extraction**: LBPH (Local Binary Patterns Histograms) algorithm extracts face features
3. **Recognition**: Compares detected face features with trained model
4. **Attendance**: If confidence score is below 70, marks attendance with ID, date, and time

## Configuration

### Admin Credentials
Default credentials in `admin.json`:
- **Admin**: username: `admin`, password: `admin123`
- **Teacher**: username: `teacher`, password: `teacher123`

### Students Data
Add student information to `students.csv` in the format:
```
ID,Name,Email,Department
1,John Doe,john@example.com,Computer Science
```

## Output

Attendance is saved to `Attendance.csv` with the following format:
```
ID,Date,Time
1,25-01-2025,09:30:15
2,25-01-2025,09:31:22
```

## Troubleshooting

1. **Haar Cascade not loaded**: Ensure `haarcascade_frontalface_default.xml` is in the project directory
2. **Camera not working**: Check if webcam is connected and accessible
3. **Model not found**: Run `trainer.py` before running `attendance.py`
4. **Low recognition accuracy**: Ensure good lighting and clear face images during dataset creation

## License

This project is for educational purposes.

## Author

Created for face recognition-based attendance management.
