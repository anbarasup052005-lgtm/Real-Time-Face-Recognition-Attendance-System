import pandas as pd
import os
from datetime import datetime

def mark_attendance(name):
    file = "attendance/attendance.csv"

    if not os.path.exists(file):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_csv(file)

    today = datetime.now().strftime("%Y-%m-%d")

    # 🔥 Prevent duplicate
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        print("⚠️ Already Marked")
        return

    now = datetime.now()
    new_row = {
        "Name": name,
        "Date": today,
        "Time": now.strftime("%H:%M:%S")
    }

    df = df.append(new_row, ignore_index=True)
    df.to_csv(file, index=False)
    print("✅ Attendance Marked")