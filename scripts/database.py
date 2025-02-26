import psycopg2
from datetime import datetime
import os


def init_db():
    conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="2572004", host="localhost", port="5432")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (student_id TEXT, date TEXT, in_time TEXT, out_time TEXT, status TEXT, in_photo TEXT, out_photo TEXT,
                  dept TEXT, year TEXT, section TEXT,
                  PRIMARY KEY (student_id, date))''')
    conn.commit()
    conn.close()

def update_attendance(student_id, status, timestamp, photo_path, result, dept, year, section):
    conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="2572004", host="localhost", port="5432")
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute("SELECT * FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
    entry = c.fetchone()
    
    # Default status is 'undefined' if no entry exists
    if not entry and result in ["present", "pending_manual"]:
        c.execute("INSERT INTO attendance (student_id, date, status, dept, year, section) VALUES (%s, %s, %s, %s, %s, %s)",
                  (student_id, today, "undefined", dept, year, section))
    
    if result == "present":
        if status == "in":
            c.execute("UPDATE attendance SET in_time = %s, in_photo = %s, status = %s WHERE student_id = %s AND date = %s",
                      (timestamp, photo_path, result, student_id, today))
        elif status == "out":
            c.execute("UPDATE attendance SET out_time = %s, out_photo = %s, status = %s WHERE student_id = %s AND date = %s",
                      (timestamp, photo_path, result, student_id, today))
    elif result == "pending_manual":
        c.execute("UPDATE attendance SET status = %s WHERE student_id = %s AND date = %s",
                  (result, student_id, today))
    
    conn.commit()
    conn.close()

def get_attendance():
    conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="2572004", host="localhost", port="5432")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    rows = c.fetchall()
    conn.close()
    return rows

# Placeholder for time-limit absent marking (off for now)
def mark_absent_after_timeout():
    # Disabled - Enable later
    pass
    # conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="admin123", host="localhost", port="5432")
    # c = conn.cursor()
    # today = datetime.now().strftime('%Y-%m-%d')
    # c.execute("UPDATE attendance SET status = 'absent' WHERE status = 'undefined' AND date = %s", (today,))
    # conn.commit()
    # conn.close()

if __name__ == "__main__":
    init_db()
    print(get_attendance())