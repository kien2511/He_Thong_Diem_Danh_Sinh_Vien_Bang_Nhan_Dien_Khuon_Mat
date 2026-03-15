import sqlite3
from datetime import datetime
import pandas as pd

DB_FILE = 'attendance.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mssv TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            time TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()

def add_student(name, mssv):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO students (name, mssv) VALUES (?, ?)", (name, mssv))
    student_id = c.lastrowid
    conn.commit()
    conn.close()
    return student_id

def get_all_students():
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    return [dict(ix) for ix in students]

def get_student_by_id(student_id):
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (student_id,)).fetchone()
    conn.close()
    return dict(student) if student else None

def delete_student(student_id):
    conn = get_db_connection()
    c = conn.cursor()
    # Delete attendance records first due to foreign key
    c.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    # Delete student
    c.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    
    # Delete all capture images in dataset/ directory
    import glob
    import os
    file_pattern = f"dataset/User.{student_id}.*.jpg"
    files = glob.glob(file_pattern)
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
            
    return True

def record_attendance(student_id):
    conn = get_db_connection()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Check if already recorded today
    existing = conn.execute('SELECT * FROM attendance WHERE student_id = ? AND date = ?', (student_id, date_str)).fetchone()
    if not existing:
        conn.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)", (student_id, date_str, time_str))
        conn.commit()
        recorded = True
    else:
        recorded = False
    conn.close()
    return recorded

def get_attendance_by_date(date_str):
    conn = get_db_connection()
    query = '''
    SELECT a.time, s.name, s.mssv 
    FROM attendance a 
    JOIN students s ON a.student_id = s.id 
    WHERE a.date = ?
    ORDER BY a.time DESC
    '''
    records = conn.execute(query, (date_str,)).fetchall()
    conn.close()
    return [dict(ix) for ix in records]

def get_attendance_stats():
    # Number of distinct students attended per day for the last 7 days
    conn = get_db_connection()
    query = '''
    SELECT date, COUNT(DISTINCT student_id) as count 
    FROM attendance 
    GROUP BY date 
    ORDER BY date DESC LIMIT 7
    '''
    records = conn.execute(query).fetchall()
    
    total_students = conn.execute('SELECT COUNT(*) as count FROM students').fetchone()['count']
    
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    today_attendance = conn.execute('SELECT COUNT(DISTINCT student_id) as count FROM attendance WHERE date = ?', (today_str,)).fetchone()['count']
    
    conn.close()
    
    return {
        'total_students': total_students,
        'today_attendance': today_attendance,
        'chart_data': [dict(ix) for ix in records][::-1] # reverse to be chronological
    }

def export_attendance_to_excel(date_str, filename):
    records = get_attendance_by_date(date_str)
    if records:
        df = pd.DataFrame(records)
        df.columns = ['Time', 'Name', 'MSSV']
        df.to_excel(filename, index=False)
        return True
    return False

if __name__ == '__main__':
    init_db()
    
    # Import from CSV if students exist and db is empty
    import os
    if os.path.exists('students.csv'):
        try:
            df = pd.read_csv('students.csv')
            conn = get_db_connection()
            count = conn.execute('SELECT COUNT(*) FROM students').fetchone()[0]
            if count == 0 and not df.empty:
                for _, row in df.iterrows():
                    # explicitly insert with id
                    conn.execute("INSERT INTO students (id, name, mssv) VALUES (?, ?, ?)", (row['id'], row['name'], row['mssv']))
                conn.commit()
        except Exception as e:
            print("Error importing csv", e)
        finally:
            if 'conn' in locals(): conn.close()
