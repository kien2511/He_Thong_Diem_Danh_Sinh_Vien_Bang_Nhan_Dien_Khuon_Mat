from flask import Flask, render_template, Response, request, jsonify, send_file
import os
import subprocess
import sys
import database
from camera import camera

app = Flask(__name__)

# Initialize database on startup
database.init_db()

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send a dummy black frame or just sleep
            import time
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/toggle', methods=['POST'])
def toggle_camera():
    data = request.json
    action = data.get('action') # "start" or "stop"
    mode = data.get('mode', 'RECOGNIZE') # "RECOGNIZE" or "CAPTURE"
    student_id = data.get('student_id')
    
    if action == "start":
        camera.start(mode=mode, student_id=student_id)
        return jsonify({"status": "started", "mode": mode})
    elif action == "stop":
        camera.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Invalid action"}), 400

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    return jsonify(camera.get_status())

@app.route('/api/students', methods=['GET', 'POST'])
def handle_students():
    if request.method == 'POST':
        data = request.json
        name = data.get('name')
        mssv = data.get('mssv')
        if not name or not mssv:
            return jsonify({"error": "Name and MSSV required"}), 400
            
        student_id = database.add_student(name, mssv)
        
        # Optionally, auto-start camera in capture mode here:
        camera.start(mode="CAPTURE", student_id=student_id)
        
        return jsonify({"message": "Student added successfully", "id": student_id, "action": "capturing_started"}), 201
    else:
        students = database.get_all_students()
        return jsonify(students)

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    success = database.delete_student(student_id)
    if success:
        return jsonify({"message": "Đã xoá sinh viên thành công"})
    return jsonify({"error": "Lỗi xoá sinh viên"}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    # Run training logic in subprocess to not block Flask totally, or block and return when done
    subprocess.run([sys.executable, "train.py"])
    # Reload model in camera
    camera.load_model()
    return jsonify({"message": "Model trained successfully"})

@app.route('/api/attendance/logs', methods=['GET'])
def get_live_logs():
    return jsonify(camera.get_logs())

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(database.get_attendance_stats())

@app.route('/api/attendance/date/<date_str>', methods=['GET'])
def get_attendance_by_date(date_str):
    records = database.get_attendance_by_date(date_str)
    return jsonify(records)

@app.route('/api/export', methods=['GET'])
def export_excel():
    date_str = request.args.get('date')
    if not date_str:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    filename = f"attendance_{date_str}.xlsx"
    success = database.export_attendance_to_excel(date_str, filename)
    
    if success and os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "No records found for this date to export.", 404

if __name__ == "__main__":
    app.run(debug=True, threaded=True)