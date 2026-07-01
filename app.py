import os
import uuid
import json
import threading
import traceback
import functools
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, session, redirect, url_for

from detector import process_video
from excel_export import build_excel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
INDEX_FILE = os.path.join(OUTPUT_DIR, 'reports_index.json')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB (kept modest for free-tier 512MB RAM)

# Set these in Render's Environment settings, not in code
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = SECRET_KEY

# In-memory job store: job_id -> {status, progress, error, result_file, video_name}
JOBS = {}


def load_reports_index():
    if not os.path.exists(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def add_to_reports_index(entry):
    reports = load_reports_index()
    reports.insert(0, entry)  # newest first
    with open(INDEX_FILE, 'w') as f:
        json.dump(reports, f, indent=2)


def require_admin(view_func):
    @functools.wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return view_func(*args, **kwargs)
    return wrapped


def process_job(job_id, video_path, original_name, city, garage, auditor, audit_date):
    try:
        JOBS[job_id]['status'] = 'processing'

        def progress_cb(pct):
            JOBS[job_id]['progress'] = pct

        results = process_video(video_path, sample_fps=2, progress_cb=progress_cb)

        out_name = f'{job_id}.xlsx'
        out_path = os.path.join(OUTPUT_DIR, out_name)
        build_excel(results, original_name, out_path, city=city, garage=garage, auditor=auditor, audit_date=audit_date)

        JOBS[job_id]['status'] = 'done'
        JOBS[job_id]['progress'] = 100
        JOBS[job_id]['result_file'] = out_name
        JOBS[job_id]['count'] = len(results)

        add_to_reports_index({
            'job_id': job_id,
            'result_file': out_name,
            'video_name': original_name,
            'city': city,
            'garage': garage,
            'auditor': auditor,
            'date': audit_date,
            'count': len(results),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    city = request.form.get('city', '').strip()
    garage = request.form.get('garage', '').strip()
    auditor = request.form.get('auditor', '').strip()
    audit_date = request.form.get('date', '').strip()
    if not city or not garage or not auditor or not audit_date:
        return jsonify({'error': 'City, Garage/Location, Auditor Name and Date are required'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'Unsupported file type: {ext}. Please upload a video file.'}), 400

    job_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f'{job_id}{ext}')
    f.save(saved_path)

    JOBS[job_id] = {
        'status': 'queued',
        'progress': 0,
        'error': None,
        'result_file': None,
        'video_name': f.filename,
        'count': 0,
    }

    thread = threading.Thread(target=process_job,
                               args=(job_id, saved_path, f.filename, city, garage, auditor, audit_date),
                               daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'error': job['error'],
        'count': job['count'],
    })


@app.route('/download/<job_id>')
def download(job_id):
    job = JOBS.get(job_id)
    if job and job['status'] == 'done':
        path = os.path.join(OUTPUT_DIR, job['result_file'])
        base = os.path.splitext(job['video_name'])[0]
        return send_file(path, as_attachment=True, download_name=f'vehicle_numbers_{base}.xlsx')

    # Fall back to the persistent index (covers server restarts / admin downloads)
    for entry in load_reports_index():
        if entry['job_id'] == job_id:
            path = os.path.join(OUTPUT_DIR, entry['result_file'])
            if os.path.exists(path):
                base = os.path.splitext(entry['video_name'])[0]
                return send_file(path, as_attachment=True, download_name=f'vehicle_numbers_{base}.xlsx')

    return jsonify({'error': 'File not ready'}), 404


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        error = 'Incorrect password.'
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
@require_admin
def admin_dashboard():
    reports = load_reports_index()
    return render_template('admin.html', reports=reports)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
