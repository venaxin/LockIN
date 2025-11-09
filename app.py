from flask import Flask, request, jsonify, redirect, url_for, session, render_template
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import datetime
import sqlite3
import random
import pickle
import json
from dateutil.parser import parse
import google.generativeai as genai

import os
from uuid import uuid4

# Force correct paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "assets"),
    static_url_path="/assets",
)

app.secret_key = os.urandom(24)          # random secret for session
CREDENTIALS_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/calendar']

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


@app.before_request
def ensure_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid4())


# Database setup
conn = sqlite3.connect('todo.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tasks
             (id INTEGER PRIMARY KEY, name TEXT, deadline DATE, days_req INTEGER, hours_daily INTEGER,
              user_id TEXT, completed BOOLEAN DEFAULT 0, missed INTEGER DEFAULT 0)''')
c.execute('''CREATE TABLE IF NOT EXISTS schedules
             (id INTEGER PRIMARY KEY, task_id INTEGER, slot_start DATETIME, slot_end DATETIME, user_id TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS reasons
             (id INTEGER PRIMARY KEY, schedule_id INTEGER, reason TEXT, timestamp DATETIME)''')
c.execute('''CREATE TABLE IF NOT EXISTS feedback
             (id INTEGER PRIMARY KEY, task_id INTEGER, extra_time_needed INTEGER, timestamp DATETIME)''')
conn.commit()

def get_calendar_service():
    if 'credentials' not in session:
        return None
    credentials = pickle.loads(session['credentials'])
    return build('calendar', 'v3', credentials=credentials)

def oauth_redirect_uri():
    return 'http://localhost:9000/oauth2callback'

@app.route('/')
def index():
    calendar_api_key = os.getenv('CALENDAR_API_KEY')
    if not calendar_api_key:
        app.logger.warning('CALENDAR_API_KEY not set; Google Calendar API calls will fail.')
    return render_template(
        'index.html',
        google_client_id=os.getenv('GOOGLE_CLIENT_ID'),
        calendar_api_key=calendar_api_key,
        google_scopes='https://www.googleapis.com/auth/calendar'
    )
    
@app.route('/debug')
def debug():
    import os
    return {
        "cwd": os.getcwd(),
        "template_folder": app.template_folder,
        "templates_dir": os.path.abspath('templates'),
        "files_in_templates": os.listdir('templates') if os.path.exists('templates') else "NOT FOUND"
    }
@app.route('/auth')
def auth():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    flow.redirect_uri = f"http://{request.host}/oauth2callback"
    auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    # ALLOW HTTP FOR LOCAL DEV
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    flow.redirect_uri = f"http://{request.host}/oauth2callback"
    flow.fetch_token(authorization_response=request.url)
    session['credentials'] = pickle.dumps(flow.credentials)
    return redirect(url_for('index'))

@app.route('/check_auth')
def check_auth():
    if 'credentials' in session:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

# Manual Task Input
@app.route('/add_task', methods=['POST'])
def add_task():
    data = request.json
    name = data['name']
    deadline = parse(data['deadline']).date()
    days_req = int(data['days_req'])
    hours_daily = int(data['hours_daily'])
    user_id = session.get('user_id', 'default')  # Simulate user

    model = genai.GenerativeModel('gemini-1.5-flash')  # Or 'gemini-1.5-pro' for more advanced
    prompt = f"Suggest a buffer time in days for task '{name}' due {deadline} that takes {days_req} days and {hours_daily} hours daily. Also suggest Pomodoro slots."
    response = model.generate_content(prompt)
    buffer_suggestion = response.text  # Parse this as needed, e.g., extract numbers
    buffer_days = int(buffer_suggestion.split()[0])  # Simplified parsing
    days_req += buffer_days

    c.execute("INSERT INTO tasks (name, deadline, days_req, hours_daily, user_id) VALUES (?, ?, ?, ?, ?)",
              (name, deadline, days_req, hours_daily, user_id))
    task_id = c.lastrowid
    conn.commit()

    # Suggest slots (Pomodoro: 25 min work + 5 min break, fit into productive times e.g., 9-5)
    slots = suggest_slots(task_id, deadline, days_req, hours_daily, user_id)
    return jsonify({'task_id': task_id, 'suggested_slots': slots, 'buffer_days': buffer_days})

def suggest_slots(task_id, deadline, days_req, hours_daily, user_id):
    slots = []
    start_date = datetime.date.today()
    for day in range(days_req):
        date = start_date + datetime.timedelta(days=day)
        if date > deadline:
            break
        # Productive slots: e.g., 9-12, 1-5, in Pomodoro chunks
        for hour in [9, 10, 11, 13, 14, 15, 16]:
            slot_start = datetime.datetime.combine(date, datetime.time(hour, 0))
            slot_end = slot_start + datetime.timedelta(hours=1)  # Simplify to 1h slots
            slots.append({'start': slot_start.isoformat(), 'end': slot_end.isoformat()})
    # Prioritize: Closer deadlines first (simulate AI)
    # Integrate with Google Calendar: Check free times
    service = get_calendar_service()
    if service:
        # Fetch events and avoid conflicts (simplified)
        pass
    return slots

@app.route('/get_gapi_token')
def get_gapi_token():
    if 'credentials' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    creds = pickle.loads(session['credentials'])
    return jsonify({
        'access_token': creds.token,
        'expires_in': creds.expiry.timestamp() - datetime.datetime.now().timestamp(),
        'scope': ' '.join(SCOPES)
    })

# User selects slots
@app.route('/add_slots', methods=['POST'])
def add_slots():
    data = request.json
    task_id = data['task_id']
    selected_slots = data['slots']  # list of {'start': iso, 'end': iso}
    user_id = session.get('user_id', 'default')
    for slot in selected_slots:
        start = parse(slot['start'])
        end = parse(slot['end'])
        c.execute("INSERT INTO schedules (task_id, slot_start, slot_end, user_id) VALUES (?, ?, ?, ?)",
                  (task_id, start, end, user_id))
    conn.commit()
    # Add to Google Calendar if integrated
    service = get_calendar_service()
    if service:
        for slot in selected_slots:
            event = {
                'summary': f"Task: {get_task_name(task_id)}",
                'start': {'dateTime': slot['start']},
                'end': {'dateTime': slot['end']}
            }
            service.events().insert(calendarId='primary', body=event).execute()
    return jsonify({'status': 'added'})

def get_task_name(task_id):
    c.execute("SELECT name FROM tasks WHERE id=?", (task_id,))
    return c.fetchone()[0]

# Upcoming deadlines and priorities
@app.route('/get_priorities')
def get_priorities():
    user_id = session.get('user_id', 'default')
    c.execute("SELECT * FROM tasks WHERE user_id=? AND completed=0 ORDER BY deadline ASC", (user_id,))
    tasks = c.fetchall()
    priorities = [{'id': t[0], 'name': t[1], 'deadline': t[2], 'priority': (datetime.date.fromisoformat(t[2]) - datetime.date.today()).days} for t in tasks]
    return jsonify(priorities)

# Analyze user list (simple: count tasks, avg time)
@app.route('/analyze_list')
def analyze_list():
    user_id = session.get('user_id', 'default')
    c.execute("SELECT COUNT(*), AVG(hours_daily) FROM tasks WHERE user_id=?", (user_id,))
    count, avg_hours = c.fetchone()
    return jsonify({'total_tasks': count, 'avg_daily_hours': avg_hours})

# Detect pattern / Procrastination
@app.route('/detect_procrastination')
def detect_procrastination():
    user_id = session.get('user_id', 'default')
    c.execute("SELECT missed FROM tasks WHERE user_id=?", (user_id,))
    misses = [row[0] for row in c.fetchall()]
    avg_miss = sum(misses) / len(misses) if misses else 0
    if avg_miss > 0.5:  # >50% missed
        # Simulate airhorn: return warning
        return jsonify({'procrastinating': True, 'message': 'You are procrastinating! Airhorn alert!'})
    return jsonify({'procrastinating': False})

# Tracking Progress (Client-side JS for timers/notifications)
# In frontend, use setInterval for 20min check-ins, Notification API.

# Daily Check
@app.route('/daily_check')
def daily_check():
    user_id = session.get('user_id', 'default')
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    # Missed slots
    c.execute("SELECT * FROM schedules WHERE user_id=? AND slot_start < ? AND NOT EXISTS (SELECT 1 FROM feedback WHERE task_id=schedules.task_id)", (user_id, yesterday.isoformat()))
    missed = c.fetchall()
    # Notify and ask reasons (predefined: tired, distracted, etc.)
    reasons_options = ['tired', 'distracted', 'unexpected event', 'other']
    # For completed, check feedback
    return jsonify({'missed': missed, 'reasons_options': reasons_options})

@app.route('/submit_reason', methods=['POST'])
def submit_reason():
    data = request.json
    schedule_id = data['schedule_id']
    reason = data['reason']
    c.execute("INSERT INTO reasons (schedule_id, reason, timestamp) VALUES (?, ?, ?)",
              (schedule_id, reason, datetime.datetime.now()))
    # Increment missed for task
    c.execute("UPDATE tasks SET missed = missed + 1 WHERE id=(SELECT task_id FROM schedules WHERE id=?)", (schedule_id,))
    conn.commit()
    return jsonify({'status': 'submitted'})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    task_id = data['task_id']
    extra_time = int(data['extra_time'])
    c.execute("INSERT INTO feedback (task_id, extra_time_needed, timestamp) VALUES (?, ?, ?)",
              (task_id, extra_time, datetime.datetime.now()))
    if extra_time > 0:
        # Adjust slots (simplified: add extra days)
        c.execute("UPDATE tasks SET days_req = days_req + ? WHERE id=?", (extra_time // 24, task_id))  # Assuming hours
    conn.commit()
    return jsonify({'status': 'submitted'})

# Task Completed
@app.route('/complete_task', methods=['POST'])
def complete_task():
    data = request.json
    task_id = data['task_id']
    c.execute("UPDATE tasks SET completed=1 WHERE id=?", (task_id,))
    # Remove remaining slots
    c.execute("DELETE FROM schedules WHERE task_id=?", (task_id,))
    conn.commit()
    # Adjust if faster: but handled in feedback
    return jsonify({'status': 'completed'})

if __name__ == '__main__':
    app.run(host='localhost',port=9000, debug=True)