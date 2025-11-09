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
from typing import Optional

# Force correct paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'todo.db')

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
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute('PRAGMA foreign_keys = ON')
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


def column_exists(table: str, column: str) -> bool:
    c.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in c.fetchall())


def compute_priority_score(deadline_value, status, missed, estimated_minutes, actual_minutes, completed) -> float:
    today = datetime.date.today()
    score = 0.0

    deadline_days = None
    if deadline_value:
        try:
            if isinstance(deadline_value, datetime.date):
                deadline_days = (deadline_value - today).days
            else:
                deadline_days = (datetime.date.fromisoformat(str(deadline_value)) - today).days
        except (ValueError, TypeError):
            deadline_days = None

    if deadline_days is None:
        score += 10.0
    elif deadline_days < 0:
        score += 120.0
    else:
        score += max(0.0, 60.0 - float(deadline_days))

    status_key = (status or 'planned').lower()
    status_weights = {
        'planned': 10.0,
        'in_progress': 20.0,
        'at_risk': 32.0,
        'stalled': 35.0,
        'completed': 0.0,
    }
    score += status_weights.get(status_key, 15.0)

    score += min(float(missed or 0) * 5.0, 30.0)

    est_minutes = float(estimated_minutes or 0)
    act_minutes = float(actual_minutes or 0)
    if est_minutes > 0:
        progress_ratio = act_minutes / est_minutes
        if progress_ratio < 0.25:
            score += 5.0
        elif progress_ratio < 0.75:
            score += 10.0
        elif progress_ratio <= 1.1:
            score += 5.0
        else:
            score += 15.0
    else:
        score += 5.0

    if completed or status_key == 'completed':
        return 0.0

    return round(min(score, 200.0), 2)


def recompute_task_priority(task_id: int) -> None:
    c.execute(
        """
        SELECT deadline, status, missed, estimated_minutes, actual_minutes,
               days_req, hours_daily, completed
        FROM tasks
        WHERE id=?
        """,
        (task_id,),
    )
    row = c.fetchone()
    if not row:
        return

    deadline, status, missed, estimated_minutes, actual_minutes, days_req, hours_daily, completed = row
    estimated_minutes = estimated_minutes or 0
    if not estimated_minutes and days_req and hours_daily:
        try:
            estimated_minutes = int(days_req) * int(hours_daily) * 60
        except (TypeError, ValueError):
            estimated_minutes = 0
        c.execute(
            "UPDATE tasks SET estimated_minutes = ? WHERE id = ?",
            (estimated_minutes, task_id),
        )

    score = compute_priority_score(
        deadline,
        status,
        missed,
        estimated_minutes,
        actual_minutes,
        completed,
    )
    c.execute(
        "UPDATE tasks SET priority_score = ? WHERE id = ?",
        (score, task_id),
    )


def record_task_activity(task_id: int, minutes: int = 0, status: Optional[str] = None,
                         activity_time: Optional[datetime.datetime] = None) -> None:
    activity_time = activity_time or datetime.datetime.now()
    updates = []
    params: list = []

    if minutes:
        updates.append("actual_minutes = COALESCE(actual_minutes, 0) + ?")
        params.append(int(minutes))

    if status:
        updates.append("status = ?")
        params.append(status)

    updates.append("last_activity_at = ?")
    params.append(activity_time)
    params.append(task_id)

    if updates:
        c.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        recompute_task_priority(task_id)


def upgrade_schema() -> None:
    new_columns = {
        'category': "TEXT DEFAULT 'general'",
        'status': "TEXT DEFAULT 'planned'",
        'estimated_minutes': 'INTEGER DEFAULT 0',
        'actual_minutes': 'INTEGER DEFAULT 0',
        'priority_score': 'REAL DEFAULT 0',
        'last_activity_at': 'DATETIME',
    }

    for column, definition in new_columns.items():
        if not column_exists('tasks', column):
            c.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")

    c.execute(
        "UPDATE tasks SET status = CASE WHEN completed=1 THEN 'completed' ELSE COALESCE(NULLIF(status, ''), 'planned') END"
    )
    c.execute(
        "UPDATE tasks SET category = COALESCE(NULLIF(category, ''), 'general')"
    )
    c.execute("UPDATE tasks SET actual_minutes = COALESCE(actual_minutes, 0)")
    c.execute(
        "UPDATE tasks SET estimated_minutes = COALESCE(estimated_minutes, 0)"
    )

    c.execute(
        '''CREATE TABLE IF NOT EXISTS task_sessions (
               id INTEGER PRIMARY KEY,
               task_id INTEGER,
               started_at DATETIME,
               ended_at DATETIME,
               duration_minutes INTEGER,
               intensity TEXT,
               notes TEXT,
               user_id TEXT,
               FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
           )'''
    )

    c.execute(
        '''CREATE TABLE IF NOT EXISTS task_metrics (
               id INTEGER PRIMARY KEY,
               task_id INTEGER,
               metric_date DATE,
               focus_score REAL,
               energy_level TEXT,
               user_id TEXT,
               notes TEXT,
               FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
           )'''
    )

    conn.commit()

    c.execute("SELECT id, days_req, hours_daily FROM tasks")
    for task_id, days_req, hours_daily in c.fetchall():
        estimated_minutes = 0
        if days_req and hours_daily:
            try:
                estimated_minutes = int(days_req) * int(hours_daily) * 60
            except (TypeError, ValueError):
                estimated_minutes = 0

        if estimated_minutes:
            c.execute(
                "UPDATE tasks SET estimated_minutes = COALESCE(NULLIF(estimated_minutes, 0), ?) WHERE id = ?",
                (estimated_minutes, task_id),
            )
        recompute_task_priority(task_id)

    conn.commit()


upgrade_schema()

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


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('credentials', None)
    return jsonify({'status': 'logged_out'})

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

    estimated_minutes = int(days_req * hours_daily * 60)
    c.execute(
        """
        UPDATE tasks
        SET estimated_minutes = ?,
            status = COALESCE(NULLIF(status, ''), 'planned'),
            category = COALESCE(NULLIF(category, ''), 'general')
        WHERE id = ?
        """,
        (estimated_minutes, task_id),
    )

    recompute_task_priority(task_id)
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
    c.execute(
        """
        SELECT id, name, deadline, missed, category, status,
               estimated_minutes, actual_minutes, priority_score
        FROM tasks
        WHERE user_id=? AND completed=0
        """,
        (user_id,),
    )
    rows = c.fetchall()

    today = datetime.date.today()
    items = []
    for row in rows:
        deadline_value = row[2]
        days_until = None
        if deadline_value:
            try:
                days_until = (datetime.date.fromisoformat(str(deadline_value)) - today).days
            except ValueError:
                days_until = None

        items.append({
            'id': row[0],
            'name': row[1],
            'deadline': row[2],
            'daysUntilDeadline': days_until,
            'missedSlots': row[3],
            'category': row[4],
            'status': row[5],
            'estimatedMinutes': row[6],
            'actualMinutes': row[7],
            'priorityScore': row[8],
        })

    items.sort(key=lambda entry: entry.get('priorityScore', 0), reverse=True)
    return jsonify(items)

@app.route('/priority_overview')
def priority_overview():
    user_id = session.get('user_id', 'default')
    c.execute(
        """
        SELECT id, name, deadline, missed, category, status,
               estimated_minutes, actual_minutes, priority_score
        FROM tasks
        WHERE user_id=?
        """,
        (user_id,),
    )
    rows = c.fetchall()

    today = datetime.date.today()
    overview = []
    for row in rows:
        task_id, name, deadline_value, missed, category, status, est_minutes, act_minutes, score = row
        days_until = None
        if deadline_value:
            try:
                days_until = (datetime.date.fromisoformat(str(deadline_value)) - today).days
            except ValueError:
                days_until = None

        est_minutes = est_minutes or 0
        act_minutes = act_minutes or 0
        progress = None
        if est_minutes > 0:
            progress = round((act_minutes / est_minutes) * 100, 1)

        tips = []
        if days_until is None:
            tips.append("Add a clear deadline so the planner can prioritize it.")
        elif days_until < 0:
            tips.append("Deadline passed—reschedule the work and notify stakeholders.")
        elif days_until <= 1:
            tips.append("Block at least one deep-focus session today.")
        elif days_until <= 3:
            tips.append("Reserve a high-energy block within the next 48 hours.")
        elif est_minutes and (est_minutes - act_minutes) <= 60:
            tips.append("You're close—schedule a final push to wrap it up.")

        if progress is not None and progress < 40:
            tips.append("Momentum is low; start with a short pomodoro to regain flow.")

        if missed:
            tips.append("Review missed slots and adjust the schedule for better fit.")

        if not tips:
            tips.append("Keep the pace steady and maintain your habit streak.")

        overview.append({
            'id': task_id,
            'name': name,
            'category': category,
            'status': status,
            'deadline': deadline_value,
            'daysUntilDeadline': days_until,
            'missedSlots': missed,
            'estimatedMinutes': est_minutes,
            'actualMinutes': act_minutes,
            'progressPercent': progress,
            'priorityScore': score,
            'suggestion': tips[0],
        })

    overview.sort(key=lambda entry: entry.get('priorityScore', 0), reverse=True)
    return jsonify({
        'generated': datetime.datetime.now().isoformat(),
        'tasks': overview,
    })

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
        c.execute(
            "UPDATE tasks SET estimated_minutes = estimated_minutes + ? WHERE id=?",
            (extra_time * 60, task_id),
        )
        record_task_activity(task_id, status='at_risk')
    else:
        record_task_activity(task_id)
    conn.commit()
    return jsonify({'status': 'submitted'})


@app.route('/log_session', methods=['POST'])
def log_session():
    payload = request.get_json(silent=True) or {}
    task_id = payload.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    user_id = session.get('user_id', 'default')
    start_raw = payload.get('start')
    end_raw = payload.get('end')
    duration = payload.get('duration_minutes')
    intensity = payload.get('intensity')
    notes = payload.get('notes')

    try:
        start_at = parse(start_raw) if start_raw else None
    except (ValueError, TypeError):
        start_at = None

    try:
        end_at = parse(end_raw) if end_raw else None
    except (ValueError, TypeError):
        end_at = None

    if duration is None and start_at and end_at:
        duration = int(max((end_at - start_at).total_seconds() / 60, 0))
    else:
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = 0

    duration = max(duration or 0, 0)

    c.execute(
        """
        INSERT INTO task_sessions (task_id, started_at, ended_at, duration_minutes, intensity, notes, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            start_at.isoformat() if start_at else None,
            end_at.isoformat() if end_at else None,
            duration,
            intensity,
            notes,
            user_id,
        ),
    )

    status_override = payload.get('status')
    record_task_activity(task_id, minutes=duration, status=status_override)
    conn.commit()

    return jsonify({'status': 'logged', 'duration_minutes': duration})


@app.route('/ai_insights', methods=['POST'])
def ai_insights():
    payload = request.get_json(silent=True) or {}
    tasks_payload = payload.get('tasks') or []
    schedule_payload = payload.get('schedule') or []
    timestamp = payload.get('timestamp') or datetime.datetime.now().isoformat()

    def _format_task(entry):
        name = entry.get('name', 'Task')
        deadline = entry.get('deadline', 'unspecified')
        remaining = entry.get('remainingHours')
        priority = entry.get('priority')
        parts = [f"- {name}"]
        if deadline:
            parts.append(f"deadline: {deadline}")
        if remaining is not None:
            parts.append(f"remaining hours: {remaining}")
        if priority is not None:
            parts.append(f"priority: {priority}")
        return ' | '.join(parts)

    def _format_block(entry):
        name = entry.get('name', 'Block')
        start = entry.get('start', 'unknown start')
        end = entry.get('end', 'unknown end')
        source = entry.get('source', 'focusflow')
        status = entry.get('status', 'scheduled')
        return f"- {name} [{source}] {start} -> {end} (status: {status})"

    task_lines = '\n'.join(_format_task(task) for task in tasks_payload) or 'No active tasks provided.'
    block_lines = '\n'.join(_format_block(block) for block in schedule_payload) or 'No scheduled focus blocks.'

    prompt = (
        "You are a concise productivity coach. Based on the user's tasks and the next focus blocks, "
        "offer one actionable tip (maximum two sentences) that will help them stay productive."
        "\n\nCurrent timestamp: "
        f"{timestamp}\n\nTasks:\n{task_lines}\n\nUpcoming focus blocks:\n{block_lines}\n"
        "Respond with direct advice only."
    )

    suggestion = "Stay focused and revisit your top priority task for a quick win today."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        if response and getattr(response, 'text', None):
            candidate = response.text.strip()
            if candidate:
                suggestion = candidate
    except Exception as exc:  # pragma: no cover - best effort logging
        app.logger.warning('AI insight generation failed: %s', exc)

    return jsonify({'message': suggestion[:500]})

# Task Completed
@app.route('/complete_task', methods=['POST'])
def complete_task():
    data = request.json
    task_id = data['task_id']
    c.execute(
        "UPDATE tasks SET completed=1, status='completed', priority_score=0 WHERE id=?",
        (task_id,),
    )
    # Remove remaining slots
    c.execute("DELETE FROM schedules WHERE task_id=?", (task_id,))
    record_task_activity(task_id, status='completed')
    conn.commit()
    # Adjust if faster: but handled in feedback
    return jsonify({'status': 'completed'})

if __name__ == '__main__':
    app.run(host='localhost',port=9000, debug=True)