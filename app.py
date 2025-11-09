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
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


@app.before_request
def ensure_user_id():
    if 'user_id' not in session:
        session['user_id'] = f"anon:{uuid4()}"


def current_user_id() -> str:
    return session.get('user_id', 'guest')


def ensure_google_identity(credentials) -> str | None:
    try:
        oauth_service = build('oauth2', 'v2', credentials=credentials)
        profile = oauth_service.userinfo().get().execute()
    except Exception as exc:  # pragma: no cover - network call
        app.logger.warning('Failed to fetch Google profile: %s', exc)
        return None

    email = (profile or {}).get('email')
    if not email:
        return None

    user_id = f"google:{email.lower()}"
    session['user_id'] = user_id
    session['user_profile'] = {
        'email': email,
        'name': (profile or {}).get('name'),
        'picture': (profile or {}).get('picture'),
    }
    return user_id


# Database setup
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute('PRAGMA foreign_keys = ON')
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS tasks
#              (id INTEGER PRIMARY KEY, name TEXT, deadline DATE, days_req INTEGER, hours_daily INTEGER,
#               user_id TEXT, completed BOOLEAN DEFAULT 0, missed INTEGER DEFAULT 0)''')
# c.execute('''CREATE TABLE IF NOT EXISTS schedules
#              (id INTEGER PRIMARY KEY, task_id INTEGER, slot_start DATETIME, slot_end DATETIME, user_id TEXT)''')
# c.execute('''CREATE TABLE IF NOT EXISTS reasons
#              (id INTEGER PRIMARY KEY, schedule_id INTEGER, reason TEXT, timestamp DATETIME)''')
# c.execute('''CREATE TABLE IF NOT EXISTS feedback
#              (id INTEGER PRIMARY KEY, task_id INTEGER, extra_time_needed INTEGER, timestamp DATETIME)''')

# c.execute('''CREATE TABLE IF NOT EXISTS today_tasks
#              (id INTEGER PRIMARY KEY,
#               user_id TEXT NOT NULL,
#               task_id INTEGER NOT NULL,
#               added_date DATE NOT NULL,
#               FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
#               UNIQUE(user_id, task_id, added_date))''')

conn.commit()

def init_db():
    c = conn.cursor() # <-- This function gets its OWN cursor
    c.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id INTEGER PRIMARY KEY, name TEXT, deadline DATE, days_req INTEGER, hours_daily INTEGER,
                  user_id TEXT, completed BOOLEAN DEFAULT 0, missed INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS schedules
                 (id INTEGER PRIMARY KEY, task_id INTEGER, slot_start DATETIME, slot_end DATETIME, user_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS reasons
                 (id INTEGER PRIMARY KEY, schedule_id INTEGER, reason TEXT, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY, task_id INTEGER, extra_time_needed INTEGER, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS today_tasks
                 (id INTEGER PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  task_id INTEGER NOT NULL,
                  added_date DATE NOT NULL,
                  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                  UNIQUE(user_id, task_id, added_date))''')
    conn.commit()


@app.route('/today_tasks')
def get_today_tasks():
    user_id = current_user_id()
    today_str = datetime.date.today().isoformat()
    c = conn.cursor()
    # 2. ADD This query to get today's tasks
    c.execute(
        """
        SELECT t.id, t.name, t.status, t.priority_score, t.estimated_minutes, t.actual_minutes
        FROM tasks t
        JOIN today_tasks tt ON t.id = tt.task_id
        WHERE tt.user_id = ? AND tt.added_date = ?
        ORDER BY t.priority_score DESC
        """,
        (user_id, today_str)
    )
    tasks = [{
        'id': row[0],
        'name': row[1],
        'status': row[2],
        'priorityScore': row[3],
        'estimatedMinutes': row[4],
        'actualMinutes': row[5],
    } for row in c.fetchall()]
    
    return jsonify(tasks)

@app.route('/today_tasks', methods=['POST'])
def add_to_today():
    data = request.json
    task_id = data.get('task_id')
    user_id = current_user_id()
    today_str = datetime.date.today().isoformat()
    c = conn.cursor()
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400
        
    try:
        # 3. ADD This insert logic
        c.execute(
            """
            INSERT INTO today_tasks (user_id, task_id, added_date)
            VALUES (?, ?, ?)
            """,
            (user_id, task_id, today_str)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Task is already on the list for today, which is fine
        pass
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
        
    return jsonify({'status': 'added', 'task_id': task_id})

@app.route('/today_tasks/<int:task_id>', methods=['DELETE'])
def remove_from_today(task_id):
    user_id = current_user_id()
    today_str = datetime.date.today().isoformat()
    c = conn.cursor()
    try:
        # 4. ADD This delete logic
        c.execute(
            """
            DELETE FROM today_tasks
            WHERE user_id = ? AND task_id = ? AND added_date = ?
            """,
            (user_id, task_id, today_str)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
        
    return jsonify({'status': 'removed', 'task_id': task_id})

def column_exists(table: str, column: str) -> bool:
    c = conn.cursor()
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
    c = conn.cursor()
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
    c = conn.cursor()
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
    c = conn.cursor()   
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

init_db()
upgrade_schema()


def seed_mock_data(user_id: str) -> None:
    if not user_id.startswith('google:'):
        return
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM tasks WHERE user_id=?", (user_id,))
    if c.fetchone()[0] > 0:
        return

    now = datetime.datetime.now()
    sample_tasks = [
        {
            'name': 'Deep Work: Product Strategy',
            'deadline': (now + datetime.timedelta(days=3)).date(),
            'days_req': 2,
            'hours_daily': 3,
            'category': 'strategy',
        },
        {
            'name': 'Team Sync Preparation',
            'deadline': (now + datetime.timedelta(days=1)).date(),
            'days_req': 1,
            'hours_daily': 2,
            'category': 'collaboration',
        },
        {
            'name': 'Prototype QA Pass',
            'deadline': (now + datetime.timedelta(days=5)).date(),
            'days_req': 3,
            'hours_daily': 1,
            'category': 'quality',
        },
    ]

    for entry in sample_tasks:
        c.execute(
            """
            INSERT INTO tasks (name, deadline, days_req, hours_daily, user_id, category, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry['name'],
                entry['deadline'],
                entry['days_req'],
                entry['hours_daily'],
                user_id,
                entry['category'],
                'planned',
            ),
        )
        task_id = c.lastrowid
        c.execute(
            "UPDATE tasks SET estimated_minutes = ?, missed = 0 WHERE id=?",
            (entry['days_req'] * entry['hours_daily'] * 60, task_id),
        )
        recompute_task_priority(task_id)

        start_base = datetime.datetime.combine(now.date(), datetime.time(9, 0))
        slot_start = start_base + datetime.timedelta(hours=random.randint(0, 4))
        slot_end = slot_start + datetime.timedelta(hours=entry['hours_daily'])
        c.execute(
            """
            INSERT INTO schedules (task_id, slot_start, slot_end, user_id)
            VALUES (?, ?, ?, ?)
            """,
            (
                task_id,
                slot_start,
                slot_end,
                user_id,
            ),
        )
        c.execute(
            """
            INSERT INTO task_sessions (task_id, started_at, ended_at, duration_minutes, intensity, notes, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                slot_start,
                slot_end,
                entry['hours_daily'] * 60,
                'moderate',
                'Sample focus block seeded for demo.',
                user_id,
            ),
        )
        record_task_activity(task_id, minutes=entry['hours_daily'] * 60)

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
    c = conn.cursor()
    calendar_api_key = os.getenv('CALENDAR_API_KEY')
    if not calendar_api_key:
        app.logger.warning('CALENDAR_API_KEY not set; Google Calendar API calls will fail.')
    user_id = current_user_id()
    if user_id.startswith('google:'):
        c.execute('SELECT COUNT(*) FROM tasks WHERE user_id=?', (user_id,))
        if (c.fetchone() or (0,))[0] == 0:
            seed_mock_data(user_id)
    return render_template(
        'index.html',
        google_client_id=os.getenv('GOOGLE_CLIENT_ID'),
        calendar_api_key=calendar_api_key,
        google_scopes='https://www.googleapis.com/auth/calendar',
        user_id=user_id,
        user_profile=session.get('user_profile', {}),
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
    session.pop('user_profile', None)
    session['user_id'] = f"anon:{uuid4()}"
    return jsonify({'status': 'logged_out'})

@app.route('/oauth2callback')
def oauth2callback():
    # ALLOW HTTP FOR LOCAL DEV
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    flow.redirect_uri = f"http://{request.host}/oauth2callback"
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session['credentials'] = pickle.dumps(credentials)
    user_id = ensure_google_identity(credentials) or current_user_id()
    seed_mock_data(user_id)
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
    c = conn.cursor()
    data = request.json
    name = data['name']
    deadline = parse(data['deadline']).date()
    days_req = int(data['days_req'])
    hours_daily = int(data['hours_daily'])
    user_id = current_user_id()

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
    c = conn.cursor()
    data = request.json
    task_id = data['task_id']
    selected_slots = data['slots']  # list of {'start': iso, 'end': iso}
    user_id = current_user_id()
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
    c = conn.cursor()
    c.execute("SELECT name FROM tasks WHERE id=?", (task_id,))
    return c.fetchone()[0]

# Upcoming deadlines and priorities
@app.route('/get_priorities')
def get_priorities():
    user_id = current_user_id()
    c = conn.cursor()
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

# Analyze user list (simple: count tasks, avg time)
@app.route('/analyze_list')
def analyze_list():
    c = conn.cursor()
    user_id = current_user_id()
    c.execute("SELECT COUNT(*), AVG(hours_daily) FROM tasks WHERE user_id=?", (user_id,))
    count, avg_hours = c.fetchone()
    return jsonify({'total_tasks': count, 'avg_daily_hours': avg_hours})


@app.route('/priority_overview')
def priority_overview():
    user_id = current_user_id()
    generated_at = datetime.datetime.now().isoformat()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, name, deadline, status, category, estimated_minutes,
               actual_minutes, missed, priority_score, last_activity_at
        FROM tasks
        WHERE user_id=?
        ORDER BY priority_score DESC, deadline ASC
        """,
        (user_id,),
    )
    rows = c.fetchall()

    tasks = []
    total_estimated = 0
    total_actual = 0
    overdue = []
    high_priority = []
    today = datetime.date.today()

    for row in rows:
        deadline_value = row[2]
        deadline_iso = None
        days_until = None
        if deadline_value:
            try:
                deadline_iso = datetime.date.fromisoformat(str(deadline_value)).isoformat()
                days_until = (datetime.date.fromisoformat(deadline_iso) - today).days
            except ValueError:
                deadline_iso = str(deadline_value)

        entry = {
            'id': row[0],
            'name': row[1],
            'deadline': deadline_iso,
            'status': row[3],
            'category': row[4],
            'estimatedMinutes': row[5],
            'actualMinutes': row[6],
            'missedSlots': row[7],
            'priorityScore': row[8],
            'lastActivityAt': row[9],
            'daysUntilDeadline': days_until,
        }
        tasks.append(entry)
        total_estimated += row[5] or 0
        total_actual += row[6] or 0

        if deadline_value and days_until is not None and days_until < 0:
            overdue.append(entry)
        if (row[8] or 0) >= 80:
            high_priority.append(entry)

    c.execute(
        "SELECT COALESCE(SUM(duration_minutes), 0) FROM task_sessions WHERE user_id=?",
        (user_id,),
    )
    logged_minutes = c.fetchone()[0] or 0

    response = {
        'generated': generated_at,
        'userId': user_id,
        'summary': {
            'totalTasks': len(tasks),
            'estimatedMinutes': int(total_estimated),
            'actualMinutes': int(total_actual),
            'loggedSessionMinutes': int(logged_minutes),
            'highPriorityCount': len(high_priority),
            'overdueCount': len(overdue),
        },
        'tasks': tasks,
        'highPriority': high_priority[:5],
        'overdue': overdue,
    }

    return jsonify(response)

# Detect pattern / Procrastination
@app.route('/detect_procrastination')
def detect_procrastination():
    c = conn.cursor()
    user_id = current_user_id()
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
    c = conn.cursor()
    user_id = current_user_id()
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
    c = conn.cursor()
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
    c = conn.cursor()
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
    c = conn.cursor()
    payload = request.get_json(silent=True) or {}
    task_id = payload.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        user_id = current_user_id()
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
    except Exception as exc:  # pragma: no cover - defensive logging
        conn.rollback()
        app.logger.exception('Failed to log session: %s', exc)
        return jsonify({'error': 'log_session_failed', 'details': str(exc)}), 500

    return jsonify({'status': 'logged', 'duration_minutes': duration})


# @app.route('/ai_insights', methods=['POST'])
# def ai_insights():
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

@app.route('/ai_today_suggestion')
def ai_today_suggestion():
    c = conn.cursor()
    user_id = current_user_id()
    today_str = datetime.date.today().isoformat()

    # Get all high-priority tasks
    c.execute(
        """
        SELECT id, name, priority_score, status, missed
        FROM tasks
        WHERE user_id=? AND completed=0
        ORDER BY priority_score DESC
        LIMIT 5
        """,
        (user_id,),
    )
    top_tasks = c.fetchall()

    if not top_tasks:
        return jsonify({'message': 'Your task list is empty. Add a task to get started!'})

    # Get tasks *already* on today's list
    c.execute(
        "SELECT task_id FROM today_tasks WHERE user_id=? AND added_date=?",
        (user_id, today_str)
    )
    today_task_ids = {row[0] for row in c.fetchall()}

    # Find the highest priority task *not* on today's list
    suggestion = None
    for task in top_tasks:
        task_id, name, score, status, missed = task
        if task_id not in today_task_ids:
            suggestion_task = {'id': task_id, 'name': name}
            
            # Generate a smart reason
            if score > 100: # Overdue
                reason = f"This is **overdue** and your highest priority."
            elif missed > 0:
                reason = f"You've procrastinated on this {missed} time(s). Let's tackle it!"
            elif status == 'at_risk':
                reason = "This task is at risk. Let's get it back on track."
            else:
                reason = "This is your next most important task."
                
            suggestion = {
                'message': f"Add **{name}** to your plan. {reason}",
                'taskToSuggest': suggestion_task
            }
            break

    if not suggestion:
        suggestion = {'message': "Your plan looks good! You've added your top priorities. Ready to start?"}

    return jsonify(suggestion)

# Task Completed
@app.route('/complete_task', methods=['POST'])
def complete_task():
    c = conn.cursor()
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