from flask import Flask, request, jsonify, redirect, url_for, session, render_template
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import datetime
import sqlite3
import random
import pickle
import json
from collections import defaultdict
from dateutil.parser import parse
import google.generativeai as genai

import os
from uuid import uuid4
from typing import Optional, Dict, Any, List, Tuple

# Force correct paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'todo.db')

TASK_CATEGORY_CHOICES = {
    'deep_work': 'Deep Work',
    'planning': 'Planning & Strategy',
    'communication': 'Communication',
    'collaboration': 'Collaboration',
    'learning': 'Learning & Growth',
    'quality': 'Quality & QA',
    'admin': 'Operations & Admin',
    'health': 'Health & Wellness',
    'creative': 'Creative Work',
    'general': 'General',
}

CATEGORY_ALIASES: Dict[str, str] = {}
for slug, label in TASK_CATEGORY_CHOICES.items():
    CATEGORY_ALIASES[slug] = slug
    CATEGORY_ALIASES[label.lower()] = slug
    CATEGORY_ALIASES[label.lower().replace('&', 'and')] = slug
    CATEGORY_ALIASES[slug.replace('_', ' ')] = slug

# Helpful alias coverage for legacy seed data
CATEGORY_ALIASES['strategy'] = 'planning'

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "assets"),
    static_url_path="/assets",
)

logger = app.logger

app.secret_key = os.urandom(24)          # random secret for session
CREDENTIALS_FILE = 'credentials.json'
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

try:
    MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as exc:  # pragma: no cover - network call
    MODEL = None
    logger.warning('Gemini model unavailable: %s', exc)


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
conn.row_factory = sqlite3.Row
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


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    return connection


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


def normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return 'general'
    value = raw.strip().lower()
    if not value:
        return 'general'
    normalized = value.replace('&', 'and')
    return CATEGORY_ALIASES.get(normalized, CATEGORY_ALIASES.get(value, 'general'))


def _extract_json_dict(response_text: str) -> Dict[str, Any]:
    try:
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('`')
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        logger.warning('Failed to decode classifier response: %s', response_text)
    return {}


def classify_task_with_ai(title: str, description: str = '') -> Dict[str, Any]:
    if not MODEL:
        return {}
    prompt = {
        'task_title': title,
        'task_description': description or '',
        'category_slugs': sorted(set(CATEGORY_ALIASES.values())),
        'category_labels': TASK_CATEGORY_CHOICES,
    }
    try:
        response = MODEL.generate_content([
            (
                'You are an executive productivity coach. '
                'Classify the task into one of the provided categories (respond with the category slug). '
                'Provide JSON with keys: category, buffer_days, focus_level, procrastination_risk, notes, pomodoro_plan. '
                'focus_level and procrastination_risk must be numbers between 0 and 1. '
                'buffer_days must be an integer (>=0). '
                'pomodoro_plan should be an array of short strings. '
                'Respond with JSON only.'
            ),
            json.dumps(prompt),
        ])
        text = getattr(response, 'text', None) or ''
        if not text and hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    text = ''.join(part.text for part in candidate.content.parts if getattr(part, 'text', None))
                    if text:
                        break
        data = _extract_json_dict(text)
        data['category'] = normalize_category(data.get('category'))
        return data
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception('Task classification failed: %s', exc)
        return {}


def serialize_task_row(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for field, value in list(data.items()):
        if isinstance(value, (datetime.datetime, datetime.date)):
            data[field] = value.isoformat()
    return data


def fetch_task_dict(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
        row = cursor.fetchone()
        return serialize_task_row(row) if row else None
    finally:
        conn.close()


def coerce_date(value: Any) -> Optional[datetime.date]:
    if value is None:
        return None
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.datetime):
        return value.date()
    try:
        return parse(str(value)).date()
    except (ValueError, TypeError):
        return None


def coerce_datetime(value: Any) -> Optional[datetime.datetime]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    try:
        return parse(str(value))
    except (ValueError, TypeError):
        return None


upgrade_schema()


def seed_mock_data(user_id: str) -> None:
    if not user_id.startswith('google:'):
        return

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
                slot_start.isoformat(),
                slot_end.isoformat(),
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
                slot_start.isoformat(),
                slot_end.isoformat(),
                entry['hours_daily'] * 60,
                'moderate',
                'Sample focus block seeded for demo.',
                user_id,
            ),
        )
        record_task_activity(task_id, minutes=entry['hours_daily'] * 60)

    c.execute("SELECT id, category FROM tasks WHERE user_id=?", (user_id,))
    seeded_tasks = [(row['id'], row['category']) for row in c.fetchall()]

    history_cutoff = (now - datetime.timedelta(days=2)).isoformat()
    c.execute(
        "SELECT COUNT(*) FROM task_sessions WHERE user_id=? AND started_at < ?",
        (user_id, history_cutoff),
    )
    if (c.fetchone() or (0,))[0] == 0:
        seed_productivity_history(seeded_tasks, user_id, now)

    conn.commit()


def seed_productivity_history(task_rows: List[Tuple[int, Optional[str]]], user_id: str,
                              anchor: datetime.datetime) -> None:
    if not task_rows:
        return

    intensity_options = ['light', 'moderate', 'deep']
    notes_templates = [
        'Deep work interval seeded for analytics.',
        'Focus sprint logged automatically.',
        'Historical session to warm up dashboard.',
    ]

    for day_offset in range(3, 15):
        day = anchor - datetime.timedelta(days=day_offset)
        session_count = random.randint(1, 3)

        for _ in range(session_count):
            task_id, category = random.choice(task_rows)
            duration = random.choice([25, 30, 45, 50, 60])
            start_hour = random.choice([8, 9, 10, 13, 14, 15, 20])
            start_time = datetime.datetime.combine(day.date(), datetime.time(start_hour, 0))
            start_time += datetime.timedelta(minutes=random.choice([0, 15, 30, 45]))
            end_time = start_time + datetime.timedelta(minutes=duration)

            c.execute(
                """
                INSERT INTO task_sessions (
                    task_id, started_at, ended_at, duration_minutes, intensity, notes, user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    start_time.isoformat(),
                    end_time.isoformat(),
                    duration,
                    random.choice(intensity_options),
                    random.choice(notes_templates),
                    user_id,
                ),
            )
            record_task_activity(task_id, minutes=duration, activity_time=end_time)

        focus_score = round(random.uniform(0.55, 0.9), 2)
        risk_score = round(random.uniform(0.05, 0.45), 2)
        exemplar_task_id, exemplar_category = random.choice(task_rows)
        metadata = {
            'category': exemplar_category or 'general',
            'focus_level': focus_score,
            'procrastination_risk': risk_score,
            'buffer_days': 0,
        }

        c.execute(
            """
            INSERT INTO task_metrics (
                task_id, metric_date, focus_score, energy_level, user_id, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                exemplar_task_id,
                day.date(),
                focus_score,
                random.choice(['low', 'medium', 'high']),
                user_id,
                json.dumps(metadata),
            ),
        )

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
    payload = request.get_json(silent=True) or {}

    name = (payload.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Task name is required'}), 400

    raw_deadline = payload.get('deadline')
    deadline: Optional[datetime.date] = None
    if raw_deadline:
        try:
            deadline = parse(raw_deadline).date()
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid deadline format'}), 400
    if deadline is None:
        deadline = datetime.date.today()

    def coerce_int(value, default=1, minimum=1) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(parsed, minimum)

    days_req = coerce_int(payload.get('days_req'), 1)
    hours_daily = coerce_int(payload.get('hours_daily'), 1)
    description = (payload.get('description') or '').strip()
    user_id = current_user_id()

    ai_result = classify_task_with_ai(name, description)
    category_slug = ai_result.get('category') or normalize_category(payload.get('category'))

    try:
        buffer_days = max(int(ai_result.get('buffer_days', 0)), 0)
    except (TypeError, ValueError):
        buffer_days = 0

    total_days = days_req + buffer_days
    estimated_minutes = total_days * hours_daily * 60

    now = datetime.datetime.now()
    c.execute(
        """
        INSERT INTO tasks (
            name, deadline, days_req, hours_daily, user_id, category, status,
            estimated_minutes, last_activity_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            deadline,
            total_days,
            hours_daily,
            user_id,
            category_slug or 'general',
            'planned',
            estimated_minutes,
            now,
        ),
    )
    task_id = c.lastrowid

    recompute_task_priority(task_id)

    def coerce_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    focus_level = coerce_float(ai_result.get('focus_level'))
    if focus_level is not None:
        focus_level = max(0.0, min(1.0, focus_level))

    procrastination_risk = coerce_float(ai_result.get('procrastination_risk'))
    if procrastination_risk is not None:
        procrastination_risk = max(0.0, min(1.0, procrastination_risk))

    metadata: Dict[str, Any] = {
        'category': category_slug or 'general',
        'focus_level': focus_level,
        'procrastination_risk': procrastination_risk,
        'buffer_days': buffer_days,
    }
    notes_value = ai_result.get('notes')
    if notes_value:
        metadata['notes'] = notes_value if isinstance(notes_value, str) else json.dumps(notes_value)
    pomodoro_plan = ai_result.get('pomodoro_plan')
    if pomodoro_plan:
        metadata['pomodoro_plan'] = pomodoro_plan

    metrics_payload = dict(metadata)

    if any(value is not None for key, value in metadata.items() if key in {'focus_level', 'procrastination_risk'}):
        c.execute(
            """
            INSERT INTO task_metrics (
                task_id, metric_date, focus_score, energy_level, user_id, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                datetime.date.today(),
                metadata.get('focus_level'),
                None,
                user_id,
                json.dumps(metrics_payload),
            ),
        )

    conn.commit()

    slots = suggest_slots(task_id, deadline, total_days, hours_daily, user_id)

    task_payload = fetch_task_dict(task_id) or {}
    task_payload['buffer_days'] = buffer_days
    task_payload['focus_level'] = focus_level
    task_payload['procrastination_risk'] = procrastination_risk
    task_payload['pomodoro_plan'] = pomodoro_plan
    task_payload['categoryLabel'] = TASK_CATEGORY_CHOICES.get(
        task_payload.get('category', 'general'),
        (task_payload.get('category') or 'general').replace('_', ' ').title(),
    )

    classification_payload = dict(metadata)
    classification_payload['categoryLabel'] = TASK_CATEGORY_CHOICES.get(
        metadata.get('category', 'general'),
        (metadata.get('category') or 'general').replace('_', ' ').title(),
    )

    return jsonify(
        {
            'task': task_payload,
            'suggested_slots': slots,
            'classification': classification_payload,
        }
    )

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
    c.execute("SELECT name FROM tasks WHERE id=?", (task_id,))
    return c.fetchone()[0]

# Upcoming deadlines and priorities
@app.route('/get_priorities')
def get_priorities():
    user_id = current_user_id()
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
    user_id = current_user_id()
    c.execute("SELECT COUNT(*), AVG(hours_daily) FROM tasks WHERE user_id=?", (user_id,))
    count, avg_hours = c.fetchone()
    return jsonify({'total_tasks': count, 'avg_daily_hours': avg_hours})


@app.route('/priority_overview')
def priority_overview():
    user_id = current_user_id()
    generated_at = datetime.datetime.now().isoformat()

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


@app.route('/analytics/productivity')
def analytics_productivity():
    user_id = current_user_id()
    conn = get_db()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.task_id, s.started_at, s.ended_at, s.duration_minutes,
                   s.intensity, s.notes,
                   t.name AS task_name,
                   COALESCE(t.category, 'general') AS category,
                   t.priority_score
            FROM task_sessions s
            LEFT JOIN tasks t ON t.id = s.task_id
            WHERE s.user_id=?
            ORDER BY COALESCE(s.started_at, s.ended_at) DESC
            """,
            (user_id,),
        )
        session_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT task_id, metric_date, focus_score, notes
            FROM task_metrics
            WHERE user_id=?
            ORDER BY metric_date DESC
            """,
            (user_id,),
        )
        metric_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, category
            FROM tasks
            WHERE user_id=? AND completed=0
            """,
            (user_id,),
        )
        task_rows = cursor.fetchall()
    finally:
        conn.close()

    session_minutes: Dict[str, int] = defaultdict(int)
    category_minutes: Dict[str, int] = defaultdict(int)
    recent_sessions: List[Dict[str, Any]] = []

    for row in session_rows:
        duration = int(row['duration_minutes'] or 0)
        if duration <= 0:
            continue
        start_dt = coerce_datetime(row['started_at']) or coerce_datetime(row['ended_at']) or datetime.datetime.now()
        end_dt = coerce_datetime(row['ended_at'])
        day_key = start_dt.date().isoformat()
        session_minutes[day_key] += duration
        category_slug = row['category'] or 'general'
        category_minutes[category_slug] += duration
        if len(recent_sessions) < 5:
            recent_sessions.append(
                {
                    'id': row['id'],
                    'taskId': row['task_id'],
                    'taskName': row['task_name'] or 'Focus Block',
                    'category': category_slug,
                    'categoryLabel': TASK_CATEGORY_CHOICES.get(category_slug, category_slug.replace('_', ' ').title()),
                    'startedAt': start_dt.isoformat(),
                    'endedAt': end_dt.isoformat() if end_dt else None,
                    'durationMinutes': duration,
                    'intensity': row['intensity'],
                    'notes': row['notes'],
                    'priorityScore': row['priority_score'],
                }
            )

    focus_map: Dict[str, List[float]] = defaultdict(list)
    risk_map: Dict[str, List[float]] = defaultdict(list)

    for row in metric_rows:
        date_obj = coerce_date(row['metric_date'])
        if not date_obj:
            continue
        iso_key = date_obj.isoformat()
        notes_payload: Dict[str, Any] = {}
        if row['notes']:
            try:
                notes_payload = json.loads(row['notes'])
            except json.JSONDecodeError:
                notes_payload = {}

        focus_val = row['focus_score']
        if focus_val is None:
            focus_val = notes_payload.get('focus_level')
        if focus_val is not None:
            try:
                focus_map[iso_key].append(float(focus_val))
            except (TypeError, ValueError):
                pass

        risk_val = notes_payload.get('procrastination_risk')
        if risk_val is not None:
            try:
                risk_map[iso_key].append(float(risk_val))
            except (TypeError, ValueError):
                pass

    daily_focus = {
        day: sum(values) / len(values)
        for day, values in focus_map.items()
        if values
    }
    daily_risk = {
        day: sum(values) / len(values)
        for day, values in risk_map.items()
        if values
    }

    today = datetime.date.today()
    timeline: List[Dict[str, Any]] = []
    focus_units: List[float] = []
    risk_units: List[float] = []

    for offset in range(6, -1, -1):
        day = today - datetime.timedelta(days=offset)
        iso_key = day.isoformat()
        focus_value = daily_focus.get(iso_key)
        risk_value = daily_risk.get(iso_key)
        timeline.append(
            {
                'date': iso_key,
                'minutes': int(session_minutes.get(iso_key, 0)),
                'focus': round(focus_value * 100, 1) if focus_value is not None else None,
                'risk': round(risk_value * 100, 1) if risk_value is not None else None,
            }
        )
        if focus_value is not None:
            focus_units.append(focus_value)
        if risk_value is not None:
            risk_units.append(risk_value)

    weekly_minutes = sum(entry['minutes'] for entry in timeline)
    focus_avg_unit = sum(focus_units) / len(focus_units) if focus_units else 0.0
    intensity_factor = min(weekly_minutes / 840.0, 1.0) if weekly_minutes else 0.0
    score = int(round(min((0.65 * focus_avg_unit) + (0.35 * intensity_factor), 1.0) * 100))

    state = 'none'
    if score >= 80:
        state = 'high'
    elif score >= 50:
        state = 'mid'
    elif score > 0:
        state = 'low'

    state_messages = {
        'high': 'Momentum is strong—protect your deep-work windows.',
        'mid': 'Solid consistency—lock one more focus block to level up.',
        'low': 'Momentum is slipping—schedule a 30-minute focus sprint today.',
        'none': 'Log your next focus block to kickstart analytics.',
    }
    message = state_messages[state]

    streak = 0
    streak_cursor = today
    while session_minutes.get(streak_cursor.isoformat(), 0) > 0:
        streak += 1
        streak_cursor -= datetime.timedelta(days=1)

    peak_day = None
    if timeline:
        candidate = max(timeline, key=lambda entry: entry['minutes'])
        if candidate and candidate['minutes'] > 0:
            peak_day = candidate

    category_counts: Dict[str, int] = defaultdict(int)
    for row in task_rows:
        category_counts[row['category'] or 'general'] += 1

    total_category_minutes = sum(category_minutes.values())
    category_breakdown: List[Dict[str, Any]] = []
    if total_category_minutes > 0:
        for slug, minutes in sorted(category_minutes.items(), key=lambda kv: kv[1], reverse=True):
            if minutes <= 0:
                continue
            category_breakdown.append(
                {
                    'category': slug,
                    'label': TASK_CATEGORY_CHOICES.get(slug, slug.replace('_', ' ').title()),
                    'minutes': int(minutes),
                    'percent': round((minutes / total_category_minutes) * 100, 1),
                    'openTasks': category_counts.get(slug, 0),
                }
            )

    if not category_breakdown and category_counts:
        total_tasks = sum(category_counts.values()) or 1
        for slug, count in sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True):
            category_breakdown.append(
                {
                    'category': slug,
                    'label': TASK_CATEGORY_CHOICES.get(slug, slug.replace('_', ' ').title()),
                    'minutes': 0,
                    'percent': round((count / total_tasks) * 100, 1),
                    'openTasks': count,
                }
            )

    average_focus_pct = round(focus_avg_unit * 100, 1) if focus_units else None
    average_risk_pct = None
    if risk_units:
        average_risk_pct = round((sum(risk_units) / len(risk_units)) * 100, 1)

    response = {
        'generatedAt': datetime.datetime.now().isoformat(),
        'userId': user_id,
        'score': score,
        'state': state,
        'message': message,
        'streakDays': streak,
        'weeklyMinutes': int(weekly_minutes),
        'averageFocus': average_focus_pct,
        'averageRisk': average_risk_pct,
        'timeline': timeline,
        'recentSessions': recent_sessions,
        'categoryBreakdown': category_breakdown,
        'peakDay': peak_day,
        'openTasks': len(task_rows),
    }

    return jsonify(response)


@app.route('/analytics/procrastination')
def analytics_procrastination():
    user_id = current_user_id()
    conn = get_db()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, deadline, missed, status, category, priority_score,
                   estimated_minutes, actual_minutes, last_activity_at
            FROM tasks
            WHERE user_id=? AND completed=0
            """,
            (user_id,),
        )
        task_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT task_id, metric_date, notes
            FROM task_metrics
            WHERE user_id=?
            ORDER BY metric_date DESC
            """,
            (user_id,),
        )
        metric_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT s.slot_start, r.reason, r.timestamp, t.name AS task_name
            FROM reasons r
            JOIN schedules s ON s.id = r.schedule_id
            LEFT JOIN tasks t ON t.id = s.task_id
            WHERE s.user_id=?
            ORDER BY r.timestamp DESC
            LIMIT 6
            """,
            (user_id,),
        )
        reason_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT started_at, duration_minutes
            FROM task_sessions
            WHERE user_id=? AND started_at IS NOT NULL AND duration_minutes IS NOT NULL
              AND started_at >= datetime('now', '-21 day')
            """,
            (user_id,),
        )
        slump_rows = cursor.fetchall()
    finally:
        conn.close()

    metrics_lookup: Dict[int, Dict[str, Any]] = {}
    for row in metric_rows:
        task_id = row['task_id']
        if task_id in metrics_lookup:
            continue
        payload: Dict[str, Any] = {}
        if row['notes']:
            try:
                payload = json.loads(row['notes'])
            except json.JSONDecodeError:
                payload = {}
        metrics_lookup[task_id] = payload

    task_insights: List[Dict[str, Any]] = []
    category_risks: Dict[str, List[float]] = defaultdict(list)
    risk_values: List[float] = []

    today = datetime.date.today()
    for row in task_rows:
        task_id = row['id']
        category_slug = row['category'] or 'general'
        deadline_date = coerce_date(row['deadline'])
        days_until = None
        if deadline_date:
            days_until = (deadline_date - today).days

        progress_total = row['estimated_minutes'] or 0
        actual_minutes = row['actual_minutes'] or 0
        progress_ratio = (actual_minutes / progress_total) if progress_total else 0.0

        fallback_risk = 0.1
        fallback_risk += min((row['missed'] or 0) * 0.12, 0.5)
        if days_until is not None:
            if days_until < 0:
                fallback_risk += 0.35
            elif days_until <= 1:
                fallback_risk += 0.25
            elif days_until <= 3:
                fallback_risk += 0.12
        if progress_ratio < 0.3:
            fallback_risk += 0.25
        elif progress_ratio < 0.6:
            fallback_risk += 0.1

        metric_payload = metrics_lookup.get(task_id, {})
        metric_risk = metric_payload.get('procrastination_risk')
        try:
            metric_risk = float(metric_risk) if metric_risk is not None else None
        except (TypeError, ValueError):
            metric_risk = None

        combined_risk = fallback_risk
        if metric_risk is not None:
            combined_risk = (0.6 * metric_risk) + (0.4 * fallback_risk)

        combined_risk = max(0.0, min(1.0, combined_risk))
        risk_values.append(combined_risk)
        category_risks[category_slug].append(combined_risk)

        task_insights.append(
            {
                'id': task_id,
                'name': row['name'],
                'deadline': deadline_date.isoformat() if deadline_date else None,
                'daysUntil': days_until,
                'missed': row['missed'],
                'status': row['status'],
                'category': category_slug,
                'priorityScore': row['priority_score'],
                'risk': combined_risk,
                'progressRatio': round(progress_ratio, 2),
                'lastActivityAt': coerce_datetime(row['last_activity_at']).isoformat()
                if coerce_datetime(row['last_activity_at']) else None,
            }
        )

    avg_risk_unit = sum(risk_values) / len(risk_values) if risk_values else 0.0
    risk_state = 'none'
    if avg_risk_unit >= 0.65:
        risk_state = 'high'
    elif avg_risk_unit >= 0.35:
        risk_state = 'mid'
    elif avg_risk_unit > 0:
        risk_state = 'low'

    if task_insights:
        at_risk_candidate = max(task_insights, key=lambda entry: (entry['risk'], entry['priorityScore'] or 0))
        at_risk_task = {
            'id': at_risk_candidate['id'],
            'name': at_risk_candidate['name'],
            'deadline': at_risk_candidate['deadline'],
            'category': at_risk_candidate['category'],
            'risk': round(at_risk_candidate['risk'] * 100, 1),
            'daysUntil': at_risk_candidate['daysUntil'],
            'missed': at_risk_candidate['missed'],
            'priorityScore': at_risk_candidate['priorityScore'],
        }
    else:
        at_risk_task = None

    category_summary: List[Dict[str, Any]] = []
    for slug, values in category_risks.items():
        if not values:
            continue
        category_summary.append(
            {
                'category': slug,
                'label': TASK_CATEGORY_CHOICES.get(slug, slug.replace('_', ' ').title()),
                'avgRisk': round((sum(values) / len(values)) * 100, 1),
                'openTasks': len(values),
            }
        )
    category_summary.sort(key=lambda entry: entry['avgRisk'], reverse=True)

    hour_totals: Dict[int, int] = defaultdict(int)
    hour_counts: Dict[int, int] = defaultdict(int)
    for row in slump_rows:
        start_dt = coerce_datetime(row['started_at'])
        if not start_dt:
            continue
        hour_totals[start_dt.hour] += int(row['duration_minutes'] or 0)
        hour_counts[start_dt.hour] += 1

    slump_windows: List[Dict[str, Any]] = []
    for hour, count in hour_counts.items():
        avg_minutes = hour_totals[hour] / max(count, 1)
        slump_windows.append(
            {
                'hour': f"{hour:02d}:00",
                'avgSessionMinutes': round(avg_minutes, 1),
                'sessions': count,
            }
        )
    slump_windows.sort(key=lambda entry: entry['avgSessionMinutes'])
    slump_windows = slump_windows[:3]

    recent_reasons: List[Dict[str, Any]] = []
    for row in reason_rows:
        timestamp_dt = coerce_datetime(row['timestamp'])
        slot_start = coerce_datetime(row['slot_start'])
        recent_reasons.append(
            {
                'reason': row['reason'],
                'timestamp': timestamp_dt.isoformat() if timestamp_dt else None,
                'slotStart': slot_start.isoformat() if slot_start else None,
                'taskName': row['task_name'] or 'Focus Block',
            }
        )

    risk_messages = {
        'high': 'Multiple tasks are slipping—schedule recovery blocks today.',
        'mid': 'A few tasks need attention—lock a catch-up session this afternoon.',
        'low': 'Risk is manageable—keep checking in on near-due tasks.',
        'none': 'No procrastination signals yet—log work to keep insights current.',
    }

    response = {
        'generatedAt': datetime.datetime.now().isoformat(),
        'userId': user_id,
        'riskIndex': int(round(avg_risk_unit * 100)) if risk_values else 0,
        'riskState': risk_state,
        'message': risk_messages[risk_state],
        'tasksAnalyzed': len(task_insights),
        'atRiskTask': at_risk_task,
        'categoryRisk': category_summary,
        'slumpWindows': slump_windows,
        'recentReasons': recent_reasons,
    }

    return jsonify(response)


# Detect pattern / Procrastination
@app.route('/detect_procrastination')
def detect_procrastination():
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


@app.route('/ai_insights', methods=['POST'])
def ai_insights():
    payload = request.get_json(silent=True) or {}
    tasks_payload = payload.get('tasks') or []
    schedule_payload = payload.get('schedule') or []
    timestamp = payload.get('timestamp') or datetime.datetime.now().isoformat()
    user_id = current_user_id()

    conn = get_db()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        cursor.execute(
            """
            SELECT name, deadline, category, priority_score
            FROM tasks
            WHERE user_id=? AND completed=0 AND deadline IS NOT NULL
              AND DATE(deadline) <= ?
            ORDER BY deadline ASC, priority_score DESC
            LIMIT 3
            """,
            (user_id, yesterday),
        )
        overdue_rows = cursor.fetchall()
    finally:
        conn.close()

    overdue_tasks = []
    overdue_lines = []
    for row in overdue_rows:
        deadline_value = coerce_date(row['deadline'])
        overdue_tasks.append(
            {
                'name': row['name'],
                'deadline': deadline_value.isoformat() if deadline_value else str(row['deadline']),
                'category': row['category'],
                'priorityScore': row['priority_score'],
            }
        )
        formatted_deadline = deadline_value.isoformat() if deadline_value else 'unscheduled'
        overdue_lines.append(
            f"- {row['name']} | deadline: {formatted_deadline} | priority: {row['priority_score'] or 0}"
        )

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
    overdue_section = '\n'.join(overdue_lines) if overdue_lines else 'None from yesterday.'

    prompt = (
        "You are a concise productivity coach. Based on the user's tasks and the next focus blocks, "
        "offer one actionable tip (maximum two sentences) that will help them stay productive."
        "\n\nCurrent timestamp: "
        f"{timestamp}\n\nTasks:\n{task_lines}\n\nUpcoming focus blocks:\n{block_lines}"
        f"\n\nPending tasks from yesterday or earlier:\n{overdue_section}\n"
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

    return jsonify({'message': suggestion[:500], 'pendingYesterday': overdue_tasks})

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