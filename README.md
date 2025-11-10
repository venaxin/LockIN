# Lock-IN

Lock-IN is an AI-assisted focus coach that syncs your tasks with Google Calendar, predicts procrastination risk, and nudges you into the right work block before momentum slips.

## Table of Contents

- [System Overview](#system-overview)
- [Key Features](#key-features)
- [Operating the App](#operating-the-app)
- [Configuration & Environment](#configuration--environment)
- [Local Development](#local-development)
- [Data Model & Telemetry](#data-model--telemetry)
- [Security & Privacy Notes](#security--privacy-notes)

## System Overview

- **Frontend:** Single-page experience rendered from `templates/index.html`, powered by vanilla JS for scheduling, analytics, and notifications.
- **Backend:** Flask (`app.py`) with SQLite storage, Google OAuth via `google-auth-oauthlib`, and Gemini Generative AI for task coaching.
- **Integrations:** Google Calendar bidirectional sync, Gemini API for task classification + coaching, browser Notification API for smart nudges.

## Key Features

### AI Task Classification & Setup

- **What it does:** Every new task runs through `classify_task_with_ai`, which asks Gemini to assign a category (e.g., `deep_work`, `planning`), estimate buffer days, and suggest Pomodoro blocks.
- **Example use case:** _"Finish Operating Systems lab"_ is tagged as `deep_work`, gets two buffer days, and suggests three 25-minute intervals. These metadata land in the task record and drive downstream planning.
- **Why it matters:** Category and buffer predictions seed consistent schedules while keeping the user’s job simple—enter a title, receive a tailored plan.

### Adaptive Priority & Risk Scoring

- **What it does:** `compute_priority_score` fuses deadline slack, streak state, missed history, and actual vs. estimated minutes to compute a 0–200 risk score that powers the dashboard ordering.
- **Example use case:** A task due tomorrow with half its planned work untouched jumps above a low-risk wellness task, ensuring the plan reflects urgency.
- **Why it matters:** Keeps attention aligned with what is likely to slip, not just what is closest in time.

### Google Calendar Sync & Conflict Resolution

- **What it does:** OAuth grants read/write access so `get_calendar_service` pulls a two-week window of events, filters conflicts, and inserts accepted focus blocks with a `Lock-IN Focus` label.
- **Example use case:** A user accepts the generated plan; meetings already on Google Calendar are respected, and deleted focus blocks in Google trigger mirror updates in Lock-IN.
- **Why it matters:** Prevents double-bookings and keeps the ecosystem (web app + calendar) in sync without manual copying.

### Smart Scheduling & Pomodoro Flow

- **What it does:** The heuristic scheduler builds 30-minute focus slots (`generateSchedule` in `templates/index.html`) around preferred windows, resolves conflicts, and attaches a live Pomodoro timer.
- **Example use case:** After adding a 6-hour project, Lock-IN automatically spreads sessions over weekdays and starts a 25-minute timer with pause/resume while tracking progress.
- **Why it matters:** Students get a ready-made plan they can immediately act on, reducing planning friction and supporting consistent focus rituals.

### Smart Notifications & Streak Protection

- **What it does:** Background sweeps (`runNotificationSweep`) fire alerts for blocks starting now, deadlines due today, and missed slots yesterday; results persist in `localStorage` to avoid duplicates.
- **Example use case:** At 9:00 AM, a toast and native browser notification remind the user their “Senior Design Draft” block starts; at noon an alert surfaces a paper due tonight.
- **Why it matters:** Time-sensitive nudges catch issues early and reinforce streak awareness without constant manual checking.

### AI Insights Coach

- **What it does:** Every ~45 minutes the client posts a snapshot to `/ai_insights`; the backend queries overdue tasks, builds a summary prompt, and asks Gemini for a two-sentence coaching tip.
- **Example use case:** After several skipped design sessions, the AI coach surfaces: _“Rescue your UX storyboard before 3 PM—swap tonight’s scrolling with a short sprint.”_
- **Why it matters:** Converts raw telemetry into motivating guidance, highlighting risk areas and recovery actions.

### Procrastination Analytics & Suggestions

- **What it does:** The dashboard tracks focus streaks, completion ratios, and category risk; `generateAISuggestions` blends heuristic signals (e.g., sliding window of misses) with AI coach output.
- **Example use case:** If four of the last five math sessions were skipped, the suggestions list a “Math rescue” card tagged as high risk with a recommended time.
- **Why it matters:** Offers actionable insight rather than passive charts, helping users recover from slumps quickly.

### Task & Schedule Management

- **What it does:** Users can add tasks manually, edit schedules via drag-and-drop (`enableCalendarDragAndDrop`), and import Google events as tasks.
- **Example use case:** Drag a block from Tuesday to Thursday, or click a calendar event to auto-populate the task form; the UI resaves to SQLite and re-renders without reloads.
- **Why it matters:** Keeps the workflow fluid so users can adjust plans on demand while maintaining data consistency.

## Operating the App

1. **Launch / Deploy**
   - Start the Flask server (see [Local Development](#local-development)).
   - For production (e.g., Render) use Gunicorn via the provided `Procfile`.
2. **Authenticate**
   - Click _Connect Google Calendar_ to trigger OAuth. After consent, Lock-IN seeds demo data if none exists and syncs your calendar.
3. **Create Tasks**
   - Use the task form or import from Google events. The AI classifier fills category, buffer days, and focus plan automatically.
4. **Review Schedule**
   - The focus planner generates Pomodoro-sized sessions in the weekly calendar, avoiding conflicts. Adjust blocks via drag-and-drop if needed.
5. **Stay on Track**
   - Start a Pomodoro session to unlock notifications, log reasons for skips, and collect streak data. Browser toasts and the AI coach keep you informed of upcoming work, deadlines, and rescue tips.
6. **Analyze & Iterate**
   - Check the analytics panels for productivity score, procrastination risk, and category hotspots. Use the suggestions list to accept recommended sessions or reprioritize tasks.

## Configuration & Environment

Set the following environment variables (or populate `.env`) before running the app:

| Variable                  | Purpose                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `FLASK_SECRET_KEY`        | Session signing key. Required in production.                                                        |
| `GEMINI_API_KEY`          | Gemini API key for task classification and coaching.                                                |
| `CALENDAR_API_KEY`        | Google API key used by the frontend JS SDK.                                                         |
| `GOOGLE_CREDENTIALS_JSON` | JSON credentials for your OAuth client (stringified). Optional in dev if `credentials.json` exists. |
| `GOOGLE_CLIENT_ID`        | Optional override for the client ID exposed to the frontend.                                        |

Additional deployment variables for Render or similar platforms should mirror the above.

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the server**
   ```bash
   python app.py
   ```
   The app listens on `http://localhost:9000` by default (update `app.run` if needed).
3. **Use Gunicorn (production-like)**
   ```bash
   gunicorn app:app --bind 0.0.0.0:9000
   ```
4. **Database**
   - SQLite file lives at `todo.db`. Existing schema migration runs automatically via `upgrade_schema()`.
   - Remove the file to reset the environment (demo seeds will reappear after Google sign-in).

## Data Model & Telemetry

- **`tasks`** captures metadata such as category, status, estimated/actual minutes, risk score, and streak-related fields.
- **`schedules`** holds planned focus blocks with start/end times and source (Lock-IN vs. Google).
- **`task_sessions`** logs actual focus sessions, duration, intensity, and notes.
- **`task_metrics`** stores daily aggregates (focus score, energy levels, AI metadata).
- **`reasons` / `feedback`** track user-entered explanations for misses and requests for more time.

These tables back the analytics panels, notification heuristics, and AI prompts.

## Security & Privacy Notes

- OAuth credentials are loaded from environment (`GOOGLE_CREDENTIALS_JSON`) to avoid bundling secrets in the repo.
- Session IDs are anonymous unless Google login succeeds; user email/profile live in `session['user_profile']` for UI display only.
- Task text and personal identifiers stay in separate tables, allowing analytics to operate on anonymized metrics when exported.
- Browser notifications require explicit consent; the app disables them gracefully if permission is denied.

---

Lock-IN keeps students “locked in” by combining habit-aware scheduling, smart reminders, and AI coaching so procrastination is caught before it snowballs.
