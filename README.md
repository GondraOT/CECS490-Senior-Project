Rust = Main Coding Language
Python/C++ = Initial Coding Language


# HoopIQ-Flask

## Setup Instructions

1. **Activate virtual environment** before running Flask:

**Windows**
python -m venv venv
venv\Scripts\activate


2. **Install dependencies** (make sure you are in folder with requirements.txt):
pip install -r requirements.txt


3. **Verification of flask (Should see 3.x.x) (You can skip if confident):**
python -c "import flask; print(flask.__version__)"


4. **Run the app:**
python hoopiq_api.py


5. **Open your browser at:**
http://localhost:5000


6. **To exit out of venv, type:**
deactivate

run this to clear auth data in website console:
clearAllAuthData();

website https://www.hoopiq.shop/
alt website? https://cecs490-senior-project.onrender.com

dev notes:


📝 Project Handoff Note — HoopIQ Basketball Shot Tracker
📌 Current Status
------------------------------------------------------------
We have implemented:
        ◘ User authentication (register, login, logout)
        ◘ Local storage persistence for:
                ◘ Users
                ◘ Session state
                ◘ User shot history (user_data)
        ◘ Stats tracking and visualization
        ◘ Backend polling from ESP32 /stats endpoint
        ◘ Shot table rendering and computed statistics
        ◘ Export functionality (CSV + JSON)
-----------------------------------------------------------
⚙️ Key Architecture Overview
Data Sources:
        1. Local Storage
                ◘ users → email/password credentials
                ◘ session_user → currently logged-in user
                ◘ user_data → per-user shot history + stats
        2. Backend (/stats)
                ◘ Provides live shot data (shot_chart)
                ◘ Provides system/sensor/heatmap info
-----------------------------------------------------------
🔄 Current Data Flow
        1. User logs in
        2. App loads stored user data from localStorage
        3. App polls backend every 2 seconds (updateStats)
        4. Backend shots are merged into local user data
        5. UI updates (table, stats, charts)
-----------------------------------------------------------
⚠️ Known Behavior / Issue Being Addressed
Problem:
        ◘ Backend returns cumulative shot history
        ◘ When user logs out and logs back in:
                ◘ Old shots reappear (from local storage)
                ◘ New backend shots may get incorrectly merged into previous session history
                ◘ No clear separation between sessions
--------------------------------------------------------------
🧠 Solution Direction Implemented (In Progress)

We introduced the concept of a session boundary:

✅ New Concept:
        ◘ Track whether a session is “new” using a flag:
        
        window.newSession = true;

Intended Behavior:
        ◘ On login → mark new session
        ◘ On first backend response → establish baseline
        ◘ Prevent old backend data from being re-merged into stored history
-------------------------------------------------------------
🔧 Recent Code Changes
1. Session Reset Logic
        ◘ Reset backend tracking on login:

        previousBackendShots = [];
        window.newSession = true;

2. Session Flag Idea
        ◘ Use a flag to detect first backend interaction after login
        ◘ Prevent mixing old backend history with new session shots
----------------------------------------------------------------
📊 Stats System
        ◘ Stats are computed dynamically using:

        computeStatsFromShots(latestShots)

        ◘ No longer relying on latestStats as a backend source
        ◘ UI is driven directly from latestShots
-------------------------------------------------------------
📤 Export Feature
CSV Export:
        ◘ Exports shot table data
JSON Export:
        ◘Should export computed stats (not backend stats)

⚠️ Note:
Export previously relied on latestStats.basketball, which is no longer valid. It should be updated to use computed stats from latestShots.
------------------------------------------------------------------
🧪 Testing Setup
        ◘ window.simulateShots = true enables fake shot generation
        ◘ Simulated shots are added over time to mimic ESP32 input
        ◘ Useful for testing UI and backend merging behavior
-----------------------------------------------------------------
📍 Files to Focus On
static/js/auth.js
        ◘ Authentication logic
        ◘ Session handling
        ◘ Local storage management
        ◘ Login/logout flows
static/js/stats.js
        ◘ Backend polling (updateStats)
        ◘ Shot merging logic
        ◘ Stats computation
        ◘ Table rendering
        ◘ Export functions
---------------------------------------------------------------
🚧 Next Steps (Recommended)

For the next developer:

        1. Finalize session handling
                ◘ Ensure backend shots do not merge incorrectly across sessions
                ◘ Confirm newSession flag works as expected
        2. Improve backend integration
                ◘ Ideally backend should:
                ◘ Return session-based data OR
                ◘ Include timestamps/session IDs
        3. Fix export JSON
                ◘ Update export to use computed stats instead of outdated latestStats
        4. (Optional) Refactor merging logic
                ◘ Make backend → frontend data flow explicitly session-aware
                ◘ Prevent duplicate merges
----------------------------------------------------------------
💡 Notes for Debugging
        ◘ Use console logs inside updateStats() to inspect:

        console.log("Backend shots:", backendShots);
        console.log("Previous backend shots:", previousBackendShots);
        console.log("Latest shots:", latestShots);
        
        ◘ Check how many shots are being merged and when
--------------------------------------------------------------------
✅ Summary
        ◘ Authentication + local storage system is working
        ◘ Backend polling and visualization are working
        ◘ Main remaining challenge: clean session separation between logins
        ◘ A session flag (window.newSession) was introduced as a solution direction
        ◘ Export feature needs minor adjustment to align with computed stats