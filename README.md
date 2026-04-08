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
user will now be able to see their old session data
still needs to fix when new data from esp32 comes in, data resets
multiple users can now create an account
table now able to update and the stats now reflect the table

chatgpt notes:
✅ ✅ What We’ve Completed
🔐 Authentication System
LocalStorage-based user system implemented
Functions:
getUsers()
getSession()
saveSession()
clearSession()
Login + register working
Session persists across refresh
Auth modal behavior fixed:
Close button works
Clicking outside works properly
Enter key triggers login/register
🧾 User Data Storage
User-specific data stored in localStorage:
shots
stats

Structure:

{
  "demo@test.com": {
    shots: [...],
    stats: {...}
  }
}
Able to load and display per-user data on login
📡 Data Flow (Frontend Logic)
Backend data (ESP32 simulation / API) is received periodically
updateStats():
Pulls backend data
Merges with user session data if logged in
Shots are stored in:
latestShots (single source of truth for UI)
📊 Shot Table
Table renders dynamically from latestShots
Each shot shows:
Shot type
Made / missed
Table updates automatically when new shots arrive
📈 Stats System
Stats are computed from the table (shots) instead of relying on backend stats

Implemented:

computeStatsFromShots(latestShots)
Stats include:
Attempts
Makes
FG%
Swishes
Backboard makes/misses
UI and table are now consistent ✅
🧪 Simulation Mode
Simulates ESP32 shot input
Fixed logic:
Swish → always made
Backboard → can be make or miss
Allows testing without hardware
🔄 Reset Functionality
Reset button:
Clears user shots
Clears stats
Updates UI
Saves cleared state to localStorage
📄 Pagination (Design Implemented)
Prepared pagination system:
Page size concept (10 shots)
Page navigation logic
Ready to display:
Page buttons
Slice table data per page
⚠️ Known Behavior / Limitations
1. Session data reset issue (not fixed yet)
When new ESP32 data arrives:
Backend data overwrites local session shots
Result:
UI appears to “reset” after login when new data comes in

👉 Cause:

No merge strategy between backend shots and user-local shots
2. Stats originally inconsistent (now fixed)
Earlier mismatch between:
Table data
Stats display
✔ Now resolved by computing stats from latestShots
3. Rim detection not supported
Hardware limitation:
Cannot distinguish rim vs backboard reliably
Decision:
Rim category effectively removed from system
4. Backend not fully authoritative
Current system is frontend-driven:
localStorage acts as database
backend acts as data feeder (ESP32)
No persistent backend user database yet
🚧 What’s Still Missing (From Full Goal)
🔁 1. Proper Data Sync / Merge Logic
When ESP32 sends new shots:
Should be appended to user’s stored shots
Not overwrite them

👉 Needed for:

Persistent user history
Multi-session consistency
🧠 2. True Backend Integration (Future Phase)
Replace localStorage user system with:
Backend database (e.g., Flask + DB)
Benefits:
Multi-device access
Persistent storage
Real authentication
📡 3. ESP32 Live Integration
Replace simulation with real device:
Continuous shot streaming
Real-time updates
Ensure:
Duplicate prevention
Data consistency
📄 4. Full Pagination UI
Not fully implemented yet:
Page buttons (Next / Prev / Page numbers)
Page indicator (e.g., “Page 2 of 5”)
Navigation controls
🎯 5. Advanced Stats Enhancements (Optional)
Currently supported:
FG%
Makes / attempts
Basic shot breakdown

Possible additions:

2PT vs 3PT (if shot metadata supports it)
Shot streaks
Heatmaps (later visualization)
Time-based stats
🔐 6. Better Session Handling
Prevent backend updates from:
Overwriting session state
Need:
Clear separation between:
“user-owned data”
“incoming device data”
🧭 Current System State (Simple View)
ESP32 (or simulation)
        ↓
Frontend receives shot data
        ↓
latestShots (source of truth)
        ↓
┌───────────────┬───────────────┐
│ Shot Table    │ Stats (computed)│
└───────────────┴───────────────┘
        ↓
Saved per user in localStorage
✅ Where You Left Off

You are currently at a working prototype stage, with:

Functional UI
Data flow working
Stats consistent
Simulation working
Authentication working
Pagination concept ready

👉 The only major unresolved issue is:
data merge / session persistence when new backend data arrives

If you come back later, the next step will likely be:

👉 Fixing the merge logic between backend shots and user session data
👉 Then finishing pagination UI
👉 Then moving toward ESP32 integration / backend persistence