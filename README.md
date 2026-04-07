Rust = Main Coding Language
Python/C++ = Initial Coding Language


# HoopIQ-Flask

## Setup Instructions

1. **Activate virtual environment** before running Flask:

**Windows**
python -m venv venv
venv\Scripts\activate


2. **Install dependencies** (first time only):
pip install -r requirements.txt


3. **Verification of flask (Should see 3.1.2) (You can skip if confident):**
python -c "import flask; print(flask.__version__)"


4. **Run the app:**
python hoopiq-api.py


5. **Open your browser at:**
http://localhost:5000


6. **To exit out of venv, type:**
deactivate

run this to clear auth data in website console:
clearAllAuthData();

website https://www.hoopiq.shop/
alt website? https://cecs490-senior-project.onrender.com

dev notes:
let user log in and see old session
moment user take a shot and esp32 sends data, reset all data and start new "session"
let user log out and log in to still see old session and export whenever