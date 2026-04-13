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
Console commands:
enable/disable simulation:
        simulateShots = true    => Starts fake shots every second
        simulateShots = false   => Stops simulation

try to add a scroll or page table of max 10 shots per page
implement proper email and password, along with username instead of email top left