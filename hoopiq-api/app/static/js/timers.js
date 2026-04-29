/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/timers.js - Handles countdown timers
*/

let sessionTimerInterval = null;
let sessionStartTime = null;
let sessionTimerRunning = false;

function startClock() {
    const el = document.getElementById('last-update');
    if (!el) return;

    const tick = () => {
        el.textContent = 'Last update: ' + new Date().toLocaleTimeString();
    };

    tick();
    setInterval(tick, 1000);
}

function updateSessionTimerDisplay() {
    const badge = document.getElementById('session-timer-badge');
    if (!badge) return;

    if (!sessionStartTime) {
        badge.textContent = '00:00:00';
        return;
    }

    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');

    badge.textContent = `${h}:${m}:${s}`;
}

function startSessionTimer() {
    if (sessionTimerRunning) return;

    sessionStartTime = Date.now();
    sessionTimerRunning = true;

    updateSessionTimerDisplay();

    sessionTimerInterval = setInterval(() => {
        updateSessionTimerDisplay();
    }, 1000);
}

function stopSessionTimer() {
    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    sessionTimerRunning = false;
    sessionStartTime = null;

    const badge = document.getElementById('session-timer-badge');
    if (badge) badge.textContent = '00:00:00';
}

function isSessionTimerRunning() {
    return sessionTimerRunning;
}

document.addEventListener('DOMContentLoaded', () => {
    startClock();
});
