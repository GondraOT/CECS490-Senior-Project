/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/timers.js - Handles countdown timers
*/

function startClock() {
    const tick = () => document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
    tick(); setInterval(tick, 1000);
}

function startSessionTimer() {
    const start = Date.now();
    const tick = () => {
        const e=Math.floor((Date.now()-start)/1000);
        const h=String(Math.floor(e/3600)).padStart(2,'0');
        const m=String(Math.floor((e%3600)/60)).padStart(2,'0');
        const s=String(e%60).padStart(2,'0');
        document.getElementById('h-session-badge').textContent = h+':'+m+':'+s;
    };
    tick(); setInterval(tick, 1000);
}

document.addEventListener('DOMContentLoaded', () => {
    startClock();
    startSessionTimer();
});