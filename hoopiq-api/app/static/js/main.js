/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/main.js - Handles app boot and live components
*/

function initApp() {
    updateAuthUI();

    startWebRTC('heatmap-img', 'heatmap-placeholder', 'heatmap-error', 'heatmap');

    startClock();

    updateStats();

    drawShotChart([]);
    if (typeof drawShotTimeline === 'function') drawShotTimeline([]);

    window.addEventListener('resize', () => {
        drawShotChart(window.latestShots || []);
        if (typeof drawShotTimeline === 'function') drawShotTimeline(window.latestShots || []);
    });
}

window.addEventListener('DOMContentLoaded', initApp);
