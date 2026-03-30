/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/main.js - Handles shot chart visualization
*/

// ── Boot ──────────────────────────────────────────────────────────
function initApp() {
    updateAuthUI();

    startWebRTC('basketball-img','basketball-placeholder','basketball-error','basketball');
    startWebRTC('heatmap-img',   'heatmap-placeholder',   'heatmap-error',   'heatmap');

    startClock();
    startSessionTimer();

    updateStats();
    setInterval(updateStats, 2000);

    // Initial empty renders
    drawShotChart([]);
    drawMakesVsMisses([]);
    updateZones([]);

    // Resize handling
    window.addEventListener('resize', () => {
        drawShotChart([]);
        drawMakesVsMisses([]);
        updateZones([]);
    });
}

// Run after page loads
window.addEventListener('DOMContentLoaded', initApp);