/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/arc_chart.js - Handles arc-based shot analysis
*/

// ── Makes vs Misses Bar Chart ────────────────────────────────────
function drawMakesVsMisses(shotData) {
    const canvas = document.getElementById('arc-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth || 400, H = 180;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const makes  = shotData.filter(s => s.made).length;
    const misses = shotData.filter(s => !s.made).length;
    const total  = makes + misses;

    if (total === 0) {
        ctx.fillStyle = 'rgba(90,122,99,0.6)';
        ctx.font = '10px IBM Plex Mono'; ctx.textAlign = 'center';
        ctx.fillText('No shots yet', W/2, H/2); return;
    }

    const pad = 40, barW = (W - pad*2 - 20) / 2, maxH = H - 60;
    const makeH  = (makes  / Math.max(makes, misses)) * maxH;
    const missH  = (misses / Math.max(makes, misses)) * maxH;
    const makeX  = pad, missX = pad + barW + 20;
    const baseY  = H - 30;

    // Grid lines
    ctx.strokeStyle = 'rgba(0,255,136,0.08)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = baseY - (i/4) * maxH;
        ctx.beginPath(); ctx.moveTo(pad - 5, y); ctx.lineTo(W - pad + 5, y); ctx.stroke();
        const val = Math.round((i/4) * Math.max(makes, misses));
        ctx.fillStyle = 'rgba(0,255,136,0.35)'; ctx.font = '9px IBM Plex Mono';
        ctx.textAlign = 'right'; ctx.fillText(val, pad - 8, y + 3);
    }

    // Makes bar
    const makeGrad = ctx.createLinearGradient(0, baseY - makeH, 0, baseY);
    makeGrad.addColorStop(0, 'rgba(0,255,136,0.9)');
    makeGrad.addColorStop(1, 'rgba(0,255,136,0.2)');
    ctx.fillStyle = makeGrad;
    ctx.beginPath();
    ctx.roundRect(makeX, baseY - makeH, barW, makeH, [4, 4, 0, 0]);
    ctx.fill();

    // Misses bar
    const missGrad = ctx.createLinearGradient(0, baseY - missH, 0, baseY);
    missGrad.addColorStop(0, 'rgba(255,68,102,0.9)');
    missGrad.addColorStop(1, 'rgba(255,68,102,0.2)');
    ctx.fillStyle = missGrad;
    ctx.beginPath();
    ctx.roundRect(missX, baseY - missH, barW, missH, [4, 4, 0, 0]);
    ctx.fill();

    // Labels
    ctx.textAlign = 'center'; ctx.font = '600 11px IBM Plex Mono';
    ctx.fillStyle = '#00ff88';
    ctx.fillText('MAKES', makeX + barW/2, baseY + 14);
    ctx.fillText(makes, makeX + barW/2, baseY - makeH - 6);
    ctx.fillStyle = '#ff4466';
    ctx.fillText('MISSES', missX + barW/2, baseY + 14);
    ctx.fillText(misses, missX + barW/2, baseY - missH - 6);

    // FG% center label
    const fgPct = (makes / total * 100).toFixed(1);
    ctx.fillStyle = 'rgba(0,255,136,0.5)'; ctx.font = '10px IBM Plex Mono';
    ctx.fillText(fgPct + '% FG', W/2, baseY + 14);
}
