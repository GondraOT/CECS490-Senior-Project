/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/shot_chart.js - Handles shot chart visualization
*/

// ── Shot Chart Canvas ─────────────────────────────────────────────
function drawShotChart(shotData) {
    const canvas = document.getElementById('shot-chart-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth || 400;
    const H = Math.max(W * 0.8, 500);
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = 20;
    const cW = W - pad*2, cH = H - pad - 40;
    const cx = pad + cW/2, baseline = pad + cH;

    // ── Real court dimensions (feet) ─────────────────────────────
    const COURT_W = 50;
    const COURT_L = 47;
    
    // scale factors
    const scaleX = cW / COURT_W;
    const scaleY = cH / COURT_L;
    
    // convert court coords → canvas
    function toCanvasX(x) {
        return cx + x * scaleX;
    }
    
    function toCanvasY(y) {
        return baseline - y * scaleY;
    }

    ctx.fillStyle = 'rgba(0,255,136,0.03)';
    ctx.fillRect(pad, pad, cW, cH);

    ctx.strokeStyle = 'rgba(0,255,136,0.25)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(pad, pad, cW, cH);

    ctx.strokeStyle = 'rgba(0,255,136,0.6)';
    ctx.lineWidth = 1.5;
    
    // ── Court boundary ─────────────────────────────────
    ctx.strokeRect(pad, pad, cW, cH);
    
    // ── Backboard ──────────────────────────────────────
    ctx.beginPath();
    ctx.moveTo(toCanvasX(-3), toCanvasY(4));
    ctx.lineTo(toCanvasX(3), toCanvasY(4));
    ctx.stroke();
    
    // ── Rim ────────────────────────────────────────────
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(5.25), 0.75 * scaleX, 0, Math.PI * 2);
    ctx.stroke();
    
    // ── Paint (key) ────────────────────────────────────
    ctx.strokeRect(
        toCanvasX(-6),
        toCanvasY(19),
        12 * scaleX,
        19 * scaleY
    );
    
    // ── Free throw circle ──────────────────────────────
    ctx.beginPath();
    for (let t = Math.PI; t >= 0; t -= 0.01) {
        const x = 6 * Math.cos(t);
        const y = 19 + 6 * Math.sin(t);
        ctx.lineTo(toCanvasX(x), toCanvasY(y));
    }
    ctx.stroke();
    
    // dashed bottom half
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let t = 0; t <= Math.PI; t += 0.01) {
        const x = 6 * Math.cos(t);
        const y = 19 + 6 * Math.sin(t);
        ctx.lineTo(toCanvasX(x), toCanvasY(y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    
    // ── 3-point line arc ─────────────────────────────────
    const radius3PT = 19.75;
    const centerY = 5.25;

    ctx.beginPath();
    for (let t = Math.PI; t >= 0; t -= 0.01) {
        const x = radius3PT * Math.cos(t);
        const y = centerY + radius3PT * Math.sin(t);
        ctx.lineTo(toCanvasX(x), toCanvasY(y));
    }
    ctx.stroke();

    // ── 3-point line side markers ─────────────────────────────
    const threePointX1 = radius3PT * Math.cos(0);
    const threePointY1 = centerY + radius3PT * Math.sin(0);
    const threePointX2 = radius3PT * Math.cos(Math.PI);
    const threePointY2 = centerY + radius3PT * Math.sin(Math.PI);
    
    ctx.beginPath();
    ctx.moveTo(toCanvasX(threePointX1), toCanvasY(threePointY1));
    ctx.lineTo(toCanvasX(threePointX1), toCanvasY(0));
    ctx.moveTo(toCanvasX(threePointX2), toCanvasY(threePointY2));
    ctx.lineTo(toCanvasX(threePointX2), toCanvasY(0));
    ctx.stroke();

    if (!shotData || shotData.length === 0) {
        ctx.fillStyle = 'rgba(90,122,99,0.5)'; ctx.font = '10px IBM Plex Mono'; ctx.textAlign = 'center';
        ctx.fillText('No shot data yet', cx, pad + cH/2);
        drawLegend(ctx, W, H, pad, 0, 0); return;
    }

    let makes = 0, misses = 0;
    shotData.forEach(shot => {
        const x = toCanvasX((shot.x - 50) * (COURT_W / 100));
        const y = toCanvasY(shot.y * (COURT_L / 100));
        const made = shot.made;
        const isSwish = shot.shot_type === 'Swish';
        if (made) makes++; else misses++;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2);
        if (isSwish) {
            ctx.fillStyle = 'rgba(0,229,255,0.85)';
            ctx.strokeStyle = '#00e5ff';
        } else if (made) {
            ctx.fillStyle = 'rgba(255,140,0,0.85)';
            ctx.strokeStyle = '#ff8c00';
        } else {
            ctx.fillStyle = 'rgba(255,68,102,0.6)';
            ctx.strokeStyle = '#ff4466';
        }
        ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
    });

    drawLegend(ctx, W, H, pad, makes, misses);
    document.getElementById('shot-count-badge').textContent = shotData.length + ' shots';
}

function drawLegend(ctx, W, H, pad, makes, misses) {
    const legendY = H - 22;
    ctx.font = '10px IBM Plex Mono'; ctx.textAlign = 'left';

    // Swish dot
    ctx.beginPath(); ctx.arc(pad + 8, legendY, 5, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(0,229,255,0.9)'; ctx.fill();
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = 'rgba(0,229,255,1)'; ctx.fillText('swish', pad + 18, legendY + 4);

    // Made dot
    ctx.beginPath(); ctx.arc(pad + 75, legendY, 5, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,140,0,0.9)'; ctx.fill();
    ctx.strokeStyle = '#ff8c00'; ctx.stroke();
    ctx.fillStyle = 'rgba(255,140,0,1)'; ctx.fillText('made' + (makes ? ' ('+makes+')' : ''), pad + 85, legendY + 4);

    // Missed dot
    ctx.beginPath(); ctx.arc(pad + 155, legendY, 5, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,68,102,0.7)'; ctx.fill();
    ctx.strokeStyle = '#ff4466'; ctx.stroke();
    ctx.fillStyle = 'rgba(255,68,102,0.9)'; ctx.fillText('missed' + (misses ? ' ('+misses+')' : ''), pad + 165, legendY + 4);
}