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
    const H = 260;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const pad = 20;
    const cW = W - pad*2, cH = H - pad - 40;
    const cx = pad + cW/2, baseline = pad + cH;

    ctx.fillStyle = 'rgba(0,255,136,0.03)';
    ctx.fillRect(pad, pad, cW, cH);

    ctx.strokeStyle = 'rgba(0,255,136,0.25)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(pad, pad, cW, cH);

    const arcR = cW * 0.44;
    ctx.beginPath();
    ctx.arc(cx, baseline, arcR, Math.PI, 0);
    const cornerX = cx - arcR, cornerX2 = cx + arcR, cornerH = cH * 0.28;
    ctx.moveTo(pad, baseline - cornerH); ctx.lineTo(cornerX, baseline - cornerH);
    ctx.moveTo(cornerX2, baseline - cornerH); ctx.lineTo(pad + cW, baseline - cornerH);
    ctx.stroke();

    const laneW = cW * 0.22, laneH = cH * 0.32, laneX = cx - laneW/2;
    ctx.strokeRect(laneX, baseline - laneH, laneW, laneH);
    ctx.beginPath(); ctx.arc(cx, baseline - laneH, laneW/2, Math.PI, 0); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, baseline - laneH, laneW/2, 0, Math.PI); ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(cx, baseline - cH*0.05, cW*0.03, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - cW*0.07, baseline - cH*0.035); ctx.lineTo(cx + cW*0.07, baseline - cH*0.035); ctx.stroke();

    if (!shotData || shotData.length === 0) {
        ctx.fillStyle = 'rgba(90,122,99,0.5)'; ctx.font = '10px IBM Plex Mono'; ctx.textAlign = 'center';
        ctx.fillText('No shot data yet', cx, pad + cH/2);
        drawLegend(ctx, W, H, pad, 0, 0); return;
    }

    let makes = 0, misses = 0;
    shotData.forEach(shot => {
        const x = pad + (shot.x / 100) * cW;
        const y = pad + (shot.y / 100) * cH;
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