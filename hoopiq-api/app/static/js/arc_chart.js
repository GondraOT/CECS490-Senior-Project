/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/arc_chart.js - Handles shot activity over time
*/

function drawShotTimeline(shotData) {
    const canvas = document.getElementById('arc-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth || 400;
    const H = 220;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = H + 'px';

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const padL = 42;
    const padR = 20;
    const padT = 22;
    const padB = 36;
    const innerW = W - padL - padR;
    const innerH = H - padT - padB;
    const baseline = padT + innerH / 2;

    ctx.strokeStyle = 'rgba(0,255,136,0.15)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(padL, baseline);
    ctx.lineTo(W - padR, baseline);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, H - padB);
    ctx.stroke();

    ctx.font = '10px IBM Plex Mono';
    ctx.fillStyle = 'rgba(0,255,136,0.45)';
    ctx.textAlign = 'left';
    ctx.fillText('Miss', 8, baseline + 18);
    ctx.fillText('Make', 8, baseline - 10);

    if (!shotData || shotData.length === 0) {
        ctx.fillStyle = 'rgba(90,122,99,0.6)';
        ctx.font = '10px IBM Plex Mono';
        ctx.textAlign = 'center';
        ctx.fillText('No shot activity yet', W / 2, H / 2);
        return;
    }

    const count = shotData.length;
    const stepX = count > 1 ? innerW / (count - 1) : innerW / 2;

    ctx.fillStyle = 'rgba(0,255,136,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('Shot Sequence', W / 2, H - 10);

    ctx.beginPath();
    shotData.forEach((shot, i) => {
        const x = padL + (count === 1 ? innerW / 2 : i * stepX);
        const y = shot.made ? baseline - 42 : baseline + 42;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'rgba(0,255,136,0.2)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    shotData.forEach((shot, i) => {
        const x = padL + (count === 1 ? innerW / 2 : i * stepX);
        const y = shot.made ? baseline - 42 : baseline + 42;

        ctx.beginPath();
        ctx.moveTo(x, baseline);
        ctx.lineTo(x, y);
        ctx.strokeStyle = shot.made ? 'rgba(0,255,136,0.25)' : 'rgba(255,68,102,0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = shot.made ? 'rgba(0,255,136,0.9)' : 'rgba(255,68,102,0.85)';
        ctx.fill();
        ctx.strokeStyle = shot.made ? '#00ff88' : '#ff4466';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    ctx.fillStyle = 'rgba(0,255,136,0.35)';
    ctx.font = '9px IBM Plex Mono';
    ctx.textAlign = 'center';

    const labelEvery = Math.max(1, Math.ceil(count / 8));
    shotData.forEach((shot, i) => {
        if (i % labelEvery !== 0 && i !== count - 1) return;
        const x = padL + (count === 1 ? innerW / 2 : i * stepX);
        ctx.fillText(String(i + 1), x, H - 20);
    });
}
