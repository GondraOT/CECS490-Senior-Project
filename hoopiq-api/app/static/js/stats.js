/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/stats.js - Handles shot statistics and charting
*/

// ── Constants ───────────────────────────────────────────────────────
const CLOUD = 'https://cecs490-senior-project.onrender.com';

// ── Export State ────────────────────────────────────────────────
let latestShots = [];
let latestStats = {};

// ── Helpers ────────────────────────────────────────────────────────
function fmt(v, suffix = '') {
    return (v === null || v === undefined) ? '--' : v + suffix;
}

function getLastShotClass(type) {
    if (!type || type === '—') return '';
    if (type === 'Swish') return 'swish';
    if (type.includes('Make') || type === 'Swish') return 'make';
    return 'miss';
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);
}

// ── Main Stats Function ────────────────────────────────────────────
async function updateStats() {
    try {
        const r = await fetch(`${CLOUD}/stats`, { cache: 'no-store' });
        if (!r.ok) throw new Error('Failed to fetch stats');
        const d = await r.json();

        const online = d.system?.online ?? false;
        setText('status-dot', online ? '●' : '○'); // optional visual
        const statusDotEl = document.getElementById('status-dot');
        if (statusDotEl) statusDotEl.className = 'status-dot' + (online ? ' online' : '');
        setText('status-text', online ? 'Online' : 'Offline');

        const b = d.basketball || {};
        const s = d.sensors || {};
        const h = d.heatmap || {};

        // // Temp data
        // b.shot_chart = [
        //     { made: true, shot_type: 'Swish' },
        //     { made: true, shot_type: 'Backboard Make' },
        //     { made: false, shot_type: 'Backboard Miss' },
        //     { made: false, shot_type: 'Rim Hit' }
        // ];

        latestStats = d;
        latestShots = b.shot_chart || [];

        console.log("FULL DATA:", d);
        console.log("SHOT CHART:", b.shot_chart);

        setText('attempts', fmt(b.attempts));
        setText('makes', fmt(b.makes));
        setText('fg-pct', b.fg_percent != null ? b.fg_percent + '%' : '--');
        setText('two-pct', b.two_pt_percent != null ? b.two_pt_percent + '%' : '--');
        setText('three-pct', b.three_pt_percent != null ? b.three_pt_percent + '%' : '--');
        setText('swishes', fmt(b.swishes));
        setText('rim', fmt(s.rim_hits));
        setText('backboard', fmt(s.backboard_hits));
        setText('streak', fmt(b.streak));

        // Shot type breakdown
        setText('swish-makes', fmt(b.swishes));
        setText('backboard-makes', fmt(s.backboard_makes));
        setText('backboard-misses', fmt(s.backboard_misses));

        // Last shot type with color
        const lastShot = b.last_shot_type || '—';
        const lastShotEl = document.getElementById('last-shot-type');
        if (lastShotEl) {
            lastShotEl.textContent = lastShot;
            lastShotEl.className = 'val small ' + getLastShotClass(lastShot);
        }

        setText('players', fmt(h.current_players));
        setText('avg-arc', b.avg_arc != null ? b.avg_arc + '°' : '--');
        setText('avg-entry', b.avg_entry_angle != null ? b.avg_entry_angle + '°' : '--');

        const bPointsBadge = document.getElementById('b-points-badge');
        if (bPointsBadge) bPointsBadge.textContent = fmt(b.makes) + ' pts';
        const arcStreakBadge = document.getElementById('arc-streak-badge');
        if (arcStreakBadge) arcStreakBadge.textContent = 'Streak: ' + fmt(b.streak);

        // Optional chart/zone functions (must be defined elsewhere)
        if (typeof drawShotChart === 'function') drawShotChart(latestShots);
        if (typeof updateZones === 'function') updateZones(latestShots);
        if (typeof updateArcTrend === 'function') updateArcTrend(b.avg_arc, b.avg_entry_angle);

        updateShotTable(latestShots);

    } catch (err) {
        console.error('Stats update error:', err);
        const statusDotEl = document.getElementById('status-dot');
        if (statusDotEl) statusDotEl.className = 'status-dot';
        setText('status-text', 'Offline');
    }
}

// ── Shot Table Update ─────────────────────────────────────────────
function updateShotTable(shots) {
    const tbody = document.getElementById('shot-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    shots.forEach((shot, index) => {
        const row = document.createElement('tr');

        const result = shot.made ? 'Make' : 'Miss';
        const swish = shot.shot_type === 'Swish' ? 'Swish' : 'Miss';
        const backboard = shot.shot_type.includes('Backboard') ? 'Hit' : 'Miss';

        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${backboard}</td>
            <td>${swish}</td>
            <td>${result}</td>
        `;

        tbody.appendChild(row);
    });
}

// ── Export Functions ─────────────────────────────────────────────
function exportCSV() {
    if (!latestShots.length) {
        alert('No shot data to export.');
        return;
    }

    const header = ['#', 'Backboard', 'Swish', 'Result'];

    const rows = latestShots.map((shot, i) => [
        i + 1,
        shot.shot_type.includes('Backboard') ? 'Hit' : 'Miss',
        shot.shot_type === 'Swish' ? 'Swish' : 'Miss',
        shot.made ? 'Make' : 'Miss'
    ]);

    const csv = [header, ...rows]
        .map(r => r.join(','))
        .join('\n');

    downloadFile(csv, 'shot_table.csv', 'text/csv');
}

function exportStatsJSON() {
    if (!latestStats || !latestStats.basketball) {
        alert('No stats available.');
        return;
    }

    const json = JSON.stringify(latestStats, null, 2);
    downloadFile(json, 'stats.json', 'application/json');
}

// ── Auto-run on DOM ready ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    setInterval(updateStats, 2000);

    // Export buttons
    document.getElementById('export-csv-btn')
        ?.addEventListener('click', exportCSV);

    document.getElementById('export-json-btn')
        ?.addEventListener('click', exportStatsJSON);
});