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
let previousBackendShots = [];

// Code for testing wihtout esp32
window.simulateShots = true;
window.simulatedShotsAdded = 0;

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

        // ── System Status ─────────────────────────────
        const online = d.system?.online ?? false;
        setText('status-dot', online ? '●' : '○');

        const statusDotEl = document.getElementById('status-dot');
        if (statusDotEl) {
            statusDotEl.className = 'status-dot' + (online ? ' online' : '');
        }

        setText('status-text', online ? 'Online' : 'Offline');

        const b = d.basketball || {};
        const s = d.sensors || {};
        const h = d.heatmap || {};

        let backendShots = b.shot_chart || [];

        // ── Simulate Shots for Testing (comment to turn off) ─────────────────────────────
        if (window.simulateShots) {
            if (backendShots.length === window.simulatedShotsAdded) {
            
                const isSwish = Math.random() > 0.75;
            
                let newShot;
            
                if (isSwish) {
                    // ✅ Swish is always a make
                    newShot = {
                        made: true,
                        shot_type: 'Swish'
                    };
                } else {
                    // ✅ Backboard can be make or miss
                    const made = Math.random() > 0.5;
                
                    newShot = {
                        made: made,
                        shot_type: made ? 'Backboard Make' : 'Backboard Miss'
                    };
                }
            
                // Store simulated shots separately
                if (!window._simulatedShotBuffer) {
                    window._simulatedShotBuffer = [];
                }
            
                window._simulatedShotBuffer.push(newShot);
                window.simulatedShotsAdded++;
            
                console.log("🧪 Simulated new shot:", newShot);
            }
        
            // Merge simulated shots into backend view
            if (window._simulatedShotBuffer) {
                backendShots = [...backendShots, ...window._simulatedShotBuffer];
            }
        }

        // ── User session handling ─────────────────────
        const user = getSession();

        if (user) {
            const data = getUserData();
            const userData = data[user];

            if (userData) {
                const storedShots = userData.shots || [];

                // Detect new backend shots
                if (backendShots.length > previousBackendShots.length) {
                    const newShots = backendShots.slice(previousBackendShots.length);

                    console.log("New shots detected:", newShots);

                    // Merge into user storage
                    data[user].shots = [...storedShots, ...newShots];
                    saveUserData(data);
                }

                // Always render from merged local history
                latestShots = data[user].shots || [];

                previousBackendShots = backendShots;
            }
        } else {
            // Guest mode → use backend directly
            latestShots = backendShots;
        }

        const computedStats = computeStatsFromShots(latestShots);

        latestStats = {};

        // ── Stats UI (NOW DERIVED FROM TABLE) ──────────────────────
        setText('attempts', fmt(computedStats.attempts));
        setText('makes', fmt(computedStats.makes));
        setText('fg-pct', computedStats.fg_percent ? computedStats.fg_percent + '%' : '--');

        setText('swishes', fmt(computedStats.swishes));
        setText('rim', fmt(computedStats.rim_hits));
        setText('backboard', fmt(computedStats.backboard_hits));

        // Breakdown
        setText('swish-makes', fmt(computedStats.swishes));
        setText('backboard-makes', fmt(computedStats.backboard_makes));
        setText('backboard-misses', fmt(computedStats.backboard_misses));

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

        // ── Visual Components ─────────────────────────
        if (typeof drawShotChart === 'function') drawShotChart(latestShots);
        if (typeof updateZones === 'function') updateZones(latestShots);
        if (typeof updateArcTrend === 'function') updateArcTrend(b.avg_arc, b.avg_entry_angle);

        // ── Table ─────────────────────────────────────
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

        const result = shot.made ? '✔ Make' : '✖ Miss';
        const swish = shot.shot_type === 'Swish' ? '✔ Swish' : '✖ Miss';
        const backboard = shot.shot_type.includes('Backboard') ? '✔ Hit' : '✖ Miss';
            
        const resultClass = shot.made ? 'make' : 'miss';
        const swishClass = shot.shot_type === 'Swish' ? 'swish' : 'miss';
        const backboardClass = shot.shot_type.includes('Backboard') ? 'make' : 'miss';
            
        row.innerHTML = `
            <td>${index + 1}</td>
            <td class="${backboardClass}">${backboard}</td>
            <td class="${swishClass}">${swish}</td>
            <td class="${resultClass}">${result}</td>
        `;

        tbody.appendChild(row);
    });
}

// ── Compute Stats from Shots (for local storage) ─────────────────
function computeStatsFromShots(shots) {
    const stats = {
        attempts: shots.length,
        makes: 0,
        swishes: 0,
        backboard_hits: 0,
        backboard_makes: 0,
        backboard_misses: 0,
        rim_hits: 0
    };

    shots.forEach(shot => {
        if (shot.made) stats.makes++;

        if (shot.shot_type === 'Swish') {
            stats.swishes++;
        }

        if (shot.shot_type.includes('Backboard')) {
            stats.backboard_hits++;

            if (shot.made) stats.backboard_makes++;
            else stats.backboard_misses++;
        }

        if (shot.shot_type === 'Rim Hit') {
            stats.rim_hits++;
        }
    });

    stats.fg_percent = stats.attempts
        ? ((stats.makes / stats.attempts) * 100).toFixed(1)
        : null;

    return stats;
}

// ── Load User Session Data ─────────────────────────────────────────
function loadUserSessionData(username) {
    const data = getUserData(); // from auth.js
    const userData = data[username];

    if (!userData) {
        console.log("No saved data for user:", username);

        latestShots = [];
        updateShotTable([]);
        return;
    }

    console.log("Loaded user session:", userData);

    // Set global state
    latestShots = userData.shots || [];

    // Render table
    updateShotTable(latestShots);

    // (Optional future)
    // updateStatsUI(userData.stats);
}

// ── Reset Data (for testing) ─────────────────────────────────────
function resetAllData() {
    const user = getSession();

    if (user) {
        const data = getUserData();

        if (data[user]) {
            data[user].shots = [];
            data[user].stats = {};
            saveUserData(data);
        }
    }

    latestShots = [];
    latestStats = {};

    // Clear table
    updateShotTable([]);

    // Reset stats UI
    setText('attempts', '--');
    setText('makes', '--');
    setText('fg-pct', '--');
    setText('swishes', '--');
    setText('rim', '--');
    setText('backboard', '--');
    setText('swish-makes', '--');
    setText('backboard-makes', '--');
    setText('backboard-misses', '--');

    console.log("✅ Reset complete");
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

    // Reset button
    document.getElementById('reset-btn')
        ?.addEventListener('click', resetAllData);

    const user = getSession();
    if (user) {
        loadUserSessionData(user);
    }
});

window.updateStats = updateStats;