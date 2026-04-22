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

// ── Export State ───────────────────────────────
let latestShots = [];
let latestStats = {};
let previousBackendShots = [];

// Simulation
window.simulateShots = false;
window.simulatedShotsAdded = 0;
window.guestShots = [];
window.newSession = true;

// Pagination
let currentPage = 1;
const shotsPerPage = 10;

// 🔥 render lock (prevents UI overwrite bugs)
let isRendering = false;
let shotsToRender = [];

// ── Helpers ────────────────────────────────────────────────────────
function fmt(v, suffix = '') {
    return (v === null || v === undefined) ? '--' : v + suffix;
}

function resetSessionState() {
    latestShots = [];
    latestStats = {};
    previousBackendShots = [];

    window._simulatedShotBuffer = [];
    window._lastSimTime = 0;

    updateShotTable([]);
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

        let shotsToRender = [];

        if (window.newSession) {
            console.log("🟡 Waiting for first new shot...");
                
            // Just establish baseline — DO NOT CLEAR
            if (previousBackendShots.length === 0) {
                previousBackendShots = backendShots;
            }
        }

        // ── Simulate Shots for Testing (time-based) (Comment if not using simulateShots) ─────────────────────────
        if (!window.simulateShots) {
            window._simulatedShotBuffer = [];
        }
        
        if (window.simulateShots) {
        
            // Initialize timer
            if (!window._lastSimTime) {
                window._lastSimTime = Date.now();
            }
        
            const now = Date.now();
        
            // Add a new shot every 1 second
            if (now - window._lastSimTime > 1000) {
            
                const rand = Math.random(); // 👈 single random value
            
                let newShot;
            
                if (rand > 0.75) {
                    newShot = {
                        made: true,
                        shot_type: 'Swish'
                    };
                
                } else if (rand > 0.15) {
                    const made = Math.random() > 0.25;
                
                    newShot = {
                        made: made,
                        shot_type: made ? 'Backboard Make' : 'Backboard Miss'
                    };
                
                } else {
                    newShot = {
                        made: false,
                        shot_type: 'Airball'
                    };
                }
            
                if (!window._simulatedShotBuffer) {
                    window._simulatedShotBuffer = [];
                }
            
                window._simulatedShotBuffer.push(newShot);
            
                window._lastSimTime = now;
            
                console.log("🧪 Simulated new shot:", newShot);
            }
        
            // Merge simulated shots into backend view
            if (window.simulateShots && window._simulatedShotBuffer?.length) {
                backendShots = [...backendShots, ...window._simulatedShotBuffer];
            }
        
            // 👇 Store for guests
            const user = getSession();
            if (!user) {
                window.guestShots = backendShots;
                latestShots = window.guestShots;
            
            }
        }

        // ── User session handling ─────────────────────
        const user = getSession();

        if (user) {
            const data = getUserData();
            const userData = data[user];
        
            if (!userData) {
                shotsToRender = [];
            } else {
            
                if (backendShots.length > previousBackendShots.length) {
                    const newShots = backendShots.slice(previousBackendShots.length);
                
                    if (window.newSession) {
                        data[user].shots = [];
                        window.newSession = false;
                    }
                
                    data[user].shots = [...(data[user].shots || []), ...newShots];
                
                    saveUserData(data);
                
                    const totalShots = data[user].shots.length;
                    const totalPages = Math.ceil(totalShots / shotsPerPage);
                
                    currentPage = totalPages; // auto jump to latest page
                    updateShotTable(data[user].shots);
                }
            
                shotsToRender = [...(data[user].shots || [])];
            }
        
            previousBackendShots = backendShots;
        
        } else {
            if (!window.guestShots) window.guestShots = [];
        
            if (backendShots.length > window.guestShots.length) {
                const newShots = backendShots.slice(window.guestShots.length);
                window.guestShots = [...window.guestShots, ...newShots];
            }
        
            shotsToRender = [...window.guestShots];
        }

        latestShots = shotsToRender;
        const computedStats = computeStatsFromShots(latestShots);

        latestStats = {};

        // ── Stats UI (NOW DERIVED FROM TABLE) ──────────────────────
        setText('attempts', fmt(computedStats.attempts));
        setText('makes', fmt(computedStats.makes));
        setText('fg-pct', computedStats.fg_percent ? computedStats.fg_percent + '%' : '--');

        setText('swishes', fmt(computedStats.swishes));
        setText('backboard', fmt(computedStats.backboard_hits));

        // Breakdown
        setText('swish-makes', fmt(computedStats.swishes));
        setText('backboard-makes', fmt(computedStats.backboard_makes));
        setText('backboard-misses', fmt(computedStats.backboard_misses));

        const airballs = computedStats.airballs;
        setText('airballs', fmt(airballs));

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
        latestShots = shotsToRender;
        renderAll(latestShots);

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

    const start = (currentPage - 1) * shotsPerPage;
    const end = start + shotsPerPage;

    const pageShots = shots.slice(start, end);

    pageShots.forEach((shot, index) => {
        const row = document.createElement('tr');

        const type = shot.shot_type || '';
        const result    = shot.made ? '✔ Make' : '✖ Miss';
        const swish     = type === 'Swish' ? '✔ Swish' : '✖ Miss';
        const backboard = (type.includes('Backboard') || type === 'Make' || type === 'Rim/Board Make') ? '✔ Hit' : '✖ Miss';
      
        const resultClass    = shot.made ? 'make' : 'miss';
        const swishClass     = type === 'Swish' ? 'swish' : 'miss';
        const backboardClass = (type.includes('Backboard') || type === 'Make' || type === 'Rim/Board Make') ? 'make' : 'miss';

        row.innerHTML = `
            <td>${start + index + 1}</td>
            <td class="${backboardClass}">${backboard}</td>
            <td class="${swishClass}">${swish}</td>
            <td class="${resultClass}">${result}</td>
        `;

        tbody.appendChild(row);
    });

    // Fill remaining rows to always show 10
    for (let i = pageShots.length; i < shotsPerPage; i++) {
        const row = document.createElement('tr');
    
        row.innerHTML = `
            <td>${start + i + 1}</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
        `;
    
        tbody.appendChild(row);
    }

    updatePaginationControls(shots.length);
    renderPageNumbers(shots.length);
    currentPage = Math.max(1, currentPage);
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
        airballs: 0
    };

    shots.forEach(shot => {
        if (shot.made) stats.makes++;

        if ((shot.shot_type || '') === 'Swish') {
            stats.swishes++;
        }

        if ((shot.shot_type || '').includes('Backboard')) {
            stats.backboard_hits++;

            if (shot.made) stats.backboard_makes++;
            else stats.backboard_misses++;
        }
        
        if ((shot.shot_type || '') === 'Airball') {
            stats.airballs++;
        }
    });

    stats.fg_percent = stats.attempts
        ? ((stats.makes / stats.attempts) * 100).toFixed(1)
        : null;

    return stats;
}

// ── Load User Session Data ─────────────────────────────────────────
function loadUserSessionData(email) {
    const data = getUserData(); // from auth.js
    const userData = data[email];

    if (!userData) {
        console.log("No saved data for user:", email);

        latestShots = [];
        updateShotTable([]);
        previousBackendShots = [];
        return;
    }

    console.log("Loaded user session:", userData);

    // Set global state
    latestShots = userData.shots || [];

    // Render table
    renderAll(shotsToRender);

    // (Optional future)
    // updateStatsUI(userData.stats);

    previousBackendShots = [];
}

function updatePaginationControls(totalShots) {
    const totalPages = Math.ceil(totalShots / shotsPerPage);

    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (prevBtn) prevBtn.disabled = currentPage === 1;
    if (nextBtn) nextBtn.disabled = currentPage === totalPages;
}

function renderPageNumbers(totalShots) {
    const container = document.getElementById('page-numbers');
    if (!container) return;

    container.innerHTML = '';

    const totalPages = Math.ceil(totalShots / shotsPerPage);

    for (let i = 1; i <= totalPages; i++) {
        const btn = document.createElement('button');
        btn.textContent = i;

        if (i === currentPage) {
            btn.classList.add('active-page');
        }

        btn.addEventListener('click', () => {
            currentPage = i;
            updateShotTable(latestShots);
        });

        container.appendChild(btn);
    }
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
    setText('backboard', '--');
    setText('swish-makes', '--');
    setText('backboard-makes', '--');
    setText('backboard-misses', '--');

    console.log("✅ Reset complete");
}

function renderAll(shots) {
    if (isRendering) return;
    isRendering = true;

    try {
        if (!Array.isArray(shots)) shots = [];

        updateShotTable(shots);

        if (typeof drawShotChart === 'function') drawShotChart(latestShots);
        if (typeof updateZones === 'function') updateZones(latestShots);

    } finally {
        isRendering = false;
    }
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
        (shot.shot_type || '').includes('Backboard') ? 'Hit' : 'Miss',
        (shot.shot_type || '') === 'Swish' ? 'Swish' : 'Miss',
        shot.made ? 'Make' : 'Miss'
    ]);

    const csv = [header, ...rows]
        .map(r => r.join(','))
        .join('\n');

    downloadFile(csv, 'shot_table.csv', 'text/csv');
}

function exportStatsJSON() {
    const stats = computeStatsFromShots(latestShots);

    if (!stats) {
        alert('No stats available.');
        return;
    }

    const json = JSON.stringify(stats, null, 2);
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

document.getElementById('prev-btn')?.addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        updateShotTable(latestShots);
    }
});

document.getElementById('next-btn')?.addEventListener('click', () => {
    const totalPages = Math.ceil(latestShots.length / shotsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        updateShotTable(latestShots);
    }
});

window.updateStats = updateStats;
