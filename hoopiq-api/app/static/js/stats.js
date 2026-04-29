/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/stats.js - Handles shot statistics and charting
*/

const CLOUD = 'https://cecs490-senior-project.onrender.com';

let latestShots = [];
let latestStats = {};
let previousBackendShots = [];

window.latestShots = latestShots;

// Simulation
window.simulateShots = false;
window.simulatedShotsAdded = 0;
window.guestShots = [];
window.newSession = true;

// Pagination
let currentPage = 1;
const shotsPerPage = 10;

// Render lock
let isRendering = false;

// Helpers
function fmt(v, suffix = '') {
    return (v === null || v === undefined) ? '--' : v + suffix;
}

function resetSessionState() {
    latestShots = [];
    latestStats = {};
    previousBackendShots = [];

    window.latestShots = [];
    window._simulatedShotBuffer = [];
    window._lastSimTime = 0;

    updateShotTable([]);
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

function normalizeShotType(shot) {
    if (shot.points === 3 || shot.shot_value === 3 || shot.shot_type === '3PT' || shot.shot_type === '3pt') {
        return '3PT';
    }
    return '2PT';
}

function normalizeShotQuality(shot) {
    const type = shot.shot_type || '';
    const quality = shot.shot_quality || '';

    if (quality) return quality;
    if (type === 'Swish') return 'Swish';
    if (type.includes('Backboard')) return 'Backboard';
    if (type === 'Airball') return 'Airball';

    return 'Standard';
}

// Main Stats Function
async function updateStats() {
    try {
        const r = await fetch(`${CLOUD}/stats`, { cache: 'no-store' });
        if (!r.ok) throw new Error('Failed to fetch stats');

        const d = await r.json();

        const online = d.system?.online ?? false;
        setText('status-dot', online ? '●' : '○');

        const statusDotEl = document.getElementById('status-dot');
        if (statusDotEl) {
            statusDotEl.className = 'status-dot' + (online ? ' online' : '');
        }

        setText('status-text', online ? 'Online' : 'Offline');

        if (online && typeof isSessionTimerRunning === 'function' && !isSessionTimerRunning()) {
            startSessionTimer();
        }

        if (!online && typeof isSessionTimerRunning === 'function' && isSessionTimerRunning()) {
            stopSessionTimer();
        }

        const b = d.basketball || {};
        const h = d.heatmap || {};

        let backendShots = b.shot_chart || [];
        let shotsToRender = [];

        if (window.newSession) {
            if (previousBackendShots.length === 0) {
                previousBackendShots = backendShots;
            }
        }

        if (!window.simulateShots) {
            window._simulatedShotBuffer = [];
        }

        if (window.simulateShots) {
            if (!window._lastSimTime) {
                window._lastSimTime = Date.now();
            }

            const now = Date.now();

            if (now - window._lastSimTime > 1000) {
                const rand = Math.random();
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
            }

            if (window.simulateShots && window._simulatedShotBuffer?.length) {
                backendShots = [...backendShots, ...window._simulatedShotBuffer];
            }

            const guestUser = getSession();
            if (!guestUser) {
                window.guestShots = backendShots;
                latestShots = window.guestShots;
            }
        }

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

                    currentPage = totalPages || 1;
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
        window.latestShots = latestShots;

        const computedStats = computeStatsFromShots(latestShots);
        latestStats = computedStats;

        setText('makes', fmt(computedStats.makes));
        setText('fg-pct', computedStats.fg_percent != null ? computedStats.fg_percent + '%' : '--');
        setText('two-pct', computedStats.two_pt_percent != null ? computedStats.two_pt_percent + '%' : '--');
        setText('three-pct', computedStats.three_pt_percent != null ? computedStats.three_pt_percent + '%' : '--');
        setText('streak', fmt(computedStats.streak));
        setText('airballs', fmt(computedStats.airballs));

        setText('players', fmt(h.current_players));

        const arcStreakBadge = document.getElementById('arc-streak-badge');
        if (arcStreakBadge) arcStreakBadge.textContent = 'Streak: ' + fmt(computedStats.streak);

        if (typeof drawShotChart === 'function') drawShotChart(latestShots);
        if (typeof drawShotTimeline === 'function') drawShotTimeline(latestShots);

        renderAll(latestShots);
    } catch (err) {
        console.error('Stats update error:', err);

        const statusDotEl = document.getElementById('status-dot');
        if (statusDotEl) statusDotEl.className = 'status-dot';

        setText('status-text', 'Offline');

        if (typeof isSessionTimerRunning === 'function' && isSessionTimerRunning()) {
            stopSessionTimer();
        }
    }
}

function updateShotTable(shots) {
    const tbody = document.getElementById('shot-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    const start = (currentPage - 1) * shotsPerPage;
    const end = start + shotsPerPage;
    const pageShots = shots.slice(start, end);

    pageShots.forEach((shot, index) => {
        const row = document.createElement('tr');

        const result = shot.made ? '✔ Make' : '✖ Miss';
        const resultClass = shot.made ? 'make' : 'miss';
        const shotType = normalizeShotType(shot);
        const shotQuality = normalizeShotQuality(shot);

        row.innerHTML = `
            <td>${start + index + 1}</td>
            <td class="${resultClass}">${result}</td>
            <td>${shotType}</td>
            <td>${shotQuality}</td>
        `;

        tbody.appendChild(row);
    });

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

function computeStatsFromShots(shots) {
    const stats = {
        attempts: shots.length,
        makes: 0,
        airballs: 0,
        streak: 0,
        two_pt_attempts: 0,
        two_pt_makes: 0,
        three_pt_attempts: 0,
        three_pt_makes: 0
    };

    let runningStreak = 0;

    shots.forEach(shot => {
        if (shot.made) {
            stats.makes++;
            runningStreak++;
        } else {
            runningStreak = 0;
        }

        const shotType = normalizeShotType(shot);

        if (shotType === '3PT') {
            stats.three_pt_attempts++;
            if (shot.made) stats.three_pt_makes++;
        } else {
            stats.two_pt_attempts++;
            if (shot.made) stats.two_pt_makes++;
        }

        if ((shot.shot_type || '') === 'Airball' || (shot.shot_quality || '') === 'Airball') {
            stats.airballs++;
        }
    });

    stats.streak = runningStreak;

    stats.fg_percent = stats.attempts
        ? ((stats.makes / stats.attempts) * 100).toFixed(1)
        : null;

    stats.two_pt_percent = stats.two_pt_attempts
        ? ((stats.two_pt_makes / stats.two_pt_attempts) * 100).toFixed(1)
        : null;

    stats.three_pt_percent = stats.three_pt_attempts
        ? ((stats.three_pt_makes / stats.three_pt_attempts) * 100).toFixed(1)
        : null;

    return stats;
}

function loadUserSessionData(email) {
    const data = getUserData();
    const userData = data[email];

    if (!userData) {
        latestShots = [];
        window.latestShots = [];
        updateShotTable([]);
        previousBackendShots = [];
        return;
    }

    latestShots = userData.shots || [];
    window.latestShots = latestShots;
    renderAll(latestShots);
    previousBackendShots = [];
}

function updatePaginationControls(totalShots) {
    const totalPages = Math.ceil(totalShots / shotsPerPage) || 1;

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
    window.latestShots = [];

    updateShotTable([]);

    setText('makes', '--');
    setText('fg-pct', '--');
    setText('two-pct', '--');
    setText('three-pct', '--');
    setText('streak', '--');
    setText('airballs', '--');
}

function renderAll(shots) {
    if (isRendering) return;
    isRendering = true;

    try {
        if (!Array.isArray(shots)) shots = [];
        updateShotTable(shots);

        if (typeof drawShotChart === 'function') drawShotChart(shots);
        if (typeof drawShotTimeline === 'function') drawShotTimeline(shots);
    } finally {
        isRendering = false;
    }
}

function exportCSV() {
    if (!latestShots.length) {
        alert('No shot data to export.');
        return;
    }

    const header = ['Shot #', 'Result', 'Shot Type', 'Shot Quality'];

    const rows = latestShots.map((shot, i) => [
        i + 1,
        shot.made ? 'Make' : 'Miss',
        normalizeShotType(shot),
        normalizeShotQuality(shot)
    ]);

    const csv = [header, ...rows]
        .map(r => r.join(','))
        .join('\n');

    downloadFile(csv, 'shot_table.csv', 'text/csv');
}

function exportStatsJSON() {
    const stats = computeStatsFromShots(latestShots);
    const json = JSON.stringify(stats, null, 2);
    downloadFile(json, 'stats.json', 'application/json');
}

document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    setInterval(updateStats, 2000);

    document.getElementById('export-csv-btn')?.addEventListener('click', exportCSV);
    document.getElementById('export-json-btn')?.addEventListener('click', exportStatsJSON);
    document.getElementById('reset-btn')?.addEventListener('click', resetAllData);

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
