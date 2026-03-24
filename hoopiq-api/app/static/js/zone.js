/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/zone.js - Handles zone-based shot analysis
*/

// ── Zone Accuracy ──────────────────────────────────────────────────
 const ZONE_MAP = {
    'Left Corner 3':  { pct:'z-pct-lc', shots:'z-shots-lc', bar:'z-bar-lc' },
    'Right Corner 3': { pct:'z-pct-rc', shots:'z-shots-rc', bar:'z-bar-rc' },
    'Above Break 3':  { pct:'z-pct-ab', shots:'z-shots-ab', bar:'z-bar-ab' },
    'Mid-Range':      { pct:'z-pct-mr', shots:'z-shots-mr', bar:'z-bar-mr' },
    'Paint':          { pct:'z-pct-pt', shots:'z-shots-pt', bar:'z-bar-pt' },
};

 function updateZones(shotData) {
    if (!shotData || shotData.length === 0) return;

    // Aggregate by zone
    const zones = {};
    for (const z of Object.keys(ZONE_MAP)) {
        zones[z] = { makes: 0, attempts: 0 };
    }
    shotData.forEach(shot => {
        const z = shot.zone;
        if (zones[z]) {
            zones[z].attempts++;
            if (shot.made) zones[z].makes++;
        }
    });

    // Find best zone for badge
    let bestZone = null, bestPct = -1;
    for (const [name, data] of Object.entries(zones)) {
        if (data.attempts >= 2) {
            const pct = data.makes / data.attempts * 100;
            if (pct > bestPct) { bestPct = pct; bestZone = name; }
        }
    }
    const badge = document.getElementById('best-zone-badge');
    if (badge) badge.textContent = bestZone ? `Best: ${bestZone} (${bestPct.toFixed(0)}%)` : 'Best zone: --';

    // Update each zone card
    for (const [name, ids] of Object.entries(ZONE_MAP)) {
        const data = zones[name];
        const pctEl    = document.getElementById(ids.pct);
        const shotsEl  = document.getElementById(ids.shots);
        const barEl    = document.getElementById(ids.bar);
        if (!pctEl) continue;

        if (data.attempts === 0) {
            pctEl.textContent   = '--';
            shotsEl.textContent = '0 shots';
            barEl.style.width   = '0%';
            pctEl.className     = 'zone-pct zone-cold';
            continue;
        }

        const pct = data.makes / data.attempts * 100;
        pctEl.textContent   = pct.toFixed(0) + '%';
        shotsEl.textContent = data.attempts + ' shot' + (data.attempts !== 1 ? 's' : '');
        barEl.style.width   = pct + '%';

        // Color by performance
        if (pct >= 50) {
            pctEl.className   = 'zone-pct zone-hot';
            barEl.style.background = 'var(--orange)';
        } else if (pct >= 30) {
            pctEl.className   = 'zone-pct zone-warm';
            barEl.style.background = 'var(--green)';
        } else {
            pctEl.className   = 'zone-pct zone-cold';
            barEl.style.background = 'rgba(0,255,136,0.3)';
        }
    }
}
