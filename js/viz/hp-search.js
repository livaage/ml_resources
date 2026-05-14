/* Interactive hyperparameter-search viz.
 * 2D objective with one "important" axis (left-right) and one "unimportant"
 * axis (up-down). Show 3 strategies and which points they sample.
 *   grid    — 5x5 — wastes half the budget on the unimportant axis
 *   random  — 25 uniform samples — better axis coverage
 *   bayes   — 10 random + 15 around the best so far */

(function () {
    const canvas    = document.getElementById('viz-hp-canvas');
    const gridBtn   = document.getElementById('viz-hp-grid');
    const randBtn   = document.getElementById('viz-hp-random');
    const bayesBtn  = document.getElementById('viz-hp-bayes');
    const resetBtn  = document.getElementById('viz-hp-reset');
    const captionEl = document.getElementById('viz-hp-caption');
    if (!canvas) return;

    let mode = 'grid';
    let seed = 1;
    let ctx;
    let W = 0, H = 0;

    // Objective: strong dependence on x (important), weak on y (unimportant)
    // Peak at (0.3, 0.5)
    function objective(x, y) {
        const dx = x - 0.3, dy = y - 0.5;
        return Math.exp(-(dx * dx) / 0.04) * (0.85 + 0.15 * Math.exp(-(dy * dy) / 0.5));
    }
    function mulberry32(s) {
        return function() {
            s |= 0; s = (s + 0x6D2B79F5) | 0;
            let t = s;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    function trials() {
        const out = [];
        if (mode === 'grid') {
            for (let i = 0; i < 5; i++) {
                for (let j = 0; j < 5; j++) {
                    out.push({ x: (i + 0.5) / 5, y: (j + 0.5) / 5 });
                }
            }
        } else if (mode === 'random') {
            const rng = mulberry32(seed);
            for (let i = 0; i < 25; i++) out.push({ x: rng(), y: rng() });
        } else if (mode === 'bayes') {
            const rng = mulberry32(seed + 100);
            // 10 random exploratory
            for (let i = 0; i < 10; i++) out.push({ x: rng(), y: rng() });
            // 15 informed — sample around the best so far with shrinking radius
            for (let i = 0; i < 15; i++) {
                let best = out[0], bestV = objective(out[0].x, out[0].y);
                for (const p of out) {
                    const v = objective(p.x, p.y);
                    if (v > bestV) { bestV = v; best = p; }
                }
                const r = 0.18 * Math.exp(-i / 10);
                const dx = (rng() - 0.5) * 2 * r;
                const dy = (rng() - 0.5) * 2 * r;
                out.push({
                    x: Math.max(0.02, Math.min(0.98, best.x + dx)),
                    y: Math.max(0.02, Math.min(0.98, best.y + dy)),
                });
            }
        }
        for (const p of out) p._v = objective(p.x, p.y);
        return out;
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.55)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 32;
        const size = Math.min(W - 320, H - 60);
        const box = { x: pad, y: 30, w: size, h: size };

        // Background heatmap
        const G = 60;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const xv = (i + 0.5) / G, yv = (j + 0.5) / G;
                const v = objective(xv, yv);
                const t = v;
                const r = 251 + (234 - 251) * t * 0.6;
                const g = 250 + (121 - 250) * t * 0.6;
                const b = 247 + (89 - 247) * t * 0.6;
                ctx.fillStyle = `rgb(${r|0}, ${g|0}, ${b|0})`;
                const px = box.x + xv * box.w, py = box.y + (1 - yv) * box.h;
                ctx.fillRect(px - box.w / G / 2 - 1, py - box.h / G / 2 - 1,
                             box.w / G + 2, box.h / G + 2);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Trial points
        const tr = trials();
        let bestX, bestY, bestV = -Infinity;
        for (const p of tr) if (p._v > bestV) { bestV = p._v; bestX = p.x; bestY = p.y; }

        for (const p of tr) {
            const px = box.x + p.x * box.w;
            const py = box.y + (1 - p.y) * box.h;
            ctx.fillStyle = '#4f46e5';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }
        // Best
        const bpx = box.x + bestX * box.w, bpy = box.y + (1 - bestY) * box.h;
        ctx.fillStyle = '#ea7959';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(bpx, bpy, 7, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();

        // Axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('important hyperparameter (e.g., lr)',
                     box.x + box.w / 2, box.y + box.h + 22);
        ctx.save();
        ctx.translate(box.x - 18, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('unimportant hp (e.g., random seed)', 0, 0);
        ctx.restore();

        // Title + best
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { grid: 'GRID SEARCH — 5×5 = 25 trials',
                         random: 'RANDOM SEARCH — 25 trials',
                         bayes: 'BAYESIAN — 10 random + 15 informed' };
        ctx.fillText(titles[mode], box.x, box.y - 8);

        // Sidebar
        let lx = box.x + box.w + 24, ly = box.y + 12;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.fillText('BEST', lx, ly); ly += 16;
        ctx.font = '700 24px "JetBrains Mono", monospace';
        ctx.fillStyle = '#ea7959';
        ctx.fillText(`${(bestV * 100).toFixed(1)}%`, lx, ly + 8);
        ly += 28;
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText(`x = ${bestX.toFixed(2)}`, lx, ly); ly += 14;
        ctx.fillText(`y = ${bestY.toFixed(2)}`, lx, ly); ly += 24;
        // Unique x values (proxy for "important axis coverage")
        const uniqueX = new Set(tr.map(p => p.x.toFixed(2))).size;
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText(`unique x: ${uniqueX} / 25`, lx, ly); ly += 14;
        ctx.fillText(`trials: ${tr.length}`, lx, ly);

        updateCaption(bestV, uniqueX, tr.length);
    }

    function updateCaption(bestV, uniqueX, n) {
        if (!captionEl) return;
        const notes = {
            grid:   `<strong>Grid 5×5.</strong> 5 unique values of each axis. Of the 25 trials, only 5 are exploring the "important" dimension — the other 20 are duplicating exploration on the unimportant axis.`,
            random: `<strong>Random 25.</strong> All 25 x-values are distinct, so the important axis is fully explored. Best objective ${(bestV * 100).toFixed(0)}% — usually beats grid for the same budget.`,
            bayes:  `<strong>Bayesian.</strong> 10 random exploratory trials map the landscape; the next 15 cluster around the best regions. Best ${(bestV * 100).toFixed(0)}%. The dense cluster of points is the model "zooming in" on the peak.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [gridBtn, randBtn, bayesBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    gridBtn?.addEventListener('click',  () => setMode('grid', gridBtn));
    randBtn?.addEventListener('click',  () => setMode('random', randBtn));
    bayesBtn?.addEventListener('click', () => setMode('bayes', bayesBtn));
    resetBtn?.addEventListener('click', () => { seed = Math.floor(Math.random() * 1000); draw(); });

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
