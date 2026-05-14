/* Interactive k-means clustering viz.
 * Step button alternates between the two halves of Lloyd's algorithm:
 *   (1) ASSIGN — each point goes to its nearest centroid (colour update)
 *   (2) UPDATE — each centroid moves to the mean of its assigned points
 * The split makes the convergence visible: assignment redraws colours,
 * update slides the X markers. Watch the algorithm settle, then try the
 * Moons preset to see where k-means fails. */

(function () {
    const canvas    = document.getElementById('viz-cluster-canvas');
    const stepBtn   = document.getElementById('viz-cluster-step');
    const autoBtn   = document.getElementById('viz-cluster-auto');
    const resetBtn  = document.getElementById('viz-cluster-reset');
    const kSel      = document.getElementById('viz-cluster-k');
    const dataSel   = document.getElementById('viz-cluster-data');
    const phaseEl   = document.getElementById('viz-cluster-phase');
    const captionEl = document.getElementById('viz-cluster-caption');
    if (!canvas) return;

    // ----- State -----
    let K = 3;
    let dataset = 'blobs';
    let points = [];               // each: { x, y, c (cluster idx) }
    let centroids = [];            // each: { x, y }
    let phase = 'assign';          // 'assign' next, then 'update', etc.
    let iter = 0;
    let converged = false;
    let autoPlaying = false;
    let lastStep = 0;
    const STEP_MS = 850;

    // Colour palette per cluster
    const PALETTE = [
        '#4f46e5',  // indigo
        '#ea7959',  // terracotta
        '#10847e',  // teal
        '#d4a13c',  // mustard
        '#a05bb0',  // plum
        '#688f5b',  // sage
    ];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'blobs') {
            const centers = [[-0.55, 0.45], [0.55, 0.5], [0.0, -0.55], [-0.65, -0.4]];
            const counts  = [60, 55, 65, 0];                  // last one silent on 3-blob
            for (let c = 0; c < centers.length; c++) {
                for (let i = 0; i < counts[c]; i++) {
                    pts.push({
                        x: centers[c][0] + randn() * 0.13,
                        y: centers[c][1] + randn() * 0.13,
                        c: 0,
                    });
                }
            }
        } else if (dataset === 'moons') {
            for (let i = 0; i < 80; i++) {
                const a = Math.PI * (i / 79);
                pts.push({
                    x: Math.cos(a) * 0.7 - 0.25 + randn() * 0.05,
                    y: Math.sin(a) * 0.5 + 0.0 + randn() * 0.05,
                    c: 0,
                });
            }
            for (let i = 0; i < 80; i++) {
                const a = Math.PI * (i / 79);
                pts.push({
                    x:  Math.cos(a) * 0.7 + 0.25 + randn() * 0.05,
                    y: -Math.sin(a) * 0.5 + 0.15 + randn() * 0.05,
                    c: 0,
                });
            }
        } else if (dataset === 'anisotropic') {
            // Tilted elongated blobs — k-means struggles even with the right K
            for (let i = 0; i < 90; i++) {
                const t = randn();
                pts.push({
                    x:  0.6 * t + randn() * 0.08 - 0.45,
                    y: -0.25 * t + randn() * 0.08 + 0.3,
                    c: 0,
                });
            }
            for (let i = 0; i < 90; i++) {
                const t = randn();
                pts.push({
                    x: 0.6 * t + randn() * 0.08 + 0.45,
                    y: 0.25 * t + randn() * 0.08 - 0.3,
                    c: 0,
                });
            }
        } else if (dataset === 'uneven') {
            // One dense and one sparse cluster of very different sizes
            for (let i = 0; i < 120; i++) {
                pts.push({ x: -0.45 + randn() * 0.09, y: 0.2 + randn() * 0.09, c: 0 });
            }
            for (let i = 0; i < 28; i++) {
                pts.push({ x: 0.5 + randn() * 0.25, y: -0.35 + randn() * 0.25, c: 0 });
            }
        }
        return pts;
    }

    function initCentroids() {
        // k-means++ flavoured: first random point, subsequent picked
        // proportional to distance² from nearest existing centroid.
        if (points.length === 0) { centroids = []; return; }
        const cs = [];
        cs.push({ ...points[Math.floor(Math.random() * points.length)] });
        while (cs.length < K) {
            const d2s = points.map(p => {
                let m = Infinity;
                for (const c of cs) {
                    const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
                    if (d < m) m = d;
                }
                return m;
            });
            const total = d2s.reduce((a, b) => a + b, 0);
            let r = Math.random() * total, i = 0;
            for (; i < d2s.length && r > d2s[i]; i++) r -= d2s[i];
            cs.push({ x: points[Math.min(i, points.length - 1)].x,
                      y: points[Math.min(i, points.length - 1)].y });
        }
        centroids = cs;
    }

    function reset() {
        points = buildDataset();
        initCentroids();
        for (const p of points) p.c = 0;
        phase = 'assign';
        iter = 0;
        converged = false;
        autoPlaying = false;
        updateAutoBtn();
    }

    // ----- One half-iteration of Lloyd's algorithm -----
    function assignStep() {
        for (const p of points) {
            let best = 0, bestD = Infinity;
            for (let i = 0; i < centroids.length; i++) {
                const d = (p.x - centroids[i].x) ** 2 + (p.y - centroids[i].y) ** 2;
                if (d < bestD) { bestD = d; best = i; }
            }
            p.c = best;
        }
    }
    function updateStep() {
        let totalShift = 0;
        for (let i = 0; i < centroids.length; i++) {
            let sx = 0, sy = 0, n = 0;
            for (const p of points) if (p.c === i) { sx += p.x; sy += p.y; n++; }
            if (n === 0) continue;
            const nx = sx / n, ny = sy / n;
            totalShift += Math.hypot(nx - centroids[i].x, ny - centroids[i].y);
            centroids[i].x = nx;
            centroids[i].y = ny;
        }
        return totalShift;
    }

    function step() {
        if (converged) return;
        if (phase === 'assign') {
            assignStep();
            phase = 'update';
        } else {
            const shift = updateStep();
            phase = 'assign';
            iter++;
            if (shift < 0.003) converged = true;
        }
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(320, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Faint frame
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.lineWidth = 1;
        const m = Math.min(W, H) / 2 - 12;
        ctx.strokeRect(W / 2 - m - 4, H / 2 - m - 4, 2 * m + 8, 2 * m + 8);

        // Lines from each point to its centroid (very faint), only after first
        // assign step so it doesn't look meaningless at iter 0.
        if (iter > 0 || phase === 'update') {
            ctx.lineWidth = 0.8;
            for (const p of points) {
                const c = centroids[p.c];
                if (!c) continue;
                const [px, py] = toPx(p.x, p.y);
                const [cx, cy] = toPx(c.x, c.y);
                ctx.strokeStyle = PALETTE[p.c % PALETTE.length] + '22';
                ctx.beginPath();
                ctx.moveTo(px, py); ctx.lineTo(cx, cy);
                ctx.stroke();
            }
        }

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            // Before any assignment, points are neutral grey
            const colour = (iter === 0 && phase === 'assign')
                ? 'rgba(70, 70, 70, 0.55)'
                : PALETTE[p.c % PALETTE.length];
            ctx.fillStyle = colour;
            ctx.beginPath();
            ctx.arc(px, py, 3.2, 0, Math.PI * 2);
            ctx.fill();
        }

        // Centroids — large X with halo
        for (let i = 0; i < centroids.length; i++) {
            const [cx, cy] = toPx(centroids[i].x, centroids[i].y);
            ctx.fillStyle = PALETTE[i % PALETTE.length] + '33';
            ctx.beginPath(); ctx.arc(cx, cy, 13, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = PALETTE[i % PALETTE.length];
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(cx - 7, cy - 7); ctx.lineTo(cx + 7, cy + 7);
            ctx.moveTo(cx + 7, cy - 7); ctx.lineTo(cx - 7, cy + 7);
            ctx.stroke();
        }

        // Phase / iter label inside the canvas, top-left
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const label = converged
            ? `converged after ${iter} iterations`
            : `iter ${iter} · next: ${phase.toUpperCase()}`;
        ctx.fillText(label, 12, 16);

        if (phaseEl) {
            phaseEl.textContent = converged ? `Converged · iter ${iter}`
                : `iter ${iter} · ${phase}`;
        }
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (iter === 0 && phase === 'assign') {
            captionEl.innerHTML =
                `<strong>K = ${K}</strong> centroids placed (k-means++). Every point is still uncoloured. ` +
                `Click <strong>Step</strong> — each point will jump to its nearest centroid's colour.`;
        } else if (converged) {
            const data = {
                blobs: `Perfect fit — k-means is at its best on round, well-separated blobs.`,
                moons: `Notice how the algorithm cut both moons in half? k-means assumes ` +
                       `convex, roughly equal-sized clusters — concentric shapes break it. ` +
                       `For this, try DBSCAN or spectral clustering.`,
                anisotropic: `The clusters are tilted ellipses, but k-means draws spherical ` +
                             `Voronoi regions — so the boundary slices through the middle. ` +
                             `A Gaussian Mixture would handle this.`,
                uneven: `The bigger cluster captured neighbours from the smaller one. ` +
                        `k-means doesn't account for cluster <em>density</em>, only distance.`,
            }[dataset] || '';
            captionEl.innerHTML = `<strong>Converged after ${iter} iterations.</strong> ${data}`;
        } else if (phase === 'update') {
            captionEl.innerHTML =
                `<strong>Iter ${iter}, assignment done.</strong> Every point now wears its ` +
                `nearest centroid's colour. Step again — each centroid will slide to the ` +
                `mean position of the points that just joined it.`;
        } else {
            captionEl.innerHTML =
                `<strong>Iter ${iter}, centroids moved.</strong> The X markers shifted to ` +
                `their new means. Step again to re-assign each point — some near the ` +
                `boundary may change colour.`;
        }
    }

    // ----- Animation -----
    function updateAutoBtn() {
        if (autoBtn) autoBtn.textContent = autoPlaying ? 'Pause' : 'Auto';
    }
    function loop(now) {
        if (autoPlaying && !converged && now - lastStep >= STEP_MS) {
            step();
            lastStep = now;
        }
        if (converged) { autoPlaying = false; updateAutoBtn(); }
        requestAnimationFrame(loop);
    }

    // ----- Controls -----
    stepBtn?.addEventListener('click', step);
    autoBtn?.addEventListener('click', () => {
        if (converged) return;
        autoPlaying = !autoPlaying;
        updateAutoBtn();
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', () => { reset(); draw(); });

    if (kSel) {
        kSel.innerHTML = `<option value="2">K = 2</option>
            <option value="3" selected>K = 3</option>
            <option value="4">K = 4</option>
            <option value="5">K = 5</option>
            <option value="6">K = 6</option>`;
        kSel.addEventListener('change', () => {
            K = parseInt(kSel.value, 10);
            reset();
            draw();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="blobs">Blobs</option>
            <option value="moons">Two moons</option>
            <option value="anisotropic">Tilted ellipses</option>
            <option value="uneven">Uneven sizes</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
            draw();
        });
    }

    // ----- Init -----
    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    requestAnimationFrame(loop);
})();
