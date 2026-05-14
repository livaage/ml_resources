/* Interactive Gaussian naive Bayes viz.
 * Each class is modelled by a per-feature Gaussian (diagonal covariance — the
 * "naive" independence assumption). The decision regions are coloured by
 * P(class | x); the per-class 1σ ellipses are drawn as axis-aligned shapes.
 * Marginal histograms along each axis show why the model is "naive" — every
 * 2D class density is just the product of two 1D densities. */

(function () {
    const canvas    = document.getElementById('viz-nb-canvas');
    const dataSel   = document.getElementById('viz-nb-data');
    const resetBtn  = document.getElementById('viz-nb-reset');
    const captionEl = document.getElementById('viz-nb-caption');
    if (!canvas) return;

    let dataset = 'separated';
    let points = [];        // { x, y, c }
    let stats = [];         // { mean: [mx, my], var: [vx, vy], prior }
    let numClasses = 2;
    let heatCache = null;
    const PALETTE = ['#4f46e5', '#ea7959', '#10847e'];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'separated') {
            for (let i = 0; i < 70; i++) pts.push({ x: -0.45 + randn() * 0.18, y:  0.30 + randn() * 0.18, c: 0 });
            for (let i = 0; i < 70; i++) pts.push({ x:  0.45 + randn() * 0.18, y: -0.30 + randn() * 0.18, c: 1 });
            numClasses = 2;
        } else if (dataset === 'overlap') {
            for (let i = 0; i < 80; i++) pts.push({ x: -0.20 + randn() * 0.25, y:  0.15 + randn() * 0.25, c: 0 });
            for (let i = 0; i < 80; i++) pts.push({ x:  0.20 + randn() * 0.25, y: -0.15 + randn() * 0.25, c: 1 });
            numClasses = 2;
        } else if (dataset === 'multiclass') {
            const cs = [[-0.45, 0.4], [0.45, 0.4], [0, -0.4]];
            for (let c = 0; c < 3; c++) {
                for (let i = 0; i < 60; i++) {
                    pts.push({ x: cs[c][0] + randn() * 0.13, y: cs[c][1] + randn() * 0.13, c });
                }
            }
            numClasses = 3;
        } else if (dataset === 'tilted') {
            // Tilted cluster — naive Bayes' axis-aligned assumption fails
            for (let i = 0; i < 80; i++) {
                const t = randn();
                pts.push({ x:  0.5 * t + randn() * 0.08 - 0.3, y:  0.3 * t + randn() * 0.08 + 0.2, c: 0 });
            }
            for (let i = 0; i < 80; i++) {
                const t = randn();
                pts.push({ x:  0.5 * t + randn() * 0.08 + 0.3, y:  0.3 * t + randn() * 0.08 - 0.2, c: 1 });
            }
            numClasses = 2;
        }
        return pts;
    }

    function fit() {
        const stats = [];
        for (let c = 0; c < numClasses; c++) {
            const cl = points.filter(p => p.c === c);
            const n = cl.length;
            if (n === 0) { stats.push({ mean: [0, 0], var: [0.1, 0.1], prior: 1 / numClasses }); continue; }
            const mx = cl.reduce((s, p) => s + p.x, 0) / n;
            const my = cl.reduce((s, p) => s + p.y, 0) / n;
            const vx = cl.reduce((s, p) => s + (p.x - mx) ** 2, 0) / n + 1e-4;
            const vy = cl.reduce((s, p) => s + (p.y - my) ** 2, 0) / n + 1e-4;
            stats.push({ mean: [mx, my], var: [vx, vy], prior: n / points.length });
        }
        return stats;
    }

    function logLikelihood(x, y, s) {
        const dx = x - s.mean[0], dy = y - s.mean[1];
        return -0.5 * (dx * dx / s.var[0] + dy * dy / s.var[1])
             - 0.5 * Math.log(s.var[0] * s.var[1]) + Math.log(s.prior + 1e-12);
    }

    function posterior(x, y) {
        const lls = stats.map(s => logLikelihood(x, y, s));
        const mx = Math.max(...lls);
        const exps = lls.map(l => Math.exp(l - mx));
        const z = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / z);
    }

    function reset() {
        points = buildDataset();
        stats = fit();
        heatCache = null;
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(330, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        heatCache = null;
        draw();
    }
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function classMix(probs) {
        // Mix PALETTE colours by probability
        let r = 251, g = 250, b = 247;
        const palRGB = [
            [79,  70,  229],
            [234, 121, 89],
            [16,  132, 126],
        ];
        let tot = 0;
        let mr = 0, mg = 0, mb = 0;
        for (let i = 0; i < probs.length; i++) {
            const p = probs[i] * 0.55;
            mr += palRGB[i][0] * p;
            mg += palRGB[i][1] * p;
            mb += palRGB[i][2] * p;
            tot += p;
        }
        const f = tot;
        return `rgb(${251 * (1 - f) + mr | 0}, ${250 * (1 - f) + mg | 0}, ${247 * (1 - f) + mb | 0})`;
    }

    function buildHeatmap() {
        if (heatCache) return heatCache;
        const G = 60;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.05;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.05;
                const probs = posterior(dx, dy);
                const col = classMix(probs);
                const m = col.match(/rgb\((\d+), (\d+), (\d+)\)/);
                const idx = (j * G + i) * 4;
                img.data[idx]     = +m[1];
                img.data[idx + 1] = +m[2];
                img.data[idx + 2] = +m[3];
                img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatCache = off;
        return off;
    }

    function drawEllipse(s, sigma, col, lw) {
        const m = Math.min(W, H) / 2 - 16;
        const [cx, cy] = toPx(s.mean[0], s.mean[1]);
        const rx = Math.sqrt(s.var[0]) * sigma * m;
        const ry = Math.sqrt(s.var[1]) * sigma * m;
        ctx.strokeStyle = col;
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        // Heatmap
        const m = Math.min(W, H) / 2 - 16;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(buildHeatmap(), W / 2 - m * 1.05, H / 2 - m * 1.05, m * 2.1, m * 2.1);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m * 1.05 - 0.5, H / 2 - m * 1.05 - 0.5, m * 2.1 + 1, m * 2.1 + 1);

        // Class-conditional 1σ + 2σ axis-aligned ellipses
        for (let c = 0; c < stats.length; c++) {
            drawEllipse(stats[c], 2, PALETTE[c] + '44', 1);
            drawEllipse(stats[c], 1, PALETTE[c],         2);
        }

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = PALETTE[p.c];
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 3.2, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`GAUSSIAN NAIVE BAYES — ${numClasses} classes, diagonal covariance`,
            W / 2 - m * 1.05, H / 2 - m * 1.05 - 6);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        let acc = 0;
        for (const p of points) {
            const probs = posterior(p.x, p.y);
            const pred = probs.indexOf(Math.max(...probs));
            if (pred === p.c) acc++;
        }
        const pct = (acc / points.length * 100).toFixed(0);
        const tiltedNote = dataset === 'tilted'
            ? ` Notice how the boundary tilts ⟂ to the data's true direction — that's the cost of the diagonal-covariance assumption.`
            : '';
        captionEl.innerHTML =
            `<strong>${pct}% train accuracy.</strong> Each ellipse is one class's axis-aligned 1σ/2σ region — ` +
            `axis-aligned because naive Bayes assumes features are conditionally independent. ` +
            `The background heatmap shows the posterior P(class | x), so the boundary is implicit ` +
            `(it's the curve where the two highest posteriors meet).${tiltedNote}`;
    }

    if (dataSel) {
        dataSel.innerHTML = `
            <option value="separated">Two separated blobs</option>
            <option value="overlap">Overlapping</option>
            <option value="multiclass">Three classes</option>
            <option value="tilted">Tilted (NB struggles)</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }
    resetBtn?.addEventListener('click', reset);

    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
