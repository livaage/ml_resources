/* Interactive feature-engineering viz.
 * Concentric-rings dataset shown four ways:
 *   raw (x, y)          — linear model fails
 *   polynomial 2D       — same scatter, but show a quadratic boundary (achievable
 *                         because the model sees x², y², xy too)
 *   RBF basis           — pick a few centres, plot the RBF-feature space colour
 *   radial              — collapse to r = √(x² + y²), 1D — a single threshold works
 * Train a logistic regression in each feature space and draw the boundary. */

(function () {
    const canvas    = document.getElementById('viz-fe-canvas');
    const rawBtn    = document.getElementById('viz-fe-raw');
    const polyBtn   = document.getElementById('viz-fe-poly');
    const rbfBtn    = document.getElementById('viz-fe-rbf');
    const radBtn    = document.getElementById('viz-fe-radial');
    const captionEl = document.getElementById('viz-fe-caption');
    if (!canvas) return;

    let mode = 'raw';
    let points = [];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    (function init() {
        for (let i = 0; i < 60; i++) {
            const a = Math.random() * Math.PI * 2;
            const r = 0.3 + randn() * 0.04;
            points.push({ x: r * Math.cos(a), y: r * Math.sin(a), c: 0 });
        }
        for (let i = 0; i < 60; i++) {
            const a = Math.random() * Math.PI * 2;
            const r = 0.75 + randn() * 0.04;
            points.push({ x: r * Math.cos(a), y: r * Math.sin(a), c: 1 });
        }
    })();

    function features(x, y) {
        if (mode === 'raw')    return [x, y, 1];
        if (mode === 'poly')   return [x, y, x * x, y * y, x * y, 1];
        if (mode === 'rbf') {
            const centres = [[0, 0], [0.6, 0], [-0.6, 0], [0, 0.6], [0, -0.6]];
            const g = 6;
            const feats = centres.map(([cx, cy]) =>
                Math.exp(-g * ((x - cx) ** 2 + (y - cy) ** 2)));
            feats.push(1);
            return feats;
        }
        if (mode === 'radial') return [Math.sqrt(x * x + y * y), 1];
        return [x, y, 1];
    }

    function fitLR(pts) {
        const sample = features(pts[0].x, pts[0].y);
        const D = sample.length;
        let w = Array(D).fill(0);
        for (let it = 0; it < 400; it++) {
            const lr = 0.4 / (1 + it / 100);
            const grad = Array(D).fill(0);
            for (const p of pts) {
                const f = features(p.x, p.y);
                let z = 0;
                for (let j = 0; j < D; j++) z += w[j] * f[j];
                const s = 1 / (1 + Math.exp(-z));
                const e = s - p.c;
                for (let j = 0; j < D; j++) grad[j] += e * f[j];
            }
            for (let j = 0; j < D; j++) w[j] -= lr * grad[j] / pts.length;
        }
        return w;
    }
    function predict(w, x, y) {
        const f = features(x, y);
        let z = 0;
        for (let j = 0; j < w.length; j++) z += w[j] * f[j];
        return z > 0 ? 1 : 0;
    }

    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.62)));
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

        const m = Math.min(W, H) / 2 - 16;
        ctx.fillStyle = '#fff';
        ctx.fillRect(W / 2 - m, H / 2 - m, 2 * m, 2 * m);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m - 0.5, H / 2 - m - 0.5, 2 * m + 1, 2 * m + 1);

        const w = fitLR(points);
        // Background tinted decision regions
        const G = 50;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx = -1.05 + (i + 0.5) / G * 2.1;
                const dy = -1.05 + (j + 0.5) / G * 2.1;
                const cls = predict(w, dx, dy);
                ctx.fillStyle = cls === 0 ? '#eef2ff' : '#fdf0eb';
                const [px, py] = toPx(dx, dy);
                ctx.fillRect(px - m * 1.05 / G - 1, py - m * 1.05 / G - 1,
                             m * 2.1 / G + 2, m * 2.1 / G + 2);
            }
        }

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = p.c === 0 ? '#4f46e5' : '#ea7959';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 3.4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Compute accuracy
        let correct = 0;
        for (const p of points) if (predict(w, p.x, p.y) === p.c) correct++;
        const acc = (correct / points.length * 100).toFixed(0);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = {
            raw: 'RAW (x, y) — linear model in original space',
            poly: 'POLYNOMIAL — linear model in (x, y, x², y², xy)',
            rbf: 'RBF FEATURES — linear model on 5 Gaussian basis functions',
            radial: 'RADIAL — single feature r = √(x²+y²)',
        };
        ctx.fillText(titles[mode], W / 2 - m, H / 2 - m - 6);
        ctx.fillStyle = '#4f46e5';
        ctx.font = '600 11px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`accuracy ${acc}%`, W / 2 + m, H / 2 - m - 6);

        updateCaption(acc);
    }

    function updateCaption(acc) {
        if (!captionEl) return;
        const notes = {
            raw:    `Raw (x, y). A linear classifier cuts the plane with a straight line — but concentric classes can't be separated that way. Accuracy ${acc}% — barely better than chance.`,
            poly:   `Polynomial features. The same linear model now sees x², y², xy — and a linear combination of those is a quadratic. The boundary curves to fit the rings. Accuracy ${acc}%.`,
            rbf:    `RBF basis. Five Gaussian features placed by hand. The linear model now sees how close each point is to each centre — which is enough to carve a near-perfect boundary. Accuracy ${acc}%.`,
            radial: `Single radial feature. <em>r = √(x² + y²)</em> turns the 2D problem into a 1D threshold. The optimal classifier is "is r above some value?" — accuracy ${acc}%.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [rawBtn, polyBtn, rbfBtn, radBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    rawBtn?.addEventListener('click',  () => setMode('raw', rawBtn));
    polyBtn?.addEventListener('click', () => setMode('poly', polyBtn));
    rbfBtn?.addEventListener('click',  () => setMode('rbf', rbfBtn));
    radBtn?.addEventListener('click',  () => setMode('radial', radBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
