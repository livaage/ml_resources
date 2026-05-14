/* Interactive Gaussian Mixture Model viz with EM training.
 * Each component is a 2D Gaussian (full covariance), drawn as a 1- and 2σ
 * ellipse. The E step computes soft responsibilities γ_ik for each point;
 * the M step updates each component's μ, Σ, weight from those. Step does
 * one full EM iteration. Notice how GMM handles tilted / overlapping
 * clusters that vanilla k-means can't. */

(function () {
    const canvas    = document.getElementById('viz-gmm-canvas');
    const stepBtn   = document.getElementById('viz-gmm-step');
    const autoBtn   = document.getElementById('viz-gmm-auto');
    const resetBtn  = document.getElementById('viz-gmm-reset');
    const kSel      = document.getElementById('viz-gmm-k');
    const dataSel   = document.getElementById('viz-gmm-data');
    const counterEl = document.getElementById('viz-gmm-counter');
    const captionEl = document.getElementById('viz-gmm-caption');
    if (!canvas) return;

    let K = 3;
    let dataset = 'tilted';
    let points = [];           // { x, y }
    let components = [];       // { mu: [x, y], cov: [a, b, c], w }  cov = [[a,b],[b,c]]
    let resp = [];             // responsibilities [N][K]
    let iter = 0;
    let playing = false, lastStep = 0;
    const STEP_MS = 700;
    const PALETTE = ['#4f46e5', '#ea7959', '#10847e', '#d4a13c', '#a05bb0'];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'tilted') {
            // Two tilted ellipses
            for (let i = 0; i < 70; i++) {
                const t = randn();
                pts.push({ x:  0.55 * t + randn() * 0.08 - 0.35,
                           y:  0.25 * t + randn() * 0.08 + 0.25 });
            }
            for (let i = 0; i < 70; i++) {
                const t = randn();
                pts.push({ x: -0.55 * t + randn() * 0.08 + 0.35,
                           y:  0.25 * t + randn() * 0.08 - 0.25 });
            }
        } else if (dataset === 'overlap') {
            for (let i = 0; i < 70; i++) pts.push({ x: -0.2 + randn() * 0.22, y:  0.1 + randn() * 0.22 });
            for (let i = 0; i < 70; i++) pts.push({ x:  0.2 + randn() * 0.22, y: -0.1 + randn() * 0.22 });
            for (let i = 0; i < 60; i++) pts.push({ x:  0.0 + randn() * 0.10, y:  0.4 + randn() * 0.10 });
        } else if (dataset === 'blobs') {
            const cs = [[-0.45, 0.4], [0.4, 0.35], [-0.05, -0.45], [0.55, -0.4]];
            const counts = [55, 55, 55, 0];
            for (let c = 0; c < cs.length; c++) {
                for (let i = 0; i < counts[c]; i++) {
                    pts.push({ x: cs[c][0] + randn() * 0.11, y: cs[c][1] + randn() * 0.11 });
                }
            }
        } else if (dataset === 'banana') {
            for (let i = 0; i < 80; i++) {
                const t = (i / 79) * Math.PI - 0.2;
                pts.push({ x: 0.65 * Math.cos(t) + randn() * 0.05 - 0.05,
                           y: 0.5 * Math.sin(t) + randn() * 0.05 + 0.05 });
            }
            for (let i = 0; i < 80; i++) {
                pts.push({ x: -0.1 + randn() * 0.1, y: -0.45 + randn() * 0.10 });
            }
        }
        return pts;
    }

    function initComponents() {
        // k-means++ for means; identity*0.04 for covs; uniform weights
        components = [];
        if (points.length === 0) return;
        const chosen = [{ ...points[Math.floor(Math.random() * points.length)] }];
        while (chosen.length < K) {
            const d2s = points.map(p => {
                let m = Infinity;
                for (const c of chosen) {
                    const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
                    if (d < m) m = d;
                }
                return m;
            });
            const total = d2s.reduce((a, b) => a + b, 0);
            let r = Math.random() * total, i = 0;
            for (; i < d2s.length && r > d2s[i]; i++) r -= d2s[i];
            chosen.push({ ...points[Math.min(i, points.length - 1)] });
        }
        for (const c of chosen) {
            components.push({ mu: [c.x, c.y], cov: [0.04, 0, 0.04], w: 1 / K });
        }
    }

    function gaussian2D(x, y, mu, cov) {
        const [a, b, c] = cov;
        const det = a * c - b * b;
        if (det <= 1e-9) return 0;
        const inv00 =  c / det;
        const inv01 = -b / det;
        const inv11 =  a / det;
        const dx = x - mu[0], dy = y - mu[1];
        const m = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy);
        return Math.exp(-0.5 * m) / (2 * Math.PI * Math.sqrt(det));
    }

    // ----- One EM iteration -----
    function emStep() {
        const N = points.length;
        // E step
        resp = Array.from({ length: N }, () => new Float64Array(K));
        for (let i = 0; i < N; i++) {
            let total = 0;
            const r = resp[i];
            for (let k = 0; k < K; k++) {
                const c = components[k];
                r[k] = c.w * gaussian2D(points[i].x, points[i].y, c.mu, c.cov);
                total += r[k];
            }
            if (total > 0) for (let k = 0; k < K; k++) r[k] /= total;
            else            for (let k = 0; k < K; k++) r[k] = 1 / K;
        }
        // M step
        for (let k = 0; k < K; k++) {
            let Nk = 0, sx = 0, sy = 0;
            for (let i = 0; i < N; i++) {
                const ri = resp[i][k];
                Nk += ri;
                sx += ri * points[i].x;
                sy += ri * points[i].y;
            }
            if (Nk < 1e-6) continue;
            const mux = sx / Nk, muy = sy / Nk;
            let cxx = 0, cyy = 0, cxy = 0;
            for (let i = 0; i < N; i++) {
                const ri = resp[i][k];
                const dx = points[i].x - mux;
                const dy = points[i].y - muy;
                cxx += ri * dx * dx;
                cxy += ri * dx * dy;
                cyy += ri * dy * dy;
            }
            // Small ridge so covariance never collapses
            components[k].mu = [mux, muy];
            components[k].cov = [cxx / Nk + 1e-5, cxy / Nk, cyy / Nk + 1e-5];
            components[k].w = Nk / N;
        }
        iter++;
    }

    function reset() {
        points = buildDataset();
        initComponents();
        resp = [];
        iter = 0;
        playing = false;
        updateAutoBtn();
        draw();
    }

    // ----- Canvas -----
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

    // Eigendecomposition of [[a,b],[b,c]] — returns {lambdas: [l1, l2], angle}
    function eig2x2(a, b, c) {
        const tr = a + c;
        const det = a * c - b * b;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        // Eigenvector for l1
        let angle;
        if (Math.abs(b) > 1e-9) {
            angle = Math.atan2(l1 - a, b);
        } else if (a >= c) {
            angle = 0;
        } else {
            angle = Math.PI / 2;
        }
        return { l1: Math.max(0, l1), l2: Math.max(0, l2), angle };
    }

    function drawEllipse(mu, cov, sigma, colour, weight) {
        const { l1, l2, angle } = eig2x2(cov[0], cov[1], cov[2]);
        const [cx, cy] = toPx(mu[0], mu[1]);
        const m = Math.min(W, H) / 2 - 16;
        const rx = Math.sqrt(l1) * sigma * m;
        const ry = Math.sqrt(l2) * sigma * m;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-angle);
        ctx.strokeStyle = colour;
        ctx.lineWidth = weight;
        ctx.beginPath();
        ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Frame
        const m = Math.min(W, H) / 2 - 12;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(W / 2 - m - 4, H / 2 - m - 4, 2 * m + 8, 2 * m + 8);

        // Points coloured by max-responsibility component
        for (let i = 0; i < points.length; i++) {
            const [px, py] = toPx(points[i].x, points[i].y);
            let cls = 0, mx = -1;
            if (resp[i]) {
                for (let k = 0; k < K; k++) if (resp[i][k] > mx) { mx = resp[i][k]; cls = k; }
            } else {
                cls = -1;
            }
            ctx.fillStyle = cls < 0 ? 'rgba(80, 80, 80, 0.6)' : PALETTE[cls % PALETTE.length];
            ctx.beginPath();
            ctx.arc(px, py, 2.6, 0, Math.PI * 2);
            ctx.fill();
        }

        // Components: 1σ (bold) and 2σ (faint) ellipses + centre
        for (let k = 0; k < components.length; k++) {
            const col = PALETTE[k % PALETTE.length];
            drawEllipse(components[k].mu, components[k].cov, 2, col + '44', 1);
            drawEllipse(components[k].mu, components[k].cov, 1, col,         2);
            const [cx, cy] = toPx(components[k].mu[0], components[k].mu[1]);
            ctx.fillStyle = col;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Iteration label
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`iter ${iter}`, 12, 16);

        if (counterEl) counterEl.textContent = `iter ${iter}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (iter === 0) {
            captionEl.innerHTML =
                `<strong>K = ${K}</strong> components placed via k-means++ with circular initial covariances. ` +
                `Click <strong>Step</strong> — one EM iteration computes soft responsibilities for every point, then ` +
                `updates each component's mean, covariance, and weight.`;
        } else if (iter < 4) {
            captionEl.innerHTML =
                `<strong>iter ${iter}.</strong> Each ellipse is the 1σ (and faint 2σ) contour of a 2D Gaussian. ` +
                `Points are coloured by their dominant component; near the boundaries they're split between two — ` +
                `that's soft assignment, the key difference from k-means.`;
        } else {
            captionEl.innerHTML =
                `<strong>iter ${iter}.</strong> The ellipses have rotated and stretched to match the data's shape — ` +
                `something k-means' spherical Voronoi cells can't do. Try the <em>Banana</em> dataset for a case ` +
                `where even GMM struggles: a Gaussian can't bend.`;
        }
    }

    // ----- Loop -----
    function updateAutoBtn() {
        if (autoBtn) autoBtn.textContent = playing ? 'Pause' : 'Auto';
    }
    function loop(now) {
        if (playing && now - lastStep >= STEP_MS) {
            emStep();
            draw();
            lastStep = now;
            if (iter >= 60) { playing = false; updateAutoBtn(); }
        }
        requestAnimationFrame(loop);
    }

    // ----- Controls -----
    stepBtn?.addEventListener('click', () => { emStep(); draw(); });
    autoBtn?.addEventListener('click', () => {
        playing = !playing;
        updateAutoBtn();
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', reset);
    if (kSel) {
        kSel.innerHTML = `
            <option value="2">K = 2</option>
            <option value="3" selected>K = 3</option>
            <option value="4">K = 4</option>
            <option value="5">K = 5</option>
        `;
        kSel.addEventListener('change', () => {
            K = parseInt(kSel.value, 10);
            reset();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="tilted">Tilted ellipses</option>
            <option value="overlap">Overlapping</option>
            <option value="blobs">Blobs</option>
            <option value="banana">Banana + blob</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }

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
