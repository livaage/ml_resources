/* Interactive Gaussian Process regression viz.
 * Click anywhere in the panel to add an observation; shift+click to remove.
 * The bold line is the posterior mean; the shaded band is ±2σ. Thin lines
 * are sample functions from the posterior. With zero observations the band
 * is the prior — switch the kernel or slide the lengthscale to see how the
 * prior changes. */

(function () {
    const canvas    = document.getElementById('viz-gp-canvas');
    const kSel      = document.getElementById('viz-gp-kernel');
    const lSlider   = document.getElementById('viz-gp-length');
    const lLbl      = document.getElementById('viz-gp-length-lbl');
    const nSlider   = document.getElementById('viz-gp-noise');
    const nLbl      = document.getElementById('viz-gp-noise-lbl');
    const resampBtn = document.getElementById('viz-gp-resample');
    const clearBtn  = document.getElementById('viz-gp-clear');
    const captionEl = document.getElementById('viz-gp-caption');
    if (!canvas) return;

    let kernel = 'rbf';
    let lengthscale = 0.25;
    let noise = 0.05;
    let observations = [];      // { x, y }
    const G = 80;               // grid points across [-1, 1]
    let posterior = null;       // { mu: Float, var: Float, samples: [][] }
    let sampleSeed = Math.random() * 1000;

    function kernelFn(x1, x2) {
        const r = Math.abs(x1 - x2);
        if (kernel === 'rbf') {
            return Math.exp(-0.5 * (r * r) / (lengthscale * lengthscale));
        }
        // Matern 3/2
        const s = Math.sqrt(3) * r / lengthscale;
        return (1 + s) * Math.exp(-s);
    }

    // ----- Linear algebra helpers (small dense matrices) -----
    function solve(A, b) {
        const n = b.length;
        const M = A.map((row, i) => [...row, b[i]]);
        for (let i = 0; i < n; i++) {
            let piv = i;
            for (let k = i + 1; k < n; k++)
                if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
            if (piv !== i) [M[i], M[piv]] = [M[piv], M[i]];
            const d = M[i][i];
            if (Math.abs(d) < 1e-12) return null;
            for (let j = i; j <= n; j++) M[i][j] /= d;
            for (let k = 0; k < n; k++) {
                if (k === i) continue;
                const f = M[k][i];
                if (f === 0) continue;
                for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
            }
        }
        return M.map(row => row[n]);
    }
    function cholesky(M) {
        const n = M.length;
        const L = Array.from({length: n}, () => new Float64Array(n));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let s = 0;
                for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
                if (i === j) {
                    L[i][j] = Math.sqrt(Math.max(0, M[i][i] - s));
                } else {
                    L[i][j] = (M[i][j] - s) / (L[j][j] || 1e-9);
                }
            }
        }
        return L;
    }
    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // ----- Compute posterior on the grid -----
    function compute() {
        const xs = new Float64Array(G);
        for (let i = 0; i < G; i++) xs[i] = -1 + (2 * i) / (G - 1);
        const mu = new Float64Array(G);
        const variance = new Float64Array(G);
        // Prior variance: kernel(x, x) = 1
        for (let i = 0; i < G; i++) variance[i] = 1.0;

        const N = observations.length;
        if (N === 0) {
            // Prior samples via Cholesky of K_GG + 1e-6 I
            const K = Array.from({length: G}, () => new Float64Array(G));
            for (let i = 0; i < G; i++) {
                for (let j = 0; j <= i; j++) {
                    const v = kernelFn(xs[i], xs[j]);
                    K[i][j] = K[j][i] = v;
                }
                K[i][i] += 1e-6;
            }
            const L = cholesky(K);
            // Draw 4 samples
            const samples = [];
            const rng = mulberry32(Math.floor(sampleSeed));
            for (let s = 0; s < 4; s++) {
                const eps = new Float64Array(G);
                for (let i = 0; i < G; i++) eps[i] = randnRng(rng);
                const y = new Float64Array(G);
                for (let i = 0; i < G; i++) {
                    let v = 0;
                    for (let j = 0; j <= i; j++) v += L[i][j] * eps[j];
                    y[i] = v;
                }
                samples.push(y);
            }
            posterior = { xs, mu, std: variance.map(v => Math.sqrt(v)), samples };
            return;
        }

        // K_NN + σ²_n I
        const Knn = Array.from({length: N}, () => new Float64Array(N));
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                Knn[i][j] = kernelFn(observations[i].x, observations[j].x);
            }
            Knn[i][i] += noise * noise + 1e-6;
        }
        const y = observations.map(o => o.y);
        const alpha = solve(Knn.map(r => [...r]), y);
        if (!alpha) {
            posterior = { xs, mu, std: variance.map(v => Math.sqrt(v)), samples: [] };
            return;
        }

        // Posterior mean and variance per grid point
        const KgN = Array.from({length: G}, () => new Float64Array(N));
        for (let i = 0; i < G; i++) {
            for (let j = 0; j < N; j++) {
                KgN[i][j] = kernelFn(xs[i], observations[j].x);
            }
        }
        for (let i = 0; i < G; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += KgN[i][j] * alpha[j];
            mu[i] = s;
            // σ² = k(x,x) - K_gN K_NN^-1 K_Ng[i]
            const v = solve(Knn.map(r => [...r]), Array.from(KgN[i]));
            let dot = 0;
            if (v) for (let j = 0; j < N; j++) dot += KgN[i][j] * v[j];
            variance[i] = Math.max(1e-5, 1 - dot);
        }

        // Posterior samples — build full posterior covariance over the grid
        const Kgg = Array.from({length: G}, () => new Float64Array(G));
        for (let i = 0; i < G; i++) {
            for (let j = 0; j <= i; j++) {
                Kgg[i][j] = Kgg[j][i] = kernelFn(xs[i], xs[j]);
            }
        }
        // Σ = K_gg - K_gN K_NN^-1 K_Ng
        for (let i = 0; i < G; i++) {
            // KgN_inv_row = K_NN^-1 K_Ng[i]
            const v = solve(Knn.map(r => [...r]), Array.from(KgN[i]));
            if (!v) continue;
            for (let j = 0; j <= i; j++) {
                let dot = 0;
                for (let k = 0; k < N; k++) dot += KgN[j][k] * v[k];
                Kgg[i][j] -= dot;
                Kgg[j][i] = Kgg[i][j];
            }
            Kgg[i][i] += 1e-6;
        }
        const L = cholesky(Kgg);
        const samples = [];
        const rng = mulberry32(Math.floor(sampleSeed));
        for (let s = 0; s < 4; s++) {
            const eps = new Float64Array(G);
            for (let i = 0; i < G; i++) eps[i] = randnRng(rng);
            const yS = new Float64Array(G);
            for (let i = 0; i < G; i++) {
                let v = 0;
                for (let j = 0; j <= i; j++) v += L[i][j] * eps[j];
                yS[i] = mu[i] + v;
            }
            samples.push(yS);
        }
        posterior = { xs, mu, std: variance.map(v => Math.sqrt(v)), samples };
    }

    // Seeded random — keeps the sample curves stable across redraws
    function mulberry32(seed) {
        let a = seed | 0;
        return function () {
            a |= 0; a = a + 0x6D2B79F5 | 0;
            let t = a;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }
    function randnRng(rng) {
        const u1 = Math.max(rng(), 1e-9), u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(320, cssW * 0.50)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function toPx(x, y) {
        const pad = 30;
        return [
            pad + ((x + 1) / 2) * (W - 2 * pad),
            H / 2 - y * (H - 2 * pad) / 5,
        ];
    }
    function pxToData(px, py) {
        const pad = 30;
        return [
            ((px - pad) / (W - 2 * pad)) * 2 - 1,
            -(py - H / 2) * 5 / (H - 2 * pad),
        ];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Axes
        const pad = 30;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad, H / 2); ctx.lineTo(W - pad, H / 2); ctx.stroke();
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.strokeRect(pad - 0.5, pad - 0.5, W - 2 * pad + 1, H - 2 * pad + 1);

        if (!posterior) return;
        const { xs, mu, std, samples } = posterior;

        // ±2σ uncertainty band
        ctx.fillStyle = 'rgba(79, 70, 229, 0.13)';
        ctx.beginPath();
        for (let i = 0; i < xs.length; i++) {
            const [px, py] = toPx(xs[i], mu[i] + 2 * std[i]);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        for (let i = xs.length - 1; i >= 0; i--) {
            const [px, py] = toPx(xs[i], mu[i] - 2 * std[i]);
            ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();

        // Posterior samples (thin)
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.32)';
        ctx.lineWidth = 1.1;
        for (const s of samples) {
            ctx.beginPath();
            for (let i = 0; i < xs.length; i++) {
                const [px, py] = toPx(xs[i], s[i]);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Mean (bold)
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        for (let i = 0; i < xs.length; i++) {
            const [px, py] = toPx(xs[i], mu[i]);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Observations
        for (const obs of observations) {
            const [px, py] = toPx(obs.x, obs.y);
            ctx.fillStyle = '#ea7959';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Title + axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${kernel.toUpperCase()} kernel — ℓ = ${lengthscale.toFixed(2)} · σ_n = ${noise.toFixed(2)}`,
            pad, pad - 8);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('click to add · shift+click to remove', W - pad, pad - 8);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const N = observations.length;
        if (N === 0) {
            captionEl.innerHTML =
                `<strong>No observations yet.</strong> The shaded band is the <em>prior</em> ±2σ; the thin curves ` +
                `are sample functions from it. Each kernel gives a different idea of what "smooth" means — try the ` +
                `Matérn 3/2 to see rougher samples than RBF's silky ones.`;
        } else {
            captionEl.innerHTML =
                `<strong>${N} observation${N === 1 ? '' : 's'}.</strong> The band collapses to the orange points ` +
                `(within noise σ_n) and inflates where you have no data — that's the GP's calibrated uncertainty. ` +
                `Drop the lengthscale to make the fit more wiggly; raise it to force smoother extrapolation.`;
        }
    }

    // ----- Interactions -----
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }
    function nearObs(px, py) {
        for (let i = 0; i < observations.length; i++) {
            const [opx, opy] = toPx(observations[i].x, observations[i].y);
            if (Math.hypot(px - opx, py - opy) < 10) return i;
        }
        return -1;
    }
    canvas.addEventListener('click', (e) => {
        const [px, py] = getPos(e);
        const pad = 30;
        if (px < pad || px > W - pad || py < pad || py > H - pad) return;
        const nearby = nearObs(px, py);
        if (e.shiftKey && nearby >= 0) {
            observations.splice(nearby, 1);
        } else if (nearby >= 0) {
            // Drag-update value
            const [dx, dy] = pxToData(px, py);
            observations[nearby].y = dy;
        } else {
            const [dx, dy] = pxToData(px, py);
            observations.push({ x: dx, y: dy });
        }
        compute(); draw();
    });

    // ----- Controls -----
    if (kSel) {
        kSel.innerHTML = `
            <option value="rbf">RBF (silky)</option>
            <option value="matern">Matérn 3/2 (rougher)</option>
        `;
        kSel.addEventListener('change', () => {
            kernel = kSel.value;
            compute(); draw();
        });
    }
    if (lSlider) {
        lSlider.min = 0.05; lSlider.max = 0.8; lSlider.step = 0.02; lSlider.value = 0.25;
        lSlider.addEventListener('input', () => {
            lengthscale = parseFloat(lSlider.value);
            if (lLbl) lLbl.textContent = `ℓ = ${lengthscale.toFixed(2)}`;
            compute(); draw();
        });
    }
    if (nSlider) {
        nSlider.min = 0; nSlider.max = 0.3; nSlider.step = 0.01; nSlider.value = 0.05;
        nSlider.addEventListener('input', () => {
            noise = parseFloat(nSlider.value);
            if (nLbl) nLbl.textContent = `σₙ = ${noise.toFixed(2)}`;
            compute(); draw();
        });
    }
    resampBtn?.addEventListener('click', () => {
        sampleSeed = Math.random() * 1000;
        compute(); draw();
    });
    clearBtn?.addEventListener('click', () => {
        observations = [];
        compute(); draw();
    });

    // ----- Init: one example observation to make the prior→posterior story clear -----
    observations = [
        { x: -0.5, y: 0.6 }, { x: 0.0, y: -0.3 }, { x: 0.5, y: 0.8 },
    ];
    compute();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
