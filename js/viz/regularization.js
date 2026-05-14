/* Interactive regularization viz.
 * A linear regression problem with 12 features — only 4 are informative,
 * the rest are noise. Bars show each fit coefficient. Slide λ and watch:
 *   L2 / Ridge      → every bar shrinks smoothly, none reach exact zero
 *   L1 / LASSO      → bars snap to zero (sparsity), exactly
 *   Elastic Net     → mixture of both
 * The true informative coefficients are drawn faint behind the bars so you
 * can tell signal from noise. */

(function () {
    const canvas    = document.getElementById('viz-reg-canvas');
    const l2Btn     = document.getElementById('viz-reg-l2');
    const l1Btn     = document.getElementById('viz-reg-l1');
    const enBtn     = document.getElementById('viz-reg-en');
    const lSlider   = document.getElementById('viz-reg-lambda');
    const lLbl      = document.getElementById('viz-reg-lambda-lbl');
    const resetBtn  = document.getElementById('viz-reg-reset');
    const captionEl = document.getElementById('viz-reg-caption');
    if (!canvas) return;

    let penalty = 'l2';
    let lambda = 0.1;
    const D = 12, N = 60;
    let X = null, y = null, trueW = null;
    let coefs = null;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function regenData() {
        trueW = Array(D).fill(0);
        // Pick 4 informative features
        const idx = [];
        while (idx.length < 4) {
            const k = Math.floor(Math.random() * D);
            if (!idx.includes(k)) idx.push(k);
        }
        for (const k of idx) trueW[k] = (Math.random() < 0.5 ? -1 : 1) * (0.7 + Math.random() * 0.6);
        X = Array.from({length: N}, () => Array.from({length: D}, () => randn()));
        y = Array(N);
        for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < D; j++) s += X[i][j] * trueW[j];
            y[i] = s + randn() * 0.4;
        }
    }

    // ----- Ridge via normal equations: w = (XᵀX + λI)⁻¹ Xᵀy -----
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
    function fitRidge(lam) {
        const A = Array.from({length: D}, () => Array(D).fill(0));
        const b = Array(D).fill(0);
        for (let j = 0; j < D; j++) {
            for (let k = 0; k < D; k++) {
                let s = 0;
                for (let i = 0; i < N; i++) s += X[i][j] * X[i][k];
                A[j][k] = s;
            }
            A[j][j] += lam * N;
            for (let i = 0; i < N; i++) b[j] += X[i][j] * y[i];
        }
        return solve(A, b) || Array(D).fill(0);
    }

    // ----- LASSO via coordinate descent (soft thresholding) -----
    function softThreshold(z, gamma) {
        if (z > gamma) return z - gamma;
        if (z < -gamma) return z + gamma;
        return 0;
    }
    function fitLasso(lam, l1ratio = 1.0) {
        // Pre-compute feature norms
        const w = Array(D).fill(0);
        const res = y.slice();
        const featNorm2 = Array(D).fill(0);
        for (let j = 0; j < D; j++)
            for (let i = 0; i < N; i++) featNorm2[j] += X[i][j] * X[i][j];
        const iters = 200;
        for (let it = 0; it < iters; it++) {
            for (let j = 0; j < D; j++) {
                // Add wj * xj back to residual
                for (let i = 0; i < N; i++) res[i] += X[i][j] * w[j];
                let dot = 0;
                for (let i = 0; i < N; i++) dot += X[i][j] * res[i];
                const denom = featNorm2[j] + N * lam * (1 - l1ratio);
                const num = softThreshold(dot, N * lam * l1ratio);
                w[j] = denom > 1e-12 ? num / denom : 0;
                // Subtract back
                for (let i = 0; i < N; i++) res[i] -= X[i][j] * w[j];
            }
        }
        return w;
    }

    function refit() {
        if (penalty === 'l2')      coefs = fitRidge(lambda);
        else if (penalty === 'l1') coefs = fitLasso(lambda, 1.0);
        else                       coefs = fitLasso(lambda, 0.5);
        draw();
    }
    function reset() {
        regenData();
        refit();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(480, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(320, cssW * 0.46)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function draw() {
        if (!ctx || !coefs) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        // Plot area
        const pad = 28;
        const box = { x: pad + 28, y: 32, w: W - 2 * pad - 28, h: H - 70 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Find y range
        let ymax = 1.5;
        for (const v of trueW) if (Math.abs(v) > ymax) ymax = Math.abs(v);
        for (const v of coefs) if (Math.abs(v) > ymax) ymax = Math.abs(v);
        ymax *= 1.05;

        const midY = box.y + box.h / 2;
        // Zero line
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.beginPath(); ctx.moveTo(box.x, midY); ctx.lineTo(box.x + box.w, midY); ctx.stroke();
        // Gridlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.setLineDash([2, 3]);
        for (const v of [-ymax / 2, ymax / 2]) {
            const py = midY - (v / ymax) * (box.h / 2 - 6);
            ctx.beginPath(); ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.stroke();
        }
        ctx.setLineDash([]);

        // Tick labels on Y
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText('0', box.x - 6, midY + 3);
        ctx.fillText('+' + (ymax / 2).toFixed(1), box.x - 6, midY - (box.h / 2 - 6) / 2 + 3);
        ctx.fillText(((-ymax / 2).toFixed(1)), box.x - 6, midY + (box.h / 2 - 6) / 2 + 3);

        // Bars
        const slot = box.w / D;
        const barW = slot * 0.7;
        for (let j = 0; j < D; j++) {
            const cx = box.x + (j + 0.5) * slot;
            // True coefficient: faint ghost bar (outlined)
            const trueH = (trueW[j] / ymax) * (box.h / 2 - 6);
            ctx.fillStyle = 'rgba(79, 70, 229, 0.10)';
            ctx.strokeStyle = 'rgba(79, 70, 229, 0.45)';
            ctx.lineWidth = 1;
            ctx.fillRect(cx - barW / 2, midY - Math.max(0, trueH), barW, Math.abs(trueH));
            ctx.strokeRect(cx - barW / 2 - 0.5, midY - Math.max(0, trueH) - 0.5,
                           barW + 1, Math.abs(trueH) + 1);

            // Fitted coefficient: solid bar
            const fitH = (coefs[j] / ymax) * (box.h / 2 - 6);
            const informative = Math.abs(trueW[j]) > 1e-6;
            ctx.fillStyle = informative ? '#4f46e5' : '#ea7959';
            ctx.fillRect(cx - barW / 4, midY - Math.max(0, fitH), barW / 2, Math.abs(fitH));

            // X-axis feature label
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.font = '500 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`x${j + 1}`, cx, box.y + box.h + 14);
        }

        // Titles + legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const pName = { l2: 'L2 / Ridge', l1: 'L1 / LASSO', en: 'Elastic Net (α=0.5)' }[penalty];
        ctx.fillText(`COEFFICIENTS — ${pName}, λ = ${lambda.toFixed(2)}`, box.x, box.y - 8);

        // Legend
        const lx = box.x + box.w - 220, ly = box.y - 8;
        ctx.fillStyle = 'rgba(79, 70, 229, 0.18)';
        ctx.fillRect(lx, ly - 5, 8, 8);
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.45)';
        ctx.strokeRect(lx - 0.5, ly - 5.5, 9, 9);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText('true (informative)', lx + 12, ly + 2);
        ctx.fillStyle = '#4f46e5';
        ctx.fillRect(lx + 100, ly - 5, 8, 8);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('fit (signal)', lx + 112, ly + 2);
        ctx.fillStyle = '#ea7959';
        ctx.fillRect(lx + 165, ly - 5, 8, 8);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('fit (noise)', lx + 177, ly + 2);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        // Count "exact zeros" and "near zeros"
        let zeros = 0, nearZero = 0;
        for (const v of coefs) {
            if (v === 0) zeros++;
            else if (Math.abs(v) < 0.05) nearZero++;
        }
        const sparsityNote = penalty === 'l1'
            ? `${zeros}/${D} coefficients are <strong>exactly zero</strong>.`
            : penalty === 'en'
                ? `${zeros}/${D} exactly zero, ${nearZero} more are near-zero.`
                : `No coefficients are exactly zero (L2 only shrinks).`;
        captionEl.innerHTML =
            `<strong>${({l2:'Ridge', l1:'LASSO', en:'Elastic Net'})[penalty]}</strong> at λ = ${lambda.toFixed(2)}. ` +
            `${sparsityNote} Indigo bars are signal (the model picked up an actually-informative feature); ` +
            `terracotta bars are noise (the model thinks a noise feature matters). ` +
            (lambda < 0.05
                ? `λ is small — the model is barely regularized; expect noise bars to be non-trivial.`
                : lambda > 0.4
                    ? `λ is large — even some signal coefficients get shrunk to zero (over-regularization).`
                    : `Around this λ is usually the sweet spot for this dataset.`);
    }

    // ----- Controls -----
    function setPenalty(p, btn) {
        penalty = p;
        for (const b of [l2Btn, l1Btn, enBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        refit();
    }
    l2Btn?.addEventListener('click', () => setPenalty('l2', l2Btn));
    l1Btn?.addEventListener('click', () => setPenalty('l1', l1Btn));
    enBtn?.addEventListener('click', () => setPenalty('en', enBtn));
    if (lSlider) {
        lSlider.min = 0; lSlider.max = 1.5; lSlider.step = 0.01; lSlider.value = 0.1;
        lSlider.addEventListener('input', () => {
            lambda = parseFloat(lSlider.value);
            if (lLbl) lLbl.textContent = `λ = ${lambda.toFixed(2)}`;
            refit();
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
