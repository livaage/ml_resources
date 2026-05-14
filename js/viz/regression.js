/* Interactive linear regression viz.
 * Drag points · click empty space to add · shift-click to remove. */

(function () {
    const canvas = document.getElementById('viz-regression-canvas');
    if (!canvas) return;

    // ----- Coordinate system -----
    const PAD   = 36;
    const MIN_X = 0, MAX_X = 10;
    const MIN_Y = 0, MAX_Y = 10;

    let W, H;  // CSS pixels
    let ctx;

    // DPR-aware sizing — sharp on retina.
    // The canvas may live inside a hidden tab when this script runs, so we
    // also wire up a ResizeObserver below to handle the becomes-visible case.
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const W_css = Math.round(rect.width);
        if (W_css <= 0) return;                       // not laid out yet
        const H_css = Math.round(W_css * 9 / 16);
        canvas.style.height = H_css + 'px';

        const dpr = window.devicePixelRatio || 1;
        W = W_css;
        H = H_css;
        canvas.width  = W_css * dpr;
        canvas.height = H_css * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function toPx(x, y) {
        return [
            PAD + (x - MIN_X) / (MAX_X - MIN_X) * (W - 2 * PAD),
            H - PAD - (y - MIN_Y) / (MAX_Y - MIN_Y) * (H - 2 * PAD),
        ];
    }
    function toData(px, py) {
        return [
            MIN_X + (px - PAD) / (W - 2 * PAD) * (MAX_X - MIN_X),
            MIN_Y + (H - PAD - py) / (H - 2 * PAD) * (MAX_Y - MIN_Y),
        ];
    }

    // ----- State -----
    const defaultPoints = () => ([
        { x: 1.0, y: 2.1 }, { x: 2.0, y: 3.5 }, { x: 3.0, y: 4.2 },
        { x: 4.0, y: 5.8 }, { x: 5.0, y: 6.1 }, { x: 6.0, y: 7.5 },
        { x: 7.0, y: 7.9 }, { x: 8.0, y: 9.1 },
    ]);
    let points = defaultPoints();
    let dragging = null;
    let degree = 1;   // 1 = linear OLS; >1 = polynomial via normal equations

    // ----- Small dense linear solver: Gauss–Jordan on [A | b] -----
    function solve(A, b) {
        const n = b.length;
        const M = A.map((row, i) => [...row, b[i]]);
        for (let i = 0; i < n; i++) {
            // Partial pivot
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

    // ----- Polynomial least-squares fit: (X^T X) β = X^T y -----
    function fitPoly(deg) {
        const n = points.length;
        const k = deg + 1;
        if (n < 2) return Array(k).fill(0);
        // Centre x to keep the normal equations well-conditioned at high degree
        let sumX = 0;
        for (const p of points) sumX += p.x;
        const xC = sumX / n;
        const A = Array.from({length: k}, () => Array(k).fill(0));
        const b = Array(k).fill(0);
        for (const p of points) {
            const xs = p.x - xC;
            const xp = Array(2 * k - 1);
            xp[0] = 1;
            for (let j = 1; j < 2 * k - 1; j++) xp[j] = xp[j - 1] * xs;
            for (let i = 0; i < k; i++) {
                for (let j = 0; j < k; j++) A[i][j] += xp[i + j];
                b[i] += xp[i] * p.y;
            }
        }
        // Tiny ridge for numerical stability at high degree
        for (let i = 0; i < k; i++) A[i][i] += 1e-6;
        const beta = solve(A, b);
        if (!beta) return Array(k).fill(0);
        return { beta, xC };
    }

    function evalPoly(model, x) {
        const xs = x - model.xC;
        let y = 0, p = 1;
        for (const c of model.beta) { y += c * p; p *= xs; }
        return y;
    }

    // ----- OLS fit (linear, returns slope/intercept/R² for the stats strip) -----
    function fit() {
        const n = points.length;
        if (n < 2) return { slope: 0, intercept: 0, r2: 0 };

        let sumX = 0, sumY = 0;
        for (const p of points) { sumX += p.x; sumY += p.y; }
        const xMean = sumX / n;
        const yMean = sumY / n;

        let num = 0, denX = 0, denY = 0;
        for (const p of points) {
            const dx = p.x - xMean;
            const dy = p.y - yMean;
            num  += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        if (denX === 0) return { slope: 0, intercept: yMean, r2: 0 };
        const slope     = num / denX;
        const intercept = yMean - slope * xMean;
        const r2        = (denY === 0) ? 1 : (num * num) / (denX * denY);
        return { slope, intercept, r2 };
    }

    // ----- Draw -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);

        // Background panel
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Gridlines
        ctx.strokeStyle = '#e8e5dd';
        ctx.lineWidth = 1;
        for (let i = MIN_X; i <= MAX_X; i++) {
            const [px] = toPx(i, 0);
            ctx.beginPath();
            ctx.moveTo(px, PAD);
            ctx.lineTo(px, H - PAD);
            ctx.stroke();
        }
        for (let i = MIN_Y; i <= MAX_Y; i++) {
            const [, py] = toPx(0, i);
            ctx.beginPath();
            ctx.moveTo(PAD, py);
            ctx.lineTo(W - PAD, py);
            ctx.stroke();
        }

        // Axes
        ctx.strokeStyle = '#5f5f5f';
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(PAD, PAD);
        ctx.lineTo(PAD, H - PAD);
        ctx.lineTo(W - PAD, H - PAD);
        ctx.stroke();

        // Axis labels
        ctx.fillStyle = '#5f5f5f';
        ctx.font = '11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('x', W - PAD + 16, H - PAD + 4);
        ctx.fillText('y', PAD - 4, PAD - 8);

        // Fit (line for stats, polynomial for the drawn curve)
        const { slope, intercept, r2 } = fit();
        let model = null;
        if (points.length >= 2) {
            model = fitPoly(degree);
            ctx.save();
            ctx.beginPath();
            ctx.rect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
            ctx.clip();

            // Residuals (faint), using the polynomial fit
            ctx.strokeStyle = 'rgba(234, 121, 89, 0.4)';
            ctx.lineWidth = 1;
            for (const p of points) {
                const yHat = evalPoly(model, p.x);
                const [pxA, pyA] = toPx(p.x, p.y);
                const [pxB, pyB] = toPx(p.x, yHat);
                ctx.beginPath();
                ctx.moveTo(pxA, pyA);
                ctx.lineTo(pxB, pyB);
                ctx.stroke();
            }

            // The fitted curve, sampled densely
            ctx.strokeStyle = '#4f46e5';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            const steps = 240;
            for (let i = 0; i <= steps; i++) {
                const x = MIN_X + (i / steps) * (MAX_X - MIN_X);
                const y = evalPoly(model, x);
                const [px, py] = toPx(x, y);
                if (i === 0) ctx.moveTo(px, py);
                else         ctx.lineTo(px, py);
            }
            ctx.stroke();

            ctx.restore();
        }

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            const active = (p === dragging);
            ctx.fillStyle   = active ? '#ea7959' : '#4f46e5';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth   = 2;
            ctx.beginPath();
            ctx.arc(px, py, active ? 7 : 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Compute MSE on the actually-drawn (poly) curve for the stats strip
        let mse = 0;
        if (model && points.length) {
            for (const p of points) {
                const r = p.y - evalPoly(model, p.x);
                mse += r * r;
            }
            mse /= points.length;
        }

        // Stats panel
        const stats = document.getElementById('viz-regression-stats');
        if (stats) {
            const fitLabel = degree === 1 ? 'linear OLS' : `polynomial deg ${degree}`;
            stats.innerHTML = `
                <span><span class="label">fit</span>${fitLabel}</span>
                <span><span class="label">MSE</span>${mse.toFixed(3)}</span>
                <span><span class="label">R² (linear)</span>${r2.toFixed(3)}</span>
                <span><span class="label">n</span>${points.length}</span>
            `;
        }
    }

    // ----- Interaction -----
    function pointAt(px, py) {
        for (const p of points) {
            const [pxx, pyy] = toPx(p.x, p.y);
            if (Math.hypot(px - pxx, py - pyy) < 12) return p;
        }
        return null;
    }

    function getEventPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }

    function onDown(e) {
        e.preventDefault();
        const [px, py] = getEventPos(e);
        const p = pointAt(px, py);
        if (p && e.shiftKey) {
            points = points.filter(q => q !== p);
            draw();
            return;
        }
        if (p) {
            dragging = p;
            draw();
            return;
        }
        const [dx, dy] = toData(px, py);
        if (dx >= MIN_X && dx <= MAX_X && dy >= MIN_Y && dy <= MAX_Y) {
            const np = { x: dx, y: dy };
            points.push(np);
            dragging = np;
            draw();
        }
    }

    function onMove(e) {
        if (!dragging) return;
        e.preventDefault();
        const [px, py] = getEventPos(e);
        const [dx, dy] = toData(px, py);
        dragging.x = Math.max(MIN_X, Math.min(MAX_X, dx));
        dragging.y = Math.max(MIN_Y, Math.min(MAX_Y, dy));
        draw();
    }

    function onUp() {
        if (dragging) { dragging = null; draw(); }
    }

    canvas.addEventListener('mousedown',  onDown);
    canvas.addEventListener('mousemove',  onMove);
    canvas.addEventListener('mouseup',    onUp);
    canvas.addEventListener('mouseleave', onUp);
    canvas.addEventListener('touchstart', onDown);
    canvas.addEventListener('touchmove',  onMove);
    canvas.addEventListener('touchend',   onUp);

    document.getElementById('viz-regression-reset')?.addEventListener('click', () => {
        points = defaultPoints();
        draw();
    });
    document.getElementById('viz-regression-noisy')?.addEventListener('click', () => {
        // Linear trend plus heavy noise — illustrates R² < 1
        points = Array.from({ length: 14 }, (_, i) => {
            const x = 0.5 + i * (9 / 13);
            const y = 0.8 * x + 1 + (Math.random() - 0.5) * 4;
            return { x, y: Math.max(MIN_Y, Math.min(MAX_Y, y)) };
        });
        draw();
    });
    document.getElementById('viz-regression-clear')?.addEventListener('click', () => {
        points = [];
        draw();
    });
    document.getElementById('viz-regression-curvy')?.addEventListener('click', () => {
        // Sine-shaped trend — linear fit struggles, degree 3+ explains it.
        points = Array.from({ length: 16 }, (_, i) => {
            const x = 0.4 + i * (9.2 / 15);
            const y = 5 + 3 * Math.sin((x - 0.4) * 0.85) + (Math.random() - 0.5) * 0.6;
            return { x, y: Math.max(MIN_Y, Math.min(MAX_Y, y)) };
        });
        draw();
    });
    const degSel = document.getElementById('viz-regression-degree');
    if (degSel) {
        degSel.innerHTML = `
            <option value="1">Linear (deg 1)</option>
            <option value="2">Polynomial (deg 2)</option>
            <option value="3">Polynomial (deg 3)</option>
            <option value="5">Polynomial (deg 5)</option>
            <option value="9">Polynomial (deg 9)</option>
        `;
        degSel.addEventListener('change', () => {
            degree = parseInt(degSel.value, 10);
            draw();
        });
    }

    if (typeof ResizeObserver !== 'undefined') {
        let debounce = null;
        const ro = new ResizeObserver(() => {
            clearTimeout(debounce);
            debounce = setTimeout(resize, 60);
        });
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
