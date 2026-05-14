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

    // DPR-aware sizing — sharp on retina
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        W = rect.width;
        H = rect.height;
        canvas.width  = W * dpr;
        canvas.height = H * dpr;
        ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
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

    // ----- OLS fit -----
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

        // Fit line and residuals
        const { slope, intercept, r2 } = fit();
        if (points.length >= 2) {
            // Clip to plot area so the line doesn't escape
            ctx.save();
            ctx.beginPath();
            ctx.rect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
            ctx.clip();

            // Residual segments (faint)
            ctx.strokeStyle = 'rgba(234, 121, 89, 0.4)';
            ctx.lineWidth = 1;
            for (const p of points) {
                const yHat = slope * p.x + intercept;
                const [pxA, pyA] = toPx(p.x, p.y);
                const [pxB, pyB] = toPx(p.x, yHat);
                ctx.beginPath();
                ctx.moveTo(pxA, pyA);
                ctx.lineTo(pxB, pyB);
                ctx.stroke();
            }

            // The fitted line
            const yL = slope * MIN_X + intercept;
            const yR = slope * MAX_X + intercept;
            const [x1, y1] = toPx(MIN_X, yL);
            const [x2, y2] = toPx(MAX_X, yR);
            ctx.strokeStyle = '#4f46e5';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
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

        // Stats panel
        const stats = document.getElementById('viz-regression-stats');
        if (stats) {
            stats.innerHTML = `
                <span><span class="label">slope β₁</span>${slope.toFixed(3)}</span>
                <span><span class="label">intercept β₀</span>${intercept.toFixed(3)}</span>
                <span><span class="label">R²</span>${r2.toFixed(3)}</span>
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

    window.addEventListener('resize', resize);
    resize();
})();
