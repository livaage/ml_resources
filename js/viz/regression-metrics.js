/* Interactive regression metrics viz.
 * Scatter of points roughly on a line, with one prominent orange outlier
 * the user can drag up/down. Live metric strip shows RMSE, MAE, R², MAPE
 * — and you watch RMSE balloon as the outlier moves away from the trend,
 * while MAE rises only linearly. */

(function () {
    const canvas    = document.getElementById('viz-rm-canvas');
    const oSlider   = document.getElementById('viz-rm-outlier');
    const oLbl      = document.getElementById('viz-rm-out-lbl');
    const resetBtn  = document.getElementById('viz-rm-reset');
    const captionEl = document.getElementById('viz-rm-caption');
    if (!canvas) return;

    let points = [];           // {x, y}
    let outlierY = 0;
    const N = 18;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function regenerate() {
        points = [];
        for (let i = 0; i < N; i++) {
            const x = (i + 0.5) / N * 2 - 1;
            const y = 0.7 * x + 0.05 + randn() * 0.08;
            points.push({ x, y });
        }
        // The "outlier" is the middle point, controlled by the slider
        const idx = Math.floor(N / 2);
        outlierY = points[idx].y;
        if (oSlider) oSlider.value = outlierY;
    }
    function reset() {
        regenerate();
        draw();
    }

    // ----- OLS on the current data (including the live outlier) -----
    function ols() {
        const pts = points.map((p, i) => i === Math.floor(N / 2)
            ? { x: p.x, y: outlierY } : p);
        let sx = 0, sy = 0;
        for (const p of pts) { sx += p.x; sy += p.y; }
        const mx = sx / N, my = sy / N;
        let num = 0, den = 0;
        for (const p of pts) {
            num += (p.x - mx) * (p.y - my);
            den += (p.x - mx) * (p.x - mx);
        }
        const slope = num / (den || 1);
        const intercept = my - slope * mx;
        return { slope, intercept, pts };
    }

    function metrics() {
        const { slope, intercept, pts } = ols();
        let sse = 0, sae = 0, sst = 0, smape = 0, nMAPE = 0;
        let sy = 0;
        for (const p of pts) sy += p.y;
        const my = sy / pts.length;
        for (const p of pts) {
            const yh = slope * p.x + intercept;
            const r = p.y - yh;
            sse += r * r;
            sae += Math.abs(r);
            sst += (p.y - my) ** 2;
            if (Math.abs(p.y) > 0.05) { smape += Math.abs(r / p.y); nMAPE++; }
        }
        const n = pts.length;
        const mse = sse / n;
        const rmse = Math.sqrt(mse);
        const mae = sae / n;
        const r2 = 1 - sse / (sst || 1);
        const mape = nMAPE > 0 ? smape / nMAPE : NaN;
        return { rmse, mae, r2, mape, slope, intercept };
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(340, cssW * 0.52)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function toPx(x, y) {
        const pad = 40;
        return [
            pad + ((x + 1) / 2) * (W - 2 * pad),
            H / 2 - 30 - y * (H - 130) / 4,
        ];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 40;
        const plotBot = H - 60;
        ctx.fillStyle = '#fff';
        ctx.fillRect(pad, 20, W - 2 * pad, plotBot - 20);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(pad - 0.5, 19.5, W - 2 * pad + 1, plotBot - 19);

        // Axes
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.setLineDash([2, 3]);
        const [, y0p] = toPx(0, 0);
        ctx.beginPath();
        ctx.moveTo(pad, y0p); ctx.lineTo(W - pad, y0p); ctx.stroke();
        ctx.setLineDash([]);

        // OLS line
        const m = metrics();
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        const [px0, py0] = toPx(-1, m.slope * -1 + m.intercept);
        const [px1, py1] = toPx( 1, m.slope *  1 + m.intercept);
        ctx.moveTo(px0, py0); ctx.lineTo(px1, py1);
        ctx.stroke();

        // Residual segments
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const isOutlier = (i === Math.floor(N / 2));
            const py = isOutlier ? outlierY : p.y;
            const yHat = m.slope * p.x + m.intercept;
            const [ax, ay] = toPx(p.x, py);
            const [bx, by] = toPx(p.x, yHat);
            ctx.strokeStyle = 'rgba(234, 121, 89, 0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        }

        // Points
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const isOutlier = (i === Math.floor(N / 2));
            const py = isOutlier ? outlierY : p.y;
            const [px, ppy] = toPx(p.x, py);
            if (isOutlier) {
                ctx.fillStyle = 'rgba(234, 121, 89, 0.25)';
                ctx.beginPath(); ctx.arc(px, ppy, 12, 0, Math.PI * 2); ctx.fill();
            }
            ctx.fillStyle = isOutlier ? '#ea7959' : '#4f46e5';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px, ppy, isOutlier ? 7 : 4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Metric strip
        const sy = plotBot + 22;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('METRICS', pad, plotBot + 10);
        const items = [
            ['RMSE', m.rmse.toFixed(3), '#4f46e5'],
            ['MAE',  m.mae.toFixed(3),  '#ea7959'],
            ['R²',   m.r2.toFixed(3),   '#10847e'],
            ['MAPE', isFinite(m.mape) ? `${(m.mape * 100).toFixed(0)}%` : '—', '#d4a13c'],
        ];
        const slot = (W - 2 * pad) / 4;
        for (let i = 0; i < items.length; i++) {
            const [lbl, val, col] = items[i];
            const x = pad + i * slot;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.fillText(lbl, x, sy);
            ctx.font = '600 14px "JetBrains Mono", monospace';
            ctx.fillStyle = col;
            ctx.fillText(val, x, sy + 18);
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('LINEAR FIT — drag the orange outlier', pad, 14);

        if (oLbl) oLbl.textContent = `y = ${outlierY.toFixed(2)}`;
        updateCaption(m);
    }

    function updateCaption(m) {
        if (!captionEl) return;
        const trueOutlierY = points[Math.floor(N / 2)].y;
        const dev = Math.abs(outlierY - trueOutlierY);
        captionEl.innerHTML =
            `<strong>RMSE ${m.rmse.toFixed(2)}, MAE ${m.mae.toFixed(2)}, R² ${m.r2.toFixed(2)}.</strong> ` +
            (dev > 1.2
                ? `The outlier is far from the trend. RMSE has ballooned — its squared term punishes that one bad point a lot — while MAE barely moves. R² has also collapsed because the variance the model has to explain is dominated by that single point.`
                : `Pull the orange dot upward (or downward) and watch the metrics diverge. RMSE rises quadratically; MAE rises linearly; R² drops sharply.`);
    }

    // ----- Controls -----
    if (oSlider) {
        oSlider.min = -2; oSlider.max = 2; oSlider.step = 0.05; oSlider.value = 0;
        oSlider.addEventListener('input', () => {
            outlierY = parseFloat(oSlider.value);
            draw();
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
