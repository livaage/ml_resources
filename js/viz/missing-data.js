/* Interactive missing-data viz.
 * 2D scatter with 30% of points missing their y-coordinate.
 * Toggle the imputation strategy — see ghost (observed-y) vs imputed (faded-y) values,
 * plus the resulting linear fit. */

(function () {
    const canvas    = document.getElementById('viz-md-canvas');
    const meanBtn   = document.getElementById('viz-md-mean');
    const medianBtn = document.getElementById('viz-md-median');
    const knnBtn    = document.getElementById('viz-md-knn');
    const iterBtn   = document.getElementById('viz-md-iter');
    const flagBtn   = document.getElementById('viz-md-flag');
    const captionEl = document.getElementById('viz-md-caption');
    if (!canvas) return;

    let mode = 'mean';
    let points = [];     // {x, y_true, observed (bool)}

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    (function init() {
        for (let i = 0; i < 60; i++) {
            const x = -1 + Math.random() * 2;
            const y = 0.7 * x + 0.05 + randn() * 0.18;
            const observed = Math.random() > 0.30;     // 30% missing
            points.push({ x, y_true: y, observed });
        }
    })();

    function imputeAll() {
        const obs = points.filter(p => p.observed);
        const out = points.map(p => ({ ...p }));
        if (mode === 'mean') {
            const m = obs.reduce((s, p) => s + p.y_true, 0) / obs.length;
            for (const p of out) if (!p.observed) p.y_imp = m;
        } else if (mode === 'median') {
            const ys = obs.map(p => p.y_true).sort((a, b) => a - b);
            const md = ys[Math.floor(ys.length / 2)];
            for (const p of out) if (!p.observed) p.y_imp = md;
        } else if (mode === 'knn') {
            for (const p of out) {
                if (p.observed) continue;
                // 5 nearest observed by x
                const ds = obs.map(q => ({ d: Math.abs(q.x - p.x), y: q.y_true }))
                              .sort((a, b) => a.d - b.d)
                              .slice(0, 5);
                p.y_imp = ds.reduce((s, q) => s + q.y, 0) / ds.length;
            }
        } else if (mode === 'iter' || mode === 'flag') {
            // Fit y = β₀ + β₁ x on observed, predict missing
            let sx = 0, sy = 0;
            for (const p of obs) { sx += p.x; sy += p.y_true; }
            const mx = sx / obs.length, my = sy / obs.length;
            let num = 0, den = 0;
            for (const p of obs) {
                num += (p.x - mx) * (p.y_true - my);
                den += (p.x - mx) ** 2;
            }
            const slope = num / (den || 1);
            const intercept = my - slope * mx;
            for (const p of out) if (!p.observed) p.y_imp = slope * p.x + intercept;
        }
        return out;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
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

        const pad = 40;
        const box = { x: pad, y: 30, w: W - 2 * pad, h: H - 100 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        function toPx(x, y) {
            return [
                box.x + (x + 1.05) / 2.1 * box.w,
                box.y + box.h - (y + 1.5) / 3 * box.h,
            ];
        }

        // Zero line
        const [, y0] = toPx(0, 0);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(box.x, y0); ctx.lineTo(box.x + box.w, y0); ctx.stroke();
        ctx.setLineDash([]);

        const imputed = imputeAll();

        // Imputed values — connect observed (truth) ghost to the imputed value
        for (const p of imputed) {
            if (p.observed) continue;
            const [ax, ay] = toPx(p.x, p.y_true);
            const [bx, by] = toPx(p.x, p.y_imp);
            ctx.strokeStyle = 'rgba(234, 121, 89, 0.35)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
            // Ghost true value (faint outline)
            ctx.fillStyle = 'rgba(0, 0, 0, 0.0)';
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.beginPath(); ctx.arc(ax, ay, 3.5, 0, Math.PI * 2); ctx.stroke();
            // Imputed (terracotta filled)
            ctx.fillStyle = '#ea7959';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(bx, by, 3.5, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Observed values (indigo)
        for (const p of imputed) {
            if (!p.observed) continue;
            const [px, py] = toPx(p.x, p.y_true);
            ctx.fillStyle = '#4f46e5';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.arc(px, py, 3.4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Fit line on observed+imputed using OLS, draw it
        let sx = 0, sy = 0;
        const dataForFit = imputed.map(p => ({ x: p.x, y: p.observed ? p.y_true : p.y_imp }));
        for (const p of dataForFit) { sx += p.x; sy += p.y; }
        const mx = sx / dataForFit.length, my = sy / dataForFit.length;
        let num = 0, den = 0;
        for (const p of dataForFit) {
            num += (p.x - mx) * (p.y - my);
            den += (p.x - mx) ** 2;
        }
        const slope = num / (den || 1);
        const intercept = my - slope * mx;

        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const [a, b] = toPx(-1, -slope + intercept);
        const [c, d] = toPx( 1,  slope + intercept);
        ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();

        // Title + legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { mean: 'MEAN', median: 'MEDIAN', knn: 'k-NN (k=5)',
                         iter: 'ITERATIVE / MICE', flag: 'ITERATIVE + missingness flag' };
        ctx.fillText(titles[mode] + ' IMPUTATION', box.x, box.y - 8);

        // Legend
        const ly = box.y + box.h + 22;
        ctx.fillStyle = '#4f46e5';
        ctx.fillRect(box.x, ly - 9, 10, 10);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillText('observed', box.x + 16, ly);
        ctx.fillStyle = '#ea7959';
        ctx.fillRect(box.x + 110, ly - 9, 10, 10);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('imputed', box.x + 126, ly);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(box.x + 215, ly - 4, 3.5, 0, Math.PI * 2); ctx.stroke();
        ctx.fillText('true (unobserved)', box.x + 226, ly);

        updateCaption(slope);
    }

    function updateCaption(slope) {
        if (!captionEl) return;
        const trueSlope = 0.7;
        const slopeErr = Math.abs(slope - trueSlope);
        const notes = {
            mean:   `<strong>Mean imputation</strong> pulls every missing point to the column mean — visibly flattens the slope (recovered ${slope.toFixed(2)} vs true 0.70). MSE looks lower because variance shrank, but inference is biased.`,
            median: `<strong>Median imputation</strong> is more robust to outliers but has the same variance-shrinkage problem — slope ${slope.toFixed(2)}.`,
            knn:    `<strong>k-NN</strong> imputes from nearby points in the observed dimensions. Slope ${slope.toFixed(2)} — much closer to truth because the structure of x → y is preserved locally.`,
            iter:   `<strong>Iterative / MICE</strong> regresses each missing column on the others and converges. On this 2D problem it's effectively OLS — slope ${slope.toFixed(2)}, nearly perfect.`,
            flag:   `<strong>Iterative + missingness flag</strong>. The model also sees a 0/1 indicator per missing cell — so it can learn from the pattern of missingness itself. Same slope (${slope.toFixed(2)}) but downstream models get an extra useful feature.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [meanBtn, medianBtn, knnBtn, iterBtn, flagBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    meanBtn?.addEventListener('click',   () => setMode('mean', meanBtn));
    medianBtn?.addEventListener('click', () => setMode('median', medianBtn));
    knnBtn?.addEventListener('click',    () => setMode('knn', knnBtn));
    iterBtn?.addEventListener('click',   () => setMode('iter', iterBtn));
    flagBtn?.addEventListener('click',   () => setMode('flag', flagBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
