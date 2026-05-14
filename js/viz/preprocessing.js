/* Interactive preprocessing viz.
 * Same labelled 2D dataset with feature scales deliberately ~1000:1. Toggle
 * the scaler and watch the k-NN decision boundary go from "ignores y entirely"
 * (raw) to "balanced" (standardised / min-max / robust). The pre-scaled axes
 * are drawn faintly behind to make the transform legible. */

(function () {
    const canvas    = document.getElementById('viz-pre-canvas');
    const rawBtn    = document.getElementById('viz-pre-raw');
    const stdBtn    = document.getElementById('viz-pre-std');
    const mmBtn     = document.getElementById('viz-pre-minmax');
    const robBtn    = document.getElementById('viz-pre-robust');
    const captionEl = document.getElementById('viz-pre-caption');
    if (!canvas) return;

    let mode = 'raw';
    let points = [];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    (function init() {
        for (let i = 0; i < 50; i++) {
            // x in [0, 1000], y in [0, 1], correlated with class
            const c = Math.random() < 0.5 ? 0 : 1;
            const x = (c === 0 ? 280 : 720) + randn() * 110;
            const y = (c === 0 ? 0.32 : 0.68) + randn() * 0.10;
            points.push({ x, y, c });
        }
    })();

    function transform(p) {
        if (mode === 'raw') return [p.x, p.y];
        const xs = points.map(q => q.x), ys = points.map(q => q.y);
        if (mode === 'std') {
            const mx = mean(xs), my = mean(ys);
            const sx = std(xs, mx), sy = std(ys, my);
            return [(p.x - mx) / sx, (p.y - my) / sy];
        }
        if (mode === 'minmax') {
            const xmin = Math.min(...xs), xmax = Math.max(...xs);
            const ymin = Math.min(...ys), ymax = Math.max(...ys);
            return [(p.x - xmin) / (xmax - xmin), (p.y - ymin) / (ymax - ymin)];
        }
        if (mode === 'robust') {
            const xs_ = [...xs].sort((a, b) => a - b);
            const ys_ = [...ys].sort((a, b) => a - b);
            const mx = xs_[Math.floor(xs_.length / 2)];
            const my = ys_[Math.floor(ys_.length / 2)];
            const iqx = xs_[Math.floor(xs_.length * 0.75)] - xs_[Math.floor(xs_.length * 0.25)];
            const iqy = ys_[Math.floor(ys_.length * 0.75)] - ys_[Math.floor(ys_.length * 0.25)];
            return [(p.x - mx) / (iqx || 1), (p.y - my) / (iqy || 1)];
        }
        return [p.x, p.y];
    }
    function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
    function std(a, m) {
        const v = a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length;
        return Math.sqrt(v) || 1;
    }

    function transformAll() {
        return points.map(p => {
            const [x, y] = transform(p);
            return { x, y, c: p.c };
        });
    }

    function predictKNN(x, y, pts, k = 5) {
        const ds = pts.map(p => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, c: p.c }));
        ds.sort((a, b) => a.d - b.d);
        let n0 = 0, n1 = 0;
        for (let i = 0; i < k && i < ds.length; i++) (ds[i].c === 0 ? n0++ : n1++);
        return n0 >= n1 ? 0 : 1;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.50)));
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

        const pts = transformAll();
        // Compute bounds with a little padding
        const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
        const xmin = Math.min(...xs), xmax = Math.max(...xs);
        const ymin = Math.min(...ys), ymax = Math.max(...ys);
        const dx = (xmax - xmin) * 0.1, dy = (ymax - ymin) * 0.1;
        const lo = [xmin - dx, ymin - dy], hi = [xmax + dx, ymax + dy];

        const pad = 36;
        const plot = { x: pad, y: 30, w: W - 2 * pad, h: H - 80 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(plot.x - 0.5, plot.y - 0.5, plot.w + 1, plot.h + 1);

        function toPx(x, y) {
            return [
                plot.x + (x - lo[0]) / (hi[0] - lo[0]) * plot.w,
                plot.y + plot.h - (y - lo[1]) / (hi[1] - lo[1]) * plot.h,
            ];
        }

        // Background heatmap of k-NN decision
        const G = 50;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const xv = lo[0] + (i + 0.5) / G * (hi[0] - lo[0]);
                const yv = lo[1] + (j + 0.5) / G * (hi[1] - lo[1]);
                const cls = predictKNN(xv, yv, pts);
                ctx.fillStyle = cls === 0 ? '#eef2ff' : '#fdf0eb';
                const [px, py] = toPx(xv, yv);
                ctx.fillRect(px - plot.w / G / 2, py - plot.h / G / 2,
                             plot.w / G + 1, plot.h / G + 1);
            }
        }

        // Axes
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(lo[0].toFixed(1), plot.x, plot.y + plot.h + 12);
        ctx.fillText(hi[0].toFixed(1), plot.x + plot.w, plot.y + plot.h + 12);
        ctx.textAlign = 'right';
        ctx.fillText(lo[1].toFixed(2), plot.x - 4, plot.y + plot.h + 3);
        ctx.fillText(hi[1].toFixed(2), plot.x - 4, plot.y + 3);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('feature 1', plot.x + plot.w / 2, plot.y + plot.h + 26);
        ctx.save();
        ctx.translate(plot.x - 22, plot.y + plot.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('feature 2', 0, 0);
        ctx.restore();

        // Points
        for (const p of pts) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = p.c === 0 ? '#4f46e5' : '#ea7959';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 3.6, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { raw: 'RAW DATA — feature 1 dominates by 1000:1',
                         std: 'STANDARDISED — mean 0, std 1',
                         minmax: 'MIN-MAX — squashed to [0, 1]',
                         robust: 'ROBUST — median + IQR' };
        ctx.fillText(titles[mode], plot.x, plot.y - 8);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            raw:    `<strong>Raw data.</strong> Feature 1 ranges over [0, 1000]; feature 2 is in [0, 1]. Euclidean distance is dominated by feature 1 — the k-NN boundary becomes a vertical strip that ignores feature 2 entirely.`,
            std:    `<strong>Standardised.</strong> Both features now have mean 0 and std 1. Distance treats them equally — the boundary diagonal reflects the actual class separation.`,
            minmax: `<strong>Min-max scaled.</strong> Both features squashed to [0, 1]. Similar effect to standardisation but bounded — useful when you need bounded inputs or are sensitive to outliers in the std calculation.`,
            robust: `<strong>Robust scaled.</strong> Subtract the median, divide by the IQR. Outliers don't change the scale, which means it stays sensible when the data has heavy tails.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [rawBtn, stdBtn, mmBtn, robBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    rawBtn?.addEventListener('click', () => setMode('raw', rawBtn));
    stdBtn?.addEventListener('click', () => setMode('std', stdBtn));
    mmBtn?.addEventListener('click',  () => setMode('minmax', mmBtn));
    robBtn?.addEventListener('click', () => setMode('robust', robBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
