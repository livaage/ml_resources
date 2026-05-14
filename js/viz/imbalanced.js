/* Interactive imbalanced-data viz.
 * 2D Gaussian blobs with an adjustable class ratio. Fit a logistic
 * regression under four regimes: unfixed, class weights, oversample,
 * undersample. Show the boundary, the confusion matrix, and the
 * difference between accuracy (misleading) and recall (the actual story). */

(function () {
    const canvas    = document.getElementById('viz-imb-canvas');
    const rSlider   = document.getElementById('viz-imb-ratio');
    const rLbl      = document.getElementById('viz-imb-r-lbl');
    const noneBtn   = document.getElementById('viz-imb-none');
    const weightBtn = document.getElementById('viz-imb-weight');
    const overBtn   = document.getElementById('viz-imb-over');
    const underBtn  = document.getElementById('viz-imb-under');
    const captionEl = document.getElementById('viz-imb-caption');
    if (!canvas) return;

    let fix = 'none';
    let ratio = 0.05;
    let points = [];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function regenerate() {
        points = [];
        const N = 200;
        const nPos = Math.max(2, Math.round(N * ratio));
        for (let i = 0; i < N - nPos; i++) {
            points.push({ x: -0.4 + randn() * 0.25, y: 0.0 + randn() * 0.25, c: 0 });
        }
        for (let i = 0; i < nPos; i++) {
            points.push({ x:  0.5 + randn() * 0.22, y: 0.0 + randn() * 0.22, c: 1 });
        }
    }
    regenerate();

    function fitLR(pts, weights) {
        let w = [0, 0], b = 0;
        for (let it = 0; it < 250; it++) {
            const lr = 0.4 / (1 + it / 80);
            let gw0 = 0, gw1 = 0, gb = 0;
            let denom = 0;
            for (let i = 0; i < pts.length; i++) {
                const p = pts[i];
                const z = w[0] * p.x + w[1] * p.y + b;
                const s = 1 / (1 + Math.exp(-z));
                const wgt = weights ? weights[i] : 1;
                const e = wgt * (s - p.c);
                gw0 += e * p.x; gw1 += e * p.y; gb += e;
                denom += wgt;
            }
            w[0] -= lr * gw0 / denom;
            w[1] -= lr * gw1 / denom;
            b    -= lr * gb / denom;
        }
        return { w, b };
    }
    function predict(model, x, y) {
        return (model.w[0] * x + model.w[1] * y + model.b > 0) ? 1 : 0;
    }

    function applyFix() {
        if (fix === 'none')   return fitLR(points);
        if (fix === 'weight') {
            const n0 = points.filter(p => p.c === 0).length;
            const n1 = points.filter(p => p.c === 1).length;
            const weights = points.map(p => p.c === 0 ? 1 : n0 / Math.max(1, n1));
            return fitLR(points, weights);
        }
        if (fix === 'over') {
            // Duplicate positives until balanced
            const pos = points.filter(p => p.c === 1);
            const neg = points.filter(p => p.c === 0);
            const dup = [];
            let k = 0;
            while (dup.length + pos.length < neg.length) {
                dup.push({ ...pos[k % pos.length] });
                k++;
            }
            return fitLR([...neg, ...pos, ...dup]);
        }
        if (fix === 'under') {
            const pos = points.filter(p => p.c === 1);
            const neg = points.filter(p => p.c === 0);
            const sample = neg.slice(0, Math.max(pos.length, 5));
            return fitLR([...sample, ...pos]);
        }
        return fitLR(points);
    }

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
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 20;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const model = applyFix();
        const m = Math.min(W, H) / 2 - 20;
        ctx.fillStyle = '#fff';
        ctx.fillRect(W / 2 - m, H / 2 - m, 2 * m, 2 * m);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m - 0.5, H / 2 - m - 0.5, 2 * m + 1, 2 * m + 1);

        // Background decision regions
        const G = 50;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx = -1.05 + (i + 0.5) / G * 2.1;
                const dy = -1.05 + (j + 0.5) / G * 2.1;
                const cls = predict(model, dx, dy);
                ctx.fillStyle = cls === 0 ? '#eef2ff' : '#fdf0eb';
                const [px, py] = toPx(dx, dy);
                ctx.fillRect(px - m * 1.05 / G - 1, py - m * 1.05 / G - 1,
                             m * 2.1 / G + 2, m * 2.1 / G + 2);
            }
        }

        // Decision line w·x + b = 0
        if (Math.abs(model.w[1]) > 1e-6) {
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            const xL = -1.05, xR = 1.05;
            const yL = (-model.b - model.w[0] * xL) / model.w[1];
            const yR = (-model.b - model.w[0] * xR) / model.w[1];
            const [a, b] = toPx(xL, yL), [c, d] = toPx(xR, yR);
            ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
        }

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = p.c === 0 ? '#4f46e5' : '#ea7959';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, p.c === 1 ? 4 : 2.8, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Metrics
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (const p of points) {
            const pred = predict(model, p.x, p.y);
            if (pred === 1 && p.c === 1) tp++;
            else if (pred === 1 && p.c === 0) fp++;
            else if (pred === 0 && p.c === 0) tn++;
            else fn++;
        }
        const acc = (tp + tn) / points.length;
        const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;

        // Title + metrics
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { none: 'NO FIX', weight: 'CLASS WEIGHTS',
                         over: 'OVERSAMPLED', under: 'UNDERSAMPLED' };
        ctx.fillText(titles[fix], W / 2 - m, H / 2 - m - 6);
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.textAlign = 'right';
        ctx.fillText(
            `acc=${acc.toFixed(2)}  prec=${prec.toFixed(2)}  rec=${rec.toFixed(2)}`,
            W / 2 + m, H / 2 - m - 6);

        updateCaption({ acc, rec, prec, tp, fp, tn, fn });
    }

    function updateCaption(m) {
        if (!captionEl) return;
        const pct = (ratio * 100).toFixed(0);
        const notes = {
            none: m.rec < 0.3
                ? `<strong>No fix at ${pct}% positives.</strong> Accuracy ${m.acc.toFixed(2)} looks fine but recall is only ${m.rec.toFixed(2)} — the model is missing most of the rare class because the loss is dominated by negatives.`
                : `<strong>No fix at ${pct}% positives.</strong> Accuracy ${m.acc.toFixed(2)}, recall ${m.rec.toFixed(2)}. At this ratio the rare class still has enough signal.`,
            weight: `<strong>Class weights.</strong> Each positive contributes (n_neg / n_pos) more to the loss. The boundary pulls back toward the positives — recall jumps to ${m.rec.toFixed(2)} (precision drops some). Cheapest, most reliable first fix.`,
            over:   `<strong>Oversampled.</strong> Duplicated positives until balanced. The model treats each pos as multiple training points; recall ${m.rec.toFixed(2)}. Risk: overfitting to those duplicates.`,
            under:  `<strong>Undersampled.</strong> Dropped most negatives. The model sees a balanced training set; recall ${m.rec.toFixed(2)}. Cheapest in compute, but you lose information from the discarded negatives.`,
        };
        captionEl.innerHTML = notes[fix];
    }

    if (rSlider) {
        rSlider.min = 0.01; rSlider.max = 0.5; rSlider.step = 0.01; rSlider.value = 0.05;
        rSlider.addEventListener('input', () => {
            ratio = parseFloat(rSlider.value);
            if (rLbl) rLbl.textContent = `${(ratio * 100).toFixed(0)}%`;
            regenerate();
            draw();
        });
    }
    function setFix(f, btn) {
        fix = f;
        for (const b of [noneBtn, weightBtn, overBtn, underBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    noneBtn?.addEventListener('click',   () => setFix('none', noneBtn));
    weightBtn?.addEventListener('click', () => setFix('weight', weightBtn));
    overBtn?.addEventListener('click',   () => setFix('over', overBtn));
    underBtn?.addEventListener('click',  () => setFix('under', underBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
