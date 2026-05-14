/* Interactive interpretability viz — SHAP-style local explanation.
 * One tabular instance, six features, each with a contribution that pushes
 * the prediction up (terracotta) or down (indigo). The bars are drawn in
 * the canonical "waterfall" style — start at the base rate, accumulate
 * each contribution, end at the prediction. */

(function () {
    const canvas    = document.getElementById('viz-int-canvas');
    const resampleBtn = document.getElementById('viz-int-resample');
    const captionEl = document.getElementById('viz-int-caption');
    if (!canvas) return;

    const FEATURES = [
        { name: 'age',                weight:  0.6 },
        { name: 'income',             weight:  1.2 },
        { name: 'credit_score',       weight:  1.4 },
        { name: 'debt_to_income',     weight: -1.0 },
        { name: 'employment_years',   weight:  0.7 },
        { name: 'recent_defaults',    weight: -1.6 },
    ];

    let values = [];
    let contributions = [];
    let ctx;
    let W = 0, H = 0;
    const BASE_RATE = 0.5;       // model average prediction

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function resample() {
        values = FEATURES.map(() => randn() * 0.8);
        contributions = FEATURES.map((f, i) => f.weight * values[i] * 0.08);
        draw();
    }
    resample();

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(280, cssW * 0.45)));
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

        const pad = 50;
        const box = { x: 130, y: 50, w: W - 230, h: H - 90 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // X axis: predictions 0 to 1
        function xToPx(p) { return box.x + p * box.w; }

        // Reference lines at 0.5 (base) and the final prediction
        const final = BASE_RATE + contributions.reduce((s, v) => s + v, 0);

        // Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('SHAP-STYLE WATERFALL — base rate + contributions = this prediction',
                     box.x, box.y - 30);

        // Base rate marker
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        const bx = xToPx(BASE_RATE);
        ctx.beginPath(); ctx.moveTo(bx, box.y); ctx.lineTo(bx, box.y + box.h); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('base 0.5', bx, box.y - 6);

        // Bars
        const rowH = box.h / FEATURES.length;
        let running = BASE_RATE;
        for (let i = 0; i < FEATURES.length; i++) {
            const yMid = box.y + (i + 0.5) * rowH;
            const startX = xToPx(running);
            running += contributions[i];
            const endX = xToPx(running);
            const isPos = contributions[i] > 0;
            ctx.fillStyle = isPos ? 'rgba(234, 121, 89, 0.85)' : 'rgba(79, 70, 229, 0.85)';
            ctx.fillRect(Math.min(startX, endX), yMid - rowH * 0.32,
                         Math.abs(endX - startX), rowH * 0.64);
            // Connecting line to next row
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(endX, yMid + rowH * 0.32);
            if (i < FEATURES.length - 1) ctx.lineTo(endX, yMid + rowH);
            ctx.stroke();

            // Feature label + value
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.font = '500 11px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${FEATURES[i].name}`, box.x - 8, yMid + 3);
            ctx.font = '500 9px "JetBrains Mono", monospace';
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.fillText(`${values[i].toFixed(2)}`, box.x - 8, yMid + 16);

            // Contribution value
            ctx.font = '600 10px "JetBrains Mono", monospace';
            ctx.fillStyle = isPos ? '#ea7959' : '#4f46e5';
            ctx.textAlign = 'left';
            const off = isPos ? endX + 4 : startX + 4;
            ctx.fillText(`${contributions[i] >= 0 ? '+' : ''}${contributions[i].toFixed(3)}`,
                         box.x + box.w + 4, yMid + 3);
        }

        // Final prediction marker
        const fx = xToPx(Math.max(0, Math.min(1, final)));
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(fx, box.y); ctx.lineTo(fx, box.y + box.h); ctx.stroke();
        ctx.fillStyle = '#1a1a1a';
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`pred ${final.toFixed(3)}`, fx, box.y - 6);

        // Axis ticks
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const v of [0, 0.25, 0.5, 0.75, 1]) {
            ctx.fillText(v.toFixed(2), xToPx(v), box.y + box.h + 14);
        }
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('predicted probability', box.x + box.w / 2, box.y + box.h + 30);

        updateCaption(final);
    }

    function updateCaption(final) {
        if (!captionEl) return;
        const pos = contributions.filter(c => c > 0).length;
        captionEl.innerHTML =
            `<strong>Final prediction: ${(final * 100).toFixed(1)}%.</strong> ` +
            `${pos} features pushed it up (orange bars), ${FEATURES.length - pos} pushed it down (indigo). ` +
            `Sum of all contributions + 0.5 base rate = the prediction — that's the SHAP additivity property. ` +
            `This kind of plot is the standard "explain this loan decision" output for regulated tabular ML.`;
    }

    resampleBtn?.addEventListener('click', resample);
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
