/* Interactive fairness viz.
 * Two groups (A and B) with different score distributions (different base
 * rates + different separability). One shared threshold τ. Show per-group
 * histograms with τ overlaid, plus live per-group metrics.
 * Slide τ and watch demographic parity, equal opportunity, and predictive
 * parity all move — usually in conflicting directions. */

(function () {
    const canvas    = document.getElementById('viz-fair-canvas');
    const tSlider   = document.getElementById('viz-fair-threshold');
    const tLbl      = document.getElementById('viz-fair-thr-lbl');
    const resampleBtn = document.getElementById('viz-fair-resample');
    const captionEl = document.getElementById('viz-fair-caption');
    if (!canvas) return;

    let threshold = 0.5;
    let groupA = [], groupB = [];
    let ctx;
    let W = 0, H = 0;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function regenerate() {
        groupA = []; groupB = [];
        // Group A: higher base rate (50% positive), separable
        for (let i = 0; i < 150; i++) {
            const y = i < 75 ? 1 : 0;
            const mu = y === 1 ? 1.3 : -1.0;
            const p = sigmoid(mu + 0.85 * randn());
            groupA.push({ p, y });
        }
        // Group B: lower base rate (~25% positive), less separable
        for (let i = 0; i < 150; i++) {
            const y = i < 38 ? 1 : 0;
            const mu = y === 1 ? 0.5 : -0.7;
            const p = sigmoid(mu + 1.05 * randn());
            groupB.push({ p, y });
        }
    }
    regenerate();

    function metrics(group, tau) {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (const s of group) {
            const pred = s.p >= tau ? 1 : 0;
            if (pred === 1 && s.y === 1) tp++;
            else if (pred === 1 && s.y === 0) fp++;
            else if (pred === 0 && s.y === 0) tn++;
            else fn++;
        }
        const sel = (tp + fp) / group.length;
        const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
        const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
        return { sel, tpr, fpr, prec };
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(560, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.46)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function drawHist(box, group, label) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        const nb = 22;
        const bin0 = new Float32Array(nb), bin1 = new Float32Array(nb);
        for (const s of group) {
            const i = Math.min(nb - 1, Math.max(0, Math.floor(s.p * nb)));
            (s.y === 1 ? bin1 : bin0)[i]++;
        }
        const mx = Math.max(...bin0, ...bin1) || 1;
        const histH = box.h - 30;
        const barW = box.w / nb;
        for (let i = 0; i < nb; i++) {
            const h0 = (bin0[i] / mx) * histH;
            const h1 = (bin1[i] / mx) * histH;
            ctx.fillStyle = 'rgba(79, 70, 229, 0.5)';
            ctx.fillRect(box.x + i * barW, box.y + histH - h0 + 6, barW - 1, h0);
            ctx.fillStyle = 'rgba(234, 121, 89, 0.5)';
            ctx.fillRect(box.x + i * barW + 1, box.y + histH - h1 + 6, barW - 2, h1);
        }

        // Threshold line
        const tx = box.x + threshold * box.w;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx, box.y + 4); ctx.lineTo(tx, box.y + histH + 4); ctx.stroke();

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, box.x, box.y - 6);

        // Group label sits in the top-right
        const m = metrics(group, threshold);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`sel ${(m.sel * 100).toFixed(0)}%  tpr ${(m.tpr * 100).toFixed(0)}%  prec ${(m.prec * 100).toFixed(0)}%`,
                     box.x + box.w, box.y - 6);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 16;
        const histH = (H - 3 * pad - 90) / 2;
        const histW = W - 2 * pad;
        const topA = { x: pad, y: pad + 22, w: histW, h: histH };
        const topB = { x: pad, y: pad + 22 + histH + 30, w: histW, h: histH };
        drawHist(topA, groupA, 'GROUP A — higher base rate (50% positive)');
        drawHist(topB, groupB, 'GROUP B — lower base rate (~25% positive)');

        // Metric comparison strip
        const mA = metrics(groupA, threshold);
        const mB = metrics(groupB, threshold);
        const stripY = topB.y + topB.h + 24;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('FAIRNESS DIFFERENCES (A − B)', pad, stripY);

        const items = [
            ['demographic parity', mA.sel - mB.sel],
            ['equal opportunity',   mA.tpr - mB.tpr],
            ['predictive parity',   mA.prec - mB.prec],
        ];
        const slot = (W - 2 * pad) / 3;
        for (let i = 0; i < items.length; i++) {
            const [lbl, gap] = items[i];
            const x = pad + i * slot;
            const col = Math.abs(gap) < 0.05 ? '#10847e' : (Math.abs(gap) < 0.15 ? '#d4a13c' : '#ea7959');
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.fillText(lbl, x, stripY + 14);
            ctx.font = '700 18px "JetBrains Mono", monospace';
            ctx.fillStyle = col;
            ctx.fillText(`${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(0)}pp`, x, stripY + 36);
        }

        if (tLbl) tLbl.textContent = `τ = ${threshold.toFixed(2)}`;
        updateCaption(mA, mB);
    }

    function updateCaption(mA, mB) {
        if (!captionEl) return;
        const selGap  = Math.abs(mA.sel - mB.sel) * 100;
        const tprGap  = Math.abs(mA.tpr - mB.tpr) * 100;
        const precGap = Math.abs(mA.prec - mB.prec) * 100;
        captionEl.innerHTML =
            `<strong>At τ = ${threshold.toFixed(2)}.</strong> Demographic parity gap ${selGap.toFixed(0)}pp, ` +
            `equal opportunity gap ${tprGap.toFixed(0)}pp, predictive parity gap ${precGap.toFixed(0)}pp. ` +
            `The classifier and threshold are the <em>same</em> for both groups — but because the underlying score distributions and base rates differ, the metrics diverge. ` +
            `Slide τ — you can close one gap, but watch the others open. This is the impossibility theorem made tangible.`;
    }

    if (tSlider) {
        tSlider.min = 0.05; tSlider.max = 0.95; tSlider.step = 0.01; tSlider.value = 0.5;
        tSlider.addEventListener('input', () => {
            threshold = parseFloat(tSlider.value);
            draw();
        });
    }
    resampleBtn?.addEventListener('click', () => { regenerate(); draw(); });

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
