/* Interactive classification metrics viz.
 * Three panels:
 *   Left:        predicted-score distributions for class 0 (indigo) and 1 (terracotta),
 *                with the threshold τ as a vertical line.
 *   Top-right:   2×2 confusion matrix, live with τ.
 *   Bottom-right ROC curve with the current operating point marked.
 * Plus a row of headline metrics — accuracy, precision, recall, F1, AUC.  */

(function () {
    const canvas    = document.getElementById('viz-clm-canvas');
    const tSlider   = document.getElementById('viz-clm-threshold');
    const tLbl      = document.getElementById('viz-clm-thr-lbl');
    const imbSel    = document.getElementById('viz-clm-imbalance');
    const resetBtn  = document.getElementById('viz-clm-reset');
    const captionEl = document.getElementById('viz-clm-caption');
    if (!canvas) return;

    let threshold = 0.5;
    let imbalance = 0.5;        // fraction positive
    const N = 300;
    let scores = [];            // {p, y}
    let aucCache = null;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function regenerate() {
        scores = [];
        const nPos = Math.round(N * imbalance);
        for (let i = 0; i < N; i++) {
            const y = i < nPos ? 1 : 0;
            // Logits with separation ~ 2σ
            const mu = y === 1 ? 1.2 : -0.8;
            const p = sigmoid(mu + 0.9 * randn());
            scores.push({ p, y });
        }
        aucCache = null;
    }
    function reset() {
        regenerate();
        draw();
    }

    function confusion(tau) {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (const s of scores) {
            const pred = s.p >= tau ? 1 : 0;
            if (pred === 1 && s.y === 1) tp++;
            else if (pred === 1 && s.y === 0) fp++;
            else if (pred === 0 && s.y === 0) tn++;
            else fn++;
        }
        const acc = (tp + tn) / N;
        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
        const rec  = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
        return { tp, fp, tn, fn, acc, prec, rec, f1 };
    }

    function rocPoints() {
        // Sort by descending score; sweep threshold
        const sorted = [...scores].sort((a, b) => b.p - a.p);
        let tp = 0, fp = 0;
        const pos = sorted.filter(s => s.y === 1).length;
        const neg = N - pos;
        const pts = [{ fpr: 0, tpr: 0, p: 1 }];
        for (const s of sorted) {
            if (s.y === 1) tp++; else fp++;
            pts.push({ fpr: fp / neg, tpr: tp / pos, p: s.p });
        }
        // AUC via trapezoidal rule
        let auc = 0;
        for (let i = 1; i < pts.length; i++) {
            auc += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
        }
        return { pts, auc };
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(360, cssW * 0.50)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function layout() {
        const pad = 14;
        const leftW = (W - 3 * pad) * 0.5;
        const left   = { x: pad, y: pad + 14, w: leftW, h: H - 2 * pad - 14 };
        const rightX = pad + leftW + pad;
        const rightW = W - rightX - pad;
        const cm     = { x: rightX, y: pad + 14, w: rightW, h: (H - 2 * pad - 14 - pad) * 0.45 };
        const roc    = { x: rightX, y: cm.y + cm.h + pad + 14, w: rightW, h: H - cm.y - cm.h - pad - pad - 14 };
        return { left, cm, roc };
    }

    function drawDistributions(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Histograms
        const nb = 24;
        const bin0 = new Float32Array(nb), bin1 = new Float32Array(nb);
        for (const s of scores) {
            const i = Math.min(nb - 1, Math.max(0, Math.floor(s.p * nb)));
            (s.y === 1 ? bin1 : bin0)[i]++;
        }
        const mx = Math.max(...bin0, ...bin1) || 1;
        const histH = box.h - 60;
        const histTop = box.y + 30;
        const barW = box.w / nb;
        for (let i = 0; i < nb; i++) {
            const h0 = (bin0[i] / mx) * histH;
            const h1 = (bin1[i] / mx) * histH;
            ctx.fillStyle = 'rgba(79, 70, 229, 0.55)';
            ctx.fillRect(box.x + i * barW, histTop + histH - h0, barW - 1, h0);
            ctx.fillStyle = 'rgba(234, 121, 89, 0.55)';
            ctx.fillRect(box.x + i * barW + 1, histTop + histH - h1, barW - 2, h1);
        }

        // Threshold line
        const tx = box.x + threshold * box.w;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx, box.y + 18); ctx.lineTo(tx, box.y + box.h - 26);
        ctx.stroke();
        ctx.fillStyle = '#1a1a1a';
        ctx.font = '600 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`τ=${threshold.toFixed(2)}`, tx, box.y + 12);

        // Axis
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const v of [0, 0.5, 1]) {
            ctx.fillText(v.toFixed(1), box.x + v * box.w, box.y + box.h - 12);
        }
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText('predicted probability', box.x + box.w / 2, box.y + box.h - 1);

        // Title + legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('SCORE DISTRIBUTIONS', box.x, box.y - 5);
        // Legend
        ctx.fillStyle = 'rgba(79, 70, 229, 0.6)';
        ctx.fillRect(box.x + box.w - 130, box.y + 22, 10, 10);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText('class 0', box.x + box.w - 116, box.y + 30);
        ctx.fillStyle = 'rgba(234, 121, 89, 0.6)';
        ctx.fillRect(box.x + box.w - 75, box.y + 22, 10, 10);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('class 1', box.x + box.w - 61, box.y + 30);
    }

    function drawConfusion(box, m) {
        // 2×2 grid
        const cw = (box.w - 50) / 2;
        const ch = (box.h - 56) / 2;
        const labels = [['TN', 'FP'], ['FN', 'TP']];
        const counts = [[m.tn, m.fp], [m.fn, m.tp]];
        const mxCount = Math.max(m.tn, m.fp, m.fn, m.tp);
        for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 2; c++) {
                const x = box.x + 50 + c * cw;
                const y = box.y + 40 + r * ch;
                const v = counts[r][c];
                const intensity = mxCount > 0 ? v / mxCount : 0;
                ctx.fillStyle = (r === c)
                    ? `rgba(79, 70, 229, ${0.1 + intensity * 0.55})`
                    : `rgba(234, 121, 89, ${0.1 + intensity * 0.55})`;
                ctx.fillRect(x, y, cw - 4, ch - 4);
                ctx.fillStyle = intensity > 0.5 ? '#fff' : 'rgba(0, 0, 0, 0.75)';
                ctx.font = '600 12px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`${labels[r][c]}: ${v}`, x + cw / 2 - 2, y + ch / 2 + 2);
            }
        }
        // Header rows
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('CONFUSION MATRIX', box.x, box.y - 5);
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('pred 0', box.x + 50 + cw / 2 - 2, box.y + 32);
        ctx.fillText('pred 1', box.x + 50 + cw + cw / 2 - 2, box.y + 32);
        ctx.save();
        ctx.translate(box.x + 8, box.y + 40 + ch);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('true', 0, 0);
        ctx.restore();
        ctx.textAlign = 'right';
        ctx.fillText('0', box.x + 46, box.y + 40 + ch / 2 + 3);
        ctx.fillText('1', box.x + 46, box.y + 40 + ch + ch / 2 + 3);

        // Metric strip below the matrix
        const my = box.y + box.h - 12;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(
            `acc=${m.acc.toFixed(2)}  prec=${m.prec.toFixed(2)}  rec=${m.rec.toFixed(2)}  F1=${m.f1.toFixed(2)}`,
            box.x, my,
        );
    }

    function drawROC(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        const pad = 26;
        const inner = { x: box.x + pad, y: box.y + 16, w: box.w - pad - 8, h: box.h - 32 };
        function toPx(fpr, tpr) {
            return [inner.x + fpr * inner.w, inner.y + (1 - tpr) * inner.h];
        }

        // Diagonal (chance)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        const [d0x, d0y] = toPx(0, 0), [d1x, d1y] = toPx(1, 1);
        ctx.moveTo(d0x, d0y); ctx.lineTo(d1x, d1y); ctx.stroke();
        ctx.setLineDash([]);

        // ROC curve
        if (!aucCache) aucCache = rocPoints();
        const { pts, auc } = aucCache;
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
            const [px, py] = toPx(pts[i].fpr, pts[i].tpr);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Operating point — find the ROC point at this threshold
        let best = pts[0];
        for (const p of pts) if (Math.abs(p.p - threshold) < Math.abs(best.p - threshold)) best = p;
        const [opx, opy] = toPx(best.fpr, best.tpr);
        ctx.fillStyle = '#ea7959';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(opx, opy, 6, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();

        // Axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('0', inner.x, inner.y + inner.h + 12);
        ctx.fillText('1', inner.x + inner.w, inner.y + inner.h + 12);
        ctx.textAlign = 'right';
        ctx.fillText('0', inner.x - 4, inner.y + inner.h + 3);
        ctx.fillText('1', inner.x - 4, inner.y + 3);
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'center';
        ctx.fillText('FPR', inner.x + inner.w / 2, inner.y + inner.h + 22);
        ctx.save();
        ctx.translate(box.x + 8, inner.y + inner.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('TPR', 0, 0);
        ctx.restore();

        // Title + AUC
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('ROC', box.x, box.y - 5);
        ctx.fillStyle = '#4f46e5';
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`AUC = ${auc.toFixed(3)}`, box.x + box.w, box.y - 5);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const lay = layout();
        const m = confusion(threshold);
        drawDistributions(lay.left);
        drawConfusion(lay.cm, m);
        drawROC(lay.roc);

        updateCaption(m);
    }

    function updateCaption(m) {
        if (!captionEl) return;
        captionEl.innerHTML =
            `<strong>τ = ${threshold.toFixed(2)}.</strong> ` +
            `Accuracy ${m.acc.toFixed(2)}, precision ${m.prec.toFixed(2)}, recall ${m.rec.toFixed(2)}, F1 ${m.f1.toFixed(2)}. ` +
            (imbalance < 0.2
                ? `Imbalance is heavy (${(imbalance * 100).toFixed(0)}% positive). Notice how the optimal τ shifts ` +
                  `away from 0.5 — and accuracy stays high even when recall is low. AUC tells a less rosy story.`
                : `Drag τ to see precision and recall trade off in real time, and the operating point slide along the ROC.`);
    }

    // ----- Controls -----
    if (tSlider) {
        tSlider.min = 0.02; tSlider.max = 0.98; tSlider.step = 0.01; tSlider.value = 0.5;
        tSlider.addEventListener('input', () => {
            threshold = parseFloat(tSlider.value);
            if (tLbl) tLbl.textContent = `τ = ${threshold.toFixed(2)}`;
            draw();
        });
    }
    if (imbSel) {
        imbSel.innerHTML = `
            <option value="0.5">50% balanced</option>
            <option value="0.3">30% positive</option>
            <option value="0.1">10% positive</option>
            <option value="0.05">5% positive (rare)</option>
        `;
        imbSel.addEventListener('change', () => {
            imbalance = parseFloat(imbSel.value);
            reset();
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
