/* Interactive gradient-boosting viz.
 * 1D regression. Each Step fits a small depth-2 regression tree to the
 * current residuals, then adds it (scaled by the learning rate) to the
 * ensemble. The top panel shows the data + the ensemble's current fit.
 * The bottom panel shows the residuals — they should shrink toward zero
 * as more trees are added. */

(function () {
    const canvas    = document.getElementById('viz-boost-canvas');
    const stepBtn   = document.getElementById('viz-boost-step');
    const autoBtn   = document.getElementById('viz-boost-auto');
    const resetBtn  = document.getElementById('viz-boost-reset');
    const dataSel   = document.getElementById('viz-boost-data');
    const lrSlider  = document.getElementById('viz-boost-lr');
    const lrLbl     = document.getElementById('viz-boost-lr-lbl');
    const counterEl = document.getElementById('viz-boost-counter');
    const captionEl = document.getElementById('viz-boost-caption');
    if (!canvas) return;

    // ----- State -----
    let dataset = 'sine';
    let points = [];       // { x, y }
    let ensemble = [];     // list of depth-2 stumps
    let preds = [];        // current f(x_i) per training point
    let lr = 0.4;
    let autoPlaying = false;
    let lastStep = 0;
    const STEP_MS = 600;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function targetFn(x) {
        if (dataset === 'sine')   return 0.6 * Math.sin(2.2 * x);
        if (dataset === 'step')   return x < -0.2 ? -0.4 : (x < 0.4 ? 0.5 : -0.1);
        if (dataset === 'cubic')  return 0.45 * x ** 3 - 0.35 * x;
        if (dataset === 'spike')  return Math.exp(-12 * (x - 0.15) ** 2) - 0.35;
        return 0;
    }
    function buildDataset() {
        const N = 60;
        const pts = [];
        for (let i = 0; i < N; i++) {
            const x = -0.95 + (i / (N - 1)) * 1.9;
            const y = targetFn(x) + randn() * 0.06;
            pts.push({ x, y });
        }
        return pts;
    }

    // ----- Depth-2 regression tree (4 leaves max) -----
    function buildStump(pts, depth) {
        const mean = (arr) => arr.reduce((s, p) => s + p.y, 0) / arr.length;
        if (pts.length < 4 || depth === 0) {
            return { leaf: true, value: mean(pts) };
        }
        const sorted = [...pts].sort((a, b) => a.x - b.x);
        let bestErr = Infinity, bestSplit = null;
        for (let i = 1; i < sorted.length; i++) {
            if (sorted[i].x === sorted[i - 1].x) continue;
            const t = (sorted[i].x + sorted[i - 1].x) / 2;
            const left = sorted.slice(0, i);
            const right = sorted.slice(i);
            const lM = mean(left), rM = mean(right);
            let err = 0;
            for (const p of left)  err += (p.y - lM) ** 2;
            for (const p of right) err += (p.y - rM) ** 2;
            if (err < bestErr) {
                bestErr = err;
                bestSplit = { t, left, right };
            }
        }
        if (!bestSplit) return { leaf: true, value: mean(pts) };
        return {
            leaf: false,
            threshold: bestSplit.t,
            left:  buildStump(bestSplit.left,  depth - 1),
            right: buildStump(bestSplit.right, depth - 1),
        };
    }
    function predictStump(node, x) {
        if (node.leaf) return node.value;
        return x <= node.threshold ? predictStump(node.left, x) : predictStump(node.right, x);
    }
    function ensemblePredict(x) {
        let y = 0;
        for (const t of ensemble) y += lr * predictStump(t, x);
        return y;
    }

    function step() {
        if (ensemble.length >= 60) return;
        const residuals = points.map((p, i) => ({ x: p.x, y: p.y - preds[i] }));
        const tree = buildStump(residuals, 2);  // 4-leaf tree
        ensemble.push(tree);
        for (let i = 0; i < points.length; i++) {
            preds[i] += lr * predictStump(tree, points[i].x);
        }
        draw();
    }

    function reset() {
        points = buildDataset();
        ensemble = [];
        // Initial prediction f_0(x) = mean(y)
        const yMean = points.reduce((s, p) => s + p.y, 0) / points.length;
        preds = points.map(() => yMean);
        // Add a "leaf 0" pseudo-tree to keep things consistent
        ensemble = [];  // we'll show round count = ensemble.length
        // Recover the initial constant by initializing preds correctly; the
        // viz line uses ensemblePredict(x) + yMean.
        autoPlaying = false;
        updateAutoBtn();
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(330, cssW * 0.58)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function layout() {
        const pad = 14;
        const splitY = pad + Math.round((H - 2 * pad) * 0.66);
        const main = { x: pad + 28, y: pad + 14, w: W - 2 * pad - 28, h: splitY - pad - 14 };
        const resid = { x: pad + 28, y: splitY + 12, w: W - 2 * pad - 28, h: H - splitY - pad - 6 };
        return { main, resid };
    }

    function fitConstant() {
        // f_0 = mean(y)
        return points.length ? points.reduce((s, p) => s + p.y, 0) / points.length : 0;
    }

    function drawMain(box) {
        // Frame + Y=0 line
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Y range
        const yMin = -1, yMax = 1;
        function dataToPx(x, y) {
            return [
                box.x + ((x + 1) / 2) * box.w,
                box.y + ((yMax - y) / (yMax - yMin)) * box.h,
            ];
        }

        // Axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('DATA & PREDICTION', box.x, box.y - 5);

        // Target curve (faint dashed) — what the ensemble is trying to learn
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.25)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const steps = 200;
        for (let i = 0; i <= steps; i++) {
            const x = -1 + 2 * i / steps;
            const [px, py] = dataToPx(x, targetFn(x));
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Ensemble fit
        const f0 = fitConstant();
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
            const x = -1 + 2 * i / steps;
            const y = f0 + ensemblePredict(x);
            const [px, py] = dataToPx(x, y);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Points
        ctx.fillStyle = '#4f46e5';
        for (const p of points) {
            const [px, py] = dataToPx(p.x, p.y);
            ctx.beginPath();
            ctx.arc(px, py, 2.4, 0, Math.PI * 2);
            ctx.fill();
        }

        // Legend
        ctx.fillStyle = 'rgba(79, 70, 229, 0.7)';
        ctx.beginPath(); ctx.arc(box.x + box.w - 130, box.y - 9, 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText('data', box.x + box.w - 122, box.y - 5);
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.4)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(box.x + box.w - 92, box.y - 9); ctx.lineTo(box.x + box.w - 76, box.y - 9); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText('target', box.x + box.w - 72, box.y - 5);
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(box.x + box.w - 36, box.y - 9); ctx.lineTo(box.x + box.w - 20, box.y - 9); ctx.stroke();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('fit', box.x + box.w - 16, box.y - 5);
    }

    function drawResid(box) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('RESIDUALS — what the next tree will try to fit', box.x, box.y - 5);

        // Zero line
        const midY = box.y + box.h / 2;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.beginPath(); ctx.moveTo(box.x, midY); ctx.lineTo(box.x + box.w, midY); ctx.stroke();

        // Compute residuals = y - (f0 + ensemblePredict(x))
        const f0 = fitConstant();
        const rs = points.map((p, i) => p.y - preds[i]);
        // Y scale: cap at ±maxAbs (or 0.8 minimum)
        let maxAbs = 0.05;
        for (const r of rs) if (Math.abs(r) > maxAbs) maxAbs = Math.abs(r);

        for (let i = 0; i < points.length; i++) {
            const x = points[i].x;
            const r = rs[i];
            const px = box.x + ((x + 1) / 2) * box.w;
            const py = midY - (r / maxAbs) * (box.h / 2 - 6);
            ctx.strokeStyle = r > 0 ? 'rgba(234, 121, 89, 0.7)' : 'rgba(79, 70, 229, 0.7)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px, midY);
            ctx.lineTo(px, py);
            ctx.stroke();
            ctx.fillStyle = r > 0 ? '#ea7959' : '#4f46e5';
            ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2); ctx.fill();
        }

        // MSE label
        let mse = 0;
        for (const r of rs) mse += r * r;
        mse /= rs.length || 1;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`MSE = ${mse.toFixed(4)}`, box.x + box.w - 4, box.y + 13);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        const lay = layout();
        drawMain(lay.main);
        drawResid(lay.resid);
        if (counterEl) counterEl.textContent = `Tree ${ensemble.length}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const n = ensemble.length;
        if (n === 0) {
            captionEl.innerHTML =
                `<strong>Round 0.</strong> The fit is a constant: <em>f₀(x) = mean(y)</em>. ` +
                `Every residual is just the deviation of each point from that mean. ` +
                `Click <strong>Step</strong> — a tiny depth-2 tree will fit those residuals.`;
        } else if (n < 5) {
            captionEl.innerHTML =
                `<strong>${n} tree${n === 1 ? '' : 's'} added.</strong> Each step fits a depth-2 tree ` +
                `to the current residuals and adds it (scaled by η = ${lr.toFixed(2)}) to the ensemble. ` +
                `The fit is still coarse — keep stepping.`;
        } else {
            captionEl.innerHTML =
                `<strong>${n} trees.</strong> The orange fit is converging to the dashed target. ` +
                `Each new tree is a small correction; the learning rate ${lr.toFixed(2)} controls ` +
                `the step size. Drop η for slower but more stable improvement.`;
        }
    }

    // ----- Loop -----
    function updateAutoBtn() {
        if (autoBtn) autoBtn.textContent = autoPlaying ? 'Pause' : 'Auto';
    }
    function loop(now) {
        if (autoPlaying && now - lastStep >= STEP_MS) {
            step();
            lastStep = now;
            if (ensemble.length >= 60) { autoPlaying = false; updateAutoBtn(); }
        }
        requestAnimationFrame(loop);
    }

    // ----- Controls -----
    stepBtn?.addEventListener('click', step);
    autoBtn?.addEventListener('click', () => {
        autoPlaying = !autoPlaying;
        updateAutoBtn();
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', reset);
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="sine">Sine curve</option>
            <option value="step">Step function</option>
            <option value="cubic">Cubic</option>
            <option value="spike">Spike</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }
    if (lrSlider) {
        lrSlider.min = 0.05; lrSlider.max = 1; lrSlider.step = 0.05; lrSlider.value = 0.4;
        lrSlider.addEventListener('input', () => {
            lr = parseFloat(lrSlider.value);
            if (lrLbl) lrLbl.textContent = `η = ${lr.toFixed(2)}`;
            // Recompute current ensemble predictions under new η
            preds = points.map(p => {
                let y = 0;
                for (const t of ensemble) y += lr * predictStump(t, p.x);
                return fitConstant() + y;
            });
            // Actually we want preds to track ensemble's contribution + f0 already.
            // Recompute properly:
            preds = points.map((p, i) => {
                const f0 = fitConstant();
                let y = f0;
                for (const t of ensemble) y += lr * predictStump(t, p.x);
                return y;
            });
            draw();
        });
    }

    // ----- Init -----
    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    requestAnimationFrame(loop);
})();
