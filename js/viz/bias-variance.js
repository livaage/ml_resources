/* Interactive bias-variance trade-off viz.
 * We resample a noisy training set from a fixed target M times and fit a
 * polynomial of the chosen degree to each. The faded indigo curves are those
 * individual fits; their pointwise average is the bold mean fit. The squared
 * gap between mean fit and target is the BIAS²; the spread of the fits is
 * the VARIANCE. They trade off as the degree increases. */

(function () {
    const canvas    = document.getElementById('viz-bv-canvas');
    const dSlider   = document.getElementById('viz-bv-degree');
    const dLbl      = document.getElementById('viz-bv-degree-lbl');
    const nSlider   = document.getElementById('viz-bv-noise');
    const nLbl      = document.getElementById('viz-bv-noise-lbl');
    const targetSel = document.getElementById('viz-bv-target');
    const resetBtn  = document.getElementById('viz-bv-reset');
    const captionEl = document.getElementById('viz-bv-caption');
    if (!canvas) return;

    let degree = 4;
    let noiseSd = 0.18;
    let target = 'sine';
    const N_TRAIN = 16;
    const N_DATASETS = 30;
    const GRID = 200;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function targetFn(x) {
        if (target === 'sine')  return Math.sin(3 * x);
        if (target === 'cubic') return 1.6 * x ** 3 - 0.6 * x;
        if (target === 'step')  return x < -0.1 ? -0.5 : (x < 0.4 ? 0.4 : -0.2);
        return 0;
    }

    // ----- Linear solver (same as regression viz) -----
    function solve(A, b) {
        const n = b.length;
        const M = A.map((row, i) => [...row, b[i]]);
        for (let i = 0; i < n; i++) {
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
    function fitPoly(xs, ys, deg) {
        const k = deg + 1;
        const A = Array.from({length: k}, () => Array(k).fill(0));
        const b = Array(k).fill(0);
        for (let i = 0; i < xs.length; i++) {
            const pw = Array(2 * k - 1);
            pw[0] = 1;
            for (let j = 1; j < 2 * k - 1; j++) pw[j] = pw[j - 1] * xs[i];
            for (let a = 0; a < k; a++) {
                for (let bj = 0; bj < k; bj++) A[a][bj] += pw[a + bj];
                b[a] += pw[a] * ys[i];
            }
        }
        for (let i = 0; i < k; i++) A[i][i] += 1e-6;  // ridge
        return solve(A, b) || Array(k).fill(0);
    }
    function evalPoly(beta, x) {
        let y = 0, p = 1;
        for (const c of beta) { y += c * p; p *= x; }
        return y;
    }

    // ----- Sample many datasets and fit each -----
    let fits = [];          // [M][G+1] grid of predictions
    let meanFit = null;     // [G+1]
    let bias2 = 0, variance = 0;
    function recompute() {
        const xsGrid = new Float64Array(GRID + 1);
        for (let i = 0; i <= GRID; i++) xsGrid[i] = -1 + 2 * i / GRID;

        fits = [];
        const targets = xsGrid.map(targetFn);

        for (let d = 0; d < N_DATASETS; d++) {
            const xs = [], ys = [];
            for (let i = 0; i < N_TRAIN; i++) {
                const x = -0.95 + (i + 0.5) / N_TRAIN * 1.9 + randn() * 0.04;
                xs.push(x);
                ys.push(targetFn(x) + randn() * noiseSd);
            }
            const beta = fitPoly(xs, ys, degree);
            const preds = new Float64Array(GRID + 1);
            for (let i = 0; i <= GRID; i++) preds[i] = evalPoly(beta, xsGrid[i]);
            fits.push(preds);
        }

        // Mean fit
        meanFit = new Float64Array(GRID + 1);
        for (let i = 0; i <= GRID; i++) {
            let s = 0;
            for (const f of fits) s += f[i];
            meanFit[i] = s / N_DATASETS;
        }

        // Bias² and variance averaged over the grid
        bias2 = 0; variance = 0;
        for (let i = 0; i <= GRID; i++) {
            const tgt = targets[i];
            bias2 += (meanFit[i] - tgt) ** 2;
            let v = 0;
            for (const f of fits) v += (f[i] - meanFit[i]) ** 2;
            variance += v / N_DATASETS;
        }
        bias2 /= (GRID + 1);
        variance /= (GRID + 1);

        return xsGrid;
    }

    let xsGrid = recompute();

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(340, cssW * 0.52)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function layout() {
        const pad = 14;
        const totalW = W - 2 * pad;
        const fitsW = Math.round(totalW * 0.62);
        const fitsBox = { x: pad, y: pad + 14, w: fitsW, h: H - 2 * pad - 14 };
        const errBox = { x: pad + fitsW + 18, y: pad + 14, w: totalW - fitsW - 18, h: H - 2 * pad - 14 };
        return { fitsBox, errBox };
    }
    function dataToPx(x, y, box, ymin, ymax) {
        return [
            box.x + ((x + 1) / 2) * box.w,
            box.y + ((ymax - y) / (ymax - ymin)) * box.h,
        ];
    }

    function drawFits(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        const ymin = -1.5, ymax = 1.5;

        // Target (faint dashed)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i <= GRID; i++) {
            const [px, py] = dataToPx(xsGrid[i], targetFn(xsGrid[i]), box, ymin, ymax);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Individual fits
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.15)';
        ctx.lineWidth = 1;
        for (const f of fits) {
            ctx.beginPath();
            for (let i = 0; i <= GRID; i++) {
                const y = Math.max(ymin, Math.min(ymax, f[i]));
                const [px, py] = dataToPx(xsGrid[i], y, box, ymin, ymax);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Mean fit (bold)
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        for (let i = 0; i <= GRID; i++) {
            const y = Math.max(ymin, Math.min(ymax, meanFit[i]));
            const [px, py] = dataToPx(xsGrid[i], y, box, ymin, ymax);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Legend / title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${N_DATASETS} fits at degree ${degree} — overlaid`, box.x, box.y - 5);
        ctx.fillStyle = '#ea7959';
        ctx.fillText('— mean fit', box.x + box.w - 80, box.y - 5);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText('-- target', box.x + box.w - 150, box.y - 5);
    }

    function drawErrors(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Sweep degree 1..12 and re-fit (lighter — only 8 datasets)
        const maxDeg = 12;
        const bias2Arr = new Float64Array(maxDeg + 1);
        const varArr   = new Float64Array(maxDeg + 1);
        const Mfast = 8;
        for (let deg = 1; deg <= maxDeg; deg++) {
            const fitsLocal = [];
            for (let d = 0; d < Mfast; d++) {
                const xs = [], ys = [];
                for (let i = 0; i < N_TRAIN; i++) {
                    const x = -0.95 + (i + 0.5) / N_TRAIN * 1.9;
                    xs.push(x);
                    ys.push(targetFn(x) + randn() * noiseSd);
                }
                const beta = fitPoly(xs, ys, deg);
                const preds = new Float64Array(GRID + 1);
                for (let i = 0; i <= GRID; i++) preds[i] = evalPoly(beta, xsGrid[i]);
                fitsLocal.push(preds);
            }
            const meanLocal = new Float64Array(GRID + 1);
            for (let i = 0; i <= GRID; i++) {
                let s = 0;
                for (const f of fitsLocal) s += f[i];
                meanLocal[i] = s / Mfast;
            }
            let b = 0, v = 0;
            for (let i = 0; i <= GRID; i++) {
                b += (meanLocal[i] - targetFn(xsGrid[i])) ** 2;
                let s = 0;
                for (const f of fitsLocal) s += (f[i] - meanLocal[i]) ** 2;
                v += s / Mfast;
            }
            bias2Arr[deg] = b / (GRID + 1);
            varArr[deg]   = v / (GRID + 1);
        }
        // Y axis: log scale to handle high-variance values
        const errMax = Math.max(...bias2Arr.slice(1), ...varArr.slice(1)) * 1.05 + 0.001;
        function toY(v) { return box.y + box.h - 12 - (v / errMax) * (box.h - 30); }

        // Bias² curve
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let d = 1; d <= maxDeg; d++) {
            const x = box.x + 8 + ((d - 1) / (maxDeg - 1)) * (box.w - 16);
            const y = toY(bias2Arr[d]);
            if (d === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Variance curve
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let d = 1; d <= maxDeg; d++) {
            const x = box.x + 8 + ((d - 1) / (maxDeg - 1)) * (box.w - 16);
            const y = toY(varArr[d]);
            if (d === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Total = bias² + var
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let d = 1; d <= maxDeg; d++) {
            const x = box.x + 8 + ((d - 1) / (maxDeg - 1)) * (box.w - 16);
            const y = toY(bias2Arr[d] + varArr[d]);
            if (d === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Current-degree marker
        const cx = box.x + 8 + ((degree - 1) / (maxDeg - 1)) * (box.w - 16);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, box.y + 16); ctx.lineTo(cx, box.y + box.h - 4); ctx.stroke();

        // Title / legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('ERROR vs DEGREE', box.x, box.y - 5);
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = '#4f46e5'; ctx.fillText('— bias²', box.x + 8, box.y + 12);
        ctx.fillStyle = '#ea7959'; ctx.fillText('— variance', box.x + 8, box.y + 26);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.fillText('-- total', box.x + 8, box.y + 40);

        // X axis ticks
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const d of [1, 4, 8, 12]) {
            const x = box.x + 8 + ((d - 1) / (maxDeg - 1)) * (box.w - 16);
            ctx.fillText(`${d}`, x, box.y + box.h - 2);
        }
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        const lay = layout();
        drawFits(lay.fitsBox);
        drawErrors(lay.errBox);
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        captionEl.innerHTML =
            `<strong>Degree ${degree}:</strong> bias² = ${bias2.toFixed(3)}, variance = ${variance.toFixed(3)}. ` +
            (degree <= 2
                ? `Low variance (every fit looks similar) but the mean fit can't follow the target — high bias.`
                : degree >= 9
                    ? `Each fit chases the noise of its own training set — high variance — even though the mean fit ` +
                      `is right on target (low bias).`
                    : `Around degree 4–6 is the sweet spot: total error is minimised because bias² and variance ` +
                      `are both moderate. Slide left and right to feel the U-curve.`);
    }

    function reseed() {
        xsGrid = recompute();
        draw();
    }

    // ----- Controls -----
    if (dSlider) {
        dSlider.min = 1; dSlider.max = 12; dSlider.step = 1; dSlider.value = 4;
        dSlider.addEventListener('input', () => {
            degree = parseInt(dSlider.value, 10);
            if (dLbl) dLbl.textContent = `deg = ${degree}`;
            reseed();
        });
    }
    if (nSlider) {
        nSlider.min = 0; nSlider.max = 0.4; nSlider.step = 0.02; nSlider.value = 0.18;
        nSlider.addEventListener('input', () => {
            noiseSd = parseFloat(nSlider.value);
            if (nLbl) nLbl.textContent = `σ = ${noiseSd.toFixed(2)}`;
            reseed();
        });
    }
    if (targetSel) {
        targetSel.innerHTML = `
            <option value="sine">Sine</option>
            <option value="cubic">Cubic</option>
            <option value="step">Step</option>
        `;
        targetSel.addEventListener('change', () => {
            target = targetSel.value;
            reseed();
        });
    }
    resetBtn?.addEventListener('click', reseed);

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
