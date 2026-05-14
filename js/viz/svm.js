/* Interactive SVM viz.
 * Linear mode: gradient descent on the hinge loss to find the max-margin
 * hyperplane. Margin lines (dashed) sit at f = ±1; support vectors
 * (highlighted) are the points closest to the boundary — only these
 * influence the fit.
 *
 * RBF mode: a kernel-density classifier. The decision function
 * f(x) = (1/N) Σ y_i exp(-γ ‖x - x_i‖²) gives a smooth surface; we
 * draw the f=0 contour as the boundary. (A real RBF SVM weights each
 * training point by α_i — only the support vectors survive — but the
 * shape of the boundary is identical to what we draw here.) */

(function () {
    const canvas    = document.getElementById('viz-svm-canvas');
    const kernelSel = document.getElementById('viz-svm-kernel');
    const dataSel   = document.getElementById('viz-svm-data');
    const cSlider   = document.getElementById('viz-svm-c');
    const gSlider   = document.getElementById('viz-svm-gamma');
    const cLbl      = document.getElementById('viz-svm-c-lbl');
    const gLbl      = document.getElementById('viz-svm-gamma-lbl');
    const resetBtn  = document.getElementById('viz-svm-reset');
    const captionEl = document.getElementById('viz-svm-caption');
    if (!canvas) return;

    let kernel = 'linear';        // 'linear' | 'rbf'
    let dataset = 'linear';
    let points = [];              // each: { x: [x, y], y: -1 | +1 }
    let model = { w: [0, 0], b: 0 };
    let C = 1.0;
    let gamma = 3.0;
    let heatCache = null;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'linear') {
            for (let i = 0; i < 30; i++) {
                pts.push({ x: [-0.45 + randn() * 0.16,  0.25 + randn() * 0.16], y: +1 });
                pts.push({ x: [ 0.45 + randn() * 0.16, -0.25 + randn() * 0.16], y: -1 });
            }
        } else if (dataset === 'overlap') {
            for (let i = 0; i < 35; i++) {
                pts.push({ x: [-0.25 + randn() * 0.25,  0.15 + randn() * 0.25], y: +1 });
                pts.push({ x: [ 0.25 + randn() * 0.25, -0.15 + randn() * 0.25], y: -1 });
            }
        } else if (dataset === 'circles') {
            for (let i = 0; i < 35; i++) {
                const a1 = Math.random() * Math.PI * 2;
                const r1 = 0.25 + randn() * 0.05;
                pts.push({ x: [r1 * Math.cos(a1), r1 * Math.sin(a1)], y: +1 });
                const a2 = Math.random() * Math.PI * 2;
                const r2 = 0.7 + randn() * 0.05;
                pts.push({ x: [r2 * Math.cos(a2), r2 * Math.sin(a2)], y: -1 });
            }
        } else if (dataset === 'xor') {
            for (let i = 0; i < 60; i++) {
                const x = -0.75 + Math.random() * 1.5;
                const y = -0.75 + Math.random() * 1.5;
                pts.push({ x: [x, y], y: (x * y > 0) ? +1 : -1 });
            }
        }
        return pts;
    }

    // ----- Train -----
    function trainLinear(pts, C, iters = 350) {
        let w = [0.01, 0.01], b = 0;
        const N = pts.length;
        for (let it = 0; it < iters; it++) {
            const lr = 0.04 / (1 + it / 80);
            let dwX = 0, dwY = 0, dB = 0;
            for (const p of pts) {
                const f = w[0] * p.x[0] + w[1] * p.x[1] + b;
                if (p.y * f < 1) {
                    dwX += p.y * p.x[0];
                    dwY += p.y * p.x[1];
                    dB  += p.y;
                }
            }
            w[0] += lr * (C * dwX / N - w[0]);
            w[1] += lr * (C * dwY / N - w[1]);
            b    += lr * (C * dB  / N);
        }
        return { w, b };
    }

    function rbfPredict(x, pts, g) {
        let s = 0;
        for (const p of pts) {
            const d2 = (x[0] - p.x[0]) ** 2 + (x[1] - p.x[1]) ** 2;
            s += p.y * Math.exp(-g * d2);
        }
        return s / pts.length;
    }

    function predict(x) {
        if (kernel === 'linear') {
            return model.w[0] * x[0] + model.w[1] * x[1] + model.b;
        }
        return rbfPredict(x, points, gamma);
    }

    function retrain() {
        if (kernel === 'linear') model = trainLinear(points, C);
        heatCache = null;
    }
    function reset() {
        points = buildDataset();
        retrain();
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(330, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        heatCache = null;
        draw();
    }

    function dataToPx(x, y) {
        const m = Math.min(W, H) / 2 - 18;
        return [W / 2 + x * m, H / 2 - y * m];
    }
    function pxToData(px, py) {
        const m = Math.min(W, H) / 2 - 18;
        return [(px - W / 2) / m, -(py - H / 2) / m];
    }

    // ----- Drawing -----
    function decisionColour(f) {
        const a = Math.tanh(f * 1.2);  // squash to [-1, 1] for colour
        if (a > 0) {
            const t = a * 0.55;
            return `rgb(${251 + (234 - 251) * t | 0}, ${250 + (121 - 250) * t | 0}, ${247 + (89 - 247) * t | 0})`;
        }
        const t = -a * 0.55;
        return `rgb(${251 + (79  - 251) * t | 0}, ${250 + (70  - 250) * t | 0}, ${247 + (229 - 247) * t | 0})`;
    }

    function buildHeatmap() {
        if (heatCache) return heatCache;
        const G = 60;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.1;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.1;
                const f = predict([dx, dy]);
                const col = decisionColour(f);
                const m = col.match(/rgb\((\d+), (\d+), (\d+)\)/);
                const idx = (j * G + i) * 4;
                img.data[idx] = +m[1]; img.data[idx + 1] = +m[2];
                img.data[idx + 2] = +m[3]; img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatCache = off;
        return off;
    }

    function drawHeatmap() {
        const hm = buildHeatmap();
        // Draw scaled to the plot region
        const m = Math.min(W, H) / 2 - 18;
        const x0 = W / 2 - m * 1.1, y0 = H / 2 - m * 1.1;
        const w  = m * 2 * 1.1,     h  = m * 2 * 1.1;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(hm, x0, y0, w, h);
    }

    function drawLinearBoundary() {
        // Draw f(x) = -1, 0, +1
        if (model.w[0] === 0 && model.w[1] === 0) return;
        for (const level of [-1, 0, +1]) {
            // Line: w0*x + w1*y + b = level → solve for y given x
            // y = (level - b - w0*x) / w1
            const isBoundary = (level === 0);
            ctx.strokeStyle = isBoundary ? 'rgba(0, 0, 0, 0.55)' : 'rgba(0, 0, 0, 0.22)';
            ctx.lineWidth = isBoundary ? 2 : 1;
            if (!isBoundary) ctx.setLineDash([4, 4]);
            ctx.beginPath();
            const xs = [-1.1, 1.1];
            for (let i = 0; i < 2; i++) {
                const x = xs[i];
                let y;
                if (Math.abs(model.w[1]) > 1e-6) {
                    y = (level - model.b - model.w[0] * x) / model.w[1];
                } else {
                    y = (level - model.b) / 1e-6;
                }
                const [px, py] = dataToPx(x, y);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function drawRBFBoundary() {
        // Marching-squares-light: sample at grid corners, draw line segments
        // where the f=0 isoline crosses. We just scan a fine grid horizontally
        // and emit short segments wherever the sign of f flips.
        const G = 72;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.lineWidth = 2;
        // Build sign grid first to avoid re-evaluating f in two scan passes
        const fGrid = new Float32Array(G * G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.1;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.1;
                fGrid[j * G + i] = predict([dx, dy]);
            }
        }
        ctx.beginPath();
        for (let j = 0; j < G - 1; j++) {
            for (let i = 0; i < G - 1; i++) {
                const a = fGrid[j * G + i];
                const b = fGrid[j * G + i + 1];
                const c = fGrid[(j + 1) * G + i];
                if ((a > 0) !== (b > 0)) {
                    const t = a / (a - b);
                    const x0 =  ((i + t + 0.5) / G * 2 - 1) * 1.1;
                    const y0 = -((j + 0.5) / G * 2 - 1) * 1.1;
                    const [px, py] = dataToPx(x0, y0);
                    ctx.moveTo(px - 1, py); ctx.lineTo(px + 1, py);
                }
                if ((a > 0) !== (c > 0)) {
                    const t = a / (a - c);
                    const x0 =  ((i + 0.5) / G * 2 - 1) * 1.1;
                    const y0 = -((j + t + 0.5) / G * 2 - 1) * 1.1;
                    const [px, py] = dataToPx(x0, y0);
                    ctx.moveTo(px, py - 1); ctx.lineTo(px, py + 1);
                }
            }
        }
        ctx.stroke();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        drawHeatmap();

        // Frame
        const m = Math.min(W, H) / 2 - 18;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(W / 2 - m * 1.1, H / 2 - m * 1.1, m * 2.2, m * 2.2);

        if (kernel === 'linear') drawLinearBoundary();
        else                     drawRBFBoundary();

        // Identify and highlight support vectors.
        // Linear: |f| ≤ 1 (margin region).  RBF: closest 25% to the f=0 surface.
        const supports = [];
        if (kernel === 'linear') {
            for (const p of points) if (Math.abs(predict(p.x)) <= 1.05) supports.push(p);
        } else {
            const scored = points.map(p => ({ p, m: Math.abs(predict(p.x)) }));
            scored.sort((a, b) => a.m - b.m);
            const n = Math.max(4, Math.round(points.length * 0.25));
            for (let i = 0; i < n; i++) supports.push(scored[i].p);
        }

        // Points
        for (const p of points) {
            const [px, py] = dataToPx(p.x[0], p.x[1]);
            const colour = p.y > 0 ? '#ea7959' : '#4f46e5';
            const isSV = supports.includes(p);
            if (isSV) {
                ctx.fillStyle = 'rgba(0, 0, 0, 0.12)';
                ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill();
            }
            ctx.fillStyle = colour;
            ctx.strokeStyle = isSV ? '#1a1a1a' : 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = isSV ? 1.8 : 1;
            ctx.beginPath();
            ctx.arc(px, py, isSV ? 4.5 : 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Title strip
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const title = kernel === 'linear'
            ? `LINEAR SVM — solid = boundary · dashed = ±1 margin`
            : `RBF KERNEL — γ = ${gamma.toFixed(1)}`;
        ctx.fillText(title, 12, 16);

        updateCaption(supports.length);
    }

    function updateCaption(svCount) {
        if (!captionEl) return;
        if (kernel === 'linear') {
            captionEl.innerHTML =
                `<strong>Linear SVM.</strong> ${svCount} support vectors (outlined points) — ` +
                `the only ones that influence the boundary. The dashed lines are the margins ` +
                `at f = ±1; the boundary lives midway between them.` +
                (dataset === 'circles' || dataset === 'xor'
                    ? ` Switch the kernel to <strong>RBF</strong> — this dataset isn't linearly separable.`
                    : ` Drop C to widen the margin; raise it to push the boundary closer to the points.`);
        } else {
            captionEl.innerHTML =
                `<strong>RBF kernel.</strong> Each training point casts a Gaussian "vote" around itself ` +
                `with width <em>1/√γ</em>. Raise γ and the boundary tightens around individual points ` +
                `(overfit); lower it and the boundary smooths out. ` +
                `${svCount} support vectors sit near the f=0 surface.`;
        }
    }

    // ----- Controls -----
    if (kernelSel) {
        kernelSel.innerHTML = `
            <option value="linear">Linear</option>
            <option value="rbf">RBF kernel</option>
        `;
        kernelSel.addEventListener('change', () => {
            kernel = kernelSel.value;
            retrain();
            draw();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="linear">Linearly separable</option>
            <option value="overlap">Overlapping</option>
            <option value="circles">Concentric</option>
            <option value="xor">XOR</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }
    if (cSlider) {
        cSlider.min = -2; cSlider.max = 2; cSlider.step = 0.1; cSlider.value = 0;
        cSlider.addEventListener('input', () => {
            C = Math.pow(10, parseFloat(cSlider.value));
            if (cLbl) cLbl.textContent = `C = ${C.toFixed(2)}`;
            if (kernel === 'linear') { retrain(); draw(); }
        });
    }
    if (gSlider) {
        gSlider.min = 0.5; gSlider.max = 30; gSlider.step = 0.5; gSlider.value = 3;
        gSlider.addEventListener('input', () => {
            gamma = parseFloat(gSlider.value);
            if (gLbl) gLbl.textContent = `γ = ${gamma.toFixed(1)}`;
            if (kernel === 'rbf') { heatCache = null; draw(); }
        });
    }
    resetBtn?.addEventListener('click', reset);

    // ----- Init -----
    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
