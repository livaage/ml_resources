/* Interactive CNN convolution viz.
 *   - Pick an image; pick a preset filter; watch the filter slide over the
 *     whole image and produce a feature map (the same 9 weights at every
 *     position — that's the weight-sharing thing students need to see).
 *   - Or switch to "Learn the filter" mode: a random kernel is gradient-
 *     descended toward the target output, so students see the weights
 *     are NOT magic — they're learned from the loss signal.
 */

(function () {
    const inputCanvas  = document.getElementById('viz-cnn-input');
    const outputCanvas = document.getElementById('viz-cnn-output');
    const kernelCanvas = document.getElementById('viz-cnn-kernel');
    const targetCanvas = document.getElementById('viz-cnn-target');
    const lossCanvas   = document.getElementById('viz-cnn-loss');
    const readoutEl    = document.getElementById('viz-cnn-readout');
    // Pooling (optional — only present if the page includes the pool section)
    const poolInCanvas  = document.getElementById('viz-cnn-pool-input');
    const poolOutCanvas = document.getElementById('viz-cnn-pool-output');
    if (!inputCanvas || !outputCanvas || !kernelCanvas) return;

    // ---------- Constants ----------
    const SIZE     = 24;                       // input is SIZE × SIZE pixels
    const KSIZE    = 3;                        // kernel is 3 × 3
    const OUT_SIZE = SIZE - KSIZE + 1;         // 22 × 22 feature map
    const CELL_IN  = 9;                        // pixels per input cell on canvas
    const CELL_OUT = 9;                        // pixels per output cell on canvas
    const CELL_K   = 50;                       // pixels per kernel cell
    const LOSS_H   = 80;                       // loss-curve canvas height
    const POOL_SZ  = 2;                        // 2 × 2 pooling window, stride 2
    const POOL_OUT = Math.floor(OUT_SIZE / POOL_SZ);   // 11 × 11 pooled map
    const CELL_PIN = 9;                        // pool input (same as feature map)
    const CELL_POUT = 18;                      // pool output (bigger so it's clear it's downsampled)

    // ---------- Image presets ----------
    // Procedurally drawn so the inputs look like little iconic pictures, not
    // synthetic binary grids. Anti-aliasing from <canvas> gives natural
    // grayscale gradient at edges.
    function makeProcImage(drawFn) {
        const tmp = document.createElement('canvas');
        tmp.width = SIZE;
        tmp.height = SIZE;
        const tx = tmp.getContext('2d');
        tx.imageSmoothingEnabled = true;
        tx.fillStyle = '#fff';
        tx.fillRect(0, 0, SIZE, SIZE);
        tx.fillStyle = '#000';
        drawFn(tx, SIZE);
        const data = tx.getImageData(0, 0, SIZE, SIZE).data;
        const a = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < SIZE * SIZE; i++) {
            a[i] = 1 - data[i * 4] / 255;       // dark pixel → high value
        }
        return a;
    }

    const IMAGES = {
        cat: makeProcImage((ctx, S) => {
            // Face oval
            ctx.beginPath();
            ctx.ellipse(S * 0.50, S * 0.62, S * 0.30, S * 0.26, 0, 0, Math.PI * 2);
            ctx.fill();
            // Ears (triangles)
            ctx.beginPath();
            ctx.moveTo(S * 0.28, S * 0.42);
            ctx.lineTo(S * 0.18, S * 0.20);
            ctx.lineTo(S * 0.40, S * 0.36);
            ctx.closePath();
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(S * 0.72, S * 0.42);
            ctx.lineTo(S * 0.82, S * 0.20);
            ctx.lineTo(S * 0.60, S * 0.36);
            ctx.closePath();
            ctx.fill();
            // Eyes (white dots)
            ctx.fillStyle = '#fff';
            ctx.beginPath(); ctx.arc(S * 0.40, S * 0.58, S * 0.045, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(S * 0.60, S * 0.58, S * 0.045, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#000';
            // Nose (small dark dot)
            ctx.beginPath(); ctx.arc(S * 0.50, S * 0.68, S * 0.025, 0, Math.PI * 2); ctx.fill();
        }),
        house: makeProcImage((ctx, S) => {
            // Walls
            ctx.fillRect(S * 0.22, S * 0.46, S * 0.56, S * 0.42);
            // Roof
            ctx.beginPath();
            ctx.moveTo(S * 0.16, S * 0.48);
            ctx.lineTo(S * 0.50, S * 0.18);
            ctx.lineTo(S * 0.84, S * 0.48);
            ctx.closePath();
            ctx.fill();
            // Door (cut out)
            ctx.fillStyle = '#fff';
            ctx.fillRect(S * 0.42, S * 0.62, S * 0.16, S * 0.26);
            // Window
            ctx.fillRect(S * 0.28, S * 0.54, S * 0.10, S * 0.10);
        }),
        face: makeProcImage((ctx, S) => {
            // Head circle (just the outline)
            ctx.lineWidth = S * 0.06;
            ctx.strokeStyle = '#000';
            ctx.beginPath();
            ctx.arc(S * 0.5, S * 0.5, S * 0.36, 0, Math.PI * 2);
            ctx.stroke();
            // Eyes
            ctx.fillStyle = '#000';
            ctx.beginPath(); ctx.arc(S * 0.38, S * 0.42, S * 0.05, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(S * 0.62, S * 0.42, S * 0.05, 0, Math.PI * 2); ctx.fill();
            // Mouth (smile arc)
            ctx.lineWidth = S * 0.05;
            ctx.beginPath();
            ctx.arc(S * 0.5, S * 0.54, S * 0.14, Math.PI * 0.15, Math.PI * 0.85);
            ctx.stroke();
        }),
        arrow: makeProcImage((ctx, S) => {
            // Shaft
            ctx.fillRect(S * 0.18, S * 0.42, S * 0.55, S * 0.16);
            // Arrow head (triangle)
            ctx.beginPath();
            ctx.moveTo(S * 0.66, S * 0.22);
            ctx.lineTo(S * 0.92, S * 0.50);
            ctx.lineTo(S * 0.66, S * 0.78);
            ctx.closePath();
            ctx.fill();
        }),
        stripes: makeProcImage((ctx, S) => {
            // Vertical bars — clean test case for Sobel-x
            const bw = S * 0.14;
            for (let x = S * 0.10; x < S * 0.95; x += bw * 2) {
                ctx.fillRect(x, S * 0.10, bw, S * 0.80);
            }
        }),
        digit: makeProcImage((ctx, S) => {
            // Hand-shaped 5
            ctx.lineWidth = S * 0.10;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000';
            // Top bar
            ctx.beginPath();
            ctx.moveTo(S * 0.30, S * 0.20);
            ctx.lineTo(S * 0.72, S * 0.20);
            ctx.stroke();
            // Left vertical
            ctx.beginPath();
            ctx.moveTo(S * 0.30, S * 0.20);
            ctx.lineTo(S * 0.30, S * 0.48);
            ctx.stroke();
            // Curve to bottom
            ctx.beginPath();
            ctx.moveTo(S * 0.30, S * 0.48);
            ctx.bezierCurveTo(
                S * 0.78, S * 0.42,
                S * 0.84, S * 0.86,
                S * 0.34, S * 0.84
            );
            ctx.stroke();
        }),
    };

    // ---------- Kernel presets ----------
    const KERNELS = {
        'edge — Sobel x':[[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
        'edge — Sobel y':[[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
        'blur (box)':    [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
        'sharpen':       [[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]],
        'identity':      [[ 0,  0,  0], [ 0,  1,  0], [ 0,  0,  0]],
    };

    // ---------- State ----------
    let mode        = 'preset';          // 'preset' | 'learn'
    let imageName   = 'cat';
    let image       = IMAGES[imageName];
    let presetName  = 'edge — Sobel x';
    let targetName  = 'edge — Sobel x';
    let learnedKernel = randomKernel();
    let trainStepNum = 0;
    let lossHistory  = [];
    let autoTraining = false;

    let activeRow = 0, activeCol = 0;
    let dragging  = false;
    let playing   = true;
    let lastSlide = 0;
    const SLIDE_MS = 90;

    let output       = new Float32Array(OUT_SIZE * OUT_SIZE);
    let targetOutput = new Float32Array(OUT_SIZE * OUT_SIZE);
    let pooled       = new Float32Array(POOL_OUT * POOL_OUT);
    let outMin = 0, outMax = 0;
    let tgtMin = 0, tgtMax = 0;
    let poolMin = 0, poolMax = 0;
    let poolMode = 'max';   // 'max' | 'avg'
    let hoverPoolRow = -1, hoverPoolCol = -1;

    function randomKernel() {
        const k = [];
        for (let r = 0; r < KSIZE; r++) {
            const row = [];
            for (let c = 0; c < KSIZE; c++) row.push((Math.random() - 0.5) * 0.4);
            k.push(row);
        }
        return k;
    }

    function activeKernel() {
        return mode === 'preset' ? KERNELS[presetName] : learnedKernel;
    }

    // ---------- Convolution ----------
    function convolveAll(kernel, dst) {
        let mn = Infinity, mx = -Infinity;
        for (let or = 0; or < OUT_SIZE; or++) {
            for (let oc = 0; oc < OUT_SIZE; oc++) {
                let s = 0;
                for (let kr = 0; kr < KSIZE; kr++)
                    for (let kc = 0; kc < KSIZE; kc++)
                        s += image[(or + kr) * SIZE + (oc + kc)] * kernel[kr][kc];
                dst[or * OUT_SIZE + oc] = s;
                if (s < mn) mn = s;
                if (s > mx) mx = s;
            }
        }
        return [mn, mx];
    }

    function recompute() {
        const [omn, omx] = convolveAll(activeKernel(), output);
        outMin = omn; outMax = omx;
        if (mode === 'learn') {
            const [tmn, tmx] = convolveAll(KERNELS[targetName], targetOutput);
            tgtMin = tmn; tgtMax = tmx;
        }
        // Always re-pool — the feature map is what students see being downsampled
        computePool();
    }

    // ---------- Pooling ----------
    function computePool() {
        let mn = Infinity, mx = -Infinity;
        for (let pr = 0; pr < POOL_OUT; pr++) {
            for (let pc = 0; pc < POOL_OUT; pc++) {
                const r0 = pr * POOL_SZ, c0 = pc * POOL_SZ;
                let v;
                if (poolMode === 'max') {
                    v = -Infinity;
                    for (let dr = 0; dr < POOL_SZ; dr++) {
                        for (let dc = 0; dc < POOL_SZ; dc++) {
                            const x = output[(r0 + dr) * OUT_SIZE + (c0 + dc)];
                            if (x > v) v = x;
                        }
                    }
                } else { // avg
                    v = 0;
                    for (let dr = 0; dr < POOL_SZ; dr++) {
                        for (let dc = 0; dc < POOL_SZ; dc++) {
                            v += output[(r0 + dr) * OUT_SIZE + (c0 + dc)];
                        }
                    }
                    v /= POOL_SZ * POOL_SZ;
                }
                pooled[pr * POOL_OUT + pc] = v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        poolMin = mn; poolMax = mx;
    }

    // One step of gradient descent on the learned kernel (MSE vs target).
    function trainStep() {
        const N = OUT_SIZE * OUT_SIZE;
        const lr = 0.6;
        const grad = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        let loss = 0;
        for (let i = 0; i < OUT_SIZE; i++) {
            for (let j = 0; j < OUT_SIZE; j++) {
                const diff = output[i * OUT_SIZE + j] - targetOutput[i * OUT_SIZE + j];
                loss += 0.5 * diff * diff;
                for (let kr = 0; kr < KSIZE; kr++) {
                    for (let kc = 0; kc < KSIZE; kc++) {
                        grad[kr][kc] += diff * image[(i + kr) * SIZE + (j + kc)];
                    }
                }
            }
        }
        loss /= N;
        for (let kr = 0; kr < KSIZE; kr++) {
            for (let kc = 0; kc < KSIZE; kc++) {
                learnedKernel[kr][kc] -= lr * grad[kr][kc] / N;
            }
        }
        trainStepNum++;
        lossHistory.push(loss);
        if (lossHistory.length > 300) lossHistory.shift();
        recompute();
        return loss;
    }

    function resetLearn() {
        learnedKernel = randomKernel();
        trainStepNum  = 0;
        lossHistory   = [];
        autoTraining  = false;
        if (autoBtn) autoBtn.textContent = 'Auto-train';
        recompute();
        redraw();
    }

    // ---------- Colour ramps ----------
    function inputColour(v) {
        const g = Math.round(255 * (1 - v));
        return `rgb(${g}, ${g}, ${g})`;
    }
    function outputColour(v, min, max) {
        const lim = Math.max(Math.abs(min), Math.abs(max), 1e-6);
        const t = v / lim;                              // [-1, 1]
        if (t >= 0) {
            const r = 255, g = Math.round(255 - 95 * t), b = Math.round(247 - 187 * t);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            const a = -t;
            const r = Math.round(251 - 173 * a);
            const g = Math.round(250 - 188 * a);
            const b = Math.round(247 -  78 * a);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    // ---------- Drawing ----------
    function sizeCanvas(canvas, w, h) {
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== w * dpr) {
            canvas.width  = w * dpr;
            canvas.height = h * dpr;
            canvas.style.width  = w + 'px';
            canvas.style.height = h + 'px';
            const ctx = canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        return canvas.getContext('2d');
    }

    function drawInput() {
        const w = SIZE * CELL_IN, h = SIZE * CELL_IN;
        const c = sizeCanvas(inputCanvas, w, h);
        c.clearRect(0, 0, w, h);
        for (let i = 0; i < SIZE; i++) {
            for (let j = 0; j < SIZE; j++) {
                c.fillStyle = inputColour(image[i * SIZE + j]);
                c.fillRect(j * CELL_IN, i * CELL_IN, CELL_IN, CELL_IN);
            }
        }
        // Very faint grid — keeps the "pixel" feel without distracting from the picture
        c.strokeStyle = 'rgba(0, 0, 0, 0.04)';
        c.lineWidth = 1;
        for (let i = 0; i <= SIZE; i++) {
            c.beginPath(); c.moveTo(0, i * CELL_IN); c.lineTo(w, i * CELL_IN); c.stroke();
            c.beginPath(); c.moveTo(i * CELL_IN, 0); c.lineTo(i * CELL_IN, h); c.stroke();
        }
        // Filter overlay — the SAME 3 × 3 box sliding across the image
        const fx = activeCol * CELL_IN;
        const fy = activeRow * CELL_IN;
        const fs = KSIZE * CELL_IN;
        c.strokeStyle = '#ea7959';
        c.lineWidth = 2.5;
        c.strokeRect(fx + 1, fy + 1, fs - 2, fs - 2);
        c.fillStyle = 'rgba(234, 121, 89, 0.10)';
        c.fillRect(fx + 1, fy + 1, fs - 2, fs - 2);
    }

    function drawKernel() {
        const w = KSIZE * CELL_K, h = KSIZE * CELL_K;
        const c = sizeCanvas(kernelCanvas, w, h);
        c.clearRect(0, 0, w, h);
        const k = activeKernel();
        for (let i = 0; i < KSIZE; i++) {
            for (let j = 0; j < KSIZE; j++) {
                const v = k[i][j];
                const a = Math.min(1, Math.abs(v));
                c.fillStyle = v >= 0
                    ? `rgba(255, 160, 68, ${0.18 + a * 0.5})`
                    : `rgba(79, 70, 229, ${0.18 + a * 0.5})`;
                c.fillRect(j * CELL_K, i * CELL_K, CELL_K, CELL_K);
                c.strokeStyle = 'rgba(0, 0, 0, 0.22)';
                c.lineWidth = 1;
                c.strokeRect(j * CELL_K + 0.5, i * CELL_K + 0.5, CELL_K - 1, CELL_K - 1);
                c.fillStyle = '#1a1a1a';
                c.font = '600 14px "Inter", system-ui, sans-serif';
                c.textAlign = 'center';
                c.textBaseline = 'middle';
                c.fillText(fmt(v), j * CELL_K + CELL_K / 2, i * CELL_K + CELL_K / 2);
            }
        }
    }

    function drawMap(canvas, data, min, max) {
        const w = OUT_SIZE * CELL_OUT, h = OUT_SIZE * CELL_OUT;
        const c = sizeCanvas(canvas, w, h);
        c.clearRect(0, 0, w, h);
        for (let i = 0; i < OUT_SIZE; i++) {
            for (let j = 0; j < OUT_SIZE; j++) {
                c.fillStyle = outputColour(data[i * OUT_SIZE + j], min, max);
                c.fillRect(j * CELL_OUT, i * CELL_OUT, CELL_OUT, CELL_OUT);
            }
        }
        c.strokeStyle = 'rgba(0, 0, 0, 0.04)';
        c.lineWidth = 1;
        for (let i = 0; i <= OUT_SIZE; i++) {
            c.beginPath(); c.moveTo(0, i * CELL_OUT); c.lineTo(w, i * CELL_OUT); c.stroke();
            c.beginPath(); c.moveTo(i * CELL_OUT, 0); c.lineTo(i * CELL_OUT, h); c.stroke();
        }
    }

    function drawOutput() {
        drawMap(outputCanvas, output, outMin, outMax);
        // Highlight active cell to match the sliding filter
        const c = outputCanvas.getContext('2d');
        const ox = activeCol * CELL_OUT;
        const oy = activeRow * CELL_OUT;
        c.strokeStyle = '#ea7959';
        c.lineWidth = 2;
        c.strokeRect(ox + 0.5, oy + 0.5, CELL_OUT - 1, CELL_OUT - 1);
    }

    function drawTarget() {
        if (!targetCanvas) return;
        drawMap(targetCanvas, targetOutput, tgtMin, tgtMax);
    }

    // Pool-input: the feature map again, but with 2 × 2 cells outlined so the
    // reader can see what becomes one output cell. The hovered block (if any)
    // is highlighted in orange to match the hovered pool-output cell.
    function drawPoolInput() {
        if (!poolInCanvas) return;
        const w = OUT_SIZE * CELL_PIN, h = OUT_SIZE * CELL_PIN;
        const c = sizeCanvas(poolInCanvas, w, h);
        c.clearRect(0, 0, w, h);
        for (let i = 0; i < OUT_SIZE; i++) {
            for (let j = 0; j < OUT_SIZE; j++) {
                c.fillStyle = outputColour(output[i * OUT_SIZE + j], outMin, outMax);
                c.fillRect(j * CELL_PIN, i * CELL_PIN, CELL_PIN, CELL_PIN);
            }
        }
        // Cell grid (faint)
        c.strokeStyle = 'rgba(0, 0, 0, 0.04)';
        c.lineWidth = 1;
        for (let i = 0; i <= OUT_SIZE; i++) {
            c.beginPath(); c.moveTo(0, i * CELL_PIN); c.lineTo(w, i * CELL_PIN); c.stroke();
            c.beginPath(); c.moveTo(i * CELL_PIN, 0); c.lineTo(i * CELL_PIN, h); c.stroke();
        }
        // 2×2 block grid (visible)
        c.strokeStyle = 'rgba(79, 70, 229, 0.35)';
        c.lineWidth = 1.5;
        for (let i = 0; i <= POOL_OUT; i++) {
            c.beginPath(); c.moveTo(0, i * POOL_SZ * CELL_PIN); c.lineTo(POOL_OUT * POOL_SZ * CELL_PIN, i * POOL_SZ * CELL_PIN); c.stroke();
            c.beginPath(); c.moveTo(i * POOL_SZ * CELL_PIN, 0); c.lineTo(i * POOL_SZ * CELL_PIN, POOL_OUT * POOL_SZ * CELL_PIN); c.stroke();
        }
        // Highlighted hovered block
        if (hoverPoolRow >= 0 && hoverPoolCol >= 0) {
            const bx = hoverPoolCol * POOL_SZ * CELL_PIN;
            const by = hoverPoolRow * POOL_SZ * CELL_PIN;
            const bs = POOL_SZ * CELL_PIN;
            c.strokeStyle = '#ea7959';
            c.lineWidth = 2.5;
            c.strokeRect(bx + 1, by + 1, bs - 2, bs - 2);
            c.fillStyle = 'rgba(234, 121, 89, 0.12)';
            c.fillRect(bx + 1, by + 1, bs - 2, bs - 2);
        }
    }

    function drawPoolOutput() {
        if (!poolOutCanvas) return;
        const w = POOL_OUT * CELL_POUT, h = POOL_OUT * CELL_POUT;
        const c = sizeCanvas(poolOutCanvas, w, h);
        c.clearRect(0, 0, w, h);
        for (let i = 0; i < POOL_OUT; i++) {
            for (let j = 0; j < POOL_OUT; j++) {
                c.fillStyle = outputColour(pooled[i * POOL_OUT + j], poolMin, poolMax);
                c.fillRect(j * CELL_POUT, i * CELL_POUT, CELL_POUT, CELL_POUT);
            }
        }
        c.strokeStyle = 'rgba(0, 0, 0, 0.10)';
        c.lineWidth = 1;
        for (let i = 0; i <= POOL_OUT; i++) {
            c.beginPath(); c.moveTo(0, i * CELL_POUT); c.lineTo(w, i * CELL_POUT); c.stroke();
            c.beginPath(); c.moveTo(i * CELL_POUT, 0); c.lineTo(i * CELL_POUT, h); c.stroke();
        }
        // Highlight hovered output cell
        if (hoverPoolRow >= 0 && hoverPoolCol >= 0) {
            const bx = hoverPoolCol * CELL_POUT;
            const by = hoverPoolRow * CELL_POUT;
            c.strokeStyle = '#ea7959';
            c.lineWidth = 2.5;
            c.strokeRect(bx + 1, by + 1, CELL_POUT - 2, CELL_POUT - 2);
        }
    }

    function drawLoss() {
        if (!lossCanvas) return;
        const w = lossCanvas.parentElement.clientWidth || 240;
        const h = LOSS_H;
        const c = sizeCanvas(lossCanvas, w, h);
        c.clearRect(0, 0, w, h);
        // Background
        c.fillStyle = 'rgba(0, 0, 0, 0.02)';
        c.fillRect(0, 0, w, h);
        if (!lossHistory.length) return;
        const padL = 4, padR = 6, padT = 8, padB = 14;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;
        const mx = Math.max(0.01, ...lossHistory);
        // Axis baseline (zero)
        c.strokeStyle = 'rgba(0, 0, 0, 0.18)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(padL, padT + plotH);
        c.lineTo(padL + plotW, padT + plotH);
        c.stroke();
        // Curve
        c.strokeStyle = '#4f46e5';
        c.lineWidth = 1.6;
        c.beginPath();
        const n = lossHistory.length;
        for (let i = 0; i < n; i++) {
            const x = padL + (n === 1 ? plotW / 2 : (i * plotW / (n - 1)));
            const y = padT + plotH - (lossHistory[i] / mx) * plotH;
            if (i === 0) c.moveTo(x, y);
            else         c.lineTo(x, y);
        }
        c.stroke();
        // Labels
        c.fillStyle = 'rgba(0, 0, 0, 0.6)';
        c.font = '10px var(--font-sans, system-ui), sans-serif';
        c.textAlign = 'left';
        c.fillText(`max ${mx.toFixed(3)}`, padL + 2, padT + 10);
        c.textAlign = 'right';
        c.fillText('0', w - padR, padT + plotH - 2);
    }

    function fmt(v) {
        if (Math.abs(v) < 0.005) return '0';
        if (Math.abs(v - Math.round(v)) < 0.01) return v.toFixed(0);
        return v.toFixed(2);
    }

    function redraw() {
        drawInput();
        drawKernel();
        drawOutput();
        if (mode === 'learn') {
            drawTarget();
            drawLoss();
            if (readoutEl) {
                const last = lossHistory.length ? lossHistory[lossHistory.length - 1] : 0;
                readoutEl.textContent = `step ${trainStepNum} · loss ${last.toFixed(4)}`;
            }
        }
        drawPoolInput();
        drawPoolOutput();
    }

    // ---------- Interaction: slide the filter ----------
    function setPos(r, c) {
        activeRow = Math.max(0, Math.min(OUT_SIZE - 1, r));
        activeCol = Math.max(0, Math.min(OUT_SIZE - 1, c));
    }

    function inputPosFromEvent(e) {
        const rect = inputCanvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        const x = t.clientX - rect.left;
        const y = t.clientY - rect.top;
        const col = Math.floor(x / CELL_IN) - 1;
        const row = Math.floor(y / CELL_IN) - 1;
        return [row, col];
    }

    inputCanvas.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        playing  = false;
        updatePlayBtn();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        e.preventDefault();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    window.addEventListener('mouseup', () => { dragging = false; });

    inputCanvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        dragging = true;
        playing  = false;
        updatePlayBtn();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('touchmove', (e) => {
        if (!dragging) return;
        e.preventDefault();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('touchend', () => { dragging = false; });

    // ---------- Interaction: edit a preset kernel cell (preset mode only) ----------
    kernelCanvas.addEventListener('click', (e) => {
        if (mode !== 'preset') return;
        const rect = kernelCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const cc = Math.floor(x / CELL_K);
        const rr = Math.floor(y / CELL_K);
        if (rr < 0 || rr >= KSIZE || cc < 0 || cc >= KSIZE) return;
        const cycle = [-2, -1, 0, 1, 2];
        const v = KERNELS[presetName][rr][cc];
        let idx = 0, bestD = Infinity;
        for (let i = 0; i < cycle.length; i++) {
            const d = Math.abs(v - cycle[i]);
            if (d < bestD) { bestD = d; idx = i; }
        }
        idx = (idx + 1) % cycle.length;
        // Deep-copy preset so we don't mutate the dict
        KERNELS[presetName] = KERNELS[presetName].map(r => r.slice());
        KERNELS[presetName][rr][cc] = cycle[idx];
        recompute();
        redraw();
    });

    // ---------- Dropdowns ----------
    function populateSelect(el, options, current) {
        if (!el) return;
        for (const name of Object.keys(options)) {
            const o = document.createElement('option');
            o.value = name;
            o.textContent = name;
            if (name === current) o.selected = true;
            el.appendChild(o);
        }
    }
    const imageSel  = document.getElementById('viz-cnn-image');
    const kernelSel = document.getElementById('viz-cnn-kernel-select');
    const targetSel = document.getElementById('viz-cnn-target-select');
    populateSelect(imageSel, IMAGES, imageName);
    populateSelect(kernelSel, KERNELS, presetName);
    populateSelect(targetSel, KERNELS, targetName);

    if (imageSel) imageSel.addEventListener('change', () => {
        imageName = imageSel.value;
        image = IMAGES[imageName];
        recompute();
        redraw();
    });
    if (kernelSel) kernelSel.addEventListener('change', () => {
        presetName = kernelSel.value;
        recompute();
        redraw();
    });
    if (targetSel) targetSel.addEventListener('change', () => {
        targetName = targetSel.value;
        recompute();
        redraw();
    });

    // ---------- Play/Pause ----------
    const playBtn = document.getElementById('viz-cnn-play');
    function updatePlayBtn() { if (playBtn) playBtn.textContent = playing ? 'Pause' : 'Play'; }
    if (playBtn) playBtn.addEventListener('click', () => {
        playing = !playing;
        updatePlayBtn();
        if (playing) lastSlide = performance.now();
    });

    // ---------- Mode toggle + learn controls ----------
    const modeBtns = document.querySelectorAll('[data-cnn-mode]');
    const presetPane = document.querySelector('.viz-cnn-pane-preset');
    const learnPane  = document.querySelector('.viz-cnn-pane-learn');
    const learnRow   = document.getElementById('viz-cnn-learn-row');
    const stepBtn    = document.getElementById('viz-cnn-step');
    const autoBtn    = document.getElementById('viz-cnn-auto');
    const resetBtn   = document.getElementById('viz-cnn-reset');

    function setMode(m) {
        mode = m;
        modeBtns.forEach(b => b.classList.toggle('active', b.dataset.cnnMode === m));
        if (presetPane) presetPane.hidden = (m !== 'preset');
        if (learnPane)  learnPane.hidden  = (m !== 'learn');
        if (learnRow)   learnRow.hidden   = (m !== 'learn');
        autoTraining = false;
        if (autoBtn) autoBtn.textContent = 'Auto-train';
        recompute();
        redraw();
    }
    modeBtns.forEach(b => b.addEventListener('click', () => setMode(b.dataset.cnnMode)));
    if (stepBtn)  stepBtn.addEventListener('click', () => {
        if (mode !== 'learn') return;
        trainStep();
        redraw();
    });
    if (autoBtn)  autoBtn.addEventListener('click', () => {
        autoTraining = !autoTraining;
        autoBtn.textContent = autoTraining ? 'Stop training' : 'Auto-train';
    });
    if (resetBtn) resetBtn.addEventListener('click', resetLearn);

    // ---------- Pool controls ----------
    const poolModeBtns = document.querySelectorAll('[data-pool-mode]');
    poolModeBtns.forEach(b => b.addEventListener('click', () => {
        poolMode = b.dataset.poolMode;
        poolModeBtns.forEach(other => other.classList.toggle('active', other === b));
        computePool();
        drawPoolInput();
        drawPoolOutput();
    }));

    // Hover over either pool canvas highlights the matching 2 × 2 block ↔ output cell.
    function poolHoverFrom(canvas, isInput) {
        if (!canvas) return;
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            const cell = isInput ? CELL_PIN * POOL_SZ : CELL_POUT;
            const c = Math.floor(x / cell), r = Math.floor(y / cell);
            if (r >= 0 && r < POOL_OUT && c >= 0 && c < POOL_OUT) {
                if (r !== hoverPoolRow || c !== hoverPoolCol) {
                    hoverPoolRow = r; hoverPoolCol = c;
                    drawPoolInput(); drawPoolOutput();
                }
            }
        });
        canvas.addEventListener('mouseleave', () => {
            hoverPoolRow = -1; hoverPoolCol = -1;
            drawPoolInput(); drawPoolOutput();
        });
    }
    poolHoverFrom(poolInCanvas, true);
    poolHoverFrom(poolOutCanvas, false);

    // ---------- Animation loop ----------
    function loop(now) {
        // Slide the filter (preset mode)
        if (mode === 'preset' && playing && now - lastSlide >= SLIDE_MS && !dragging) {
            let r = activeRow, c = activeCol + 1;
            if (c >= OUT_SIZE) { c = 0; r += 1; }
            if (r >= OUT_SIZE) { r = 0; c = 0; }
            setPos(r, c);
            redraw();
            lastSlide = now;
        }
        // Auto-training (learn mode) — many small steps per frame for fast convergence
        if (mode === 'learn' && autoTraining) {
            for (let i = 0; i < 3; i++) trainStep();
            redraw();
            const last = lossHistory[lossHistory.length - 1];
            if (last !== undefined && last < 1e-6) {
                autoTraining = false;
                if (autoBtn) autoBtn.textContent = 'Auto-train';
            }
        }
        requestAnimationFrame(loop);
    }

    // ---------- Init ----------
    recompute();
    redraw();
    updatePlayBtn();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => redraw());
        ro.observe(inputCanvas);
        if (lossCanvas) ro.observe(lossCanvas);
    }
    requestAnimationFrame(loop);
})();
