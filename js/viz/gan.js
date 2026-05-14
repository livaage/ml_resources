/* Interactive 2D GAN viz.
 * Real samples (indigo) come from a chosen target distribution. Fake samples
 * (terracotta) start as noise near the origin. A small 2→12→1 MLP discriminator
 * is trained to tell them apart; each fake particle is then nudged in the
 * direction that increases its "realness" score from D. The background heatmap
 * shows D(x) — blue means "D says fake", orange means "D says real" — and the
 * loss curves on the right show the adversarial dance.
 *
 * Training uses exact gradients computed by hand (the network is tiny). The
 * generator here is just the particles themselves rather than a separate
 * network — it captures the training dynamic without an extra MLP. */

(function () {
    const canvas    = document.getElementById('viz-gan-canvas');
    const stepBtn   = document.getElementById('viz-gan-step');
    const trainBtn  = document.getElementById('viz-gan-train');
    const resetBtn  = document.getElementById('viz-gan-reset');
    const distSel   = document.getElementById('viz-gan-dist');
    const counterEl = document.getElementById('viz-gan-counter');
    const captionEl = document.getElementById('viz-gan-caption');
    if (!canvas) return;

    // ----- Tiny discriminator: 2 → HIDDEN → 1 (sigmoid) -----
    const HIDDEN = 12;
    let D = newD();
    let realPts = [];
    let fakePts = [];
    let losses  = { d: [], g: [] };
    let step    = 0;
    let preset  = 'twoClusters';
    let training = false;
    const N_PTS = 140;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function newD() {
        // Glorot-ish init
        const sx = Math.sqrt(2 / (2 + HIDDEN));
        const sy = Math.sqrt(2 / (HIDDEN + 1));
        return {
            W1: Array.from({length: HIDDEN}, () =>
                [Math.random() * 2 * sx - sx, Math.random() * 2 * sx - sx]),
            b1: Array(HIDDEN).fill(0),
            W2: Array.from({length: HIDDEN}, () => Math.random() * 2 * sy - sy),
            b2: 0,
        };
    }
    function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    function dForward(x) {
        const h = new Array(HIDDEN);
        for (let i = 0; i < HIDDEN; i++) {
            const z = D.W1[i][0] * x[0] + D.W1[i][1] * x[1] + D.b1[i];
            h[i] = Math.tanh(z);
        }
        let z2 = D.b2;
        for (let i = 0; i < HIDDEN; i++) z2 += D.W2[i] * h[i];
        return { p: sigmoid(z2), h, z2 };
    }

    // ----- Target distributions -----
    function sampleTarget() {
        if (preset === 'twoClusters') {
            const c = Math.random() < 0.5 ? [-0.55, 0.0] : [0.55, 0.0];
            return [c[0] + randn() * 0.10, c[1] + randn() * 0.10];
        }
        if (preset === 'ring') {
            const a = Math.random() * Math.PI * 2;
            const r = 0.65 + randn() * 0.04;
            return [r * Math.cos(a), r * Math.sin(a)];
        }
        if (preset === 'spiral') {
            const t = Math.random() * 2.6;
            const r = 0.15 + t * 0.22;
            const a = t * 2.2;
            return [r * Math.cos(a) + randn() * 0.03, r * Math.sin(a) + randn() * 0.03];
        }
        if (preset === 'line') {
            const t = Math.random() * 2 - 1;
            return [t * 0.75, t * 0.35 + randn() * 0.04];
        }
        return [randn() * 0.2, randn() * 0.2];
    }

    function resetData() {
        realPts = Array.from({length: N_PTS}, () => sampleTarget());
        // Initial fakes: small Gaussian around origin (a "naive" generator)
        fakePts = Array.from({length: N_PTS}, () => [randn() * 0.08, randn() * 0.08]);
        D = newD();
        losses = { d: [], g: [] };
        step = 0;
        invalidateHeatmap();
    }

    // ----- Training -----
    function trainD(lr) {
        // BCE; real target=1, fake target=0
        const dW1 = Array.from({length: HIDDEN}, () => [0, 0]);
        const db1 = Array(HIDDEN).fill(0);
        const dW2 = Array(HIDDEN).fill(0);
        let db2 = 0;
        const N = realPts.length + fakePts.length;

        function accumulate(pt, target) {
            const { p, h } = dForward(pt);
            const dz2 = (p - target) / N;
            for (let i = 0; i < HIDDEN; i++) {
                dW2[i] += dz2 * h[i];
                const dh = dz2 * D.W2[i];
                const dz1 = dh * (1 - h[i] * h[i]);
                dW1[i][0] += dz1 * pt[0];
                dW1[i][1] += dz1 * pt[1];
                db1[i]   += dz1;
            }
            db2 += dz2;
        }
        for (const pt of realPts) accumulate(pt, 1);
        for (const pt of fakePts) accumulate(pt, 0);

        // Update with weight decay for stability
        const wd = 0.0008;
        for (let i = 0; i < HIDDEN; i++) {
            D.W1[i][0] -= lr * dW1[i][0] + lr * wd * D.W1[i][0];
            D.W1[i][1] -= lr * dW1[i][1] + lr * wd * D.W1[i][1];
            D.b1[i]   -= lr * db1[i];
            D.W2[i]   -= lr * dW2[i] + lr * wd * D.W2[i];
        }
        D.b2 -= lr * db2;
    }

    function trainG(lr) {
        // Non-saturating loss: minimize -log D(fake) ⇒ push each fake toward
        // higher D(x). For each particle, gradient w.r.t. x:
        //   dz2/dx = sum_i W2[i] * (1 - h[i]^2) * W1[i]
        //   dL/dx  = -(1 - p) * dz2/dx
        for (let f = 0; f < fakePts.length; f++) {
            const pt = fakePts[f];
            const { p, h } = dForward(pt);
            const dz2 = -(1 - p);
            let dx = 0, dy = 0;
            for (let i = 0; i < HIDDEN; i++) {
                const dh_i  = dz2 * D.W2[i];
                const dz1_i = dh_i * (1 - h[i] * h[i]);
                dx += dz1_i * D.W1[i][0];
                dy += dz1_i * D.W1[i][1];
            }
            // A small per-particle Gaussian "exploration" noise keeps the
            // generator from collapsing into a single point.
            pt[0] -= lr * dx + 0.003 * randn();
            pt[1] -= lr * dy + 0.003 * randn();
            // Soft clamp to keep things on-canvas
            const r = Math.hypot(pt[0], pt[1]);
            if (r > 1.4) { pt[0] *= 1.4 / r; pt[1] *= 1.4 / r; }
        }
    }

    function recordLoss() {
        let dLoss = 0, gLoss = 0;
        for (const pt of realPts) dLoss -= Math.log(dForward(pt).p + 1e-9);
        for (const pt of fakePts) {
            const p = dForward(pt).p;
            dLoss -= Math.log(1 - p + 1e-9);
            gLoss -= Math.log(p + 1e-9);
        }
        dLoss /= (realPts.length + fakePts.length);
        gLoss /= fakePts.length;
        losses.d.push(dLoss);
        losses.g.push(gLoss);
        if (losses.d.length > 200) { losses.d.shift(); losses.g.shift(); }
    }

    function trainStep() {
        // A couple of D steps per G step is standard
        trainD(0.06);
        trainD(0.06);
        trainG(0.10);
        recordLoss();
        step++;
        invalidateHeatmap();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    let heatmapCache = null;
    function invalidateHeatmap() { heatmapCache = null; }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(560, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(330, cssW * 0.54)));
        W = cssW;
        H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        invalidateHeatmap();
        draw();
    }

    // ----- Layout: left scatter, right loss plot -----
    function layout() {
        const pad = 16;
        const scatterW = Math.min(H - 2 * pad, W * 0.58);
        const scatter = {
            x: pad,
            y: pad,
            w: scatterW,
            h: H - 2 * pad,
        };
        const lossPlot = {
            x: scatter.x + scatter.w + pad + 8,
            y: pad + 26,
            w: W - (scatter.x + scatter.w + pad + 8) - pad,
            h: H * 0.45,
        };
        const stats = {
            x: lossPlot.x,
            y: lossPlot.y + lossPlot.h + 18,
            w: lossPlot.w,
            h: H - (lossPlot.y + lossPlot.h + 18) - pad,
        };
        return { scatter, lossPlot, stats };
    }

    // Data range: [-1.4, 1.4]² maps to scatter box
    function dataToCanvas(p, box) {
        const m = Math.min(box.w, box.h) / 2 - 6;
        return {
            x: box.x + box.w / 2 + p[0] * m,
            y: box.y + box.h / 2 - p[1] * m,
        };
    }
    function canvasToData(cx, cy, box) {
        const m = Math.min(box.w, box.h) / 2 - 6;
        return [
            (cx - (box.x + box.w / 2)) / m,
            -(cy - (box.y + box.h / 2)) / m,
        ];
    }

    // ----- Drawing -----
    function dColour(p) {
        // p in [0, 1]. p < 0.5 → indigo wash (fake), p > 0.5 → orange wash (real)
        if (p < 0.5) {
            const t = (0.5 - p) / 0.5;        // 0..1
            // wash toward indigo
            const r = 251 + (79  - 251) * t * 0.45;
            const g = 250 + (70  - 250) * t * 0.45;
            const b = 247 + (229 - 247) * t * 0.45;
            return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
        }
        const t = (p - 0.5) / 0.5;
        const r = 251 + (234 - 251) * t * 0.55;
        const g = 250 + (121 - 250) * t * 0.55;
        const b = 247 + (89  - 247) * t * 0.55;
        return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    }

    function renderHeatmap(box) {
        // 60×60 grid → 3600 D evaluations, fast enough.
        if (heatmapCache && heatmapCache.box.w === box.w && heatmapCache.box.h === box.h) {
            return heatmapCache.canvas;
        }
        const G = 48;
        const off = document.createElement('canvas');
        off.width = G;
        off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dataX =  ((i + 0.5) / G * 2 - 1) * 1.35;
                const dataY = -((j + 0.5) / G * 2 - 1) * 1.35;
                const p = dForward([dataX, dataY]).p;
                const col = dColour(p);
                const m = col.match(/rgb\((\d+), (\d+), (\d+)\)/);
                const idx = (j * G + i) * 4;
                img.data[idx]     = +m[1];
                img.data[idx + 1] = +m[2];
                img.data[idx + 2] = +m[3];
                img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatmapCache = { box: {w: box.w, h: box.h}, canvas: off };
        return off;
    }

    function drawScatter(box) {
        // Heatmap background (D decision surface)
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        const hm = renderHeatmap(box);
        ctx.drawImage(hm, box.x, box.y, box.w, box.h);
        ctx.restore();

        // Decision contour at D = 0.5 by overlaying a slight gradient ring
        // (visual cue; precise contour would require marching squares)

        // Frame
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Axes
        const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(box.x + 6, cy); ctx.lineTo(box.x + box.w - 6, cy);
        ctx.moveTo(cx, box.y + 6); ctx.lineTo(cx, box.y + box.h - 6);
        ctx.stroke();
        ctx.setLineDash([]);

        // Real points (indigo)
        ctx.fillStyle = 'rgba(79, 70, 229, 0.75)';
        for (const p of realPts) {
            const c = dataToCanvas(p, box);
            ctx.beginPath();
            ctx.arc(c.x, c.y, 2.4, 0, Math.PI * 2);
            ctx.fill();
        }
        // Fake points (terracotta)
        ctx.fillStyle = 'rgba(234, 121, 89, 0.85)';
        for (const p of fakePts) {
            const c = dataToCanvas(p, box);
            ctx.beginPath();
            ctx.arc(c.x, c.y, 2.4, 0, Math.PI * 2);
            ctx.fill();
        }

        // Title and legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('SAMPLE SPACE — real vs fake', box.x, box.y - 5);
        // Legend dots
        ctx.fillStyle = 'rgba(79, 70, 229, 0.85)';
        ctx.beginPath(); ctx.arc(box.x + box.w - 96, box.y - 9, 3.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText('real', box.x + box.w - 90, box.y - 5);
        ctx.fillStyle = 'rgba(234, 121, 89, 0.95)';
        ctx.beginPath(); ctx.arc(box.x + box.w - 52, box.y - 9, 3.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('fake', box.x + box.w - 46, box.y - 5);
    }

    function drawLossPlot(box) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.025)';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('LOSS', box.x, box.y - 5);

        // Find max for scale
        let mx = 0.5;
        for (const v of losses.d) if (v > mx) mx = v;
        for (const v of losses.g) if (v > mx) mx = v;
        mx *= 1.1;

        if (losses.d.length < 2) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click Train to start', box.x + box.w / 2, box.y + box.h / 2);
            return;
        }

        const N = losses.d.length;
        function drawCurve(arr, colour) {
            ctx.strokeStyle = colour;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            for (let i = 0; i < arr.length; i++) {
                const x = box.x + (i / Math.max(1, N - 1)) * box.w;
                const y = box.y + box.h - (arr[i] / mx) * box.h;
                if (i === 0) ctx.moveTo(x, y);
                else         ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        drawCurve(losses.d, '#4f46e5');
        drawCurve(losses.g, '#ea7959');

        // Legend
        ctx.fillStyle = '#4f46e5';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('— D', box.x + 6, box.y + 12);
        ctx.fillStyle = '#ea7959';
        ctx.fillText('— G', box.x + 6, box.y + 24);
    }

    function drawStats(box) {
        // Step counter, D accuracy, mode-collapse hint
        let nReal = 0, nFake = 0;
        for (const pt of realPts) if (dForward(pt).p > 0.5) nReal++;
        for (const pt of fakePts) if (dForward(pt).p < 0.5) nFake++;
        const dAcc = (nReal + nFake) / (realPts.length + fakePts.length);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('METRICS', box.x, box.y - 5);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        let y = box.y + 14;
        ctx.fillText(`step ${step}`, box.x + 6, y); y += 16;
        ctx.fillText(`D accuracy ${(dAcc * 100).toFixed(0)}%`, box.x + 6, y); y += 16;

        // Equilibrium hint
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        const tip = (dAcc < 0.6 && step > 30)
            ? '≈ 50% — G is fooling D (good!)'
            : (dAcc > 0.9 && step > 30)
                ? 'D is dominating — G needs more time'
                : 'training in progress…';
        // Word-wrap on space
        const words = tip.split(' ');
        let line = '';
        for (const w of words) {
            const candidate = line ? line + ' ' + w : w;
            if (ctx.measureText(candidate).width > box.w - 12) {
                ctx.fillText(line, box.x + 6, y);
                y += 12;
                line = w;
            } else {
                line = candidate;
            }
        }
        if (line) ctx.fillText(line, box.x + 6, y);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const lay = layout();
        drawScatter(lay.scatter);
        drawLossPlot(lay.lossPlot);
        drawStats(lay.stats);

        if (counterEl) counterEl.textContent = `Step ${step}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const fakeMean = fakePts.reduce((a, p) => [a[0] + p[0], a[1] + p[1]], [0, 0]);
        fakeMean[0] /= fakePts.length; fakeMean[1] /= fakePts.length;
        if (step === 0) {
            captionEl.innerHTML =
                `<strong>Step 0.</strong> Fake samples (orange) all sit near the origin — a naive generator. ` +
                `Click <strong>Train</strong> and watch each particle nudge toward where the discriminator says "real".`;
        } else if (step < 40) {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> The discriminator is learning the boundary between real and fake; ` +
                `each fake particle climbs the gradient of <em>D</em> toward higher "realness". The orange wash in ` +
                `the background marks where <em>D</em> currently believes real samples live.`;
        } else {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> Real and fake distributions are overlapping — <em>D</em> can no longer ` +
                `tell them apart cleanly. Try the <strong>Two clusters</strong> preset to see <em>mode collapse</em>: ` +
                `the generator often pours all its mass into one mode and abandons the other.`;
        }
    }

    // ----- Animation loop -----
    function loop() {
        if (training) {
            trainStep();
            draw();
        }
        requestAnimationFrame(loop);
    }

    // ----- Interactions -----
    stepBtn?.addEventListener('click', () => { trainStep(); draw(); });
    resetBtn?.addEventListener('click', () => { training = false; updateTrainBtn(); resetData(); draw(); });
    function updateTrainBtn() {
        if (trainBtn) trainBtn.textContent = training ? 'Pause' : 'Train';
    }
    trainBtn?.addEventListener('click', () => {
        training = !training;
        updateTrainBtn();
    });

    if (distSel) {
        distSel.innerHTML = `
            <option value="twoClusters">Two clusters</option>
            <option value="ring">Ring</option>
            <option value="spiral">Spiral</option>
            <option value="line">Line</option>
        `;
        distSel.addEventListener('change', () => {
            preset = distSel.value;
            training = false;
            updateTrainBtn();
            resetData();
            draw();
        });
    }

    // ----- Init -----
    resetData();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    updateTrainBtn();
    requestAnimationFrame(loop);
})();
