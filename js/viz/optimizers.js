/* Interactive optimizer race viz.
 * Same starting point and step budget for four optimizers on a 2D loss
 * surface: SGD, SGD+Momentum, RMSprop, Adam. The contours show the loss;
 * the four coloured paths show how each gets to the minimum. Try a narrow-
 * valley landscape (Rosenbrock-ish) to see momentum / Adam shine. */

(function () {
    const canvas    = document.getElementById('viz-opt-canvas');
    const stepBtn   = document.getElementById('viz-opt-step');
    const playBtn   = document.getElementById('viz-opt-play');
    const resetBtn  = document.getElementById('viz-opt-reset');
    const landSel   = document.getElementById('viz-opt-land');
    const lrSlider  = document.getElementById('viz-opt-lr');
    const lrLbl     = document.getElementById('viz-opt-lr-lbl');
    const stepLbl   = document.getElementById('viz-opt-step-lbl');
    const captionEl = document.getElementById('viz-opt-caption');
    if (!canvas) return;

    let landscape = 'valley';
    let lr = 0.05;
    let stepN = 0;
    let playing = false, lastStep = 0;
    const STEP_MS = 80;

    // ----- Loss surfaces (scaled to ~[-1.4, 1.4]²) -----
    function loss(x, y) {
        if (landscape === 'valley') {
            // Long narrow valley aligned to y = 0; quadratic in y, gentle in x
            return 8 * (y - 0.3 * x * x) ** 2 + 0.2 * (x - 0.7) ** 2;
        }
        if (landscape === 'saddle') {
            return 1.5 * x * x - 1.5 * y * y + 0.6 * x ** 4 + 0.6 * y ** 4;
        }
        if (landscape === 'multimodal') {
            return Math.cos(2.5 * x) + Math.cos(2.5 * y) + 0.4 * (x * x + y * y);
        }
        // bowl: simple quadratic
        return 2 * x * x + 2 * y * y;
    }
    function grad(x, y) {
        // Analytic gradients per landscape — faster than finite differences
        if (landscape === 'valley') {
            const inner = y - 0.3 * x * x;
            return [
                -16 * inner * 0.6 * x + 0.4 * (x - 0.7),
                 16 * inner,
            ];
        }
        if (landscape === 'saddle') {
            return [3 * x + 2.4 * x ** 3, -3 * y + 2.4 * y ** 3];
        }
        if (landscape === 'multimodal') {
            return [-2.5 * Math.sin(2.5 * x) + 0.8 * x,
                    -2.5 * Math.sin(2.5 * y) + 0.8 * y];
        }
        return [4 * x, 4 * y];
    }

    // ----- Optimizers -----
    function newOpt(name, p0) {
        const o = { name, x: p0[0], y: p0[1], path: [[p0[0], p0[1]]] };
        // SGD has no state. Others:
        if (name === 'momentum') { o.vx = 0; o.vy = 0; }
        if (name === 'rmsprop')  { o.sx = 0; o.sy = 0; }
        if (name === 'adam')     { o.mx = 0; o.my = 0; o.vx = 0; o.vy = 0; o.t = 0; }
        return o;
    }
    function step(o) {
        const [gx, gy] = grad(o.x, o.y);
        if (o.name === 'sgd') {
            o.x -= lr * gx; o.y -= lr * gy;
        } else if (o.name === 'momentum') {
            o.vx = 0.9 * o.vx + gx;
            o.vy = 0.9 * o.vy + gy;
            o.x -= lr * o.vx;
            o.y -= lr * o.vy;
        } else if (o.name === 'rmsprop') {
            o.sx = 0.9 * o.sx + 0.1 * gx * gx;
            o.sy = 0.9 * o.sy + 0.1 * gy * gy;
            o.x -= lr * gx / (Math.sqrt(o.sx) + 1e-6);
            o.y -= lr * gy / (Math.sqrt(o.sy) + 1e-6);
        } else if (o.name === 'adam') {
            o.t++;
            const b1 = 0.9, b2 = 0.999;
            o.mx = b1 * o.mx + (1 - b1) * gx;
            o.my = b1 * o.my + (1 - b1) * gy;
            o.vx = b2 * o.vx + (1 - b2) * gx * gx;
            o.vy = b2 * o.vy + (1 - b2) * gy * gy;
            const mhx = o.mx / (1 - b1 ** o.t);
            const mhy = o.my / (1 - b1 ** o.t);
            const vhx = o.vx / (1 - b2 ** o.t);
            const vhy = o.vy / (1 - b2 ** o.t);
            o.x -= lr * mhx / (Math.sqrt(vhx) + 1e-8);
            o.y -= lr * mhy / (Math.sqrt(vhy) + 1e-8);
        }
        // Soft clamp
        o.x = Math.max(-1.4, Math.min(1.4, o.x));
        o.y = Math.max(-1.4, Math.min(1.4, o.y));
        o.path.push([o.x, o.y]);
    }

    let optimizers = [];
    const COLOURS = {
        sgd:      '#4f46e5',
        momentum: '#ea7959',
        rmsprop:  '#10847e',
        adam:     '#d4a13c',
    };
    const LABELS = {
        sgd:      'SGD',
        momentum: 'SGD + momentum',
        rmsprop:  'RMSprop',
        adam:     'Adam',
    };

    function startPoint() {
        if (landscape === 'valley') return [-1.0, 0.7];
        if (landscape === 'saddle') return [0.04, 0.04];
        if (landscape === 'multimodal') return [-1.1, 0.9];
        return [-1.0, 0.8];
    }
    function reset() {
        const p = startPoint();
        optimizers = ['sgd', 'momentum', 'rmsprop', 'adam'].map(n => newOpt(n, p));
        stepN = 0;
        playing = false;
        if (playBtn) playBtn.textContent = 'Play';
        heatCache = null;
        draw();
    }
    function doStep() {
        for (const o of optimizers) step(o);
        stepN++;
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    let heatCache = null;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        heatCache = null;
        draw();
    }
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function buildHeatmap() {
        if (heatCache) return heatCache;
        const G = 70;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        let mx = -Infinity, mn = Infinity;
        const vals = new Float32Array(G * G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.3;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.3;
                const v = loss(dx, dy);
                vals[j * G + i] = v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        // Log-ish remap so the basin is visible
        const rng = mx - mn;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const t = (vals[j * G + i] - mn) / (rng || 1);
                const tt = Math.pow(t, 0.45);
                // cream → indigo wash
                const r = 251 + (79  - 251) * tt * 0.55;
                const g = 250 + (70  - 250) * tt * 0.55;
                const b = 247 + (229 - 247) * tt * 0.55;
                const idx = (j * G + i) * 4;
                img.data[idx] = r | 0; img.data[idx + 1] = g | 0;
                img.data[idx + 2] = b | 0; img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatCache = off;
        return off;
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const m = Math.min(W, H) / 2 - 16;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(buildHeatmap(), W / 2 - m * 1.3, H / 2 - m * 1.3, m * 2.6, m * 2.6);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m * 1.3 - 0.5, H / 2 - m * 1.3 - 0.5, m * 2.6 + 1, m * 2.6 + 1);

        // Paths
        for (const o of optimizers) {
            ctx.strokeStyle = COLOURS[o.name];
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < o.path.length; i++) {
                const [px, py] = toPx(o.path[i][0], o.path[i][1]);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
            // Current position
            const [cx, cy] = toPx(o.x, o.y);
            ctx.fillStyle = COLOURS[o.name];
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Start marker
        const sp = startPoint();
        const [sx, sy] = toPx(sp[0], sp[1]);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, Math.PI * 2); ctx.fill();

        // Legend
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        let ly = 16;
        for (const o of optimizers) {
            ctx.fillStyle = COLOURS[o.name];
            ctx.fillRect(12, ly + 4, 12, 2.6);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillText(`${LABELS[o.name]}  L = ${loss(o.x, o.y).toFixed(3)}`, 30, ly + 8);
            ly += 16;
        }

        if (stepLbl) stepLbl.textContent = `step ${stepN}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            valley:     `A narrow curved valley. SGD zig-zags across it; momentum and Adam roll along it; RMSprop's per-parameter scaling also handles the curvature well.`,
            saddle:     `A saddle at the origin. Vanilla SGD often stalls right at the saddle (gradient ≈ 0); momentum and Adam escape much faster because their state carries non-zero velocity.`,
            multimodal: `A surface with several local minima. Optimizers can end up at different ones depending on path — and the lr matters a lot.`,
            bowl:       `An easy bowl. Every optimizer converges; SGD is fastest because there's no curvature variation for momentum / adaptive scaling to exploit.`,
        };
        captionEl.innerHTML = `<strong>step ${stepN}.</strong> ${notes[landscape]} Loss values shown on the left.`;
    }

    function loop(now) {
        if (playing && now - lastStep >= STEP_MS) {
            doStep();
            lastStep = now;
            if (stepN >= 300) { playing = false; playBtn.textContent = 'Play'; }
        }
        requestAnimationFrame(loop);
    }

    // ----- Controls -----
    stepBtn?.addEventListener('click', doStep);
    playBtn?.addEventListener('click', () => {
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', reset);
    if (landSel) {
        landSel.innerHTML = `
            <option value="valley">Narrow valley</option>
            <option value="saddle">Saddle point</option>
            <option value="multimodal">Multimodal</option>
            <option value="bowl">Quadratic bowl</option>
        `;
        landSel.addEventListener('change', () => {
            landscape = landSel.value;
            reset();
        });
    }
    if (lrSlider) {
        lrSlider.min = 0.005; lrSlider.max = 0.2; lrSlider.step = 0.005; lrSlider.value = 0.05;
        lrSlider.addEventListener('input', () => {
            lr = parseFloat(lrSlider.value);
            if (lrLbl) lrLbl.textContent = `lr = ${lr.toFixed(3)}`;
            reset();
        });
    }

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
