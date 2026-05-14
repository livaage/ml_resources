/* Interactive diffusion viz.
 * Scrub the t slider — or hit Forward / Reverse — and watch a 28×28 image walk
 * between "clean" (t=0) and "pure noise" (t=T) along a cosine schedule. The
 * forward process is the literal formula: x_t = √α̅_t · x_0 + √(1-α̅_t) · ε.
 *
 * The middle panel shows the current noisy x_t. The right panel shows the
 * model's "predicted x_0" — what a perfect denoiser would reconstruct from
 * x_t alone. To simulate model imperfection it adds a small Gaussian jitter
 * to the recovered ε, which makes the prediction less confident at high t
 * (just like a real diffusion model). */

(function () {
    const canvas    = document.getElementById('viz-diff-canvas');
    const slider    = document.getElementById('viz-diff-slider');
    const tLabel    = document.getElementById('viz-diff-t');
    const fwdBtn    = document.getElementById('viz-diff-forward');
    const revBtn    = document.getElementById('viz-diff-reverse');
    const resetBtn  = document.getElementById('viz-diff-reset');
    const presetSel = document.getElementById('viz-diff-preset');
    const captionEl = document.getElementById('viz-diff-caption');
    if (!canvas) return;

    const SIZE = 28;
    const T = 32;

    // ----- Image patterns (28×28, '#' = filled, '.' = empty) -----
    function parsePattern(s) {
        const img = new Float32Array(SIZE * SIZE);
        const rows = s.trim().split('\n').map(r => r.trim());
        for (let y = 0; y < Math.min(SIZE, rows.length); y++) {
            for (let x = 0; x < Math.min(SIZE, rows[y].length); x++) {
                // Map: '#' = 1, '.' = -1  (diffusion convention is [-1, 1])
                img[y * SIZE + x] = (rows[y][x] === '#') ? 1.0 : -1.0;
            }
        }
        return img;
    }

    const PROTOS = {
        heart: parsePattern(`
            ............................
            ............................
            ............................
            ............................
            .....######.......######....
            ...##########...##########..
            ..############.############.
            .##########################.
            .##########################.
            .##########################.
            ..########################..
            ...######################...
            ....####################....
            .....##################.....
            ......################......
            .......##############.......
            ........############........
            .........##########.........
            ..........########..........
            ...........######...........
            ............####............
            .............##.............
            ............................
            ............................
            ............................
            ............................
            ............................
            ............................
        `),
        smile: parsePattern(`
            ............................
            ............................
            ............................
            ............................
            ............................
            .......####........####.....
            ......######......######....
            ......######......######....
            ......######......######....
            .......####........####.....
            ............................
            ............................
            ............................
            ............................
            ............................
            ...##....................##.
            ....##..................##..
            .....##................##...
            ......###............###....
            .......###..........###.....
            ........####......####......
            .........##############.....
            ..........############......
            ...........##########.......
            ............########........
            ............................
            ............................
            ............................
        `),
        digit5: parsePattern(`
            ............................
            ............................
            ............................
            ............................
            ......################......
            ......################......
            ......################......
            ......##....................
            ......##....................
            ......##....................
            ......##....................
            ......##....................
            ......##############........
            ......################......
            ......##################....
            ......####..........####....
            ......................####..
            ..............##......####..
            ..............####....####..
            ..............####....####..
            ......####....####....####..
            ......######..########......
            ......######..######........
            ........###############.....
            ..........############......
            ............#########.......
            ............................
            ............................
        `),
    };

    // ----- Schedule: cosine ᾱ_t (smooth, used in modern diffusion) -----
    const alphaBar = new Float32Array(T + 1);
    (function buildSchedule() {
        const s = 0.008;
        const f0 = Math.cos((s / (1 + s)) * Math.PI / 2) ** 2;
        for (let t = 0; t <= T; t++) {
            const f = Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
            alphaBar[t] = Math.max(0, Math.min(1, f / f0));
        }
    })();

    // Fixed noise across all timesteps so scrubbing the slider gives a smooth
    // visual trajectory (DDPM actually uses independent ε per t, but that
    // would make scrubbing look jittery).
    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    let noise = null;
    function resampleNoise() {
        noise = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < noise.length; i++) noise[i] = randn();
    }
    resampleNoise();

    // ----- State -----
    let presetKey = 'heart';
    let x0 = PROTOS[presetKey];
    let t = 0;
    let animating = null;          // 'forward' | 'reverse' | null
    let animStart = 0;
    let animStartT = 0;

    function noisedAt(t) {
        const ab = alphaBar[t];
        const s_x = Math.sqrt(ab);
        const s_n = Math.sqrt(1 - ab);
        const out = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < out.length; i++) out[i] = s_x * x0[i] + s_n * noise[i];
        return out;
    }

    // "Model" predicted x_0 from x_t: oracle that adds a small noise to
    // simulate imperfect denoising. At t=0 it equals x_0 exactly.
    function predictedX0(x_t, t) {
        const ab = alphaBar[t];
        const s_x = Math.max(1e-6, Math.sqrt(ab));
        const s_n = Math.sqrt(1 - ab);
        const jitterScale = 0.04 * (1 - ab);   // more jitter at small ᾱ
        const out = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < out.length; i++) {
            const eps_hat = noise[i] + jitterScale * randn();
            out[i] = (x_t[i] - s_n * eps_hat) / s_x;
        }
        return out;
    }

    function predictedNoise(x_t, t) {
        const ab = alphaBar[t];
        const s_x = Math.sqrt(ab);
        const s_n = Math.max(1e-6, Math.sqrt(1 - ab));
        const out = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < out.length; i++) {
            // ε̂ = (x_t - √α̅ · x_0) / √(1-α̅)  — uses the true x_0 (oracle)
            out[i] = (x_t[i] - s_x * x0[i]) / s_n;
        }
        return out;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(340, cssW * 0.50)));
        W = cssW;
        H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    // ----- Layout -----
    function layout() {
        const pad = 14;
        const stripH = 56;
        const stripTop = pad + 14;
        const strip = { x: pad, y: stripTop, w: W - 2 * pad, h: stripH };

        const bigTop = strip.y + strip.h + 32;
        const bigH = H - bigTop - pad;
        const bigW = Math.min(bigH, (W - 2 * pad - 16) / 3);

        const labelOffset = 14;
        const totalBigW = bigW * 3 + 16;
        const startX = (W - totalBigW) / 2;
        const xt    = { x: startX,                      y: bigTop, w: bigW, h: bigH - labelOffset };
        const xhat  = { x: startX + bigW + 8,           y: bigTop, w: bigW, h: bigH - labelOffset };
        const ehat  = { x: startX + bigW * 2 + 16,      y: bigTop, w: bigW, h: bigH - labelOffset };

        return { strip, xt, xhat, ehat };
    }

    // ----- Drawing -----
    function imgColour(t, mode = 'image') {
        // t in [-1, 1]. Map to indigo (-1) → cream (0) → terracotta (1)
        // For "noise" mode use a slightly muted version for contrast.
        const cl = Math.max(-1, Math.min(1, t));
        let r, g, b;
        if (cl < 0) {
            const a = -cl;
            r = 251 + (79  - 251) * a;
            g = 250 + (70  - 250) * a;
            b = 247 + (229 - 247) * a;
        } else {
            const a = cl;
            r = 251 + (234 - 251) * a;
            g = 250 + (121 - 250) * a;
            b = 247 + (89  - 247) * a;
        }
        return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    }

    function drawImage(box, img, label) {
        const cellW = box.w / SIZE;
        const cellH = box.h / SIZE;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE; x++) {
                ctx.fillStyle = imgColour(img[y * SIZE + x]);
                ctx.fillRect(box.x + x * cellW, box.y + y * cellH, cellW + 0.6, cellH + 0.6);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.14)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, box.x + box.w / 2, box.y - 8);
    }

    function drawStrip(box) {
        // 9 thumbnails at evenly spaced t
        const N = 9;
        const slotW = (box.w - 8 * (N - 1)) / N;
        const slotH = box.h - 16;  // leave room for tick labels and current marker
        for (let i = 0; i < N; i++) {
            const ti = Math.round(i * T / (N - 1));
            const img = noisedAt(ti);
            const slot = {
                x: box.x + i * (slotW + 8),
                y: box.y,
                w: slotW,
                h: slotH,
            };
            // Mini image
            const cellW = slot.w / SIZE;
            const cellH = slot.h / SIZE;
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    ctx.fillStyle = imgColour(img[y * SIZE + x]);
                    ctx.fillRect(slot.x + x * cellW, slot.y + y * cellH, cellW + 0.6, cellH + 0.6);
                }
            }
            // Highlight current
            const isCurrent = Math.abs(ti - t) <= Math.ceil(T / (N - 1) / 2);
            ctx.strokeStyle = isCurrent ? '#ea7959' : 'rgba(0, 0, 0, 0.14)';
            ctx.lineWidth = isCurrent ? 2 : 1;
            ctx.strokeRect(slot.x - 0.5, slot.y - 0.5, slot.w + 1, slot.h + 1);

            // Tick label
            ctx.fillStyle = isCurrent ? '#ea7959' : 'rgba(0, 0, 0, 0.4)';
            ctx.font = isCurrent
                ? '600 9px "JetBrains Mono", monospace'
                : '500 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`t=${ti}`, slot.x + slot.w / 2, slot.y + slot.h + 11);
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('FORWARD PROCESS — x₀ becomes pure noise as t grows', box.x, box.y - 7);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const lay = layout();
        drawStrip(lay.strip);

        const x_t = noisedAt(t);
        const xh  = predictedX0(x_t, t);
        const eh  = predictedNoise(x_t, t);

        drawImage(lay.xt,   x_t, `x_t  (t = ${t})`);
        drawImage(lay.xhat, xh,  'predicted x₀');
        drawImage(lay.ehat, eh,  'predicted ε̂');

        // Equation under the boxes
        const lay2 = layout();
        const eqY = lay2.xt.y + lay2.xt.h + 22;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        const ab = alphaBar[t];
        const s_x = Math.sqrt(ab).toFixed(2);
        const s_n = Math.sqrt(1 - ab).toFixed(2);
        ctx.fillText(`x_t = ${s_x} · x₀  +  ${s_n} · ε`,
            lay2.xt.x + lay2.xt.w / 2, eqY);

        if (tLabel) tLabel.textContent = `t = ${t}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const ab = alphaBar[t];
        const pct = (1 - ab) * 100;
        if (t === 0) {
            captionEl.innerHTML =
                `<strong>t = 0.</strong> The clean image x₀. No noise — the model has nothing to denoise. ` +
                `Drag the slider right (or hit <em>Forward</em>) to start adding Gaussian noise.`;
        } else if (t >= T) {
            captionEl.innerHTML =
                `<strong>t = T.</strong> Pure noise — α̅ ≈ 0, so x_t is essentially Gaussian. ` +
                `From here the model would denoise step by step back to x₀. Hit <em>Reverse</em> ` +
                `to watch the trajectory play backward.`;
        } else if (t < T / 3) {
            captionEl.innerHTML =
                `<strong>t = ${t}.</strong> Mild noise (≈ ${pct.toFixed(0)}% of the variance is noise). ` +
                `The model's x₀ estimate is sharp; the predicted ε̂ matches the true added noise.`;
        } else if (t < 2 * T / 3) {
            captionEl.innerHTML =
                `<strong>t = ${t}.</strong> The signal is half-lost — α̅ ≈ ${ab.toFixed(2)}. ` +
                `Predicting x₀ is harder now; this is where a real denoiser earns its keep.`;
        } else {
            captionEl.innerHTML =
                `<strong>t = ${t}.</strong> Nearly pure noise — α̅ ≈ ${ab.toFixed(2)}. ` +
                `The model is essentially guessing what x₀ might be from a few faint signal cues.`;
        }
    }

    // ----- Animation -----
    const ANIM_MS = 1800;
    function startAnim(dir) {
        animating = dir;
        animStart = performance.now();
        animStartT = t;
    }
    function loop(now) {
        if (animating) {
            const elapsed = now - animStart;
            const target = animating === 'forward' ? T : 0;
            const totalDist = Math.abs(target - animStartT);
            const dur = (totalDist / T) * ANIM_MS;
            const prog = Math.min(1, elapsed / Math.max(40, dur));
            const newT = Math.round(animStartT + (target - animStartT) * prog);
            if (newT !== t) {
                t = newT;
                if (slider) slider.value = t;
                draw();
            }
            if (prog >= 1) animating = null;
        }
        requestAnimationFrame(loop);
    }

    // ----- Interactions -----
    if (slider) {
        slider.min = 0;
        slider.max = T;
        slider.step = 1;
        slider.value = 0;
        slider.addEventListener('input', () => {
            animating = null;
            t = parseInt(slider.value, 10);
            draw();
        });
    }
    fwdBtn?.addEventListener('click', () => startAnim('forward'));
    revBtn?.addEventListener('click', () => startAnim('reverse'));
    resetBtn?.addEventListener('click', () => {
        animating = null;
        t = 0;
        if (slider) slider.value = 0;
        resampleNoise();
        draw();
    });

    if (presetSel) {
        presetSel.innerHTML = `
            <option value="heart">Heart</option>
            <option value="smile">Smile</option>
            <option value="digit5">Digit "5"</option>
        `;
        presetSel.addEventListener('change', () => {
            presetKey = presetSel.value;
            x0 = PROTOS[presetKey];
            resampleNoise();
            draw();
        });
    }

    // ----- Init -----
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    requestAnimationFrame(loop);
})();
