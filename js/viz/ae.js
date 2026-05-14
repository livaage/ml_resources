/* Interactive autoencoder viz.
 * Pick a preset image → it's encoded to a 2D latent point, then the decoder
 * reconstructs it on the right. Drag the cursor in the latent square to roam
 * the code space — the output morphs smoothly. Toggle "VAE mode" and the
 * encoder produces a Gaussian instead of a point; the decoder averages over
 * samples (drawn as small orange dots). The "model" here is just softmax
 * blending over 5 hand-crafted prototypes — real autoencoders learn this
 * mapping from data, but the propagation pattern is identical. */

(function () {
    const canvas     = document.getElementById('viz-ae-canvas');
    const captionEl  = document.getElementById('viz-ae-caption');
    const presetWrap = document.getElementById('viz-ae-presets');
    const vaeToggle  = document.getElementById('viz-ae-vae');
    const resetBtn   = document.getElementById('viz-ae-reset');
    if (!canvas) return;

    const IMG = 12;

    function parsePattern(s) {
        const img = new Float32Array(IMG * IMG);
        const rows = s.trim().split('\n').map(r => r.trim());
        for (let y = 0; y < Math.min(IMG, rows.length); y++) {
            for (let x = 0; x < Math.min(IMG, rows[y].length); x++) {
                img[y * IMG + x] = (rows[y][x] === '#') ? 1.0 : 0.0;
            }
        }
        return img;
    }

    // 5 hand-drawn 12×12 prototypes, each pinned to a latent position
    const PROTOS = [
        { id: 'smiley', name: 'Smile', latent: [-0.65, 0.65], img: parsePattern(`
            ............
            ............
            ...##....##.
            ...##....##.
            ............
            ............
            ............
            .#........#.
            ..##....##..
            ....####....
            ............
            ............
        `) },
        { id: 'heart', name: 'Heart', latent: [0.65, 0.6], img: parsePattern(`
            ............
            ............
            ..##....##..
            .####..####.
            .##########.
            .##########.
            ..########..
            ...######...
            ....####....
            .....##.....
            ............
            ............
        `) },
        { id: 'cross', name: 'X', latent: [-0.65, -0.6], img: parsePattern(`
            ............
            .#........#.
            ..#......#..
            ...#....#...
            ....#..#....
            .....##.....
            .....##.....
            ....#..#....
            ...#....#...
            ..#......#..
            .#........#.
            ............
        `) },
        { id: 'circle', name: 'Circle', latent: [0.65, -0.65], img: parsePattern(`
            ............
            ....####....
            ..##....##..
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            ..##....##..
            ....####....
            ............
        `) },
        { id: 'square', name: 'Square', latent: [0.0, 0.0], img: parsePattern(`
            ............
            .##########.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .#........#.
            .##########.
            ............
        `) },
    ];

    // ----- State -----
    let activePreset = 0;
    let inputImg = PROTOS[0].img;
    let latent = [...PROTOS[0].latent];
    let outputImg = null;
    let vaeNoiseSamples = [];
    let dragging = false;
    let vaeMode  = false;

    // ----- Encode / Decode -----
    // Encoder: a soft nearest-prototype lookup in image space, projected onto
    // each prototype's latent coordinate. With one of the exact prototypes as
    // input, almost all weight lands on it → latent = its anchor coordinate.
    function encode(img) {
        const sims = PROTOS.map(p => {
            let d2 = 0;
            for (let i = 0; i < img.length; i++) {
                const d = img[i] - p.img[i];
                d2 += d * d;
            }
            return Math.exp(-d2 / 8);
        });
        const Z = sims.reduce((a, b) => a + b, 0) || 1;
        let lx = 0, ly = 0;
        PROTOS.forEach((p, i) => {
            const w = sims[i] / Z;
            lx += w * p.latent[0];
            ly += w * p.latent[1];
        });
        return [lx, ly];
    }

    // Decoder: softmax over distances in latent space, blended over prototypes.
    // This is essentially a Gaussian-RBF decoder and gives smooth interpolation.
    function decode(lx, ly) {
        const sims = PROTOS.map(p => {
            const dx = lx - p.latent[0];
            const dy = ly - p.latent[1];
            return Math.exp(-(dx * dx + dy * dy) / 0.35);
        });
        const Z = sims.reduce((a, b) => a + b, 0) || 1;
        const out = new Float32Array(IMG * IMG);
        PROTOS.forEach((p, i) => {
            const w = sims[i] / Z;
            for (let j = 0; j < out.length; j++) out[j] += w * p.img[j];
        });
        return out;
    }

    // VAE: encode → Gaussian with σ around μ, sample K times, average the
    // decodes. Visually, the output is slightly blurry but more robust to the
    // exact latent position — which is the whole point of the KL regularizer.
    function decodeVAE(lx, ly, sigma = 0.16, K = 10) {
        const out = new Float32Array(IMG * IMG);
        const samples = [];
        for (let k = 0; k < K; k++) {
            // Box-Muller
            const u1 = Math.random(), u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1 + 1e-9));
            const t = 2 * Math.PI * u2;
            const ex = r * Math.cos(t), ey = r * Math.sin(t);
            const sx = lx + sigma * ex, sy = ly + sigma * ey;
            samples.push([sx, sy]);
            const d = decode(sx, sy);
            for (let j = 0; j < out.length; j++) out[j] += d[j] / K;
        }
        return { img: out, samples };
    }

    function recompute() {
        if (vaeMode) {
            const r = decodeVAE(latent[0], latent[1]);
            outputImg = r.img;
            vaeNoiseSamples = r.samples;
        } else {
            outputImg = decode(latent[0], latent[1]);
            vaeNoiseSamples = [];
        }
    }

    function setActive(i) {
        activePreset = i;
        inputImg = PROTOS[i].img;
        latent = [...PROTOS[i].latent];
        recompute();
        updatePresetChips();
        updateCaption({ kind: 'preset', name: PROTOS[i].name });
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(480, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(320, cssW * 0.50)));
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
        const usableW = W - 2 * pad;
        const imgSize = Math.min(108, Math.max(76, H - 110));
        const neckW   = Math.max(56, Math.min(94, usableW * 0.10));
        let latentSide = Math.min(H - 60, usableW - 2 * imgSize - 2 * neckW - 40);
        latentSide = Math.max(180, latentSide);

        const totalW = imgSize + neckW + latentSide + neckW + imgSize;
        const slack = Math.max(0, usableW - totalW);
        const gap = slack / 4;

        let x = pad + gap;
        const input    = { x, y: (H - imgSize) / 2, w: imgSize, h: imgSize };
        x += imgSize + gap;
        const enc      = { x, y: (H - imgSize) / 2 - 6, w: neckW, h: imgSize + 12 };
        x += neckW + gap;
        const latentBox = { x, y: (H - latentSide) / 2, w: latentSide, h: latentSide };
        x += latentSide + gap;
        const dec      = { x, y: (H - imgSize) / 2 - 6, w: neckW, h: imgSize + 12 };
        x += neckW + gap;
        const output   = { x, y: (H - imgSize) / 2, w: imgSize, h: imgSize };

        return { input, enc, latentBox, dec, output };
    }

    // ----- Drawing -----
    function imgColour(t) {
        t = Math.max(0, Math.min(1, t));
        const r = 251 + (79  - 251) * t;
        const g = 250 + (70  - 250) * t;
        const b = 247 + (229 - 247) * t;
        return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    }

    function drawImageBox(box, img, label) {
        const cellW = box.w / IMG, cellH = box.h / IMG;
        for (let y = 0; y < IMG; y++) {
            for (let x = 0; x < IMG; x++) {
                ctx.fillStyle = imgColour(img[y * IMG + x]);
                // +0.6 to avoid seams between cells
                ctx.fillRect(box.x + x * cellW, box.y + y * cellH, cellW + 0.6, cellH + 0.6);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.14)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, box.x + box.w / 2, box.y - 8);
    }

    function drawNeck(box, direction) {
        // Trapezoidal "compression / expansion" shape
        const x0 = box.x, x1 = box.x + box.w;
        const cy = box.y + box.h / 2;
        const hWide = box.h * 0.95;
        const hMid  = box.h * 0.55;
        const hNarrow = box.h * 0.22;
        const h0 = direction === 'enc' ? hWide   : hNarrow;
        const h1 = direction === 'enc' ? hNarrow : hWide;

        // Subtle fill
        ctx.fillStyle = 'rgba(79, 70, 229, 0.07)';
        ctx.beginPath();
        ctx.moveTo(x0, cy - h0 / 2);
        ctx.lineTo(x1, cy - h1 / 2);
        ctx.lineTo(x1, cy + h1 / 2);
        ctx.lineTo(x0, cy + h0 / 2);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Layer dimensionality bars at three positions
        const heights = direction === 'enc'
            ? [hWide, hMid, hNarrow]
            : [hNarrow, hMid, hWide];
        const xs = [x0 + 3, x0 + box.w / 2, x1 - 3];
        ctx.fillStyle = 'rgba(79, 70, 229, 0.32)';
        for (let i = 0; i < 3; i++) {
            ctx.fillRect(xs[i] - 1.5, cy - heights[i] / 2, 3, heights[i]);
        }

        // Arrow indicating direction
        const ay = box.y + box.h + 6;
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.6)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x0 + 4, ay);
        ctx.lineTo(x1 - 4, ay);
        ctx.stroke();
        // Arrowhead
        const ax = x1 - 4;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - 4, ay - 3);
        ctx.lineTo(ax - 4, ay + 3);
        ctx.closePath();
        ctx.fillStyle = 'rgba(79, 70, 229, 0.6)';
        ctx.fill();

        // Label
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(direction === 'enc' ? 'encoder' : 'decoder',
            box.x + box.w / 2, box.y - 8);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.font = '500 8px "JetBrains Mono", monospace';
        ctx.fillText(
            direction === 'enc' ? '144 → 32 → 2' : '2 → 32 → 144',
            box.x + box.w / 2,
            box.y + box.h + 18,
        );
    }

    function latentToCanvas(lx, ly, box) {
        const m = Math.min(box.w, box.h) / 2 - 10;
        return {
            x: box.x + box.w / 2 + lx * m,
            y: box.y + box.h / 2 - ly * m,
        };
    }
    function canvasToLatent(cx, cy, box) {
        const m = Math.min(box.w, box.h) / 2 - 10;
        return [
            (cx - (box.x + box.w / 2)) / m,
            -(cy - (box.y + box.h / 2)) / m,
        ];
    }

    function drawLatent(box) {
        // Background and frame
        ctx.fillStyle = 'rgba(79, 70, 229, 0.04)';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // Axes
        const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(box.x + 6, cy);
        ctx.lineTo(box.x + box.w - 6, cy);
        ctx.moveTo(cx, box.y + 6);
        ctx.lineTo(cx, box.y + box.h - 6);
        ctx.stroke();
        ctx.setLineDash([]);

        // Axis tick labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText('z₁', box.x + box.w - 6, cy - 5);
        ctx.textAlign = 'left';
        ctx.fillText('z₂', cx + 5, box.y + 12);

        // VAE noise samples
        if (vaeMode && vaeNoiseSamples.length) {
            ctx.fillStyle = 'rgba(234, 121, 89, 0.55)';
            for (const [sx, sy] of vaeNoiseSamples) {
                const p = latentToCanvas(sx, sy, box);
                ctx.beginPath();
                ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Prototype anchor points
        for (let i = 0; i < PROTOS.length; i++) {
            const p = latentToCanvas(PROTOS[i].latent[0], PROTOS[i].latent[1], box);
            const active = (i === activePreset);
            ctx.fillStyle = active ? '#ea7959' : 'rgba(79, 70, 229, 0.7)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, active ? 4.5 : 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = active ? '#ea7959' : 'rgba(0, 0, 0, 0.55)';
            ctx.font = active
                ? '600 10px "Inter", system-ui, sans-serif'
                : '500 10px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(PROTOS[i].name, p.x, p.y - 9);
        }

        // Current latent cursor (draggable)
        const lp = latentToCanvas(latent[0], latent[1], box);
        ctx.fillStyle = 'rgba(234, 121, 89, 0.2)';
        ctx.beginPath();
        ctx.arc(lp.x, lp.y, 13, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ea7959';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(lp.x, lp.y, 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('LATENT  z ∈ ℝ²', cx, box.y - 8);
        // Display current z value
        ctx.font = '500 8px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillText(
            `z = (${latent[0].toFixed(2)}, ${latent[1].toFixed(2)})`,
            cx, box.y + box.h + 14,
        );
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const lay = layout();
        drawImageBox(lay.input, inputImg, 'INPUT');
        drawNeck(lay.enc, 'enc');
        drawLatent(lay.latentBox);
        drawNeck(lay.dec, 'dec');
        drawImageBox(lay.output, outputImg, 'RECONSTRUCTION');
    }

    // ----- Caption -----
    function updateCaption(ev = {}) {
        if (!captionEl) return;
        if (ev.kind === 'preset') {
            captionEl.innerHTML = vaeMode
                ? `Encoded <strong>${ev.name}</strong> as a Gaussian (μ, σ). Orange dots show samples ` +
                  `drawn near μ — the decoder averages over them, so the reconstruction is smoother and ` +
                  `more robust to small latent perturbations.`
                : `Encoded <strong>${ev.name}</strong> down to a 2-number code, then decoded it back. ` +
                  `Drag the orange cursor anywhere in the latent square to see how the reconstruction ` +
                  `morphs between prototypes.`;
        } else if (ev.kind === 'drag') {
            captionEl.innerHTML =
                `Latent <strong>z = (${latent[0].toFixed(2)}, ${latent[1].toFixed(2)})</strong> — ` +
                `between the prototype anchors. The decoder interpolates: midway between two shapes, ` +
                `you get a blend of both. This is the same trick that lets a VAE walk smoothly between ` +
                `faces or digits.`;
        } else {
            captionEl.innerHTML =
                `An autoencoder squeezes each image down to a small <strong>latent code</strong>, then ` +
                `tries to reconstruct it. Here the code is only 2D — small enough to draw. ` +
                `Pick an image, then drag the cursor to explore.`;
        }
    }

    // ----- Interactions -----
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }
    function inBox(x, y, box) {
        return x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h;
    }

    function onDown(e) {
        const [x, y] = getPos(e);
        const lay = layout();
        if (!inBox(x, y, lay.latentBox)) return;
        dragging = true;
        const [lx, ly] = canvasToLatent(x, y, lay.latentBox);
        // Snap to a prototype if very close
        let near = -1, nearD = Infinity;
        for (let i = 0; i < PROTOS.length; i++) {
            const d = Math.hypot(lx - PROTOS[i].latent[0], ly - PROTOS[i].latent[1]);
            if (d < nearD) { nearD = d; near = i; }
        }
        if (nearD < 0.07) {
            setActive(near);
            return;
        }
        latent = [Math.max(-1, Math.min(1, lx)), Math.max(-1, Math.min(1, ly))];
        // No longer matches any specific preset — clear active highlight
        activePreset = -1;
        updatePresetChips();
        recompute();
        updateCaption({ kind: 'drag' });
        draw();
    }
    function onMove(e) {
        if (!dragging) return;
        const [x, y] = getPos(e);
        const lay = layout();
        const [lx, ly] = canvasToLatent(x, y, lay.latentBox);
        latent = [Math.max(-1, Math.min(1, lx)), Math.max(-1, Math.min(1, ly))];
        activePreset = -1;
        updatePresetChips();
        recompute();
        updateCaption({ kind: 'drag' });
        draw();
    }
    function onUp() { dragging = false; }

    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); onDown(e); }, { passive: false });
    canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); onMove(e); }, { passive: false });
    canvas.addEventListener('touchend',   onUp);

    // ----- Preset chips -----
    function updatePresetChips() {
        if (!presetWrap) return;
        for (const btn of presetWrap.querySelectorAll('.viz-ae-chip')) {
            btn.classList.toggle('active', +btn.dataset.idx === activePreset);
        }
    }
    if (presetWrap) {
        presetWrap.innerHTML = '';
        PROTOS.forEach((p, i) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'viz-ae-chip';
            btn.textContent = p.name;
            btn.dataset.idx = i;
            btn.addEventListener('click', () => setActive(i));
            presetWrap.appendChild(btn);
        });
    }

    if (vaeToggle) {
        vaeToggle.addEventListener('click', () => {
            vaeMode = !vaeMode;
            vaeToggle.classList.toggle('active', vaeMode);
            vaeToggle.textContent = vaeMode ? 'VAE mode: on' : 'VAE mode: off';
            recompute();
            updateCaption({
                kind: 'preset',
                name: activePreset >= 0 ? PROTOS[activePreset].name : 'the current code',
            });
            draw();
        });
    }

    if (resetBtn) resetBtn.addEventListener('click', () => setActive(0));

    // ----- Init -----
    setActive(0);
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();

    // Continuous re-sample in VAE mode so the noise dots animate
    let lastSample = 0;
    function loop(now) {
        if (vaeMode && now - lastSample > 450) {
            recompute();
            draw();
            lastSample = now;
        }
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
})();
