/* Interactive CV augmentation viz.
 * A simple 28×28 source image; toggle classic augmentations and see the
 * transformed result side-by-side with the original. */

(function () {
    const canvas    = document.getElementById('viz-cv-canvas');
    const origBtn   = document.getElementById('viz-cv-orig');
    const flipBtn   = document.getElementById('viz-cv-flip');
    const cropBtn   = document.getElementById('viz-cv-crop');
    const rotBtn    = document.getElementById('viz-cv-rotate');
    const noiseBtn  = document.getElementById('viz-cv-noise');
    const blurBtn   = document.getElementById('viz-cv-blur');
    const captionEl = document.getElementById('viz-cv-caption');
    if (!canvas) return;

    let mode = 'orig';
    let ctx;
    let W = 0, H = 0;

    const SIZE = 28;

    // Hand-drawn "digit 5"-ish pattern as a source image
    function makeImage() {
        const out = new Float32Array(SIZE * SIZE);
        function parse(rows) {
            for (let y = 0; y < rows.length; y++) {
                for (let x = 0; x < rows[y].length; x++) {
                    out[y * SIZE + x] = (rows[y][x] === '#') ? 1 : 0;
                }
            }
        }
        parse([
            '............................',
            '............................',
            '............................',
            '......################......',
            '......################......',
            '......################......',
            '......##....................',
            '......##....................',
            '......##....................',
            '......##....................',
            '......##....................',
            '......##############........',
            '......################......',
            '......##################....',
            '......####..........####....',
            '......................####..',
            '..............##......####..',
            '..............####....####..',
            '..............####....####..',
            '......####....####....####..',
            '......######..########......',
            '......######..######........',
            '........###############.....',
            '..........############......',
            '............#########.......',
            '............................',
            '............................',
            '............................',
        ]);
        return out;
    }

    const SOURCE = makeImage();

    function transform(src, mode) {
        const out = new Float32Array(SIZE * SIZE);
        if (mode === 'orig') {
            for (let i = 0; i < src.length; i++) out[i] = src[i];
        } else if (mode === 'flip') {
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    out[y * SIZE + x] = src[y * SIZE + (SIZE - 1 - x)];
                }
            }
        } else if (mode === 'crop') {
            const cx = 6 + Math.floor(Math.random() * 8) - 4;
            const cy = 6 + Math.floor(Math.random() * 8) - 4;
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    const sx = x + cx, sy = y + cy;
                    if (sx >= 0 && sx < SIZE && sy >= 0 && sy < SIZE) {
                        out[y * SIZE + x] = src[sy * SIZE + sx];
                    }
                }
            }
        } else if (mode === 'rotate') {
            const a = Math.PI / 6;        // 30°
            const c = Math.cos(a), s = Math.sin(a);
            const cx = SIZE / 2, cy = SIZE / 2;
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    const dx = x - cx, dy = y - cy;
                    const sx = Math.round( c * dx + s * dy + cx);
                    const sy = Math.round(-s * dx + c * dy + cy);
                    if (sx >= 0 && sx < SIZE && sy >= 0 && sy < SIZE) {
                        out[y * SIZE + x] = src[sy * SIZE + sx];
                    }
                }
            }
        } else if (mode === 'noise') {
            for (let i = 0; i < src.length; i++) {
                out[i] = Math.max(0, Math.min(1, src[i] + 0.4 * (Math.random() * 2 - 1)));
            }
        } else if (mode === 'blur') {
            const ksz = 1;
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    let s = 0, n = 0;
                    for (let dy = -ksz; dy <= ksz; dy++) {
                        for (let dx = -ksz; dx <= ksz; dx++) {
                            const yy = y + dy, xx = x + dx;
                            if (xx >= 0 && xx < SIZE && yy >= 0 && yy < SIZE) {
                                s += src[yy * SIZE + xx]; n++;
                            }
                        }
                    }
                    out[y * SIZE + x] = s / n;
                }
            }
        }
        return out;
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(280, cssW * 0.42)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function imgColour(t) {
        t = Math.max(0, Math.min(1, t));
        const r = 251 + (79  - 251) * t;
        const g = 250 + (70  - 250) * t;
        const b = 247 + (229 - 247) * t;
        return `rgb(${r|0}, ${g|0}, ${b|0})`;
    }

    function drawImage(img, box, label) {
        const cellW = box.w / SIZE, cellH = box.h / SIZE;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE; x++) {
                ctx.fillStyle = imgColour(img[y * SIZE + x]);
                ctx.fillRect(box.x + x * cellW, box.y + y * cellH, cellW + 0.5, cellH + 0.5);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.14)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, box.x + box.w / 2, box.y - 8);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 20;
        const size = Math.min(W / 2 - 2 * pad, H - 80);
        const left  = { x: (W / 2 - size) / 2 + 10, y: 30, w: size, h: size };
        const right = { x: W / 2 + (W / 2 - size) / 2 - 10, y: 30, w: size, h: size };

        const transformed = transform(SOURCE, mode);
        drawImage(SOURCE, left, 'ORIGINAL');
        drawImage(transformed, right, mode.toUpperCase());

        // Arrow
        const ax = W / 2 - 10, ay = 30 + size / 2;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(left.x + size + 6, ay);
        ctx.lineTo(right.x - 6, ay);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(right.x - 6, ay);
        ctx.lineTo(right.x - 14, ay - 5);
        ctx.lineTo(right.x - 14, ay + 5);
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fill();

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            orig:   `Source image. Try each augmentation — they all preserve "what" the image shows while changing pixel-level details. The model should learn to be invariant to these.`,
            flip:   `Horizontal flip. Free augmentation when left-right symmetry holds (faces, scenes). Watch out for text or asymmetric scenes.`,
            crop:   `Random crop. The most important single augmentation — forces the model to recognise objects regardless of position. Modern recipes crop and resize aggressively.`,
            rotate: `Rotation. Helpful when orientation isn't part of the label (medical imaging) — harmful when it is (digits, text).`,
            noise:  `Additive Gaussian noise. Helps with sensor robustness and acts as a regulariser. Strong noise during training often improves test-time robustness.`,
            blur:   `Gaussian blur. Augmentation against motion / focus blur. Also a common adversarial training defence.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [origBtn, flipBtn, cropBtn, rotBtn, noiseBtn, blurBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    origBtn?.addEventListener('click',  () => setMode('orig', origBtn));
    flipBtn?.addEventListener('click',  () => setMode('flip', flipBtn));
    cropBtn?.addEventListener('click',  () => setMode('crop', cropBtn));
    rotBtn?.addEventListener('click',   () => setMode('rotate', rotBtn));
    noiseBtn?.addEventListener('click', () => setMode('noise', noiseBtn));
    blurBtn?.addEventListener('click',  () => setMode('blur', blurBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
