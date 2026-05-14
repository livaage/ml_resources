/* Interactive CNN convolution viz.
 * Drag the filter on the input to see the feature map compute live.
 * Switch image / kernel presets; auto-slide animates the full pass. */

(function () {
    const inputCanvas  = document.getElementById('viz-cnn-input');
    const outputCanvas = document.getElementById('viz-cnn-output');
    const kernelCanvas = document.getElementById('viz-cnn-kernel');
    const mathEl       = document.getElementById('viz-cnn-math');
    if (!inputCanvas || !outputCanvas || !kernelCanvas) return;

    // ----- Constants -----
    const SIZE  = 24;          // input image is SIZE × SIZE pixels
    const KSIZE = 3;           // kernel is KSIZE × KSIZE
    const OUT_SIZE = SIZE - KSIZE + 1;   // valid padding
    const CELL_IN  = 13;       // pixels per input cell on canvas
    const CELL_OUT = 14;       // pixels per output cell on canvas
    const CELL_K   = 36;       // pixels per kernel cell

    // ----- Image presets -----
    // Each is a SIZE × SIZE array of values in [0, 1].
    function makeImage(fn) {
        const a = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < SIZE; i++)
            for (let j = 0; j < SIZE; j++)
                a[i * SIZE + j] = Math.max(0, Math.min(1, fn(i, j)));
        return a;
    }

    const IMAGES = {
        'vertical edge': makeImage((r, c) => c < SIZE / 2 ? 0.1 : 0.9),
        'cross':         makeImage((r, c) => {
            const mid = (SIZE - 1) / 2;
            const onV = Math.abs(c - mid) < 2;
            const onH = Math.abs(r - mid) < 2;
            return (onV || onH) ? 0.92 : 0.08;
        }),
        'circle':        makeImage((r, c) => {
            const cx = (SIZE - 1) / 2, cy = (SIZE - 1) / 2;
            const d = Math.hypot(c - cx, r - cy);
            const R = SIZE * 0.32;
            return Math.abs(d - R) < 1.5 ? 0.92 : 0.08;
        }),
        'digit 5':       makeImage((r, c) => {
            // hand-shaped 5: top bar, left vertical (upper half), middle bar,
            // right vertical (lower half), bottom bar
            const top    = r >= 4  && r <= 6  && c >= 6  && c <= 16;
            const leftV  = r >= 6  && r <= 12 && c >= 6  && c <= 8;
            const midBar = r >= 11 && r <= 13 && c >= 6  && c <= 16;
            const rightV = r >= 12 && r <= 18 && c >= 14 && c <= 16;
            const botBar = r >= 17 && r <= 19 && c >= 6  && c <= 16;
            return (top || leftV || midBar || rightV || botBar) ? 0.95 : 0.08;
        }),
        'gradient':      makeImage((r, c) => (r + c) / (2 * SIZE - 2)),
    };

    // ----- Kernel presets -----
    const KERNELS = {
        'identity':      [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        'edge — Sobel x':[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        'edge — Sobel y':[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        'blur (box)':    [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
        'sharpen':       [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        'emboss':        [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    };

    // ----- State -----
    let image  = IMAGES['vertical edge'];
    let kernel = KERNELS['edge — Sobel x'];
    let output = new Float32Array(OUT_SIZE * OUT_SIZE);
    let outMin = 0, outMax = 0;

    let activeRow = 0;        // top-left row of filter on input
    let activeCol = 0;        // top-left col
    let dragging  = false;
    let playing   = true;
    let lastStep  = 0;
    const STEP_MS = 70;

    // ----- Convolution -----
    function convolve() {
        let mn = Infinity, mx = -Infinity;
        for (let or = 0; or < OUT_SIZE; or++) {
            for (let oc = 0; oc < OUT_SIZE; oc++) {
                let s = 0;
                for (let kr = 0; kr < KSIZE; kr++)
                    for (let kc = 0; kc < KSIZE; kc++)
                        s += image[(or + kr) * SIZE + (oc + kc)] * kernel[kr][kc];
                output[or * OUT_SIZE + oc] = s;
                if (s < mn) mn = s;
                if (s > mx) mx = s;
            }
        }
        outMin = mn;
        outMax = mx;
    }

    // ----- Colour ramps -----
    // Input: black-to-white grayscale (light bg → dark = high)
    function inputColour(v) {
        const g = Math.round(255 * (1 - v));    // dark = high intensity
        return `rgb(${g}, ${g}, ${g})`;
    }
    // Output: diverging — blue (negative) → cream (zero) → red/orange (positive)
    function outputColour(v) {
        const lim = Math.max(Math.abs(outMin), Math.abs(outMax), 1e-6);
        const t   = v / lim;                     // [-1, 1]
        if (t >= 0) {
            // cream → orange
            const r = 255, g = Math.round(255 - 95 * t), b = Math.round(247 - 187 * t);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // cream → indigo
            const a = -t;
            const r = Math.round(251 - 173 * a), g = Math.round(250 - 188 * a), b = Math.round(247 - 78 * a);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    // ----- Drawing -----
    function drawInput() {
        const c = inputCanvas.getContext('2d');
        const w = SIZE * CELL_IN, h = SIZE * CELL_IN;
        const dpr = window.devicePixelRatio || 1;
        if (inputCanvas.width !== w * dpr) {
            inputCanvas.width  = w * dpr;
            inputCanvas.height = h * dpr;
            inputCanvas.style.width  = w + 'px';
            inputCanvas.style.height = h + 'px';
            c.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        c.clearRect(0, 0, w, h);
        // Pixels
        for (let i = 0; i < SIZE; i++) {
            for (let j = 0; j < SIZE; j++) {
                c.fillStyle = inputColour(image[i * SIZE + j]);
                c.fillRect(j * CELL_IN, i * CELL_IN, CELL_IN, CELL_IN);
            }
        }
        // Subtle grid
        c.strokeStyle = 'rgba(0,0,0,0.06)';
        c.lineWidth = 1;
        for (let i = 0; i <= SIZE; i++) {
            c.beginPath();
            c.moveTo(0, i * CELL_IN); c.lineTo(w, i * CELL_IN); c.stroke();
            c.beginPath();
            c.moveTo(i * CELL_IN, 0); c.lineTo(i * CELL_IN, h); c.stroke();
        }
        // Filter overlay
        const fx = activeCol * CELL_IN;
        const fy = activeRow * CELL_IN;
        const fs = KSIZE * CELL_IN;
        // Outer outline
        c.strokeStyle = '#ea7959';
        c.lineWidth = 2.5;
        c.strokeRect(fx + 1, fy + 1, fs - 2, fs - 2);
        // Inner soft tint
        c.fillStyle = 'rgba(234, 121, 89, 0.10)';
        c.fillRect(fx + 1, fy + 1, fs - 2, fs - 2);
    }

    function drawOutput() {
        const c = outputCanvas.getContext('2d');
        const w = OUT_SIZE * CELL_OUT, h = OUT_SIZE * CELL_OUT;
        const dpr = window.devicePixelRatio || 1;
        if (outputCanvas.width !== w * dpr) {
            outputCanvas.width  = w * dpr;
            outputCanvas.height = h * dpr;
            outputCanvas.style.width  = w + 'px';
            outputCanvas.style.height = h + 'px';
            c.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        c.clearRect(0, 0, w, h);
        for (let i = 0; i < OUT_SIZE; i++) {
            for (let j = 0; j < OUT_SIZE; j++) {
                c.fillStyle = outputColour(output[i * OUT_SIZE + j]);
                c.fillRect(j * CELL_OUT, i * CELL_OUT, CELL_OUT, CELL_OUT);
            }
        }
        c.strokeStyle = 'rgba(0,0,0,0.06)';
        c.lineWidth = 1;
        for (let i = 0; i <= OUT_SIZE; i++) {
            c.beginPath();
            c.moveTo(0, i * CELL_OUT); c.lineTo(w, i * CELL_OUT); c.stroke();
            c.beginPath();
            c.moveTo(i * CELL_OUT, 0); c.lineTo(i * CELL_OUT, h); c.stroke();
        }
        // Highlight the active output cell
        const ox = activeCol * CELL_OUT;
        const oy = activeRow * CELL_OUT;
        c.strokeStyle = '#ea7959';
        c.lineWidth = 2.5;
        c.strokeRect(ox + 1, oy + 1, CELL_OUT - 2, CELL_OUT - 2);
    }

    function drawKernel() {
        const c = kernelCanvas.getContext('2d');
        const w = KSIZE * CELL_K, h = KSIZE * CELL_K;
        const dpr = window.devicePixelRatio || 1;
        if (kernelCanvas.width !== w * dpr) {
            kernelCanvas.width  = w * dpr;
            kernelCanvas.height = h * dpr;
            kernelCanvas.style.width  = w + 'px';
            kernelCanvas.style.height = h + 'px';
            c.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        c.clearRect(0, 0, w, h);
        // Background per cell shows sign
        for (let i = 0; i < KSIZE; i++) {
            for (let j = 0; j < KSIZE; j++) {
                const v = kernel[i][j];
                const a = Math.min(1, Math.abs(v));
                c.fillStyle = v >= 0
                    ? `rgba(255, 160, 68, ${0.18 + a * 0.5})`
                    : `rgba(79, 70, 229, ${0.18 + a * 0.5})`;
                c.fillRect(j * CELL_K, i * CELL_K, CELL_K, CELL_K);
                c.strokeStyle = 'rgba(0, 0, 0, 0.18)';
                c.lineWidth = 1;
                c.strokeRect(j * CELL_K + 0.5, i * CELL_K + 0.5, CELL_K - 1, CELL_K - 1);
                // Value text
                c.fillStyle = '#1a1a1a';
                c.font = '600 13px "Inter", system-ui, sans-serif';
                c.textAlign = 'center';
                c.textBaseline = 'middle';
                const txt = formatKernelVal(v);
                c.fillText(txt, j * CELL_K + CELL_K / 2, i * CELL_K + CELL_K / 2);
            }
        }
    }

    function formatKernelVal(v) {
        if (Math.abs(v - Math.round(v)) < 0.001) return v.toFixed(0);
        return v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '.0');
    }

    function drawMath() {
        if (!mathEl) return;
        // Sum is output[activeRow][activeCol] — also assemble the explicit terms
        const terms = [];
        let total = 0;
        for (let kr = 0; kr < KSIZE; kr++) {
            for (let kc = 0; kc < KSIZE; kc++) {
                const pix = image[(activeRow + kr) * SIZE + (activeCol + kc)];
                const k   = kernel[kr][kc];
                total    += pix * k;
                if (Math.abs(k) > 0.001) {
                    terms.push(`${formatKernelVal(k)}·${pix.toFixed(2)}`);
                }
            }
        }
        const expr = terms.length === 0 ? '0' : terms.join(' + ');
        mathEl.innerHTML = `
            <span class="lbl">Output (${activeRow}, ${activeCol})</span>
            ${expr} = <span class="sum">${total.toFixed(2)}</span>
        `;
    }

    function redraw() {
        drawInput();
        drawOutput();
        drawKernel();
        drawMath();
    }

    // ----- Interaction -----
    function setPos(r, c) {
        activeRow = Math.max(0, Math.min(OUT_SIZE - 1, r));
        activeCol = Math.max(0, Math.min(OUT_SIZE - 1, c));
    }

    function inputPosFromEvent(e) {
        const rect = inputCanvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        const x = t.clientX - rect.left;
        const y = t.clientY - rect.top;
        const col = Math.floor(x / CELL_IN) - 1;   // center the filter on the click
        const row = Math.floor(y / CELL_IN) - 1;
        return [row, col];
    }

    inputCanvas.addEventListener('mousedown',  (e) => {
        e.preventDefault();
        dragging = true;
        playing  = false;
        updatePlayButton();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('mousemove',  (e) => {
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
        updatePlayButton();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('touchmove',  (e) => {
        if (!dragging) return;
        e.preventDefault();
        const [r, c] = inputPosFromEvent(e);
        setPos(r, c);
        redraw();
    });
    inputCanvas.addEventListener('touchend',   () => { dragging = false; });

    // Click the kernel to cycle a cell's value through {-2, -1, 0, 1, 2}
    kernelCanvas.addEventListener('click', (e) => {
        const rect = kernelCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const c = Math.floor(x / CELL_K);
        const r = Math.floor(y / CELL_K);
        if (r < 0 || r >= KSIZE || c < 0 || c >= KSIZE) return;
        const cycle = [-2, -1, 0, 1, 2];
        // Find the current value's index (use nearest), advance
        const v = kernel[r][c];
        let idx = 0, bestD = Infinity;
        for (let i = 0; i < cycle.length; i++) {
            const d = Math.abs(v - cycle[i]);
            if (d < bestD) { bestD = d; idx = i; }
        }
        idx = (idx + 1) % cycle.length;
        // Mutate the kernel (deep-copy first so we don't trash the preset)
        kernel = kernel.map(row => row.slice());
        kernel[r][c] = cycle[idx];
        convolve();
        redraw();
    });

    // ----- Dropdowns -----
    function populateSelect(el, options, current) {
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
    if (imageSel) {
        populateSelect(imageSel, IMAGES, 'vertical edge');
        imageSel.addEventListener('change', () => {
            image = IMAGES[imageSel.value];
            convolve();
            redraw();
        });
    }
    if (kernelSel) {
        populateSelect(kernelSel, KERNELS, 'edge — Sobel x');
        kernelSel.addEventListener('change', () => {
            kernel = KERNELS[kernelSel.value].map(r => r.slice());
            convolve();
            redraw();
        });
    }

    // ----- Play/Pause + animation -----
    const playBtn = document.getElementById('viz-cnn-play');
    function updatePlayButton() {
        if (playBtn) playBtn.textContent = playing ? 'Pause' : 'Auto';
    }
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            playing = !playing;
            updatePlayButton();
            if (playing) lastStep = performance.now();
        });
    }

    function loop(now) {
        if (playing && now - lastStep >= STEP_MS && !dragging) {
            // Advance filter through the feature-map cells row by row
            let r = activeRow, c = activeCol + 1;
            if (c >= OUT_SIZE) { c = 0; r += 1; }
            if (r >= OUT_SIZE) { r = 0; c = 0; }
            setPos(r, c);
            redraw();
            lastStep = now;
        }
        requestAnimationFrame(loop);
    }

    // ----- Init -----
    convolve();
    redraw();
    updatePlayButton();

    // ResizeObserver — if the tab is hidden at script run, redraw when it appears.
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => redraw());
        ro.observe(inputCanvas);
    }

    requestAnimationFrame(loop);
})();
