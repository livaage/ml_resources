/* Interactive DNN forward-pass viz.
 * 2 inputs → 4 hidden (ReLU) → 1 output (sigmoid).
 * Each hidden + output neuron is shown as a mini 2D heatmap of its activation
 * across the input space. Click anywhere to set the current input point. */

(function () {
    const canvas    = document.getElementById('viz-dnn-canvas');
    const presetSel = document.getElementById('viz-dnn-preset');
    const mathEl    = document.getElementById('viz-dnn-math');
    if (!canvas) return;

    // ----- Pre-set weight configurations -----
    // Each preset is hand-tuned to produce an interesting decision pattern.
    // W1[i][j] = weight from input i to hidden j, b1[j] = bias of hidden j
    // W2[j]   = weight from hidden j to output, b2 = output bias
    const PRESETS = {
        'XOR-like': {
            W1: [[ 2.5, 2.5, -2.5, -2.5],
                 [ 2.5, -2.5, 2.5, -2.5]],
            b1: [-0.8, -0.8, -0.8, -0.8],
            W2: [ 1.8, -1.8, -1.8,  1.8],
            b2: -0.2,
        },
        'central blob': {
            W1: [[ 3.0, -3.0,  0.0,  0.0],
                 [ 0.0,  0.0,  3.0, -3.0]],
            b1: [ 1.5,  1.5,  1.5,  1.5],
            W2: [ 1.5,  1.5,  1.5,  1.5],
            b2: -3.0,
        },
        'diagonal stripe': {
            W1: [[ 3.0, -3.0,  1.5, -1.5],
                 [ 3.0, -3.0, -1.5,  1.5]],
            b1: [-0.5, -0.5, -0.5, -0.5],
            W2: [ 2.0,  2.0, -1.5, -1.5],
            b2: -1.0,
        },
    };
    let preset = 'XOR-like';
    let W1 = PRESETS[preset].W1, b1 = PRESETS[preset].b1;
    let W2 = PRESETS[preset].W2, b2 = PRESETS[preset].b2;

    // ----- State -----
    let pt = { x: 0.4, y: 0.4 };        // current input point in [-1, 1]²
    let ctx;
    let W = 0, H = 0;

    // ----- Forward pass -----
    function relu(z) { return Math.max(0, z); }
    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function forward(x1, x2) {
        const h_pre  = new Float32Array(4);
        const h_post = new Float32Array(4);
        for (let j = 0; j < 4; j++) {
            h_pre[j]  = W1[0][j] * x1 + W1[1][j] * x2 + b1[j];
            h_post[j] = relu(h_pre[j]);
        }
        let z = b2;
        for (let j = 0; j < 4; j++) z += W2[j] * h_post[j];
        return { h_pre, h_post, z, y: sigmoid(z) };
    }

    // Activation at a 2D point — used for the mini-heatmaps
    function hiddenAt(j, x1, x2) {
        return relu(W1[0][j] * x1 + W1[1][j] * x2 + b1[j]);
    }
    function outputAt(x1, x2) {
        let z = b2;
        for (let j = 0; j < 4; j++) {
            z += W2[j] * relu(W1[0][j] * x1 + W1[1][j] * x2 + b1[j]);
        }
        return sigmoid(z);
    }

    // ----- Layout -----
    // Three columns: inputs (left), hidden (middle), output (right)
    function geom() {
        const pad = 18;
        const colsX = [W * 0.10, W * 0.50, W * 0.86];
        // Input nodes (just two circles)
        const inY = [H * 0.35, H * 0.65];
        // Hidden: 4 mini-heatmaps spaced vertically
        const hY = [H * 0.14, H * 0.38, H * 0.62, H * 0.86];
        const outY = H * 0.50;
        const miniR = Math.min(38, H * 0.13);     // heatmap "radius" (half side)
        const inR   = 14;
        const outR  = Math.min(58, H * 0.22);
        return { pad, colsX, inY, hY, outY, miniR, inR, outR };
    }

    // ----- Heatmap canvases (cached) -----
    // Each hidden neuron + the output has its own pre-rendered heatmap. We
    // recompute when the preset changes (weights stable thereafter).
    const HEAT_RES = 32;       // resolution of the cached heatmap
    let hiddenHeats = [null, null, null, null];
    let outputHeat  = null;

    function buildHeatmaps() {
        // Hidden — separate normalisation per neuron so the pattern is visible
        for (let j = 0; j < 4; j++) {
            const grid = new Float32Array(HEAT_RES * HEAT_RES);
            let mx = 0;
            for (let r = 0; r < HEAT_RES; r++) {
                const y = 1 - 2 * r / (HEAT_RES - 1);
                for (let c = 0; c < HEAT_RES; c++) {
                    const x = -1 + 2 * c / (HEAT_RES - 1);
                    const v = hiddenAt(j, x, y);
                    grid[r * HEAT_RES + c] = v;
                    if (v > mx) mx = v;
                }
            }
            hiddenHeats[j] = { grid, max: mx || 1 };
        }
        // Output is sigmoid, already in [0, 1]
        const grid = new Float32Array(HEAT_RES * HEAT_RES);
        for (let r = 0; r < HEAT_RES; r++) {
            const y = 1 - 2 * r / (HEAT_RES - 1);
            for (let c = 0; c < HEAT_RES; c++) {
                const x = -1 + 2 * c / (HEAT_RES - 1);
                grid[r * HEAT_RES + c] = outputAt(x, y);
            }
        }
        outputHeat = grid;
    }

    // ----- Colour ramps -----
    function hiddenColour(t) {
        // Cream → indigo
        const r = 251 + (79 - 251) * t;
        const g = 250 + (70 - 250) * t;
        const b = 247 + (229 - 247) * t;
        return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    }
    function outputColour(t) {
        // Diverging — indigo (<0.5) ↔ cream (0.5) ↔ orange (>0.5)
        if (t >= 0.5) {
            const u = (t - 0.5) * 2;
            const r = 251, g = 250 - 90 * u, b = 247 - 180 * u;
            return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
        } else {
            const u = (0.5 - t) * 2;
            const r = 251 - 170 * u, g = 250 - 185 * u, b = 247 - 80 * u;
            return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
        }
    }

    // ----- Drawing helpers -----
    function drawMiniHeatmap(cx, cy, size, grid, colour, marker) {
        const half = size / 2;
        const cellSize = size / HEAT_RES;
        // Clip to square
        ctx.save();
        roundedRect(ctx, cx - half, cy - half, size, size, 6);
        ctx.clip();
        for (let r = 0; r < HEAT_RES; r++) {
            for (let c = 0; c < HEAT_RES; c++) {
                ctx.fillStyle = colour(grid[r * HEAT_RES + c]);
                ctx.fillRect(cx - half + c * cellSize,
                             cy - half + r * cellSize,
                             cellSize + 0.5, cellSize + 0.5);
            }
        }
        ctx.restore();
        // Border
        ctx.strokeStyle = 'rgba(0,0,0,0.18)';
        ctx.lineWidth = 1;
        roundedRect(ctx, cx - half, cy - half, size, size, 6);
        ctx.stroke();
        // Marker for current point
        if (marker) {
            const mx = cx - half + (pt.x + 1) / 2 * size;
            const my = cy - half + (1 - pt.y) / 2 * size;
            // Halo
            ctx.fillStyle = 'rgba(234, 121, 89, 0.35)';
            ctx.beginPath();
            ctx.arc(mx, my, 6, 0, Math.PI * 2);
            ctx.fill();
            // Centre
            ctx.fillStyle = '#ea7959';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(mx, my, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }
    }

    function roundedRect(c, x, y, w, h, r) {
        c.beginPath();
        c.moveTo(x + r, y);
        c.arcTo(x + w, y,     x + w, y + h, r);
        c.arcTo(x + w, y + h, x,     y + h, r);
        c.arcTo(x,     y + h, x,     y,     r);
        c.arcTo(x,     y,     x + w, y,     r);
        c.closePath();
    }

    // ----- Draw the whole network -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const g = geom();
        const fp = forward(pt.x, pt.y);

        // Column labels
        ctx.fillStyle = '#9a9a9a';
        ctx.font      = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('INPUTS',  g.colsX[0], 8);
        ctx.fillText('HIDDEN (ReLU)', g.colsX[1], 8);
        ctx.fillText('OUTPUT (sigmoid)', g.colsX[2], 8);

        // Connections: inputs → hidden
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 4; j++) {
                drawWeightLine(g.colsX[0] + g.inR, g.inY[i],
                               g.colsX[1] - g.miniR, g.hY[j],
                               W1[i][j], 2.6);
            }
        }
        // Connections: hidden → output
        for (let j = 0; j < 4; j++) {
            drawWeightLine(g.colsX[1] + g.miniR, g.hY[j],
                           g.colsX[2] - g.outR, g.outY,
                           W2[j], 2.6);
        }

        // Input nodes (filled circles labelled x1, x2 with value)
        for (let i = 0; i < 2; i++) {
            const x_val = i === 0 ? pt.x : pt.y;
            const t = (x_val + 1) / 2;
            ctx.fillStyle = hiddenColour(t * 0.9);   // 0.9 to avoid pure indigo
            ctx.strokeStyle = '#5f5f5f';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(g.colsX[0], g.inY[i], g.inR, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            // Label
            ctx.fillStyle = '#1a1a1a';
            ctx.font      = '600 11px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(i === 0 ? 'x₁' : 'x₂', g.colsX[0] - g.inR - 4, g.inY[i]);
            // Value
            ctx.fillStyle = '#5f5f5f';
            ctx.font      = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText(x_val.toFixed(2), g.colsX[0] + g.inR + 4, g.inY[i]);
        }

        // Hidden mini-heatmaps
        for (let j = 0; j < 4; j++) {
            const heat = hiddenHeats[j];
            drawMiniHeatmap(g.colsX[1], g.hY[j], g.miniR * 2,
                            new Float32Array(heat.grid).map(v => v / heat.max),
                            hiddenColour, true);
            // Activation value chip
            const v = fp.h_post[j];
            ctx.fillStyle = v > 0 ? 'rgba(79, 70, 229, 0.85)' : 'rgba(154, 154, 154, 0.85)';
            const chipW = 34, chipH = 18;
            roundedRect(ctx,
                g.colsX[1] - chipW / 2,
                g.hY[j] + g.miniR + 3,
                chipW, chipH, 9);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.font      = '600 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(v.toFixed(2), g.colsX[1], g.hY[j] + g.miniR + 3 + chipH / 2);
            // Label
            ctx.fillStyle = '#5f5f5f';
            ctx.font      = '10px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`h${sub(j + 1)}`, g.colsX[1] - g.miniR - 6, g.hY[j]);
        }

        // Output mini-heatmap
        drawMiniHeatmap(g.colsX[2], g.outY, g.outR * 2,
                        outputHeat, outputColour, true);
        // Output value chip
        const v = fp.y;
        ctx.fillStyle = v >= 0.5
            ? 'rgba(234, 121, 89, 0.92)'
            : 'rgba(79, 70, 229, 0.92)';
        const chipW = 48, chipH = 22;
        roundedRect(ctx,
            g.colsX[2] - chipW / 2,
            g.outY + g.outR + 6,
            chipW, chipH, 11);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font      = '600 12px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`y=${v.toFixed(2)}`, g.colsX[2], g.outY + g.outR + 6 + chipH / 2);
        ctx.fillStyle = '#5f5f5f';
        ctx.font      = '10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('y', g.colsX[2] + g.outR + 6, g.outY);

        updateMath(fp);
    }

    function drawWeightLine(x1, y1, x2, y2, w, maxW) {
        const mag = Math.min(1, Math.abs(w) / maxW);
        const alpha = 0.12 + mag * 0.65;
        const colour = w >= 0 ? `rgba(79, 70, 229, ${alpha})` : `rgba(234, 121, 89, ${alpha})`;
        ctx.strokeStyle = colour;
        ctx.lineWidth   = 0.6 + mag * 3.4;
        const cx1 = (x1 + x2) / 2, cx2 = (x1 + x2) / 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(cx1, y1, cx2, y2, x2, y2);
        ctx.stroke();
    }

    function sub(n) {
        const map = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
        return String(n).split('').map(c => map[+c] ?? c).join('');
    }

    // ----- Math display -----
    function updateMath(fp) {
        if (!mathEl) return;
        const parts = [];
        for (let j = 0; j < 4; j++) {
            const w1 = W1[0][j].toFixed(1);
            const w2 = W1[1][j].toFixed(1);
            const bj = b1[j].toFixed(1);
            parts.push(
                `h${sub(j+1)} = ReLU(${w1}·${pt.x.toFixed(2)} + ${w2}·${pt.y.toFixed(2)} + ${bj}) = ` +
                `<span class="sum">${fp.h_post[j].toFixed(2)}</span>`
            );
        }
        const ySum = W2.map((w, j) => `${w.toFixed(1)}·${fp.h_post[j].toFixed(2)}`).join(' + ');
        const yLine = `y = σ(${ySum} + ${b2.toFixed(1)}) = <span class="sum">${fp.y.toFixed(3)}</span>`;
        mathEl.innerHTML = `
            <span class="lbl">Forward pass</span>
            ${parts.join('<br>')}<br>
            ${yLine}
        `;
    }

    // ----- Interactions -----
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }

    function hitHeatmap(px, py) {
        const g = geom();
        // Hidden mini-heatmaps
        for (let j = 0; j < 4; j++) {
            const dx = px - g.colsX[1], dy = py - g.hY[j];
            if (Math.abs(dx) <= g.miniR && Math.abs(dy) <= g.miniR) {
                return { cx: g.colsX[1], cy: g.hY[j], size: g.miniR * 2 };
            }
        }
        // Output
        const dx = px - g.colsX[2], dy = py - g.outY;
        if (Math.abs(dx) <= g.outR && Math.abs(dy) <= g.outR) {
            return { cx: g.colsX[2], cy: g.outY, size: g.outR * 2 };
        }
        return null;
    }
    function pickPoint(e) {
        const [px, py] = getPos(e);
        const hit = hitHeatmap(px, py);
        if (!hit) return false;
        // Map pixel → input coords in [-1, 1]
        pt.x =  ((px - hit.cx + hit.size / 2) / hit.size) * 2 - 1;
        pt.y = -((py - hit.cy + hit.size / 2) / hit.size) * 2 + 1;
        pt.x = Math.max(-1, Math.min(1, pt.x));
        pt.y = Math.max(-1, Math.min(1, pt.y));
        return true;
    }

    let dragging = false;
    canvas.addEventListener('mousedown',  (e) => { if (pickPoint(e)) { dragging = true; draw(); }});
    canvas.addEventListener('mousemove',  (e) => { if (dragging && pickPoint(e)) draw(); });
    window.addEventListener('mouseup',    () => { dragging = false; });
    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); if (pickPoint(e)) { dragging = true; draw(); }});
    canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); if (dragging && pickPoint(e)) draw(); });
    canvas.addEventListener('touchend',   () => { dragging = false; });

    if (presetSel) {
        for (const name of Object.keys(PRESETS)) {
            const o = document.createElement('option');
            o.value = name;
            o.textContent = name;
            presetSel.appendChild(o);
        }
        presetSel.value = preset;
        presetSel.addEventListener('change', () => {
            preset = presetSel.value;
            ({ W1, b1, W2, b2 } = PRESETS[preset]);
            buildHeatmaps();
            draw();
        });
    }

    // ----- Sizing -----
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(460, Math.max(360, cssW * 0.62)));
        W = cssW;
        H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    buildHeatmaps();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
