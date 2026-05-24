/* Interactive RNN recurrence viz.
 *
 * Three things the previous version hid that students must see:
 *
 *  1. Text → numbers — strings become token IDs, IDs become embedding
 *     vectors (rows of a learned matrix E). The "input" to the recurrence
 *     is a vector, not a word.
 *
 *  2. The SAME weight matrices apply at every timestep. We draw a "shared
 *     weights" panel (E, W_x, W_h, b) ONCE in the middle of the figure
 *     and connect it via dashed lines to every cell in the unrolled chain
 *     below. That sharing is what makes the RNN an RNN.
 *
 *  3. The cell computes a specific formula:
 *        h_t = tanh(W_x · x_t  +  W_h · h_{t-1}  +  b)
 *     Each arrow into a cell is colour-coded by which weight matrix it
 *     uses (W_x = indigo for inputs, W_h = orange for recurrence).
 */

(function () {
    const canvas    = document.getElementById('viz-rnn-canvas');
    const seqSelect = document.getElementById('viz-rnn-seq');
    const cellSelect = document.getElementById('viz-rnn-cell');
    const playBtn   = document.getElementById('viz-rnn-play');
    const formulaEl = document.getElementById('viz-rnn-formula');
    if (!canvas) return;

    // ---------------- Vocabulary + sequences ----------------
    // Small toy vocab so the embedding lookup is concrete: each word maps to
    // an integer id, and each id maps to a row of the embedding matrix E.
    const VOCAB = [
        'the','a','I','it','this','was',                             // 0..5
        'quick','brown','fox','jumps','over',                        // 6..10
        'really','enjoyed','loved','liked','hated',                  // 11..15
        'movie','book','cat','dog','sunset',                         // 16..20
        'big','small','old','new','warm','cold',                     // 21..26
    ];
    const ID_OF = Object.fromEntries(VOCAB.map((w, i) => [w, i]));

    const SEQUENCES = {
        'The quick brown fox jumps':       ['the', 'quick', 'brown', 'fox', 'jumps'],
        'I really enjoyed it':             ['I', 'really', 'enjoyed', 'it'],
        'I really loved this movie':       ['I', 'really', 'loved', 'this', 'movie'],
        'long sequence (vanishing demo)':  ['the','cat','was','old','and','warm','it','liked','sunset','this','movie'],
    };
    // Some sequences include words not yet in the vocab — extend at load.
    for (const seq of Object.values(SEQUENCES)) {
        for (const w of seq) {
            if (!(w in ID_OF)) { ID_OF[w] = VOCAB.length; VOCAB.push(w); }
        }
    }
    const V = VOCAB.length;
    const D = 4;          // embedding dimension
    const H = 6;          // hidden dimension

    let tokens   = SEQUENCES['The quick brown fox jumps'];
    let cellKind = 'vanilla';

    // ---------------- Deterministic weight matrices ----------------
    function seededRand(seed) {
        // Mulberry32 PRNG, seeded — same values every page load.
        let a = seed | 0;
        return () => {
            a |= 0; a = (a + 0x6d2b79f5) | 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return (((t ^ (t >>> 14)) >>> 0) & 0xffffffff) / 4294967296 - 0.5;
        };
    }

    function makeMatrix(rows, cols, seed, scale = 1.0) {
        const r = seededRand(seed);
        const M = [];
        for (let i = 0; i < rows; i++) {
            const row = new Float32Array(cols);
            for (let j = 0; j < cols; j++) row[j] = r() * 2 * scale;
            M.push(row);
        }
        return M;
    }

    function makeVector(n, seed, scale = 1.0) {
        const r = seededRand(seed);
        const v = new Float32Array(n);
        for (let i = 0; i < n; i++) v[i] = r() * 2 * scale;
        return v;
    }

    // E: V × D, W_x: D × H, b: H. W_h depends on cellKind.
    // Scales chosen so the hidden states span [-1, 1] without saturating —
    // makes the per-step evolution actually visible in the bar charts.
    const E   = makeMatrix(V, D, 7, 0.6);
    const W_x = makeMatrix(D, H, 11, 0.35);
    const b   = makeVector(H, 17, 0.1);

    function makeW_h(kind) {
        // Spectral-radius hack: random matrix, normalise rows, scale by decay.
        const decay = (kind === 'lstm') ? 0.97 : 0.65;
        const M = makeMatrix(H, H, 23);
        for (let i = 0; i < H; i++) {
            let n = 0;
            for (let j = 0; j < H; j++) n += M[i][j] * M[i][j];
            n = Math.sqrt(n) + 1e-9;
            for (let j = 0; j < H; j++) M[i][j] = (M[i][j] / n) * decay;
        }
        return M;
    }
    let W_h = makeW_h(cellKind);

    // ---------------- Forward pass ----------------
    // Returns { ids, xs, prez, hs } — same length T arrays plus h_0 in hs[0].
    function forward(seq) {
        const ids = seq.map(w => ID_OF[w] ?? 0);
        const xs   = ids.map(id => Float32Array.from(E[id]));
        const prez = [];   // pre-tanh values
        const hs   = [new Float32Array(H)];  // h_0 = 0
        for (let t = 0; t < seq.length; t++) {
            const x = xs[t];
            const hPrev = hs[hs.length - 1];
            const z = new Float32Array(H);
            for (let i = 0; i < H; i++) {
                let s = b[i];
                for (let j = 0; j < D; j++) s += W_x[j][i] * x[j];
                for (let j = 0; j < H; j++) s += W_h[i][j] * hPrev[j];
                z[i] = s;
            }
            prez.push(z);
            const h = new Float32Array(H);
            for (let i = 0; i < H; i++) h[i] = Math.tanh(z[i]);
            hs.push(h);
        }
        return { ids, xs, prez, hs };
    }
    let result = forward(tokens);

    // ---------------- Canvas state ----------------
    let activeT = 1;      // 1-indexed timestep (h_0 = step 0, h_1 = step 1, ...)
    let playing = true;
    let lastStep = 0;
    const STEP_MS = 1100;

    let ctx;
    let cssW = 0, cssH = 0;

    function resize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        cssW = Math.max(560, Math.round(rect.width - 8));
        cssH = 580;
        canvas.style.width  = cssW + 'px';
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    // ---------------- Colour helpers ----------------
    const POS = [79, 70, 229];      // indigo
    const NEG = [234, 121, 89];     // orange
    function cellColour(v, maxAbs, alphaBase = 0.18) {
        const t = Math.min(1, Math.abs(v) / (maxAbs + 1e-9));
        const c = v >= 0 ? POS : NEG;
        return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alphaBase + 0.65 * t})`;
    }
    const W_X_COLOUR = 'rgba(79, 70, 229, 0.85)';      // indigo for W_x arrows
    const W_H_COLOUR = 'rgba(234, 121, 89, 0.85)';     // orange for W_h arrows

    function drawText(x, y, text, opts = {}) {
        ctx.fillStyle = opts.color || '#1a1a1a';
        ctx.font = (opts.bold ? '600 ' : '')
                 + (opts.size || 11) + 'px '
                 + (opts.family || '"Inter", system-ui, sans-serif');
        ctx.textAlign = opts.align || 'left';
        ctx.textBaseline = opts.baseline || 'alphabetic';
        ctx.fillText(text, x, y);
    }

    function drawSectionLabel(x, y, text) {
        drawText(x, y, text, { color: '#9a9a9a', size: 9, bold: true });
    }

    function drawHeatmap(x, y, mat, rows, cols, cellW, cellH, maxAbs) {
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const v = mat[i][j];
                ctx.fillStyle = cellColour(v, maxAbs);
                ctx.fillRect(x + j * cellW, y + i * cellH, cellW, cellH);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cols * cellW - 1, rows * cellH - 1);
    }

    function drawVector(x, y, vec, cellW, cellH, maxAbs, horizontal = true) {
        const n = vec.length;
        for (let i = 0; i < n; i++) {
            const v = vec[i];
            ctx.fillStyle = cellColour(v, maxAbs);
            const cx = horizontal ? x + i * cellW : x;
            const cy = horizontal ? y : y + i * cellH;
            ctx.fillRect(cx, cy, cellW, cellH);
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5,
            horizontal ? n * cellW - 1 : cellW - 1,
            horizontal ? cellH - 1 : n * cellH - 1);
    }

    function roundedRect(x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y,     x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x,     y + h, r);
        ctx.arcTo(x,     y + h, x,     y,     r);
        ctx.arcTo(x,     y,     x + w, y,     r);
        ctx.closePath();
    }

    function arrow(x1, y1, x2, y2, color, dashed = false, lw = 1.5) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.setLineDash(dashed ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);
        // Arrowhead at (x2, y2)
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const ah = 5;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - ah * Math.cos(ang - 0.4), y2 - ah * Math.sin(ang - 0.4));
        ctx.lineTo(x2 - ah * Math.cos(ang + 0.4), y2 - ah * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
    }

    function sub(n) {
        const map = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
        return String(n).split('').map(c => map[+c] ?? c).join('');
    }

    // ---------------- Layout constants ----------------
    // Three vertical bands; each band knows its own y-extent. Computed in draw()
    // since they depend on token count.

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, cssW, cssH);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, cssW, cssH);

        const T = tokens.length;
        const padL = 18, padR = 18;

        // ---- Band 1: tokenisation strip (y = 0..115) ----
        const tokY = 16;       // section label
        const txtY = 32;       // "Text: ..." line
        const tokenBoxY = 52;
        const idBoxY    = 86;

        drawSectionLabel(padL, tokY, 'TEXT → NUMBERS');

        // The whole input string as a single highlighted strip
        const fullText = '"' + tokens.join(' ') + '"';
        drawText(padL, txtY + 9, fullText,
                 { color: '#1a1a1a', size: 12, bold: false });

        // Token boxes evenly spaced under the text — each step gets one column
        const colW   = (cssW - padL - padR) / T;
        const colXs  = Array.from({ length: T }, (_, t) => padL + colW * (t + 0.5));
        const tokenW = Math.min(80, colW - 6);
        const tokenH = 26;
        const idH    = 18;

        for (let t = 0; t < T; t++) {
            const cx = colXs[t];
            const isActive = (t + 1 === activeT);
            // Token box
            ctx.fillStyle   = isActive ? 'rgba(79, 70, 229, 0.85)' : '#ffffff';
            ctx.strokeStyle = isActive ? '#3730a3' : '#d6d3ca';
            ctx.lineWidth = 1.2;
            roundedRect(cx - tokenW / 2, tokenBoxY, tokenW, tokenH, 5);
            ctx.fill(); ctx.stroke();
            drawText(cx, tokenBoxY + 17, tokens[t], {
                color: isActive ? '#ffffff' : '#1a1a1a',
                size: 12, bold: true, align: 'center'
            });
            // ID box below
            const id = result.ids[t];
            ctx.fillStyle   = isActive ? '#fff3eb' : '#f5f3ee';
            ctx.strokeStyle = isActive ? '#ea7959' : '#d6d3ca';
            roundedRect(cx - 28, idBoxY, 56, idH, 4);
            ctx.fill(); ctx.stroke();
            drawText(cx, idBoxY + 13, `id ${id}`, {
                color: isActive ? '#8a3a1f' : '#5f5f5f',
                size: 10, family: '"JetBrains Mono", monospace',
                align: 'center', bold: true
            });
        }

        // ---- Band 2: shared weights panel (y = 120..260) ----
        const swY = 124;
        drawSectionLabel(padL, swY, 'SHARED WEIGHTS — the same matrices apply at every timestep');

        const panelY = swY + 12;
        const panelH = 130;
        // Outline a soft card so the panel reads as one unit
        ctx.fillStyle = 'rgba(79, 70, 229, 0.04)';
        roundedRect(padL, panelY, cssW - padL - padR, panelH, 8);
        ctx.fill();
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.20)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Layout matrices horizontally inside the panel
        const xCursor = { x: padL + 16 };
        const matY    = panelY + 30;

        function drawMatrixWithLabel(label, mat, rows, cols, cellW, cellH, sublabel) {
            // Label above
            drawText(xCursor.x, panelY + 22, label,
                     { color: '#1a1a1a', size: 11, bold: true,
                       family: '"JetBrains Mono", monospace' });
            // Matrix
            let maxAbs = 0;
            for (let i = 0; i < rows; i++)
                for (let j = 0; j < cols; j++)
                    if (Math.abs(mat[i][j]) > maxAbs) maxAbs = Math.abs(mat[i][j]);
            drawHeatmap(xCursor.x, matY, mat, rows, cols, cellW, cellH, maxAbs);
            // Sublabel
            drawText(xCursor.x, matY + rows * cellH + 14, sublabel,
                     { color: '#9a9a9a', size: 9 });
            xCursor.x += Math.max(cols * cellW, ctx.measureText(label).width,
                                  ctx.measureText(sublabel).width) + 32;
        }

        drawMatrixWithLabel('E', E, V, D, 7, 4, `embeddings · ${V}×${D}`);
        drawMatrixWithLabel('W_x', W_x, D, H, 11, 11, `inputs · ${D}×${H}`);
        drawMatrixWithLabel('W_h', W_h, H, H, 11, 11, `recurrent · ${H}×${H}`);
        // bias as a row vector
        drawText(xCursor.x, panelY + 22, 'b',
                 { color: '#1a1a1a', size: 11, bold: true,
                   family: '"JetBrains Mono", monospace' });
        let bMax = 0;
        for (let i = 0; i < H; i++) if (Math.abs(b[i]) > bMax) bMax = Math.abs(b[i]);
        drawVector(xCursor.x, matY, b, 11, 11, bMax, false);
        drawText(xCursor.x, matY + H * 11 + 14, `bias · ${H}`,
                 { color: '#9a9a9a', size: 9 });

        // Caption on the right — what does "shared" mean?
        const captionX = cssW - padR - 4;
        const captionLines = [
            'These four matrices are',
            'the WHOLE network.',
            'Same numbers used at',
            'every step below.',
        ];
        captionLines.forEach((line, i) => {
            drawText(captionX, panelY + 28 + i * 14, line,
                     { color: '#5f5f5f', size: 10, align: 'right' });
        });

        // ---- Band 3: unrolled chain (y = 280..560) ----
        const unrollLabelY = 280;
        drawSectionLabel(padL, unrollLabelY, 'UNROLLED RNN — h_t = tanh(W_x · x_t  +  W_h · h_{t-1}  +  b)');

        const xtRowY    = 300;     // embedding (x_t) row
        const cellRowY  = 350;     // RNN cell row
        const hRowY     = 422;     // hidden state (h_t) row
        const labelRowY = 494;     // label below h
        const cellSize  = Math.min(24, colW - 16);  // size of each x_t and h_t vector cell

        // Connection lines from the shared weights panel down to each cell.
        // Two dashed bundles: indigo (for W_x → input arrows) and orange
        // (for W_h → recurrent arrows). Drives home "same weights everywhere".
        const panelBottom = panelY + panelH;
        for (let t = 0; t < T; t++) {
            const cx = colXs[t];
            const isActive = (t + 1 === activeT);
            // W_x → input arrow line (indigo)
            ctx.strokeStyle = isActive ? 'rgba(79, 70, 229, 0.45)' : 'rgba(79, 70, 229, 0.22)';
            ctx.setLineDash([3, 4]);
            ctx.lineWidth = isActive ? 1.4 : 1;
            ctx.beginPath();
            ctx.moveTo(cx - 4, panelBottom + 2);
            ctx.lineTo(cx - 4, xtRowY - 4);
            ctx.stroke();
            // W_h → cell line (orange)
            ctx.strokeStyle = isActive ? 'rgba(234, 121, 89, 0.45)' : 'rgba(234, 121, 89, 0.22)';
            ctx.lineWidth = isActive ? 1.4 : 1;
            ctx.beginPath();
            ctx.moveTo(cx + 4, panelBottom + 2);
            ctx.lineTo(cx + 4, cellRowY + 4);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        // "↓ used here, here, here..." callout to the right of the panel
        drawText(cssW - padR - 4, panelBottom + 14,
                 '↓ ↓ ↓ same weights, every step',
                 { color: '#5f5f5f', size: 10, bold: true, align: 'right' });

        // h_0 marker on the far left of the chain (a small "h0 = 0" box)
        const h0X = padL + 10;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#9a9a9a';
        ctx.lineWidth = 1;
        roundedRect(h0X, cellRowY + 6, 36, 32, 5);
        ctx.fill(); ctx.stroke();
        drawText(h0X + 18, cellRowY + 21, 'h₀', {
            color: '#5f5f5f', size: 11, bold: true, align: 'center'
        });
        drawText(h0X + 18, cellRowY + 33, '= 0', {
            color: '#9a9a9a', size: 9, align: 'center'
        });

        // Per-timestep stuff
        for (let t = 0; t < T; t++) {
            const cx = colXs[t];
            const isActive = (t + 1 === activeT);

            // x_t embedding row — pulled from E[id_t]
            const x = result.xs[t];
            let xMax = 0;
            for (let i = 0; i < D; i++) if (Math.abs(x[i]) > xMax) xMax = Math.abs(x[i]);
            drawVector(cx - (D * cellSize) / 2, xtRowY, x, cellSize, cellSize,
                       xMax || 1);
            // Highlight the active x_t
            if (isActive) {
                ctx.strokeStyle = '#3730a3';
                ctx.lineWidth = 2;
                ctx.strokeRect(cx - (D * cellSize) / 2 + 0.5,
                               xtRowY + 0.5,
                               D * cellSize - 1, cellSize - 1);
            }
            drawText(cx + (D * cellSize) / 2 + 6, xtRowY + cellSize / 2 + 4,
                     `x${sub(t + 1)}`, {
                color: '#1a1a1a', size: 11, bold: true,
                family: '"JetBrains Mono", monospace'
            });

            // Arrow from x_t down into cell — coloured by W_x (indigo)
            arrow(cx, xtRowY + cellSize + 2,
                  cx, cellRowY - 2,
                  W_X_COLOUR, false, isActive ? 2 : 1.5);
            // Label on the input arrow ("W_x")
            if (t === 0) {
                drawText(cx + 8, (xtRowY + cellSize + cellRowY) / 2 + 3,
                         '× W_x', {
                    color: W_X_COLOUR.slice(0, -5) + '1)',
                    size: 10, bold: true,
                    family: '"JetBrains Mono", monospace'
                });
            }

            // The cell — a rounded box showing the formula
            const cellW = colW - 16;
            const cellH = 56;
            const cellX = cx - cellW / 2;
            if (isActive) {
                ctx.fillStyle = 'rgba(234, 121, 89, 0.18)';
                roundedRect(cellX - 3, cellRowY - 3, cellW + 6, cellH + 6, 9);
                ctx.fill();
            }
            const grd = ctx.createLinearGradient(0, cellRowY, 0, cellRowY + cellH);
            if (isActive) {
                grd.addColorStop(0, '#fff3eb');
                grd.addColorStop(1, '#fde0d2');
            } else {
                grd.addColorStop(0, '#ffffff');
                grd.addColorStop(1, '#f5f3ee');
            }
            ctx.fillStyle = grd;
            ctx.strokeStyle = isActive ? '#ea7959' : '#d6d3ca';
            ctx.lineWidth = isActive ? 2 : 1.2;
            roundedRect(cellX, cellRowY, cellW, cellH, 7);
            ctx.fill(); ctx.stroke();

            // Cell contents — the formula, scaled down
            drawText(cx, cellRowY + 18, cellKind === 'lstm' ? 'LSTM cell' : 'RNN cell', {
                color: isActive ? '#8a3a1f' : '#5f5f5f',
                size: 10, bold: true, align: 'center'
            });
            drawText(cx, cellRowY + 34, 'tanh(W_x·x + W_h·h + b)', {
                color: isActive ? '#1a1a1a' : '#9a9a9a',
                size: 9, family: '"JetBrains Mono", monospace',
                align: 'center'
            });
            drawText(cx, cellRowY + 48, `t = ${t + 1}`, {
                color: isActive ? '#1a1a1a' : '#9a9a9a',
                size: 9, family: '"JetBrains Mono", monospace',
                align: 'center'
            });

            // Arrow from cell down to h_t row — neutral colour (it's just the output)
            arrow(cx, cellRowY + cellH + 2,
                  cx, hRowY - 2,
                  isActive ? 'rgba(0, 0, 0, 0.55)' : 'rgba(0, 0, 0, 0.25)',
                  false, isActive ? 2 : 1.5);

            // h_t hidden state vector
            const h = result.hs[t + 1];
            drawVector(cx - (H * cellSize) / 2 / 1.2, hRowY,
                       h,
                       cellSize / 1.2, cellSize, 1.0);
            if (isActive) {
                ctx.strokeStyle = '#3730a3';
                ctx.lineWidth = 2;
                ctx.strokeRect(cx - (H * cellSize) / 2 / 1.2 + 0.5,
                               hRowY + 0.5,
                               H * cellSize / 1.2 - 1, cellSize - 1);
            }
            drawText(cx, labelRowY, `h${sub(t + 1)}`, {
                color: isActive ? '#ea7959' : '#5f5f5f',
                size: 11, bold: isActive, align: 'center'
            });
        }

        // Horizontal recurrent arrows between cells — coloured by W_h (orange)
        // Start at h_0 box (or previous cell) and go to next cell.
        for (let t = 0; t < T; t++) {
            const isActive = (t + 1 === activeT);
            const cx = colXs[t];
            const cellX1 = cx - (colW - 16) / 2;
            const x1 = (t === 0) ? h0X + 36
                     : colXs[t - 1] + (colW - 16) / 2;
            const x2 = cellX1;
            const y  = cellRowY + 56 / 2;
            arrow(x1 + 2, y, x2 - 2, y, W_H_COLOUR, false, isActive ? 2 : 1.5);
        }
        // "× W_h" label on the first recurrent arrow
        {
            const cellX1 = colXs[0] - (colW - 16) / 2;
            const x1 = h0X + 36;
            const midX = (x1 + cellX1) / 2;
            const y  = cellRowY + 56 / 2;
            drawText(midX, y - 6, '× W_h', {
                color: W_H_COLOUR.slice(0, -5) + '1)',
                size: 10, bold: true,
                family: '"JetBrains Mono", monospace',
                align: 'center'
            });
        }

        // Legend at the bottom — colour key for the two arrow families
        const legY = labelRowY + 24;
        drawText(padL, legY, '──', { color: W_X_COLOUR, bold: true, size: 12 });
        drawText(padL + 22, legY, 'inputs go through W_x', { color: '#1a1a1a', size: 10 });
        drawText(padL + 200, legY, '──', { color: W_H_COLOUR, bold: true, size: 12 });
        drawText(padL + 222, legY, 'recurrence goes through W_h', { color: '#1a1a1a', size: 10 });

        // Update the formula readout
        updateFormula();
    }

    // ---------------- Formula readout (below the canvas) ----------------
    function updateFormula() {
        if (!formulaEl) return;
        const t = activeT;
        if (t < 1) {
            formulaEl.innerHTML = '<span class="lbl">Initial state</span>h₀ = 0';
            return;
        }
        const tok = tokens[t - 1];
        const id  = result.ids[t - 1];
        const xv  = Array.from(result.xs[t - 1]).map(v => v.toFixed(2)).join(', ');
        const hv  = Array.from(result.hs[t]).map(v => v.toFixed(2)).join(', ');
        formulaEl.innerHTML = `
            <span class="lbl">Step ${t}</span>
            "${tok}" → id ${id} → x${sub(t)} = E[${id}] = [${xv}]
            &nbsp;⇒&nbsp;
            h${sub(t)} = tanh(W<sub>x</sub>·x${sub(t)} + W<sub>h</sub>·h${sub(t - 1)} + b)
            = <span class="sum">[${hv}]</span>
        `;
    }

    // ---------------- Interactions ----------------
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const T = tokens.length;
        const padL = 18, padR = 18;
        const colW = (cssW - padL - padR) / T;
        const col = Math.floor((x - padL) / colW);
        if (col >= 0 && col < T) {
            playing = false;
            updatePlayBtn();
            activeT = col + 1;
            draw();
        }
    });

    if (seqSelect) {
        for (const name of Object.keys(SEQUENCES)) {
            const o = document.createElement('option');
            o.value = name;
            o.textContent = name;
            seqSelect.appendChild(o);
        }
        seqSelect.value = 'The quick brown fox jumps';
        seqSelect.addEventListener('change', () => {
            tokens = SEQUENCES[seqSelect.value];
            result = forward(tokens);
            activeT = 1;
            draw();
        });
    }

    if (cellSelect) {
        cellSelect.innerHTML = `
            <option value="vanilla">Vanilla RNN</option>
            <option value="lstm">LSTM-like (better memory)</option>
        `;
        cellSelect.addEventListener('change', () => {
            cellKind = cellSelect.value;
            W_h = makeW_h(cellKind);
            result = forward(tokens);
            draw();
        });
    }

    if (playBtn) {
        playBtn.addEventListener('click', () => {
            playing = !playing;
            updatePlayBtn();
            if (playing) lastStep = performance.now();
        });
    }
    function updatePlayBtn() {
        if (playBtn) playBtn.textContent = playing ? 'Pause' : 'Play';
    }

    function loop(now) {
        if (playing && now - lastStep >= STEP_MS) {
            activeT = (activeT % tokens.length) + 1;
            draw();
            lastStep = now;
        }
        requestAnimationFrame(loop);
    }

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    updatePlayBtn();
    requestAnimationFrame(loop);
})();
