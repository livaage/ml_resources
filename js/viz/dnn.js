/* Two-layer network visualised as matrix multiplications.
 *
 * Layout (left → right):
 *
 *                        W₁ (2 × K)                       W₂ (K × 1)
 *                       ┌────┐                            ┌──┐
 *                       │    │  (weight matrix)           │  │
 *                       └────┘                            └──┘
 *                          │                                │
 *   X (4 × 2)              ▼                                ▼
 *   ┌────┐  ───→   Z₁ (4 × K)   ──act──→   H (4 × K)   ───→   Z₂ (4 × 1)   ──σ──→   Y (4 × 1)
 *   │ X  │
 *   └────┘
 *
 * Cells of W₁, W₂ are draggable (vertical). Click a column to inspect the
 * corresponding neuron in the "Inside one neuron" panel below.
 */

(function () {
    const canvas = document.getElementById('viz-dnn-canvas');
    if (!canvas) return;

    const actRadios = document.querySelectorAll('input[name="viz-dnn-act"]');
    const nCountEl  = document.getElementById('viz-dnn-ncount');
    const nMinusBtn = document.getElementById('viz-dnn-n-minus');
    const nPlusBtn  = document.getElementById('viz-dnn-n-plus');
    const randBtn   = document.getElementById('viz-dnn-randomize');
    const numbersCb = document.getElementById('viz-dnn-numbers');

    // ---------- Constants ----------
    const POS = [79, 70, 229];     // indigo (positive)
    const NEG = [224, 128, 53];    // orange (negative)
    const WEIGHT_RANGE = 1.5;
    const CELL = 34;               // matrix cell size in px
    // Slightly bigger in "show numbers" mode so the numeric labels fit, but
    // not so big that the right-most columns overflow the canvas.
    function cellSize3D()   { return showNumbers ? 32 : 28; }
    function cellSizeFlat() { return showNumbers ? 24 : 20; }

    // Two features chosen so each column of X looks visually distinct.
    // mood is allowed to be NEGATIVE (-1 = melancholic, +1 = upbeat), so the
    // mood column has both indigo and orange cells while tempo stays positive.
    const SONGS = [
        // tempo and mood are CENTERED around 0 — negative = slower / sadder
        // than average. Targets form an XOR-like pattern that a single linear
        // layer can't fit, so toggling activation off visibly collapses the
        // predictions.
        { name: 'Blinding Lights',   tempo:  0.70, mood:  0.60, target: 1 },
        { name: 'Clair de Lune',     tempo: -0.70, mood: -0.50, target: 1 },
        { name: 'Anti-Hero',         tempo:  0.50, mood: -0.50, target: 0 },
        { name: 'Take Five',         tempo: -0.40, mood:  0.50, target: 0 },
    ];
    const FEATURES = ['tempo', 'mood'];
    const N = SONGS.length;
    const D = FEATURES.length;     // input features

    // ---------- State ----------
    let K = 3;                     // hidden units
    let W1 = [];                   // [D][K]  ⇒ each column is a neuron
    let b1 = [];                   // [K]
    let W2 = [];                   // [K][1]  ⇒ column is the output neuron
    let b2 = 0;
    let activation = 'relu';
    let showNumbers = false;       // toggle: numeric labels inside cells
    let drag = null;               // { mat:'W1'|'W2'|'b1'|'b2', i, j, startY, startVal, range }
    let selected = { mat: 'W1', col: 0 };
    let hovered = null;            // { mat:'Z1'|'Z2', i, j } when hovering an output cell
    let layout = {};               // populated each render: positions for hit-testing

    // ---------- Isometric projection (for the 3D matmul cube) ----------
    // CELL3D / DEPTH_DX / DEPTH_DY are recomputed at the start of every render()
    // so they can grow when "show numbers" is on.
    let CELL3D   = 28;
    let DEPTH_DX = -CELL3D * 0.78;
    let DEPTH_DY = -CELL3D * 0.48;
    /* 3D coordinates (a, b, c):
     *   a = lateral, increases to the RIGHT on screen
     *   b = depth, increases INTO the screen (up-left on screen)
     *   c = vertical (matrix row), increases DOWN on screen
     * Origin is the front-top-left corner of the cube — i.e. top-left of Z₁'s front face.
     */
    function project(a, b, c, origin) {
        return {
            x: origin.x + a * CELL3D + b * DEPTH_DX,
            y: origin.y + c * CELL3D + b * DEPTH_DY,
        };
    }
    function quad(ctx, p1, p2, p3, p4) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.lineTo(p4.x, p4.y);
        ctx.closePath();
    }
    function drawCubeCell(ctx, p1, p2, p3, p4, value, maxAbs, opts = {}) {
        const t = clamp(Math.abs(value) / maxAbs, 0, 1);
        const alpha = 0.10 + t * 0.78;
        ctx.fillStyle = rgbaPN(value, alpha);
        quad(ctx, p1, p2, p3, p4); ctx.fill();
        // Border
        if (opts.highlight) {
            ctx.strokeStyle = '#1e1e2e';
            ctx.lineWidth = 2;
        } else if (opts.dim) {
            ctx.strokeStyle = 'rgba(0,0,0,0.08)';
            ctx.lineWidth = 1;
        } else {
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 1;
        }
        quad(ctx, p1, p2, p3, p4); ctx.stroke();
        // Optional number (front face only — parallelogram cells get cluttered)
        if (showNumbers && opts.showNumber !== false && opts.face === 'front') {
            ctx.fillStyle = t > 0.55 ? '#fff' : '#222';
            ctx.font = '10px var(--font-mono, ui-monospace), monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const cx = (p1.x + p2.x + p3.x + p4.x) / 4;
            const cy = (p1.y + p2.y + p3.y + p4.y) / 4;
            ctx.fillText(value.toFixed(2), cx, cy);
            ctx.textBaseline = 'alphabetic';
        }
    }
    function pointInQuad(px, py, p1, p2, p3, p4) {
        // Simple ray-casting test
        const pts = [p1, p2, p3, p4];
        let inside = false;
        for (let i = 0, j = 3; i < 4; j = i++) {
            const xi = pts[i].x, yi = pts[i].y;
            const xj = pts[j].x, yj = pts[j].y;
            if (((yi > py) !== (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    const rnd = (lo, hi) => lo + Math.random() * (hi - lo);

    function randomize() {
        W1 = []; b1 = [];
        for (let i = 0; i < D; i++) {
            const row = [];
            for (let j = 0; j < K; j++) row.push(rnd(-WEIGHT_RANGE, WEIGHT_RANGE));
            W1.push(row);
        }
        for (let j = 0; j < K; j++) b1.push(rnd(-0.3, 0.3));
        W2 = [];
        for (let j = 0; j < K; j++) W2.push([rnd(-WEIGHT_RANGE, WEIGHT_RANGE)]);
        b2 = rnd(-0.3, 0.3);
        if (selected.col >= K) selected.col = K - 1;
    }

    function activate(z) {
        if (activation === 'none')    return z;
        if (activation === 'relu')    return Math.max(0, z);
        return 1 / (1 + Math.exp(-z));   // sigmoid
    }
    const sigmoid = z => 1 / (1 + Math.exp(-z));

    function forward() {
        const Z1 = [], H = [], Z2 = [], Y = [];
        for (const s of SONGS) {
            const x = FEATURES.map(f => s[f]);
            const z1 = new Array(K);
            for (let j = 0; j < K; j++) {
                let sum = b1[j];
                for (let i = 0; i < D; i++) sum += W1[i][j] * x[i];
                z1[j] = sum;
            }
            const h = z1.map(activate);
            let zo = b2;
            for (let j = 0; j < K; j++) zo += W2[j][0] * h[j];
            Z1.push(z1); H.push(h); Z2.push([zo]); Y.push([sigmoid(zo)]);
        }
        return { Z1, H, Z2, Y };
    }

    // ---------- Drawing primitives ----------
    function rgbaPN(value, alpha = 1) {
        const c = value >= 0 ? POS : NEG;
        return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
    }
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function drawCell(ctx, x, y, value, maxAbs, opts = {}) {
        const w = opts.size || CELL;
        const h = opts.size || CELL;
        // background tint
        const t = clamp(Math.abs(value) / maxAbs, 0, 1);
        const alpha = 0.10 + t * 0.78;
        ctx.fillStyle = rgbaPN(value, alpha);
        ctx.fillRect(x, y, w, h);
        // border
        ctx.strokeStyle = opts.highlight ? '#1e1e2e' : 'rgba(0,0,0,0.18)';
        ctx.lineWidth = opts.highlight ? 1.8 : 1;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        // number (only when the global toggle is on)
        if (showNumbers && opts.showNumber !== false) {
            ctx.fillStyle = t > 0.55 ? '#fff' : '#222';
            ctx.font = '11px var(--font-mono, ui-monospace), monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const txt = value.toFixed(2);
            ctx.fillText(txt, x + w / 2, y + h / 2);
        }
        ctx.textBaseline = 'alphabetic';
    }

    function drawMatrix(ctx, x, y, mat, rows, cols, maxAbs, opts = {}) {
        // mat is rows-of-arrays or a function (i,j)->value
        const get = typeof mat === 'function' ? mat : (i, j) => mat[i][j];
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const highlight = opts.highlightCol === j || opts.highlightRow === i;
                drawCell(ctx, x + j * CELL, y + i * CELL, get(i, j), maxAbs, {
                    ...opts,
                    highlight,
                });
            }
        }
        // outline the whole matrix
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 0.5, y - 0.5, cols * CELL + 1, rows * CELL + 1);
    }

    function drawLabel(ctx, x, y, text, opts = {}) {
        ctx.fillStyle = opts.muted ? '#888' : '#333';
        ctx.font = (opts.bold ? '600 ' : '') + (opts.size || 11) + 'px var(--font-sans, system-ui), sans-serif';
        ctx.textAlign = opts.align || 'center';
        ctx.textBaseline = opts.baseline || 'alphabetic';
        ctx.fillText(text, x, y);
        ctx.textBaseline = 'alphabetic';
    }

    function drawArrow(ctx, x1, y1, x2, y2, color = 'rgba(0,0,0,0.35)') {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        // arrowhead
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const ah = 5;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - ah * Math.cos(ang - 0.4), y2 - ah * Math.sin(ang - 0.4));
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - ah * Math.cos(ang + 0.4), y2 - ah * Math.sin(ang + 0.4));
        ctx.stroke();
    }

    // Place absolutely-positioned anchor divs (one per logical figure in the
    // canvas) inside the canvas wrap. The global scroll-spy in topic-page.js
    // picks these up via [data-fig] and swaps the sidebar accordingly.
    function placeFigureAnchors(ranges) {
        const wrap = canvas.parentElement;
        if (!wrap) return;
        wrap.querySelectorAll('.viz-fig-anchor').forEach(el => el.remove());
        ranges.forEach(r => {
            const a = document.createElement('div');
            a.className = 'viz-fig-anchor';
            a.dataset.fig = r.fig;
            a.style.top = r.y + 'px';
            a.style.height = r.h + 'px';
            wrap.appendChild(a);
        });
        // Re-trigger the scroll-spy now that anchor positions have changed.
        window.dispatchEvent(new Event('scroll'));
    }

    // ---------- Render ----------
    function render() {
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.parentElement.clientWidth;

        // Recompute cell sizes (these grow when "show numbers" is on)
        CELL3D   = cellSize3D();
        DEPTH_DX = -CELL3D * 0.78;
        DEPTH_DY = -CELL3D * 0.48;

        // Layout
        const padL = 16, padR = 16;
        const gap = 18;
        const labelLeftPx = 100;    // space for song names left of cube 1
        const FLAT = cellSizeFlat();// small flat-panel cell size
        const POL_W = 56;           // polariser slab width

        // Hover overrides selected: when something is being hovered, we suppress
        // the persistent "selected" highlight so only ONE thing pops at a time.
        const hoverActive = hovered !== null;

        // ===== STACK ORDER (top → bottom) =====
        //   1. Single-neuron explainer panel (simple example)
        //   2. Main cube figure (extended view of the whole network)
        //   3. Bigger-network cubish figure (shows it scales)
        const neuronPanelY = 8;
        const neuronPanelH = 300;
        const mainFigGap   = 28;    // gap between neuron panel and main figure
        const bigFigGap    = 36;    // gap between main figure and big network
        const bigNetH      = 220;   // reserved height for the big network section

        // Top zone for the main figure — flat reference panels for X, W₁, W₂.
        // Sits below the neuron panel.
        const refTitleH    = neuronPanelY + neuronPanelH + mainFigGap;  // title baseline y
        const refXH        = N * FLAT;
        const refW1H       = D * FLAT;
        const refW2H       = K * FLAT;
        const refMaxH      = Math.max(refXH, refW1H, refW2H);
        const refTopLabel  = refTitleH + 24;
        const topZone      = refTopLabel + refMaxH + 22;

        // First matmul cube geometry — X @ W₁ = Z₁
        const cube1OffsetX = D * (-DEPTH_DX);
        const cube1OffsetY = D * (-DEPTH_DY);
        const cube1W       = K * CELL3D + cube1OffsetX;
        // Second matmul cube geometry — H @ W₂ = Z₂. Depth is K, lateral is 1.
        const cube2OffsetX = K * (-DEPTH_DX);
        const cube2OffsetY = K * (-DEPTH_DY);
        const cube2W       = 1 * CELL3D + cube2OffsetX;

        const xCols = {};
        let x = padL + labelLeftPx;
        xCols.cube1Front = x + cube1OffsetX;
        x += cube1W + gap;
        xCols.pol1       = x;  x += POL_W + gap;
        xCols.cube2Front = x + cube2OffsetX;
        x += cube2W + gap;
        xCols.pol2       = x;  x += POL_W + gap;
        xCols.Y          = x;  x += CELL3D + 6;
        xCols.target     = x;  x += CELL3D;

        // Cube top positions — cube 2 is taller than cube 1, so the data row
        // bottoms align at the same y. We choose the data row y from the
        // tallest cube.
        const cubeTopMostOffset = Math.max(cube1OffsetY, cube2OffsetY);
        const cubeFrontTop   = topZone + 4 + cubeTopMostOffset;       // y of Z₁/Z₂ front-face top
        const cube1TopY      = cubeFrontTop - cube1OffsetY;           // top of cube 1 (top of W₁ top face)
        const cube2TopY      = cubeFrontTop - cube2OffsetY;           // top of cube 2 (top of W₂ top face)

        const mainFigBottom = cubeFrontTop + N * CELL3D + 28;   // includes bias row
        const bigFigTop     = mainFigBottom + bigFigGap;
        const cssH = bigFigTop + bigNetH + 12;
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;
        canvas.style.height = cssH + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        const { Z1, H, Z2, Y } = forward();
        // Magnitudes for normalising cell intensity
        const xMax = 1.0;
        const wMax = WEIGHT_RANGE;
        const z1Max = Math.max(0.5, ...Z1.flat().map(Math.abs));
        const hMax  = Math.max(0.5, ...H.flat().map(Math.abs));
        const z2Max = Math.max(0.5, ...Z2.flat().map(Math.abs));

        // ===== TOP: single-neuron explainer (simple example) =====
        renderNeuronPanel(ctx, padL, neuronPanelY, cssW - padL - padR,
                          neuronPanelH - 12, { Z1, H, Z2, Y });

        // ===== TOP-ZONE FLAT REFERENCE PANELS =====
        // X panel sits above cube 1's left face. W₁ panel above cube 1's top
        // face. W₂ panel above cube 2's top face. Each is a true 2D matrix view.
        const xRefLeft  = padL + labelLeftPx - D * FLAT - 8;        // align approximately under the left face of cube 1
        const xRefY     = refTopLabel;
        const w1RefLeft = xCols.cube1Front;                          // align with cube 1 top columns
        const w1RefY    = refTopLabel;
        const w2RefLeft = xCols.cube2Front;
        const w2RefY    = refTopLabel;

        function drawFlatPanel(originX, originY, rows, cols, getVal, maxAbs, opts) {
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x0 = originX + c * FLAT, y0 = originY + r * FLAT;
                    drawCubeCell(ctx,
                        { x: x0, y: y0 },
                        { x: x0 + FLAT, y: y0 },
                        { x: x0 + FLAT, y: y0 + FLAT },
                        { x: x0, y: y0 + FLAT },
                        getVal(r, c), maxAbs,
                        { face: 'front',
                          highlight: opts && opts.highlightCol === c });
                }
            }
        }

        // Wrap a drawing block in an arbitrary 2x2 transform around (pivotX, pivotY).
        // Used to "fold" each flat panel toward its matching cube face — the
        // panel's edge nearest the cube stays put (the hinge), and the rest
        // shears toward the cube's depth direction.
        function withTransform(pivotX, pivotY, a, b, c, d, fn) {
            ctx.save();
            ctx.translate(pivotX, pivotY);
            ctx.transform(a, b, c, d, 0, 0);
            ctx.translate(-pivotX, -pivotY);
            fn();
            ctx.restore();
        }
        // Tilt fraction: how strongly each panel folds toward its cube face.
        // Each panel's "far edge from the cube" tilts UP-LEFT (toward the cube's
        // depth direction). The "near edge" (the hinge) stays put.
        //   X panel hinge = right edge → left side tilts up-left
        //   W₁ / W₂ hinge = bottom edge → top tilts up-left
        const TILT = 0.30;
        // X panel: (a, b, 0, 1). a > 1 stretches LEFT (further from pivot),
        // b > 0 shifts UP as you go left of the pivot.
        const xShearA = 1 + TILT * 0.78;   // ≈ 1.23
        const xShearB = TILT * 0.48;       // ≈ 0.14
        // W₁ / W₂: (1, 0, c, d). c > 0 shifts the TOP left, d > 1 shifts it UP.
        const wShearC = TILT * 0.78;       // ≈ 0.23
        const wShearD = 1 + TILT * 0.48;   // ≈ 1.14

        // For all three panels: title sits ABOVE the column labels with proper
        // spacing so it never overlaps. Column labels sit just above the cells.
        const TITLE_Y = refTitleH;          // y of the matrix-name title
        const COLLBL_PAD = 4;               // gap between column label and the cells

        // Pivot points.
        //   X panel pivots at its CENTRE — the left side rotates up-left (into
        //   depth) and the right side comes down-right toward the viewer,
        //   matching the cube face's perspective.
        //   W₁ / W₂ panels still pivot at their BOTTOM edge (hinge to the cube
        //   face directly below them).
        const xPivot  = { x: xRefLeft  + (D * FLAT) / 2, y: xRefY  + (N * FLAT) / 2 };
        const w1Pivot = { x: w1RefLeft + (K * FLAT) / 2, y: w1RefY + D * FLAT };
        const w2Pivot = { x: w2RefLeft + FLAT / 2,       y: w2RefY + K * FLAT };

        // Helper to compute where a point on a panel ends up after its shear.
        function shearedPoint(px, py, pivot, a, b, c, d) {
            const dx = px - pivot.x, dy = py - pivot.y;
            return { x: pivot.x + a * dx + c * dy, y: pivot.y + b * dx + d * dy };
        }

        // Titles drawn at the top of each panel — small + simple so they don't
        // collide with the column labels after the panel shear.
        drawLabel(ctx, w1RefLeft + (K * FLAT) / 2, TITLE_Y, 'W₁', { size: 12, bold: true });
        drawLabel(ctx, w2RefLeft +  FLAT / 2,       TITLE_Y, 'W₂', { size: 12, bold: true });

        // X panel rendered as a mini parallelogram matching the orientation of
        // cube 1's LEFT face: cells slant up-left as the feature index grows,
        // with col 0 (tempo, "front") on the right and col D-1 ("back") on the
        // left and up. Same direction as the cube cells, just smaller.
        const xFLAT  = 24;                                   // X panel cell vertical extent
        const xTILT  = 0.85;                                  // 0 flat ... 1 full match to cube
        const xDX    = xTILT * xFLAT * DEPTH_DX / CELL3D;    // ≈ −16  (left per col)
        const xDY    = xTILT * xFLAT * DEPTH_DY / CELL3D;    // ≈ −10  (up   per col)
        // Origin = top-front-left corner (b=0, c=0). Place the panel so the
        // front edge ends up roughly at the original right edge of the panel area.
        const xOrigin = { x: xRefLeft + Math.abs(D * xDX) + 4, y: xRefY + 4 };

        function projectX(b, c) {
            return { x: xOrigin.x + b * xDX, y: xOrigin.y + c * xFLAT + b * xDY };
        }

        // X title at the very top-left, well clear of the tilted feature labels
        drawLabel(ctx, padL + 8, TITLE_Y, 'X', { size: 12, bold: true, align: 'left' });

        // Cells
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < D; k++) {
                const v = SONGS[i][FEATURES[k]];
                const p1 = projectX(k,     i);
                const p2 = projectX(k + 1, i);
                const p3 = projectX(k + 1, i + 1);
                const p4 = projectX(k,     i + 1);
                drawCubeCell(ctx, p1, p2, p3, p4, v, xMax, { face: 'front' });
            }
        }
        // Feature labels along the front edge (top-front of each col)
        for (let k = 0; k < D; k++) {
            const p = projectX(k + 0.5, 0);
            drawLabel(ctx, p.x, p.y - 4, FEATURES[k], { size: 10, bold: true, muted: true });
        }
        // Song labels at the BACK-left of each row (so they sit on the panel's
        // far edge — the same side the back of the cube face lives)
        for (let i = 0; i < N; i++) {
            const p = projectX(D, i + 0.5);
            drawLabel(ctx, p.x - 4, p.y + 3, SONGS[i].name, { align: 'right', size: 10 });
        }

        // W₁ panel — vertical shear: top tilts UP-LEFT toward cube 1's top face.
        // Rows are REVERSED so the panel's visual order matches the cube top
        // face (where b=D-1 sits at the back/top edge after the fold).
        withTransform(w1Pivot.x, w1Pivot.y, 1, 0, wShearC, wShearD, () => {
            for (let j = 0; j < K; j++) {
                const isSelW = selected.mat === 'W1' && selected.col === j;
                drawLabel(ctx, w1RefLeft + j * FLAT + FLAT / 2, w1RefY - COLLBL_PAD, `n${j + 1}`,
                          { size: 10, bold: isSelW, muted: !isSelW });
            }
            drawFlatPanel(w1RefLeft, w1RefY, D, K, (k, j) => W1[D - 1 - k][j], wMax,
                           { highlightCol: hoverActive ? -1 : (selected.mat === 'W1' ? selected.col : -1) });
        });

        // W₂ panel — same vertical shear; rows reversed for the same reason.
        // No "y" header; row labels show neuron index in the reversed order so
        // n1 sits at the bottom (matching the cube's front edge of the top face).
        withTransform(w2Pivot.x, w2Pivot.y, 1, 0, wShearC, wShearD, () => {
            for (let j = 0; j < K; j++) {
                drawLabel(ctx, w2RefLeft - 4, w2RefY + j * FLAT + FLAT / 2 + 3, `n${K - j}`,
                          { align: 'right', size: 10, muted: true });
            }
            drawFlatPanel(w2RefLeft, w2RefY, K, 1, (j, _) => W2[K - 1 - j][0], wMax);
        });

        // Dashed connectors from each panel's bottom-centre (the rotation pivot,
        // so it stays at its original position) to the CENTRE of the matching
        // cube face — computed via the same project() call used to draw the face.
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.5)';
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.2;
        // X panel → cube 1 LEFT face centre. Connector starts from the
        // FRONT-middle of the new X panel (col 0, vertical middle) — that's the
        // edge closest to cube 1.
        {
            const start = (function () {
                // projectX inline — using the same formulas as in render
                return { x: xOrigin.x, y: xOrigin.y + (N / 2) * xFLAT };
            })();
            const center = project(0, D / 2, N / 2, { x: xCols.cube1Front, y: cubeFrontTop });
            ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(center.x, center.y); ctx.stroke();
        }
        // W₁ panel → cube 1 TOP face centre. Face: (a=[0,K], b=[0,D], c=0)
        {
            const center = project(K / 2, D / 2, 0, { x: xCols.cube1Front, y: cubeFrontTop });
            ctx.beginPath(); ctx.moveTo(w1Pivot.x, w1Pivot.y + 2); ctx.lineTo(center.x, center.y); ctx.stroke();
        }
        // W₂ panel → cube 2 TOP face centre. Face: (a=[0,1], b=[0,K], c=0)
        {
            const center = project(0.5, K / 2, 0, { x: xCols.cube2Front, y: cubeFrontTop });
            ctx.beginPath(); ctx.moveTo(w2Pivot.x, w2Pivot.y + 2); ctx.lineTo(center.x, center.y); ctx.stroke();
        }
        ctx.setLineDash([]);

        // ===== CUBE 1: X · W₁ → Z₁ =====
        const cube1Origin = { x: xCols.cube1Front, y: cubeFrontTop };
        const hoverRow1 = (hovered && hovered.mat === 'Z1') ? hovered.i : -1;
        const hoverCol1 = (hovered && hovered.mat === 'Z1') ? hovered.j : -1;

        // Left face — X (a=0, b∈[0,D], c∈[0,N])
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < D; k++) {
                const v = SONGS[i][FEATURES[k]];
                const p1 = project(0, k,     i,     cube1Origin);
                const p2 = project(0, k + 1, i,     cube1Origin);
                const p3 = project(0, k + 1, i + 1, cube1Origin);
                const p4 = project(0, k,     i + 1, cube1Origin);
                drawCubeCell(ctx, p1, p2, p3, p4, v, xMax,
                    { highlight: hoverRow1 === i, dim: hoverRow1 !== -1 && hoverRow1 !== i });
            }
        }
        // Top face — W₁
        for (let k = 0; k < D; k++) {
            for (let j = 0; j < K; j++) {
                const p1 = project(j,     k,     0, cube1Origin);
                const p2 = project(j + 1, k,     0, cube1Origin);
                const p3 = project(j + 1, k + 1, 0, cube1Origin);
                const p4 = project(j,     k + 1, 0, cube1Origin);
                const isHi = hoverActive
                    ? hoverCol1 === j
                    : (selected.mat === 'W1' && selected.col === j);
                drawCubeCell(ctx, p1, p2, p3, p4, W1[k][j], wMax,
                    { highlight: isHi,
                      dim: hoverCol1 !== -1 && hoverCol1 !== j });
            }
        }
        // Front face — Z₁
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < K; j++) {
                const p1 = project(j,     0, i,     cube1Origin);
                const p2 = project(j + 1, 0, i,     cube1Origin);
                const p3 = project(j + 1, 0, i + 1, cube1Origin);
                const p4 = project(j,     0, i + 1, cube1Origin);
                drawCubeCell(ctx, p1, p2, p3, p4, Z1[i][j], z1Max,
                    { highlight: hoverRow1 === i && hoverCol1 === j, face: 'front' });
            }
        }
        // Z₁ label — right of the front face, vertically centred (out of the way of W₁)
        drawLabel(ctx, xCols.cube1Front + K * CELL3D + 6,
                  cubeFrontTop + (N * CELL3D) / 2 + 5,
                  'Z₁', { size: 14, bold: true, align: 'left' });

        // b₁ row under cube 1 — "+" on the left (it's added to the matmul),
        // "b₁" label on the right of the row.
        const b1RowY = cubeFrontTop + N * CELL3D + 8;
        drawLabel(ctx, xCols.cube1Front - 6, b1RowY + CELL3D / 2 + 5, '+',
                  { align: 'right', size: 18, bold: true });
        drawLabel(ctx, xCols.cube1Front + K * CELL3D + 6, b1RowY + CELL3D / 2 + 5, 'b₁',
                  { align: 'left', size: 12, bold: true });
        for (let j = 0; j < K; j++) {
            const x0 = xCols.cube1Front + j * CELL3D, y0 = b1RowY;
            const isHi = !hoverActive && selected.mat === 'W1' && selected.col === j;
            drawCubeCell(ctx,
                { x: x0, y: y0 }, { x: x0 + CELL3D, y: y0 },
                { x: x0 + CELL3D, y: y0 + CELL3D }, { x: x0, y: y0 + CELL3D },
                b1[j], 0.6,
                { highlight: isHi, face: 'front' });
        }

        // ===== POLARISER 1 (colour-block negatives for ReLU) =====
        drawPolariser(ctx, xCols.pol1, cubeFrontTop, POL_W, N * CELL3D, activation);

        // ===== CUBE 2: H · W₂ → Z₂ (degenerate cube, depth = K) =====
        const cube2Origin = { x: xCols.cube2Front, y: cubeFrontTop };
        const hoverRow2 = (hovered && hovered.mat === 'Z2') ? hovered.i : -1;

        // Left face — H (a=0, b∈[0,K], c∈[0,N]). Depth axis = hidden-unit index.
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < K; k++) {
                const p1 = project(0, k,     i,     cube2Origin);
                const p2 = project(0, k + 1, i,     cube2Origin);
                const p3 = project(0, k + 1, i + 1, cube2Origin);
                const p4 = project(0, k,     i + 1, cube2Origin);
                drawCubeCell(ctx, p1, p2, p3, p4, H[i][k], hMax,
                    { highlight: hoverRow2 === i, dim: hoverRow2 !== -1 && hoverRow2 !== i });
            }
        }
        // Top face — W₂ (c=0, a∈[0,1], b∈[0,K])
        for (let k = 0; k < K; k++) {
            const p1 = project(0, k,     0, cube2Origin);
            const p2 = project(1, k,     0, cube2Origin);
            const p3 = project(1, k + 1, 0, cube2Origin);
            const p4 = project(0, k + 1, 0, cube2Origin);
            const isHi = !hoverActive && selected.mat === 'W2';
            drawCubeCell(ctx, p1, p2, p3, p4, W2[k][0], wMax,
                { highlight: isHi });
        }
        // Front face — Z₂ (b=0, a∈[0,1], c∈[0,N])
        for (let i = 0; i < N; i++) {
            const p1 = project(0, 0, i,     cube2Origin);
            const p2 = project(1, 0, i,     cube2Origin);
            const p3 = project(1, 0, i + 1, cube2Origin);
            const p4 = project(0, 0, i + 1, cube2Origin);
            drawCubeCell(ctx, p1, p2, p3, p4, Z2[i][0], z2Max,
                { highlight: hoverRow2 === i, face: 'front' });
        }
        // Z₂ label — right of the front face
        drawLabel(ctx, xCols.cube2Front + CELL3D + 6,
                  cubeFrontTop + (N * CELL3D) / 2 + 5,
                  'Z₂', { size: 14, bold: true, align: 'left' });

        // b₂ under cube 2 — "+" on the left, "b₂" label on the right
        const b2RowY = cubeFrontTop + N * CELL3D + 8;
        drawLabel(ctx, xCols.cube2Front - 6, b2RowY + CELL3D / 2 + 5, '+',
                  { align: 'right', size: 18, bold: true });
        drawLabel(ctx, xCols.cube2Front + CELL3D + 6, b2RowY + CELL3D / 2 + 5, 'b₂',
                  { align: 'left', size: 12, bold: true });
        {
            const x0 = xCols.cube2Front, y0 = b2RowY;
            drawCubeCell(ctx,
                { x: x0, y: y0 }, { x: x0 + CELL3D, y: y0 },
                { x: x0 + CELL3D, y: y0 + CELL3D }, { x: x0, y: y0 + CELL3D },
                b2, 0.6, { face: 'front' });
        }

        // ===== POLARISER 2 — final sigmoid =====
        drawPolariser(ctx, xCols.pol2, cubeFrontTop, POL_W, N * CELL3D, 'sigmoid');

        // ===== Y (prediction) and Target columns side by side =====
        // Signed colouring on both: indigo = add (>0.5 for Y, 1 for target),
        // orange = skip. Right of the columns shows ✓ (matched) or ✗ (wrong).
        drawLabel(ctx, xCols.Y + CELL3D / 2,      cubeFrontTop - 4, 'Ŷ',     { size: 11, bold: true });
        drawLabel(ctx, xCols.target + CELL3D / 2, cubeFrontTop - 4, 'target', { size: 10, bold: true, muted: true });
        // Column captions one line further up
        drawLabel(ctx, xCols.Y + CELL3D / 2,      cubeFrontTop - 16, 'predicted', { size: 9, muted: true });
        drawLabel(ctx, xCols.target + CELL3D / 2, cubeFrontTop - 16, 'true',      { size: 9, muted: true });
        for (let i = 0; i < N; i++) {
            const y0  = cubeFrontTop + i * CELL3D;
            const v   = Y[i][0];
            const tgt = SONGS[i].target;
            const signedY = (v - 0.5) * 2;          // map [0,1] → [-1,+1]
            const signedT = tgt === 1 ? 1 : -1;
            // Prediction cell
            const yx0 = xCols.Y;
            drawCubeCell(ctx,
                { x: yx0, y: y0 }, { x: yx0 + CELL3D, y: y0 },
                { x: yx0 + CELL3D, y: y0 + CELL3D }, { x: yx0, y: y0 + CELL3D },
                signedY, 1.0, { face: 'front' });
            // Target cell — same colour scheme but full-intensity (it's a label)
            const tx0 = xCols.target;
            drawCubeCell(ctx,
                { x: tx0, y: y0 }, { x: tx0 + CELL3D, y: y0 },
                { x: tx0 + CELL3D, y: y0 + CELL3D }, { x: tx0, y: y0 + CELL3D },
                signedT, 1.0, { face: 'front' });
            // Match indicator + label
            const predLabel = v >= 0.5 ? 'add' : 'skip';
            const tgtLabel  = tgt === 1 ? 'add' : 'skip';
            const match     = (v >= 0.5) === (tgt === 1);
            drawLabel(ctx, xCols.target + CELL3D + 6, y0 + CELL3D / 2 + 4,
                      `${match ? '✓' : '✗'}  ${predLabel}`,
                      { align: 'left', size: 11,
                        muted: !match,
                        bold: match });
        }

        // Song labels along the BACK-left edge of cube 1's left face — aligned
        // vertically with the back of each row (which is where the leftmost edge
        // of each row actually sits in screen because of the depth tilt).
        for (let i = 0; i < N; i++) {
            const backLeftCentre = project(0, D, i + 0.5, cube1Origin);
            drawLabel(ctx, backLeftCentre.x - 6, backLeftCentre.y + 4,
                      SONGS[i].name, { align: 'right', size: 10 });
        }

        // ===== BOTTOM: bigger network in the same cube style =====
        renderBigNetwork(ctx, padL, bigFigTop, cssW - padL - padR, bigNetH);

        // Publish figure y-ranges (CSS pixels, relative to canvas top) so the
        // scroll-spy can know which figure is currently in view.
        placeFigureAnchors([
            { fig: 'neuron', y: neuronPanelY, h: neuronPanelH },
            { fig: 'main',   y: refTitleH - 8, h: mainFigBottom - (refTitleH - 8) },
            { fig: 'big',    y: bigFigTop, h: bigNetH },
        ]);

        // Save layout for hit-testing
        layout = {
            xCols, cube1Origin, cube2Origin,
            cubeFrontTop, b1RowY, b2RowY,
            cube1TopY, cube2TopY,
            xRefLeft, xRefY, w1RefLeft, w1RefY, w2RefLeft, w2RefY,
            xPivot, w1Pivot, w2Pivot,
            xShearA, xShearB, wShearC, wShearD,
            FLAT,
        };
    }

    /* ===== BIGGER NETWORK PANEL =====
     * Traditional textbook-style layered NN diagram (circles + edges), with
     * miniature matmul-cubes nestled in the gaps so readers can see how the
     * cube view above corresponds to the familiar "neurons-and-wires" picture.
     * Each edge in the diagram is one cell of the corresponding W matrix —
     * the tiny cube in the same gap shows that matrix as its top face.
     */
    function renderBigNetwork(ctx, x, y, w, h) {
        // Architecture: 5 inputs → 7 hidden → 4 hidden → 1 output
        const sizes = [5, 7, 4, 1];
        const layerNames = ['inputs', 'hidden 1', 'hidden 2', 'output'];

        // Deterministic pseudo-random in [-1, 1]
        const rand = (i, j, salt) => {
            const v = Math.sin(i * 12.9898 + j * 78.233 + salt * 0.7) * 43758.5453;
            return ((v - Math.floor(v)) - 0.5) * 2;
        };
        // Weight matrices W_l with shape (sizes[l] × sizes[l+1])
        const Ws = [];
        for (let l = 0; l < sizes.length - 1; l++) {
            const M = Array.from({ length: sizes[l] }, (_, i) =>
                      Array.from({ length: sizes[l + 1] }, (_, j) => rand(i, j, l + 10)));
            Ws.push(M);
        }
        // One example input + forward pass (just to put a sign on each neuron)
        const xInput = Array.from({ length: sizes[0] }, (_, i) => rand(i, 0, 50));
        const relu = z => Math.max(0, z);
        const sig  = z => 1 / (1 + Math.exp(-z));
        const acts = [xInput];
        for (let l = 0; l < Ws.length; l++) {
            const prev = acts[l], W = Ws[l];
            const out = new Array(sizes[l + 1]).fill(0);
            for (let j = 0; j < sizes[l + 1]; j++) {
                let s = 0;
                for (let i = 0; i < sizes[l]; i++) s += W[i][j] * prev[i];
                out[j] = l === Ws.length - 1 ? sig(s) : relu(s);
            }
            acts.push(out);
        }

        // ===== Layout =====
        // Title strip
        drawLabel(ctx, x + 8, y + 14,
                  'Same network, in textbook style',
                  { align: 'left', size: 12, bold: true });
        drawLabel(ctx, x + 8, y + 28,
                  `${sizes.join(' → ')}  •  each line is one edge weight, each colour matches the same cell in a W matrix above`,
                  { align: 'left', size: 10, muted: true });

        const titleH    = 38;
        const cubeStripH = 60;        // reserved at the bottom for mini-cubes
        const labelStripH = 14;
        const circleTop = y + titleH;
        const circleBot = y + h - cubeStripH - labelStripH - 2;

        const padL = 30, padR = 20;
        const layerSpacing = (w - padL - padR) / (sizes.length - 1);
        const layerX = sizes.map((_, i) => x + padL + i * layerSpacing);

        // Circle sizing — fit the largest layer in the available height
        const maxLayer = Math.max(...sizes);
        const cellH = Math.min(18, (circleBot - circleTop) / maxLayer);
        const NEURON_R = Math.max(3, Math.min(7, cellH / 2 - 1.5));

        function neuronY(l, i) {
            const n = sizes[l];
            if (n === 1) return (circleTop + circleBot) / 2;
            const spread = Math.min(circleBot - circleTop, (n - 1) * cellH);
            const top = (circleTop + circleBot) / 2 - spread / 2;
            return top + i * spread / (n - 1);
        }

        // ===== Edges (drawn first so circles sit on top) =====
        // Each edge gets its colour from the corresponding weight cell, so the
        // edge palette matches the mini-cube tops below.
        for (let l = 0; l < Ws.length; l++) {
            const W = Ws[l];
            const inN = sizes[l], outN = sizes[l + 1];
            const x1 = layerX[l] + NEURON_R;
            const x2 = layerX[l + 1] - NEURON_R;
            for (let i = 0; i < inN; i++) {
                for (let j = 0; j < outN; j++) {
                    const wv = W[i][j];
                    const y1 = neuronY(l, i);
                    const y2 = neuronY(l + 1, j);
                    ctx.strokeStyle = rgbaPN(wv, Math.min(1, 0.22 + Math.abs(wv) * 0.45));
                    ctx.lineWidth   = Math.min(2.4, 0.4 + Math.abs(wv) * 1.1);
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }
        }

        // ===== Neurons =====
        for (let l = 0; l < sizes.length; l++) {
            const n = sizes[l];
            for (let i = 0; i < n; i++) {
                const ny = neuronY(l, i);
                // Subtle fill tinted by activation value (sign + magnitude)
                const av = acts[l][i];
                const tinted = l === sizes.length - 1 ? (av - 0.5) * 2 : av;
                const t = Math.min(0.7, Math.abs(tinted) * 0.6 + 0.05);
                ctx.fillStyle = `rgba(${tinted >= 0 ? POS[0] : NEG[0]}, ${tinted >= 0 ? POS[1] : NEG[1]}, ${tinted >= 0 ? POS[2] : NEG[2]}, ${t})`;
                ctx.strokeStyle = '#555';
                ctx.lineWidth = 1.1;
                ctx.beginPath();
                ctx.arc(layerX[l], ny, NEURON_R, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }
            // Layer name beneath the column
            drawLabel(ctx, layerX[l], circleBot + 14, layerNames[l],
                      { size: 9, muted: true });
            // n=… count
            drawLabel(ctx, layerX[l], y + titleH - 6, `${n}`,
                      { size: 9, muted: true, bold: true });
        }

        // ===== Mini-cubes between layer pairs — bridge to the cube view above =====
        const savedCELL = CELL3D, savedDX = DEPTH_DX, savedDY = DEPTH_DY;
        CELL3D   = 6;
        DEPTH_DX = -CELL3D * 0.78;
        DEPTH_DY = -CELL3D * 0.48;
        try {
            for (let l = 0; l < Ws.length; l++) {
                const W = Ws[l];
                const inN = sizes[l], outN = sizes[l + 1];
                const midX = (layerX[l] + layerX[l + 1]) / 2;
                // Cube geometry: top face is W (inN rows × outN cols), with a
                // shallow left + front face for "cube-ness". The whole thing
                // sits in the cube strip at the bottom of the panel.
                const cubeW = outN * CELL3D + inN * (-DEPTH_DX);
                const cubeH = 3 * CELL3D + inN * (-DEPTH_DY);
                const o = {
                    x: midX - cubeW / 2 + inN * (-DEPTH_DX),
                    y: y + h - cubeStripH + (cubeStripH - cubeH) / 2 + inN * (-DEPTH_DY),
                };
                // Left face — fake "X" with 3 example rows (just to read as a cube)
                for (let i = 0; i < 3; i++) {
                    for (let k = 0; k < inN; k++) {
                        const v = rand(i, k, l + 70);
                        const p1 = project(0, k,     i,     o);
                        const p2 = project(0, k + 1, i,     o);
                        const p3 = project(0, k + 1, i + 1, o);
                        const p4 = project(0, k,     i + 1, o);
                        drawCubeCell(ctx, p1, p2, p3, p4, v, 1.0, {});
                    }
                }
                // Top face — W (this is the matrix whose cells = edge weights)
                for (let k = 0; k < inN; k++) {
                    for (let j = 0; j < outN; j++) {
                        const p1 = project(j,     k,     0, o);
                        const p2 = project(j + 1, k,     0, o);
                        const p3 = project(j + 1, k + 1, 0, o);
                        const p4 = project(j,     k + 1, 0, o);
                        drawCubeCell(ctx, p1, p2, p3, p4, W[k][j], WEIGHT_RANGE, {});
                    }
                }
                // Front face — Z (a few dummy values for visual completeness)
                for (let i = 0; i < 3; i++) {
                    for (let j = 0; j < outN; j++) {
                        let s = 0;
                        for (let k = 0; k < inN; k++) s += W[k][j] * rand(i, k, l + 70);
                        const p1 = project(j,     0, i,     o);
                        const p2 = project(j + 1, 0, i,     o);
                        const p3 = project(j + 1, 0, i + 1, o);
                        const p4 = project(j,     0, i + 1, o);
                        drawCubeCell(ctx, p1, p2, p3, p4, s, 2.0, { face: 'front' });
                    }
                }
                // Label below
                drawLabel(ctx, midX, y + h - 4,
                          `W${sub(l + 1)} · ${inN}×${outN}`,
                          { size: 9, muted: true });
            }
        } finally {
            CELL3D = savedCELL; DEPTH_DX = savedDX; DEPTH_DY = savedDY;
        }
    }

    /* Colour-block polariser: a thin vertical slab that visually transforms
     * values passing through. Negative cells are absorbed (dark/opaque),
     * positive cells pass (clear or tinted indigo). For sigmoid, a continuous
     * gradient shows the squeeze into [0, 1]. */
    function drawPolariser(ctx, x, y, w, h, act) {
        const cx = x + w / 2;
        const slabW = 30;
        const slabX = cx - slabW / 2;
        const slabY = y + 6;
        const slabH = h - 12;
        const midY  = slabY + slabH / 2;

        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(slabX, slabY, slabW, slabH);

        if (act === 'none') {
            // Clear glass — barely-visible vertical "pass-through" lines
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.lineWidth = 1;
            for (let i = 1; i < 6; i++) {
                const xx = slabX + i * (slabW / 6);
                ctx.beginPath(); ctx.moveTo(xx, slabY); ctx.lineTo(xx, slabY + slabH); ctx.stroke();
            }
        } else if (act === 'relu') {
            // Top half (positive): clear, with subtle indigo tint
            ctx.fillStyle = `rgba(${POS[0]}, ${POS[1]}, ${POS[2]}, 0.10)`;
            ctx.fillRect(slabX, slabY, slabW, slabH / 2);
            // Bottom half (negative): solid orange block, opaque — "absorbs"
            ctx.fillStyle = `rgba(${NEG[0]}, ${NEG[1]}, ${NEG[2]}, 0.85)`;
            ctx.fillRect(slabX, midY, slabW, slabH / 2);
            // Diagonal hatching on the blocked half for "absorbed" feel
            ctx.save();
            ctx.beginPath();
            ctx.rect(slabX, midY, slabW, slabH / 2);
            ctx.clip();
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
            ctx.lineWidth = 1;
            for (let s = -slabH; s < slabH * 2; s += 5) {
                ctx.beginPath();
                ctx.moveTo(slabX + s, midY);
                ctx.lineTo(slabX + s + slabH, midY + slabH);
                ctx.stroke();
            }
            ctx.restore();
            // Threshold line at zero
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
            ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.moveTo(slabX, midY); ctx.lineTo(slabX + slabW, midY); ctx.stroke();
        } else { // sigmoid
            // Continuous gradient: orange at bottom (absorbed → 0), indigo at top (squashed → 1)
            const grad = ctx.createLinearGradient(0, slabY + slabH, 0, slabY);
            grad.addColorStop(0,    `rgba(${NEG[0]}, ${NEG[1]}, ${NEG[2]}, 0.85)`);
            grad.addColorStop(0.5,  `rgba(160, 140, 200, 0.40)`);
            grad.addColorStop(1,    `rgba(${POS[0]}, ${POS[1]}, ${POS[2]}, 0.10)`);
            ctx.fillStyle = grad;
            ctx.fillRect(slabX, slabY, slabW, slabH);
            // S-curve hint inside
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            for (let i = 0; i <= 20; i++) {
                const t = i / 20;
                const yv = slabY + slabH * (1 - t);
                const sigVal = 1 / (1 + Math.exp(-(t - 0.5) * 8));
                const xv = slabX + sigVal * slabW;
                if (i === 0) ctx.moveTo(xv, yv); else ctx.lineTo(xv, yv);
            }
            ctx.stroke();
        }

        // Slab outline
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 1.4;
        ctx.strokeRect(slabX + 0.5, slabY + 0.5, slabW - 1, slabH - 1);

        // Label vertically along the slab's left edge
        ctx.save();
        ctx.translate(slabX - 6, midY);
        ctx.rotate(-Math.PI / 2);
        drawLabel(ctx, 0, 0, act === 'none' ? 'no activation' : act,
                  { size: 10, bold: true });
        ctx.restore();

        // Arrows in / out
        drawArrow(ctx, x + 2, midY, slabX - 2, midY);
        drawArrow(ctx, slabX + slabW + 2, midY, x + w - 2, midY);
    }

    /* Polariser-style activation. The activation is drawn as a thin vertical
     * "polariser" slab between Z₁ and H. Its internal pattern visualises what
     * passes through:
     *   - none:    clear slab (everything passes unchanged)
     *   - ReLU:    bottom half opaque/blocked (negative values are absorbed),
     *              top half clear (positive values pass)
     *   - sigmoid: a continuous gradient from opaque-at-bottom to clear-at-top,
     *              squeezing the magnitude into [0, 1].
     * Tiny markers along the left edge show incoming Z₁ values for the selected
     * neuron column; markers along the right edge show the corresponding H. */
    function drawFilterBadge(ctx, x, y, w, h, act) {
        const cx = x + w / 2;
        // Slab geometry — tall and narrow, occupying most of the available height
        const slabW = 30;
        const slabX = cx - slabW / 2;
        const slabY = y + 4;
        const slabH = h - 8;
        const midY  = slabY + slabH / 2;   // y of "zero" on the slab's vertical axis

        // The "zero" line — splits positive (above) from negative (below)
        // Draw the slab background first
        ctx.fillStyle = '#fcfcff';
        ctx.fillRect(slabX, slabY, slabW, slabH);

        // Internal pattern depends on activation
        if (act === 'none') {
            // No filter — clear glass with a vertical line implying "pass-through"
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.lineWidth = 1;
            for (let i = 0; i < 4; i++) {
                const yy = slabY + (i + 0.5) * (slabH / 4);
                ctx.beginPath();
                ctx.moveTo(slabX, yy);
                ctx.lineTo(slabX + slabW, yy);
                ctx.stroke();
            }
        } else if (act === 'relu') {
            // Bottom half absorbs: dense dark hatch.
            ctx.save();
            ctx.beginPath();
            ctx.rect(slabX, midY, slabW, slabH / 2);
            ctx.clip();
            ctx.fillStyle = 'rgba(40, 40, 60, 0.35)';
            ctx.fillRect(slabX, midY, slabW, slabH / 2);
            // Hatched lines showing the "polariser" angle
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1;
            for (let s = -slabH; s < slabH * 2; s += 4) {
                ctx.beginPath();
                ctx.moveTo(slabX + s, midY);
                ctx.lineTo(slabX + s + slabH, midY + slabH);
                ctx.stroke();
            }
            ctx.restore();
            // Top half — clear with subtle vertical lines suggesting "open polariser"
            ctx.strokeStyle = 'rgba(79, 70, 229, 0.35)';
            ctx.lineWidth = 1;
            for (let i = 0; i < 5; i++) {
                const xx = slabX + (i + 0.5) * (slabW / 5);
                ctx.beginPath();
                ctx.moveTo(xx, slabY);
                ctx.lineTo(xx, midY);
                ctx.stroke();
            }
            // Boundary line at zero
            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.moveTo(slabX, midY); ctx.lineTo(slabX + slabW, midY);
            ctx.stroke();
        } else { // sigmoid
            // Smooth gradient — opaque at bottom, clear at top, continuous squeeze
            const grad = ctx.createLinearGradient(0, slabY + slabH, 0, slabY);
            grad.addColorStop(0, 'rgba(40, 40, 60, 0.45)');
            grad.addColorStop(0.5, 'rgba(79, 70, 229, 0.18)');
            grad.addColorStop(1, 'rgba(79, 70, 229, 0.03)');
            ctx.fillStyle = grad;
            ctx.fillRect(slabX, slabY, slabW, slabH);
            // Curved "polarisation" lines — diagonal but pinched
            ctx.strokeStyle = 'rgba(79, 70, 229, 0.4)';
            ctx.lineWidth = 1;
            for (let i = 0; i < 7; i++) {
                const t = (i + 0.5) / 7;
                const tx = slabX + t * slabW;
                ctx.beginPath();
                ctx.moveTo(tx - 3, slabY);
                ctx.quadraticCurveTo(tx, slabY + slabH / 2, tx + 3, slabY + slabH);
                ctx.stroke();
            }
        }

        // Slab border
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.lineWidth = 1.4;
        ctx.strokeRect(slabX + 0.5, slabY + 0.5, slabW - 1, slabH - 1);

        // Activation name vertically along the slab (small, rotated)
        ctx.save();
        ctx.translate(slabX - 4, slabY + slabH / 2);
        ctx.rotate(-Math.PI / 2);
        drawLabel(ctx, 0, 0, act === 'none' ? 'no activation' : act,
                  { size: 10, bold: true });
        ctx.restore();

        // Arrows: Z₁ → slab → H
        drawArrow(ctx, x + 2, midY, slabX - 2, midY);
        drawArrow(ctx, slabX + slabW + 2, midY, x + w - 2, midY);
    }

    /* Large activation graph with markers showing each row's (input, output) pair.
     * Pedagogically: students see the curve AND see where each cell of the input
     * matrix lands on it. Toggling activation re-shapes the curve and the markers
     * snap to their new positions. */
    function drawActivationGraph(ctx, x, y, w, h, act, markers, columnLabel) {
        // x,y is top-left of the graph area. w,h is the total area for the graph.
        // Save a top strip for the function name, a bottom strip for axis labels.
        const padTop = 18, padBot = 18, padX = 8;
        const gx = x + padX, gy = y + padTop;
        const gw = w - 2 * padX, gh = h - padTop - padBot;

        // Function-name label
        const fnText = act === 'none' ? 'y = x   (no activation)'
                     : act === 'relu' ? 'y = max(0, x)   (ReLU)'
                     : 'y = 1 / (1 + e⁻ˣ)   (sigmoid)';
        drawLabel(ctx, x + w / 2, y + 12, fnText, { size: 11, bold: true });

        // Determine input/output ranges that comfortably fit all markers
        const inAbs = Math.max(2.0, ...markers.map(m => Math.abs(m.z)));
        const xMin = -inAbs, xMax = inAbs;
        let yMin, yMax;
        if (act === 'sigmoid') { yMin = -0.1; yMax = 1.1; }
        else if (act === 'relu') { yMin = -0.2; yMax = Math.max(1.0, ...markers.map(m => m.h)) + 0.2; }
        else { yMin = -inAbs; yMax = inAbs; }

        const X = v => gx + ((v - xMin) / (xMax - xMin)) * gw;
        const Y = v => gy + (1 - (v - yMin) / (yMax - yMin)) * gh;

        // Background card
        ctx.fillStyle = 'rgba(0,0,0,0.025)';
        ctx.fillRect(gx, gy, gw, gh);
        ctx.strokeStyle = 'rgba(0,0,0,0.15)';
        ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);

        // Axes through zero
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 1;
        if (yMin <= 0 && yMax >= 0) {
            ctx.beginPath();
            ctx.moveTo(gx, Y(0)); ctx.lineTo(gx + gw, Y(0));
            ctx.stroke();
        }
        if (xMin <= 0 && xMax >= 0) {
            ctx.beginPath();
            ctx.moveTo(X(0), gy); ctx.lineTo(X(0), gy + gh);
            ctx.stroke();
        }

        // The activation curve
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.85)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const STEPS = 60;
        for (let i = 0; i <= STEPS; i++) {
            const xv = xMin + (i / STEPS) * (xMax - xMin);
            const yv = act === 'none' ? xv
                     : act === 'relu' ? Math.max(0, xv)
                     : 1 / (1 + Math.exp(-xv));
            const px = X(xv), py = Y(clamp(yv, yMin, yMax));
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Per-row markers: dot at (z, h) with faint guides
        markers.forEach((m, i) => {
            const zClamped = clamp(m.z, xMin, xMax);
            const hClamped = clamp(m.h, yMin, yMax);
            const px = X(zClamped), py = Y(hClamped);
            // dashed guides
            ctx.strokeStyle = 'rgba(0,0,0,0.18)';
            ctx.setLineDash([2, 2]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, Y(0)); ctx.lineTo(px, py);
            ctx.lineTo(X(0), py);
            ctx.stroke();
            ctx.setLineDash([]);
            // dot
            ctx.fillStyle = '#1e1e2e';
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        // Axis labels at the bottom
        drawLabel(ctx, gx, y + h - 4, `x = ${xMin.toFixed(1)}`, { size: 9, align: 'left', muted: true });
        drawLabel(ctx, gx + gw, y + h - 4, `${xMax.toFixed(1)}`, { size: 9, align: 'right', muted: true });
        // Caption: which column the markers correspond to
        if (columnLabel) {
            drawLabel(ctx, gx + gw / 2, y + h - 4, `dots = ${columnLabel}`,
                      { size: 9, muted: true });
        }
    }

    // ---------- Inside-one-neuron panel ----------
    // Shows the EXPLICIT mapping between a column of the weight matrix in the
    // cube and a familiar neuron diagram. Left side: the selected column drawn
    // as a vertical stack of the same coloured cells used in the cube. Right
    // side: those same numbers drawn as wires going into a single neuron, with
    // arrows linking each cell to its corresponding wire.
    function renderNeuronPanel(ctx, x, y, w, h, acts) {
        const { Z1, H, Z2, Y } = acts;
        // Background card
        ctx.fillStyle = 'rgba(79, 70, 229, 0.04)';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.18)';
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

        const isW1 = selected.mat === 'W1';
        const col  = isW1 ? selected.col : 0;
        // Which song is "the one example"? Use hovered Z1/Z2/H row if present, else song 0.
        const songIdx = (hovered && (hovered.mat === 'Z1' || hovered.mat === 'Z2' || hovered.mat === 'H'))
                        ? hovered.i : 0;
        const song = SONGS[songIdx];

        const colVals = isW1
            ? Array.from({ length: D }, (_, i) => W1[i][col])
            : Array.from({ length: K }, (_, i) => W2[i][0]);
        const xVals = isW1
            ? FEATURES.map(f => song[f])
            : H[songIdx].slice();
        const inputs = isW1
            ? FEATURES.slice()
            : Array.from({ length: K }, (_, i) => `n${i + 1}`);
        const inputSyms = isW1
            ? FEATURES.map((_, i) => `x${sub(i + 1)}`)
            : Array.from({ length: K }, (_, i) => `h${sub(i + 1)}`);
        const biasVal = isW1 ? b1[col] : b2;
        const neuronName = isW1 ? `n${col + 1}` : 'y';
        const totalNeurons = isW1 ? K : 1;
        const neuronIdx = isW1 ? col + 1 : 1;
        const n = colVals.length;
        const colSub = isW1 ? sub(col + 1) : '';

        const preAct = isW1 ? Z1[songIdx][col] : Z2[songIdx][0];
        const postAct = isW1 ? H[songIdx][col] : Y[songIdx][0];
        const actLabel = isW1 ? actSymbol() : 'σ';

        // ===== Compact title — the detailed explanation lives in the sidebar =====
        const titleA = isW1
            ? `Single-neuron view — example "${song.name}", neuron ${neuronName} (${neuronIdx} of K=${totalNeurons})`
            : `Output neuron view — example "${song.name}"`;
        drawLabel(ctx, x + 16, y + 18, titleA, { align: 'left', size: 12, bold: true });

        // ===== Equation block, ABOVE the diagram (no surrounding box) =====
        // Use cell indices like z₁,₁ that match the Z₁ matrix in the figure above.
        const songSub = sub(songIdx + 1);
        const zSym = isW1 ? `z${songSub},${colSub}` : `z${songSub}`;
        const hSym = isW1 ? `h${songSub},${colSub}` : `ŷ${songSub}`;
        const zMatrixName = isW1 ? 'Z₁' : 'Z₂';
        const cellLoc    = isW1 ? `(${songIdx + 1}, ${neuronIdx})` : `${songIdx + 1}`;

        const eqY = y + 38;
        const eqLineH = 16;
        ctx.fillStyle = '#333';
        ctx.font = '11.5px var(--font-mono, ui-monospace), monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';

        // Line 1: structural form
        const inputName = isW1 ? 'x' : 'h';
        const structParts = [];
        for (let i = 0; i < n; i++) {
            structParts.push(`w${sub(i + 1)}·${inputName}${sub(i + 1)}`);
        }
        const structLine = `${zSym} = ${structParts.join(' + ')} + b${colSub}`;
        ctx.fillText(structLine, x + 16, eqY);
        // Annotation pointing back to the cube
        ctx.fillStyle = '#888';
        ctx.font = '10px var(--font-sans, system-ui), sans-serif';
        ctx.fillText(`  ← cell ${cellLoc} of ${zMatrixName} above`,
                     x + 16 + ctx.measureText(structLine).width + 4, eqY);

        // Line 2: numbers substituted
        ctx.fillStyle = '#444';
        ctx.font = '11.5px var(--font-mono, ui-monospace), monospace';
        const termsNum = [];
        for (let i = 0; i < n; i++) {
            termsNum.push(`(${colVals[i].toFixed(2)})(${xVals[i].toFixed(2)})`);
        }
        const bStr = (biasVal >= 0 ? '+ ' : '− ') + Math.abs(biasVal).toFixed(2);
        const numLine = `       = ${termsNum.join(' + ')} ${bStr} = ${preAct.toFixed(2)}`;
        ctx.fillText(numLine, x + 16, eqY + eqLineH);

        // Line 3: after activation
        ctx.fillStyle = '#333';
        const actLine = `${hSym} = ${actLabel}(${zSym}) = ${postAct.toFixed(2)}`;
        ctx.fillText(actLine, x + 16, eqY + 2 * eqLineH);

        // ===== Diagram — simple neuron picture, colours pulled straight from above =====
        // - input "neurons" = circles containing the same coloured feature cell as in X
        // - edges = wires coloured by the corresponding weight cell from W
        // - activation gate sits just before the output neuron
        // - output neuron = circle containing the coloured h (or y) cell from H (or Y)
        const diagTop = eqY + 3 * eqLineH + 16;
        const diagBot = y + h - 14;
        const diagMidY = (diagTop + diagBot) / 2;

        const inX   = x + 110;
        const actX  = x + Math.max(360, w * 0.62);   // activation gate
        const outX  = actX + 90;                       // output neuron

        // (Song name is in the title above — no second caption needed here.)

        const NEURON_R = 22;     // radius of input / output circles
        const FT_CELL  = 22;     // size of the coloured cell inside each input neuron

        // Vertical layout for input neurons — leave room for the labels above
        // (symbol) and below (feature name) so nothing spills outside the panel.
        const vPad = NEURON_R + 18;
        const usable = Math.max(0, (diagBot - diagTop) - 2 * vPad);
        const spread = Math.min(usable, n * 70);
        const inTopY = diagMidY - spread / 2;
        const rowYs = [];
        for (let i = 0; i < n; i++) {
            rowYs.push(inTopY + (n === 1 ? spread / 2 : (i * spread / (n - 1))));
        }

        // ===== Edges (drawn first so circles sit on top) =====
        // Each edge takes its colour from the W column entry — same palette as the
        // weight matrix in the figure above.
        for (let i = 0; i < n; i++) {
            const py = rowYs[i];
            const wv = colVals[i];
            ctx.strokeStyle = rgbaPN(wv, Math.min(1, 0.55 + Math.abs(wv) * 0.35));
            ctx.lineWidth   = Math.min(6, 1.2 + Math.abs(wv) * 2.0);
            ctx.beginPath();
            ctx.moveTo(inX + NEURON_R, py);
            ctx.lineTo(actX - 18, diagMidY);
            ctx.stroke();

            // Small weight label sitting on the edge midpoint
            const mx = (inX + NEURON_R + actX - 18) / 2;
            const my = (py + diagMidY) / 2 - 8;
            // White pill background so the text reads over the line
            const wSym = isW1
                ? `w${sub(i + 1)}${sub(',')}${colSub} = ${wv.toFixed(2)}`
                : `w${sub(i + 1)} = ${wv.toFixed(2)}`;
            ctx.font = '10px var(--font-mono, ui-monospace), monospace';
            const tw = ctx.measureText(wSym).width;
            ctx.fillStyle = 'rgba(255,255,255,0.92)';
            ctx.fillRect(mx - tw / 2 - 4, my - 8, tw + 8, 14);
            ctx.fillStyle = rgbaPN(wv, 1);
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(wSym, mx, my - 1);
            ctx.textBaseline = 'alphabetic';
        }

        // Bias edge into the activation node (from below)
        ctx.strokeStyle = rgbaPN(biasVal, 0.65);
        ctx.lineWidth = Math.min(4, 1 + Math.abs(biasVal) * 1.6);
        ctx.beginPath();
        ctx.moveTo(actX, diagBot - 6);
        ctx.lineTo(actX - 6, diagMidY + 8);
        ctx.stroke();
        // Bias label
        ctx.font = '10px var(--font-mono, ui-monospace), monospace';
        ctx.fillStyle = rgbaPN(biasVal, 1);
        ctx.textAlign = 'center';
        ctx.fillText(`b${colSub} = ${biasVal.toFixed(2)}`,
                     actX + 10, diagBot - 8);

        // ===== Input neurons =====
        for (let i = 0; i < n; i++) {
            const py = rowYs[i];
            const xv = xVals[i];

            // Circle
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(inX, py, NEURON_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Coloured cell inside, same palette/intensity as the X cells above
            const cx = inX - FT_CELL / 2, cy = py - FT_CELL / 2;
            drawCubeCell(ctx,
                { x: cx,            y: cy },
                { x: cx + FT_CELL,  y: cy },
                { x: cx + FT_CELL,  y: cy + FT_CELL },
                { x: cx,            y: cy + FT_CELL },
                xv, 1.0, { face: 'front', showNumber: true });

            // Symbol next to the neuron (e.g. x₁ = tempo)
            drawLabel(ctx, inX - NEURON_R - 6, py + 3,
                      isW1 ? `x${sub(i + 1)}` : `h${sub(i + 1)}`,
                      { align: 'right', size: 11, bold: true });
            // Feature/input name below the neuron
            drawLabel(ctx, inX, py + NEURON_R + 12,
                      `${inputs[i]} = ${xv.toFixed(2)}`,
                      { size: 9, muted: true });
        }

        // ===== Activation gate (small box just before the output neuron) =====
        // Drawn as a rounded rectangle showing the activation symbol — same place
        // the polariser sits in the cube figure above.
        const gateW = 36, gateH = 30;
        const gx = actX - gateW / 2, gy = diagMidY - gateH / 2;
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        const r = 6;
        ctx.moveTo(gx + r, gy);
        ctx.lineTo(gx + gateW - r, gy);
        ctx.quadraticCurveTo(gx + gateW, gy, gx + gateW, gy + r);
        ctx.lineTo(gx + gateW, gy + gateH - r);
        ctx.quadraticCurveTo(gx + gateW, gy + gateH, gx + gateW - r, gy + gateH);
        ctx.lineTo(gx + r, gy + gateH);
        ctx.quadraticCurveTo(gx, gy + gateH, gx, gy + gateH - r);
        ctx.lineTo(gx, gy + r);
        ctx.quadraticCurveTo(gx, gy, gx + r, gy);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#222';
        ctx.font = '11px var(--font-sans, system-ui), sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(actLabel, actX, diagMidY);
        ctx.textBaseline = 'alphabetic';
        drawLabel(ctx, actX, gy - 4, 'activation', { size: 9, muted: true });

        // Wire from activation gate to output neuron
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(actX + gateW / 2, diagMidY);
        ctx.lineTo(outX - NEURON_R, diagMidY);
        ctx.stroke();

        // ===== Output neuron =====
        // Same coloured-cell-inside-circle treatment, using the H (or Y) palette.
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(outX, diagMidY, NEURON_R, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Cell inside — for the hidden layer, use h value as-is; for output, use
        // signed (v - 0.5) * 2 so above/below the 0.5 decision boundary reads as
        // pos/neg, matching the Y column above.
        const outCellVal = isW1 ? postAct : (postAct - 0.5) * 2;
        const ocx = outX - FT_CELL / 2, ocy = diagMidY - FT_CELL / 2;
        drawCubeCell(ctx,
            { x: ocx,            y: ocy },
            { x: ocx + FT_CELL,  y: ocy },
            { x: ocx + FT_CELL,  y: ocy + FT_CELL },
            { x: ocx,            y: ocy + FT_CELL },
            outCellVal, Math.max(0.5, Math.abs(outCellVal) * 1.1),
            { face: 'front', showNumber: false });
        // Numeric value over the cell so it's always legible
        ctx.fillStyle = '#222';
        ctx.font = 'bold 11px var(--font-mono, ui-monospace), monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(postAct.toFixed(2), outX, diagMidY);
        ctx.textBaseline = 'alphabetic';
        // Label this neuron's output symbol (e.g. h₁,₁ or ŷ₁) above the circle
        drawLabel(ctx, outX, diagMidY - NEURON_R - 6,
                  hSym, { size: 11, bold: true });
        drawLabel(ctx, outX, diagMidY + NEURON_R + 12,
                  isW1 ? `neuron ${neuronName}` : 'output',
                  { size: 9, muted: true });
    }

    // Tiny helper: unicode subscript digits for nicer in-canvas labels
    function sub(n) {
        const s = String(n);
        const map = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                      '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
        return s.split('').map(c => map[c] || c).join('');
    }
    function actSymbol() {
        return activation === 'none' ? 'id' : activation === 'relu' ? 'ReLU' : 'σ';
    }
    function drawNode(ctx, cx, cy, label) {
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 13, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#333';
        ctx.font = '11px var(--font-sans, system-ui), sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, cy);
        ctx.textBaseline = 'alphabetic';
    }

    // ---------- Hit-testing & interaction ----------
    function within(mx, my, rx, ry, rw, rh) {
        return mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh;
    }

    function hitTestCell(mx, my) {
        const { xCols, cube1Origin, cube2Origin, b1RowY, b2RowY,
                xRefLeft, xRefY, w1RefLeft, w1RefY, w2RefLeft, w2RefY,
                xPivot, w1Pivot, w2Pivot,
                xShearA, xShearB, wShearC, wShearD, FLAT } = layout;
        if (!cube1Origin) return null;

        // Inverse of: translate(pivot); transform(a,b,c,d,0,0); translate(-pivot)
        function untransform(mx, my, pivotX, pivotY, a, b, c, d) {
            const dx = mx - pivotX, dy = my - pivotY;
            const det = a * d - b * c;
            return {
                x: pivotX + (d * dx - c * dy) / det,
                y: pivotY + (-b * dx + a * dy) / det,
            };
        }

        // W₁ flat panel — vertical shear. Display rows are REVERSED: panel row k
        // shows W1[D-1-k][...], so when we hit a cell at physical row k we
        // return data index D-1-k.
        {
            const p = untransform(mx, my, w1Pivot.x, w1Pivot.y, 1, 0, wShearC, wShearD);
            for (let k = 0; k < D; k++) {
                for (let j = 0; j < K; j++) {
                    if (within(p.x, p.y, w1RefLeft + j * FLAT, w1RefY + k * FLAT, FLAT, FLAT)) {
                        return { mat: 'W1', i: D - 1 - k, j, range: WEIGHT_RANGE };
                    }
                }
            }
        }
        // W₂ flat panel — also reversed: panel row j shows W2[K-1-j][0]
        {
            const p = untransform(mx, my, w2Pivot.x, w2Pivot.y, 1, 0, wShearC, wShearD);
            for (let j = 0; j < K; j++) {
                if (within(p.x, p.y, w2RefLeft, w2RefY + j * FLAT, FLAT, FLAT)) {
                    return { mat: 'W2', i: K - 1 - j, j: 0, range: WEIGHT_RANGE };
                }
            }
        }
        // W₁ cells on cube top face (parallelogram hit-test)
        for (let k = 0; k < D; k++) {
            for (let j = 0; j < K; j++) {
                const p1 = project(j,     k,     0, cube1Origin);
                const p2 = project(j + 1, k,     0, cube1Origin);
                const p3 = project(j + 1, k + 1, 0, cube1Origin);
                const p4 = project(j,     k + 1, 0, cube1Origin);
                if (pointInQuad(mx, my, p1, p2, p3, p4)) {
                    return { mat: 'W1', i: k, j, range: WEIGHT_RANGE };
                }
            }
        }
        // W₂ cells on cube 2 top face
        for (let k = 0; k < K; k++) {
            const p1 = project(0, k,     0, cube2Origin);
            const p2 = project(1, k,     0, cube2Origin);
            const p3 = project(1, k + 1, 0, cube2Origin);
            const p4 = project(0, k + 1, 0, cube2Origin);
            if (pointInQuad(mx, my, p1, p2, p3, p4)) {
                return { mat: 'W2', i: k, j: 0, range: WEIGHT_RANGE };
            }
        }
        // b₁ row
        for (let j = 0; j < K; j++) {
            if (within(mx, my, xCols.cube1Front + j * CELL3D, b1RowY, CELL3D, CELL3D)) {
                return { mat: 'b1', i: 0, j, range: 0.6 };
            }
        }
        // b₂ row (single cell)
        if (within(mx, my, xCols.cube2Front, b2RowY, CELL3D, CELL3D)) {
            return { mat: 'b2', i: 0, j: 0, range: 0.6 };
        }
        return null;
    }

    function hitTestOutput(mx, my) {
        const { xCols, cubeFrontTop } = layout;
        if (cubeFrontTop === undefined) return null;
        // Z₁ front face
        if (mx >= xCols.cube1Front && mx <= xCols.cube1Front + K * CELL3D &&
            my >= cubeFrontTop      && my <= cubeFrontTop      + N * CELL3D) {
            const j = Math.floor((mx - xCols.cube1Front) / CELL3D);
            const i = Math.floor((my - cubeFrontTop)      / CELL3D);
            return { mat: 'Z1', i, j };
        }
        // Z₂ front face
        if (mx >= xCols.cube2Front && mx <= xCols.cube2Front + CELL3D &&
            my >= cubeFrontTop      && my <= cubeFrontTop      + N * CELL3D) {
            const i = Math.floor((my - cubeFrontTop) / CELL3D);
            return { mat: 'Z2', i, j: 0 };
        }
        return null;
    }

    function valueAt(hit) {
        switch (hit.mat) {
            case 'W1': return W1[hit.i][hit.j];
            case 'b1': return b1[hit.j];
            case 'W2': return W2[hit.i][hit.j];
            case 'b2': return b2;
        }
    }
    function setValueAt(hit, v) {
        v = clamp(v, -hit.range, hit.range);
        switch (hit.mat) {
            case 'W1': W1[hit.i][hit.j] = v; break;
            case 'b1': b1[hit.j] = v; break;
            case 'W2': W2[hit.i][hit.j] = v; break;
            case 'b2': b2 = v; break;
        }
    }

    function mousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    canvas.addEventListener('mousedown', e => {
        const { x, y } = mousePos(e);
        const hit = hitTestCell(x, y);
        if (!hit) return;
        // Select the neuron (column) clicked
        if (hit.mat === 'W1' || hit.mat === 'b1') selected = { mat: 'W1', col: hit.j };
        else if (hit.mat === 'W2' || hit.mat === 'b2') selected = { mat: 'W2', col: 0 };
        drag = { ...hit, startY: y, startVal: valueAt(hit) };
        canvas.style.cursor = 'grabbing';
        render();
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        const { x, y } = mousePos(e);
        if (!drag) {
            // Hover detection: highlight Z₁/Z₂ row × col contributions
            const out = hitTestOutput(x, y);
            const hoverChanged = JSON.stringify(out) !== JSON.stringify(hovered);
            hovered = out;
            const hit = hitTestCell(x, y);
            canvas.style.cursor = hit ? 'grab' : (out ? 'pointer' : '');
            if (hoverChanged) render();
            return;
        }
        const dy = drag.startY - y;
        const v = drag.startVal + (dy / 40) * drag.range;
        setValueAt(drag, v);
        render();
    });
    canvas.addEventListener('mouseleave', () => {
        if (hovered) { hovered = null; render(); }
    });
    window.addEventListener('mouseup', () => {
        drag = null;
        canvas.style.cursor = '';
    });

    // Touch support (basic)
    canvas.addEventListener('touchstart', e => {
        if (e.touches.length !== 1) return;
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const x = t.clientX - rect.left, y = t.clientY - rect.top;
        const hit = hitTestCell(x, y);
        if (!hit) return;
        if (hit.mat === 'W1' || hit.mat === 'b1') selected = { mat: 'W1', col: hit.j };
        else                                       selected = { mat: 'W2', col: 0 };
        drag = { ...hit, startY: y, startVal: valueAt(hit) };
        render();
        e.preventDefault();
    }, { passive: false });
    window.addEventListener('touchmove', e => {
        if (!drag) return;
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const y = t.clientY - rect.top;
        const dy = drag.startY - y;
        const v = drag.startVal + (dy / 40) * drag.range;
        setValueAt(drag, v);
        render();
        e.preventDefault();
    }, { passive: false });
    window.addEventListener('touchend', () => { drag = null; });

    // ---------- DOM controls ----------
    actRadios.forEach(r => r.addEventListener('change', () => {
        if (r.checked) { activation = r.value; render(); }
    }));
    if (numbersCb) {
        numbersCb.addEventListener('change', () => {
            showNumbers = numbersCb.checked;
            render();
        });
    }
    nMinusBtn.addEventListener('click', () => {
        if (K > 1) { K--; randomize(); nCountEl.textContent = K; render(); }
    });
    nPlusBtn.addEventListener('click', () => {
        if (K < 4) { K++; randomize(); nCountEl.textContent = K; render(); }
    });
    randBtn.addEventListener('click', () => { randomize(); render(); });

    // ---------- Init ----------
    // Hand-tuned starting weights for a "looks reasonable" first impression
    function presetInit() {
        K = 3;
        nCountEl.textContent = K;
        // Weights that solve the XOR-like target pattern WITH ReLU. With
        // activation set to "none", the W₂·W₁ product reduces to ~0, so all
        // predictions collapse to the same value — the visible demonstration
        // that stacking matmuls without a nonlinearity gives no extra power.
        //   n1 = ReLU( tempo + mood − 0.5)  →  fires when both are positive
        //   n2 = ReLU(−tempo − mood − 0.5)  →  fires when both are negative
        //   n3 unused (≈ 0)
        //   y  = sigmoid(2·n1 + 2·n2 − 1)
        W1 = [
            [ 1.0, -1.0,  0.0],     // tempo → n1, n2, n3
            [ 1.0, -1.0,  0.0],     // mood  → n1, n2, n3
        ];
        b1 = [-0.5, -0.5,  0.0];
        W2 = [[ 2.0], [ 2.0], [ 0.0]];
        b2 = -1.0;
        selected = { mat: 'W1', col: 0 };
    }
    presetInit();
    render();
    window.addEventListener('resize', render);
})();
