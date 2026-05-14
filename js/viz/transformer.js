/* Interactive transformer self-attention viz.
 * Four hand-crafted attention heads showing distinct patterns. Click a token
 * to see what it attends to. Click any cell in the heatmap to do the same. */

(function () {
    const TOKENS = ['The', 'quick', 'brown', 'fox', 'jumps'];
    const N      = TOKENS.length;

    // Each head is a row-stochastic N×N matrix.
    // Rows = queries; columns = keys.
    const HEADS = [
        {
            name:  'Look back',
            blurb: 'Each token attends primarily to the <strong>previous</strong> token. ' +
                   'The simplest positional pattern transformers learn, even at random init.',
            // weights[query][key]
            weights: [
                [.60, .10, .10, .10, .10],   // The      — no previous; falls back to self
                [.70, .15, .05, .05, .05],   // quick    → The
                [.05, .70, .15, .05, .05],   // brown    → quick
                [.05, .05, .70, .15, .05],   // fox      → brown
                [.05, .05, .05, .70, .15],   // jumps    → fox
            ],
        },
        {
            name:  'Look ahead',
            blurb: 'The opposite: each token attends to what comes <strong>next</strong>. ' +
                   'Useful for predicting upcoming tokens — common in BERT-style bidirectional models.',
            weights: [
                [.15, .70, .05, .05, .05],   // The      → quick
                [.05, .15, .70, .05, .05],   // quick    → brown
                [.05, .05, .15, .70, .05],   // brown    → fox
                [.05, .05, .05, .15, .70],   // fox      → jumps
                [.10, .10, .10, .10, .60],   // jumps    — no next; falls back to self
            ],
        },
        {
            name:  'Modifier → noun',
            blurb: 'Semantic linking: the adjectives <strong>"quick"</strong> and <strong>"brown"</strong> ' +
                   'both attend forward to <strong>"fox"</strong> — the noun they modify. Real transformers learn ' +
                   'patterns exactly like this, often in middle layers.',
            weights: [
                [.70, .075,.075,.075,.075],  // The      — self
                [.05, .15, .05, .70, .05],   // quick    → fox
                [.05, .05, .15, .70, .05],   // brown    → fox
                [.05, .075,.075,.70, .10],   // fox      — self (with a bit toward jumps)
                [.05, .05, .05, .15, .70],   // jumps    — self
            ],
        },
        {
            name:  'Verb → subject',
            blurb: 'Long-range structural attention: the verb <strong>"jumps"</strong> attends back to its ' +
                   'subject <strong>"fox"</strong> across the sentence. This is the kind of pattern that makes ' +
                   'transformers good at language understanding.',
            weights: [
                [.70, .075,.075,.075,.075],  // The
                [.05, .70, .05, .15, .05],   // quick
                [.05, .05, .70, .15, .05],   // brown
                [.10, .10, .10, .60, .10],   // fox      — slight self
                [.05, .05, .05, .70, .15],   // jumps    → fox
            ],
        },
    ];

    let activeHead  = 0;
    let activeQuery = 3;   // start with "fox" — most interesting under most heads

    // ----- DOM refs -----
    const canvas    = document.getElementById('viz-tx-canvas');
    const tabsEl    = document.getElementById('viz-tx-heads');
    const descEl    = document.getElementById('viz-tx-desc');
    if (!canvas || !tabsEl || !descEl) return;

    let ctx;
    let W = 0, H = 0;

    // Layout (in CSS pixels — internal canvas buffer is scaled by DPR)
    const HEIGHT_RATIO = 0.62;   // canvas height / canvas width
    // Top half: arcs view; bottom half: heatmap

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(360, Math.round(rect.width));
        const cssH = Math.round(cssW * HEIGHT_RATIO);
        W = cssW;
        H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    // ----- Layout helpers -----
    function arcsRegion() {
        return { x: 8, y: 8, w: W - 16, h: H * 0.5 - 10 };
    }
    function heatRegion() {
        const top = H * 0.5 + 6;
        return { x: 8, y: top, w: W - 16, h: H - top - 8 };
    }
    function tokenBox(i, region) {
        // 5 equally-spaced token boxes along the bottom of the arcs region
        const pad = 24;
        const w   = (region.w - pad * 2) / N;
        return {
            cx: region.x + pad + w * i + w / 2,
            cy: region.y + region.h - 22,
            w:  Math.min(78, w - 8),
            h:  28,
        };
    }

    // ----- Drawing -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);

        // Background (subtle)
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        drawArcsView();
        drawHeatmap();
    }

    function drawArcsView() {
        const reg = arcsRegion();

        // Section label
        ctx.fillStyle = '#5f5f5f';
        ctx.font      = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('ATTENTION', reg.x + 4, reg.y + 4);
        ctx.fillStyle = '#9a9a9a';
        ctx.font      = '10px "Inter", system-ui, sans-serif';
        ctx.fillText('Click a token to make it the query', reg.x + 76, reg.y + 4);

        // Arcs from active query to each key
        const w = HEADS[activeHead].weights[activeQuery];
        const queryBox = tokenBox(activeQuery, reg);
        for (let k = 0; k < N; k++) {
            const weight = w[k];
            const keyBox = tokenBox(k, reg);

            const x1 = queryBox.cx;
            const y1 = queryBox.cy - queryBox.h / 2 - 2;
            const x2 = keyBox.cx;
            const y2 = keyBox.cy - keyBox.h / 2 - 2;

            // Arc peak: higher for longer-distance arcs
            const distFrac = Math.abs(activeQuery - k) / (N - 1);
            const peakY    = reg.y + 18 + (1 - distFrac) * 38;

            const cx1 = x1;
            const cx2 = x2;

            const alpha    = 0.08 + weight * 0.85;
            const width_px = 1 + weight * 7;

            // Soft glow for the strongest arc
            if (weight > 0.45) {
                ctx.strokeStyle = `rgba(79, 70, 229, 0.18)`;
                ctx.lineWidth   = width_px + 4;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.bezierCurveTo(cx1, peakY, cx2, peakY, x2, y2);
                ctx.stroke();
            }

            ctx.strokeStyle = `rgba(79, 70, 229, ${alpha})`;
            ctx.lineWidth   = width_px;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.bezierCurveTo(cx1, peakY, cx2, peakY, x2, y2);
            ctx.stroke();

            // Weight label near the destination, if visible
            if (weight > 0.08) {
                ctx.fillStyle = `rgba(79, 70, 229, ${Math.min(1, 0.35 + weight)})`;
                ctx.font      = '11px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const lblY = peakY + (y2 - peakY) * 0.55;
                ctx.fillText(weight.toFixed(2), (x1 + x2) / 2, lblY);
            }
        }

        // Tokens
        for (let i = 0; i < N; i++) {
            const b = tokenBox(i, reg);
            const isQuery = (i === activeQuery);
            // Fill
            ctx.fillStyle = isQuery ? '#4f46e5' : '#ffffff';
            roundedRect(ctx, b.cx - b.w / 2, b.cy - b.h / 2, b.w, b.h, 8);
            ctx.fill();
            // Border
            ctx.strokeStyle = isQuery ? '#3730a3' : '#d6d3ca';
            ctx.lineWidth   = 1.25;
            roundedRect(ctx, b.cx - b.w / 2, b.cy - b.h / 2, b.w, b.h, 8);
            ctx.stroke();
            // Text
            ctx.fillStyle = isQuery ? '#ffffff' : '#1a1a1a';
            ctx.font      = '600 13px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(TOKENS[i], b.cx, b.cy + 1);
        }
    }

    function drawHeatmap() {
        const reg = heatRegion();

        // Section label
        ctx.fillStyle = '#5f5f5f';
        ctx.font      = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('ATTENTION MATRIX', reg.x + 4, reg.y);
        ctx.fillStyle = '#9a9a9a';
        ctx.font      = '10px "Inter", system-ui, sans-serif';
        ctx.fillText('rows = queries · columns = keys', reg.x + 130, reg.y);

        const labelLeft = 64;
        const labelTop  = 22;
        const gridX = reg.x + labelLeft;
        const gridY = reg.y + labelTop;
        const gridW = reg.w - labelLeft - 4;
        const gridH = reg.h - labelTop - 4;
        const cellW = gridW / N;
        const cellH = gridH / N;

        const W_ = HEADS[activeHead].weights;

        // Cells
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                const v = W_[r][c];
                ctx.fillStyle = heatColour(v);
                ctx.fillRect(gridX + c * cellW, gridY + r * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Cell text (weight value) — only if cell is large enough
        if (cellW > 40 && cellH > 22) {
            for (let r = 0; r < N; r++) {
                for (let c = 0; c < N; c++) {
                    const v = W_[r][c];
                    ctx.fillStyle = v > 0.4 ? '#ffffff' : 'rgba(26, 26, 26, 0.55)';
                    ctx.font      = '10px "JetBrains Mono", monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(v.toFixed(2), gridX + c * cellW + cellW / 2, gridY + r * cellH + cellH / 2);
                }
            }
        }

        // Highlight the active query row
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth   = 2;
        ctx.strokeRect(gridX - 0.5, gridY + activeQuery * cellH - 0.5, gridW + 1, cellH + 1);

        // Grid hairlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.lineWidth   = 1;
        for (let i = 0; i <= N; i++) {
            ctx.beginPath();
            ctx.moveTo(gridX, gridY + i * cellH); ctx.lineTo(gridX + gridW, gridY + i * cellH); ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(gridX + i * cellW, gridY); ctx.lineTo(gridX + i * cellW, gridY + gridH); ctx.stroke();
        }

        // Row labels (left)
        ctx.fillStyle = '#1a1a1a';
        ctx.font      = '11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let r = 0; r < N; r++) {
            ctx.fillStyle = (r === activeQuery) ? '#ea7959' : '#1a1a1a';
            ctx.fillText(TOKENS[r], gridX - 6, gridY + r * cellH + cellH / 2);
        }
        // Column labels (top)
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        for (let c = 0; c < N; c++) {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillText(TOKENS[c], gridX + c * cellW + cellW / 2, gridY - 4);
        }
    }

    // ----- Helpers -----
    function roundedRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y,     x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x,     y + h, r);
        ctx.arcTo(x,     y + h, x,     y,     r);
        ctx.arcTo(x,     y,     x + w, y,     r);
        ctx.closePath();
    }
    function heatColour(t) {
        // Cream → light indigo → deep indigo (perceptually monotone-ish)
        const stops = [
            { t: 0.00, c: [251, 250, 247] },
            { t: 0.30, c: [221, 218, 245] },
            { t: 0.60, c: [127, 117, 226] },
            { t: 1.00, c: [ 55,  48, 163] },
        ];
        for (let i = 1; i < stops.length; i++) {
            if (t <= stops[i].t) {
                const u = (t - stops[i-1].t) / (stops[i].t - stops[i-1].t);
                const a = stops[i-1].c, b = stops[i].c;
                return `rgb(${(a[0] + (b[0] - a[0]) * u) | 0}, ` +
                       `${(a[1] + (b[1] - a[1]) * u) | 0}, ` +
                       `${(a[2] + (b[2] - a[2]) * u) | 0})`;
            }
        }
        return 'rgb(55, 48, 163)';
    }

    function updateDesc() {
        descEl.innerHTML = HEADS[activeHead].blurb;
    }

    // ----- Interactions -----
    function tokenAt(x, y) {
        const reg = arcsRegion();
        for (let i = 0; i < N; i++) {
            const b = tokenBox(i, reg);
            if (Math.abs(x - b.cx) < b.w / 2 && Math.abs(y - b.cy) < b.h / 2 + 6) {
                return i;
            }
        }
        return -1;
    }
    function heatmapRowAt(x, y) {
        const reg = heatRegion();
        const labelLeft = 64;
        const labelTop  = 22;
        const gridX = reg.x + labelLeft;
        const gridY = reg.y + labelTop;
        const gridW = reg.w - labelLeft - 4;
        const gridH = reg.h - labelTop - 4;
        if (x < gridX || x > gridX + gridW || y < gridY || y > gridY + gridH) return -1;
        return Math.min(N - 1, Math.max(0, Math.floor((y - gridY) / (gridH / N))));
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }
    function onClick(e) {
        const [x, y] = getPos(e);
        const ti = tokenAt(x, y);
        if (ti >= 0) { activeQuery = ti; draw(); return; }
        const ri = heatmapRowAt(x, y);
        if (ri >= 0) { activeQuery = ri; draw(); return; }
    }
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); onClick(e); });

    // ----- Head tab buttons -----
    HEADS.forEach((h, i) => {
        const btn = document.createElement('button');
        btn.type        = 'button';
        btn.className   = 'viz-tx-head';
        btn.textContent = h.name;
        if (i === activeHead) btn.classList.add('active');
        btn.addEventListener('click', () => {
            activeHead = i;
            tabsEl.querySelectorAll('.viz-tx-head').forEach((b, j) =>
                b.classList.toggle('active', j === i));
            updateDesc();
            draw();
        });
        tabsEl.appendChild(btn);
    });

    // ----- Init -----
    updateDesc();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
