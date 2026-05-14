/* Interactive RNN recurrence viz.
 * A sequence unrolled across timesteps. At each step the same RNN cell takes
 * the current token + previous hidden state and produces a new hidden state.
 * Click any step to inspect it. The "Vanishing gradient" demo shows why
 * long-range dependencies are hard for vanilla RNNs. */

(function () {
    const canvas    = document.getElementById('viz-rnn-canvas');
    const seqSelect = document.getElementById('viz-rnn-seq');
    const cellSelect = document.getElementById('viz-rnn-cell');
    const playBtn   = document.getElementById('viz-rnn-play');
    const formulaEl = document.getElementById('viz-rnn-formula');
    if (!canvas) return;

    // ----- Sequences to choose from -----
    const SEQUENCES = {
        'The quick brown fox jumps':       ['The', 'quick', 'brown', 'fox', 'jumps'],
        'I really enjoyed it':             ['I', 'really', 'enjoyed', 'it'],
        'long sequence (vanishing demo)':  ['The', 'cat', 'that', 'I', 'saw', 'yesterday', 'was', 'big'],
    };
    let tokens = SEQUENCES['The quick brown fox jumps'];

    // ----- Hidden-state dimensions -----
    const H_DIM = 6;     // dimensionality of hidden state vector

    // ----- Cell choice: vanilla RNN vs (simplified) LSTM-like -----
    let cellKind = 'vanilla';   // or 'lstm'

    // ----- Hand-crafted weights so the hidden state evolves visibly -----
    // We hash each token to a deterministic input vector and apply a recurrence.
    function hashToken(tok) {
        // Stable per-token vector (not random per page-load)
        let h = 2166136261;
        for (const ch of tok) {
            h ^= ch.charCodeAt(0);
            h = (h * 16777619) >>> 0;
        }
        const v = new Float32Array(H_DIM);
        for (let i = 0; i < H_DIM; i++) {
            h = (h * 1103515245 + 12345) >>> 0;
            v[i] = ((h & 0xffff) / 0xffff - 0.5) * 1.6;
        }
        return v;
    }

    // Recurrence weight matrix W_h (H_DIM × H_DIM). Spectral-radius ≈ 0.9 for vanilla
    // (so vanishing demo is visible); approaching 1.0 for LSTM-like.
    function buildWh(decay) {
        const W = [];
        let seed = 12345;
        const rand = () => {
            seed = (seed * 1103515245 + 12345) >>> 0;
            return ((seed & 0xffff) / 0xffff - 0.5);
        };
        for (let i = 0; i < H_DIM; i++) {
            const row = new Float32Array(H_DIM);
            for (let j = 0; j < H_DIM; j++) row[j] = rand();
            W.push(row);
        }
        // Normalise rows for stability, then scale by `decay`
        for (let i = 0; i < H_DIM; i++) {
            let n = 0;
            for (let j = 0; j < H_DIM; j++) n += W[i][j] * W[i][j];
            n = Math.sqrt(n) + 1e-9;
            for (let j = 0; j < H_DIM; j++) W[i][j] = (W[i][j] / n) * decay;
        }
        return W;
    }

    // Recurrence: h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
    // We just use W_x = I and b = 0 for transparency.
    function runRecurrence(seq) {
        const decay = (cellKind === 'lstm') ? 0.97 : 0.65;
        const W_h = buildWh(decay);
        const states = [];
        let h = new Float32Array(H_DIM);   // h_0 = 0
        states.push(h.slice());
        for (let t = 0; t < seq.length; t++) {
            const x = hashToken(seq[t]);
            const next = new Float32Array(H_DIM);
            for (let i = 0; i < H_DIM; i++) {
                let s = x[i];
                for (let j = 0; j < H_DIM; j++) s += W_h[i][j] * h[j];
                next[i] = Math.tanh(s);
            }
            h = next;
            states.push(h.slice());
        }
        return states;   // length T+1 (includes h_0)
    }

    let states = runRecurrence(tokens);
    let activeT = 1;          // which timestep is highlighted (1-indexed; 0 is h_0)
    let playing = true;
    let lastStep = 0;
    const STEP_MS = 900;

    // ----- Canvas sizing -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        // Height scales with sequence length (more cells need more vertical room)
        const cssH = Math.round(Math.min(420, Math.max(280, cssW * 0.38)));
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
    function stepGeometry() {
        const T   = tokens.length;
        const pad = 16;
        const colW = (W - 2 * pad) / T;
        const rows = {
            token:  { y: 24,  h: 32 },
            cell:   { y: 80,  h: 56 },
            hidden: { y: 168, h: H - 168 - 28 },
        };
        return { pad, colW, rows };
    }

    // ----- Drawing -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const geo = stepGeometry();
        const T   = tokens.length;

        // Section labels
        ctx.fillStyle = '#9a9a9a';
        ctx.font      = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('INPUT',        geo.pad, 6);
        ctx.fillText('RNN CELL',     geo.pad, 64);
        ctx.fillText('HIDDEN STATE', geo.pad, 152);

        // Horizontal hidden-state arrows (between timesteps)
        for (let t = 0; t < T - 1; t++) {
            const x1 = geo.pad + geo.colW * t + geo.colW * 0.78;
            const x2 = geo.pad + geo.colW * (t + 1) + geo.colW * 0.22;
            const y  = geo.rows.cell.y + geo.rows.cell.h / 2;
            const accent = (t + 1 === activeT) ? '#ea7959' : 'rgba(79, 70, 229, 0.35)';
            ctx.strokeStyle = accent;
            ctx.lineWidth = (t + 1 === activeT) ? 2 : 1.5;
            ctx.beginPath();
            ctx.moveTo(x1, y);
            ctx.lineTo(x2 - 6, y);
            ctx.stroke();
            // Arrowhead
            ctx.fillStyle = accent;
            ctx.beginPath();
            ctx.moveTo(x2 - 6, y - 4);
            ctx.lineTo(x2,     y);
            ctx.lineTo(x2 - 6, y + 4);
            ctx.closePath();
            ctx.fill();
        }

        // Each timestep column
        for (let t = 0; t < T; t++) {
            const cx = geo.pad + geo.colW * t + geo.colW / 2;
            const isActive = (t + 1 === activeT);

            // Token box (top)
            drawTokenBox(cx, geo.rows.token.y, tokens[t], isActive);

            // Vertical arrow into cell
            arrowDown(cx, geo.rows.token.y + geo.rows.token.h + 2, geo.rows.cell.y - 4, isActive);

            // RNN cell (rounded box, gradient on active)
            drawRnnCell(cx, geo.rows.cell.y, geo.rows.cell.h, isActive, t + 1);

            // Vertical arrow into hidden state
            arrowDown(cx, geo.rows.cell.y + geo.rows.cell.h + 2, geo.rows.hidden.y - 4, isActive);

            // Hidden state bars
            drawHiddenState(cx, geo.rows.hidden.y, geo.rows.hidden.h, states[t + 1], isActive);

            // Subscript label below
            ctx.fillStyle = isActive ? '#ea7959' : '#5f5f5f';
            ctx.font      = isActive
                ? '600 11px "Inter", system-ui, sans-serif'
                : '11px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(`h${sub(t + 1)}`, cx, H - 24);
        }

        updateFormula();
    }

    function sub(n) {
        const map = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
        return String(n).split('').map(c => map[+c] ?? c).join('');
    }

    function drawTokenBox(cx, y, text, active) {
        const w = 64, h = 28;
        ctx.fillStyle   = active ? '#4f46e5' : '#ffffff';
        ctx.strokeStyle = active ? '#3730a3' : '#d6d3ca';
        ctx.lineWidth   = 1.25;
        roundedRect(ctx, cx - w / 2, y, w, h, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = active ? '#ffffff' : '#1a1a1a';
        ctx.font      = '600 12px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, cx, y + h / 2 + 1);
    }

    function drawRnnCell(cx, y, h, active, idx) {
        const w = 70;
        // Soft glow on active
        if (active) {
            ctx.fillStyle = 'rgba(234, 121, 89, 0.18)';
            roundedRect(ctx, cx - w / 2 - 3, y - 3, w + 6, h + 6, 10);
            ctx.fill();
        }
        // Main body
        const grd = ctx.createLinearGradient(0, y, 0, y + h);
        if (active) {
            grd.addColorStop(0, '#fff3eb');
            grd.addColorStop(1, '#fde0d2');
        } else {
            grd.addColorStop(0, '#ffffff');
            grd.addColorStop(1, '#f5f3ee');
        }
        ctx.fillStyle = grd;
        ctx.strokeStyle = active ? '#ea7959' : '#d6d3ca';
        ctx.lineWidth = active ? 2 : 1.25;
        roundedRect(ctx, cx - w / 2, y, w, h, 8);
        ctx.fill();
        ctx.stroke();

        // Cell label
        ctx.fillStyle = active ? '#8a3a1f' : '#5f5f5f';
        ctx.font      = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(cellKind === 'lstm' ? 'LSTM' : 'RNN', cx, y + 8);
        ctx.fillStyle = active ? '#1a1a1a' : '#9a9a9a';
        ctx.font      = '10px "JetBrains Mono", monospace';
        ctx.fillText(`t=${idx}`, cx, y + 24);
        // Tiny "same weights" hint on first cell
        if (idx === 1) {
            ctx.fillStyle = '#9a9a9a';
            ctx.font      = 'italic 9px "Inter", system-ui, sans-serif';
            ctx.fillText('same weights ▸', cx, y + 40);
        }
    }

    function arrowDown(cx, y1, y2, active) {
        const accent = active ? '#ea7959' : '#c8c4b8';
        ctx.strokeStyle = accent;
        ctx.lineWidth = active ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(cx, y1);
        ctx.lineTo(cx, y2 - 6);
        ctx.stroke();
        ctx.fillStyle = accent;
        ctx.beginPath();
        ctx.moveTo(cx - 4, y2 - 6);
        ctx.lineTo(cx,     y2);
        ctx.lineTo(cx + 4, y2 - 6);
        ctx.closePath();
        ctx.fill();
    }

    function drawHiddenState(cx, y, h, vec, active) {
        const total = H_DIM;
        const padX = 4;
        const cellW = Math.min(10, (70 - padX * 2) / total);
        const bx = cx - (cellW * total) / 2;
        // Background frame
        if (active) {
            ctx.fillStyle = 'rgba(79, 70, 229, 0.08)';
            roundedRect(ctx, bx - 4, y - 4, cellW * total + 8, h + 8, 6);
            ctx.fill();
        }
        // Each dimension as a vertical bar (centred at zero)
        const midY = y + h / 2;
        for (let i = 0; i < total; i++) {
            const v = vec[i];
            const barH = Math.abs(v) * (h / 2 - 4);
            const x = bx + i * cellW;
            if (v >= 0) {
                ctx.fillStyle = active ? `rgba(79, 70, 229, ${0.6 + 0.4 * v})`
                                       : `rgba(79, 70, 229, ${0.25 + 0.25 * v})`;
                ctx.fillRect(x + 1, midY - barH, cellW - 2, barH);
            } else {
                ctx.fillStyle = active ? `rgba(234, 121, 89, ${0.6 + 0.4 * (-v)})`
                                       : `rgba(234, 121, 89, ${0.25 + 0.25 * (-v)})`;
                ctx.fillRect(x + 1, midY, cellW - 2, barH);
            }
        }
        // Zero line
        ctx.strokeStyle = 'rgba(0,0,0,0.10)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx, midY);
        ctx.lineTo(bx + cellW * total, midY);
        ctx.stroke();
    }

    function roundedRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y,     x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x,     y + h, r);
        ctx.arcTo(x,     y + h, x,     y,     r);
        ctx.arcTo(x,     y,     x + w, y,     r);
        ctx.closePath();
    }

    // ----- Formula display -----
    function updateFormula() {
        if (!formulaEl) return;
        const t = activeT;
        if (t < 1) {
            formulaEl.innerHTML = '<span class="lbl">Initial state</span>h₀ = 0';
            return;
        }
        const tok    = tokens[t - 1];
        const fmtVec = v => Array.from(v).map(x => x.toFixed(2)).join(', ');
        const prev   = `h${sub(t - 1)}`;
        const next   = `h${sub(t)}`;
        formulaEl.innerHTML = `
            <span class="lbl">Step ${t}</span>
            ${next} = tanh(W<sub>h</sub>·${prev} + W<sub>x</sub>·"${tok}" + b) =
            <span class="sum">[${fmtVec(states[t])}]</span>
        `;
    }

    // ----- Interactions -----
    function colAt(x) {
        const geo = stepGeometry();
        const i = Math.floor((x - geo.pad) / geo.colW);
        return (i >= 0 && i < tokens.length) ? i + 1 : -1;
    }
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const col  = colAt(e.clientX - rect.left);
        if (col > 0) {
            playing = false;
            updatePlayBtn();
            activeT = col;
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
            states = runRecurrence(tokens);
            activeT = 1;
            resize();
        });
    }
    if (cellSelect) {
        cellSelect.innerHTML = `
            <option value="vanilla">Vanilla RNN</option>
            <option value="lstm">LSTM-like (better memory)</option>
        `;
        cellSelect.addEventListener('change', () => {
            cellKind = cellSelect.value;
            states = runRecurrence(tokens);
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
