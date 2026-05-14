/* Interactive RL viz — tabular Q-learning on a small grid.
 * 5×5 grid. Agent starts (0,0). Goal at (4,4) gives +1. Walls at a few
 * cells give -1 and end the episode. Each "Step" runs one full episode of
 * Q-learning with ε-greedy exploration. Cell colour = max Q value;
 * an arrow shows the greedy action. */

(function () {
    const canvas    = document.getElementById('viz-rl-canvas');
    const stepBtn   = document.getElementById('viz-rl-step');
    const playBtn   = document.getElementById('viz-rl-play');
    const resetBtn  = document.getElementById('viz-rl-reset');
    const epsSlider = document.getElementById('viz-rl-eps');
    const epsLbl    = document.getElementById('viz-rl-eps-lbl');
    const epLbl     = document.getElementById('viz-rl-ep-lbl');
    const captionEl = document.getElementById('viz-rl-caption');
    if (!canvas) return;

    const ROWS = 5, COLS = 5;
    const GOAL = [4, 4];
    const WALLS = [[1, 2], [2, 2], [3, 1]];
    const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]];   // U D L R
    const ARROWS = ['↑', '↓', '←', '→'];
    const ALPHA = 0.25, GAMMA = 0.95;
    let EPS = 0.2;

    let Q = null;     // [ROWS][COLS][4]
    let episode = 0;
    let playing = false, lastStep = 0;
    let ctx;
    let W = 0, H = 0;

    function reset() {
        Q = Array.from({length: ROWS}, () =>
                Array.from({length: COLS}, () => [0, 0, 0, 0]));
        episode = 0;
        playing = false;
        if (playBtn) playBtn.textContent = 'Play';
        draw();
    }
    reset();

    function isTerminal(r, c) {
        if (r === GOAL[0] && c === GOAL[1]) return true;
        return WALLS.some(([wr, wc]) => wr === r && wc === c);
    }
    function reward(r, c) {
        if (r === GOAL[0] && c === GOAL[1]) return 1;
        if (WALLS.some(([wr, wc]) => wr === r && wc === c)) return -1;
        return -0.02;       // tiny per-step penalty to encourage shorter paths
    }

    function runEpisode() {
        let [r, c] = [0, 0];
        const maxSteps = 50;
        for (let t = 0; t < maxSteps; t++) {
            let a;
            if (Math.random() < EPS) a = Math.floor(Math.random() * 4);
            else {
                const q = Q[r][c];
                let best = 0;
                for (let i = 1; i < 4; i++) if (q[i] > q[best]) best = i;
                a = best;
            }
            let [dr, dc] = ACTIONS[a];
            let nr = Math.max(0, Math.min(ROWS - 1, r + dr));
            let nc = Math.max(0, Math.min(COLS - 1, c + dc));
            const rw = reward(nr, nc);
            const done = isTerminal(nr, nc);
            const nextMax = done ? 0 : Math.max(...Q[nr][nc]);
            Q[r][c][a] += ALPHA * (rw + GAMMA * nextMax - Q[r][c][a]);
            r = nr; c = nc;
            if (done) break;
        }
        episode++;
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.65)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 30;
        const cellSize = Math.min((W - 2 * pad) / COLS, (H - 70) / ROWS);
        const ox = (W - cellSize * COLS) / 2;
        const oy = pad + 14;

        // Find max Q value to scale colours
        let mx = 0;
        for (let r = 0; r < ROWS; r++)
            for (let c = 0; c < COLS; c++)
                for (const v of Q[r][c]) if (v > mx) mx = v;
        mx = Math.max(0.01, mx);

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const x = ox + c * cellSize, y = oy + r * cellSize;
                const isWall = WALLS.some(([wr, wc]) => wr === r && wc === c);
                const isGoal = (r === GOAL[0] && c === GOAL[1]);
                if (isWall) {
                    ctx.fillStyle = 'rgba(80, 80, 80, 0.4)';
                    ctx.fillRect(x, y, cellSize, cellSize);
                    ctx.fillStyle = '#fff';
                    ctx.font = '600 14px "Inter", system-ui';
                    ctx.textAlign = 'center';
                    ctx.fillText('-1', x + cellSize / 2, y + cellSize / 2 + 5);
                } else if (isGoal) {
                    ctx.fillStyle = '#ea7959';
                    ctx.fillRect(x, y, cellSize, cellSize);
                    ctx.fillStyle = '#fff';
                    ctx.font = '700 14px "Inter", system-ui';
                    ctx.textAlign = 'center';
                    ctx.fillText('+1', x + cellSize / 2, y + cellSize / 2 + 5);
                } else {
                    // Colour by max Q
                    const v = Math.max(...Q[r][c]) / mx;
                    const t = Math.max(0, Math.min(1, v));
                    ctx.fillStyle = `rgb(${251 + (79  - 251) * t * 0.7 | 0}, ${250 + (70  - 250) * t * 0.7 | 0}, ${247 + (229 - 247) * t * 0.7 | 0})`;
                    ctx.fillRect(x, y, cellSize, cellSize);
                    // Arrow for the greedy action
                    let best = 0;
                    for (let i = 1; i < 4; i++) if (Q[r][c][i] > Q[r][c][best]) best = i;
                    if (Q[r][c][best] > 0.01) {
                        ctx.fillStyle = t > 0.5 ? '#fff' : 'rgba(0, 0, 0, 0.55)';
                        ctx.font = '600 18px "Inter", system-ui';
                        ctx.textAlign = 'center';
                        ctx.fillText(ARROWS[best], x + cellSize / 2, y + cellSize / 2 + 6);
                    }
                    // Q value
                    ctx.fillStyle = t > 0.5 ? 'rgba(255, 255, 255, 0.85)' : 'rgba(0, 0, 0, 0.45)';
                    ctx.font = '500 9px "JetBrains Mono", monospace';
                    ctx.fillText(Q[r][c][best].toFixed(2),
                                 x + cellSize / 2, y + cellSize - 6);
                }
                // Grid lines
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, cellSize, cellSize);
            }
        }
        // Agent at start
        const sx = ox + cellSize / 2, sy = oy + cellSize / 2;
        ctx.fillStyle = '#4f46e5';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx, sy, cellSize * 0.18, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Q-LEARNING — episode ${episode}, ε = ${EPS.toFixed(2)}`, ox, oy - 6);

        if (epLbl) epLbl.textContent = `episode ${episode}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (episode === 0) {
            captionEl.innerHTML =
                `<strong>Episode 0.</strong> The Q-table is all zeros — the agent has no idea what to do. Hit Step or Play; each episode runs one path from start to terminal, with ε-greedy exploration mixing random moves with greedy moves.`;
        } else if (episode < 30) {
            captionEl.innerHTML =
                `<strong>Episode ${episode}.</strong> Value is flooding backward from the goal: cells that can reach the goal in one step have positive Q first; then their predecessors; then theirs. The arrows show the current greedy policy.`;
        } else {
            captionEl.innerHTML =
                `<strong>Episode ${episode}.</strong> Most cells point toward the goal. Try dropping ε to 0 — the agent acts greedily; with no exploration, any region it never visited stays unfilled. Try raising it to 1.0 — pure random, fast learning but unstable arrows.`;
        }
    }

    function loop(now) {
        if (playing && now - lastStep > 80) {
            runEpisode();
            draw();
            lastStep = now;
            if (episode >= 500) { playing = false; if (playBtn) playBtn.textContent = 'Play'; }
        }
        requestAnimationFrame(loop);
    }

    stepBtn?.addEventListener('click', () => { runEpisode(); draw(); });
    playBtn?.addEventListener('click', () => {
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', reset);
    if (epsSlider) {
        epsSlider.min = 0; epsSlider.max = 1; epsSlider.step = 0.05; epsSlider.value = 0.2;
        epsSlider.addEventListener('input', () => {
            EPS = parseFloat(epsSlider.value);
            if (epsLbl) epsLbl.textContent = `ε = ${EPS.toFixed(2)}`;
            draw();
        });
    }

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    requestAnimationFrame(loop);
})();
