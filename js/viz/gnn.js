/* Interactive GNN message-passing viz.
 * Click any node → it becomes the "source" with signal 1.0; every other node
 * starts at 0. One round of message passing diffuses the signal to immediate
 * neighbours; K rounds reaches the K-hop neighbourhood. */

(function () {
    const canvas    = document.getElementById('viz-gnn-canvas');
    const stepBtn   = document.getElementById('viz-gnn-step');
    const resetBtn  = document.getElementById('viz-gnn-reset');
    const playBtn   = document.getElementById('viz-gnn-play');
    const aggSelect = document.getElementById('viz-gnn-agg');
    const roundEl   = document.getElementById('viz-gnn-round');
    const captionEl = document.getElementById('viz-gnn-caption');
    if (!canvas) return;

    // ----- Graph: 8 nodes laid out in two rough rings + a centre -----
    // Each entry: { label, angle, radiusFactor }  (angle in radians)
    const NODES = [
        { label: 'A', cx: 0.50, cy: 0.50 },                     // 0 — centre
        { label: 'B', cx: 0.18, cy: 0.20 },                     // 1
        { label: 'C', cx: 0.50, cy: 0.12 },                     // 2
        { label: 'D', cx: 0.82, cy: 0.20 },                     // 3
        { label: 'E', cx: 0.90, cy: 0.62 },                     // 4
        { label: 'F', cx: 0.68, cy: 0.88 },                     // 5
        { label: 'G', cx: 0.32, cy: 0.88 },                     // 6
        { label: 'H', cx: 0.10, cy: 0.62 },                     // 7
    ];

    // Edges: a hub-and-spoke from A to all, plus an outer cycle around 1..7
    const EDGES = [
        [0, 1], [0, 2], [0, 3], [0, 5], [0, 7],   // hub spokes
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1], // outer cycle
    ];

    const N = NODES.length;

    // Adjacency (with self-loops, like real GCN normalisation)
    const adj = [...Array(N)].map(() => []);
    for (const [a, b] of EDGES) {
        adj[a].push(b);
        adj[b].push(a);
    }

    // ----- State -----
    let source   = 0;            // current source node
    let round    = 0;
    let features = new Float32Array(N);   // current signal values
    let aggKind  = 'mean';                 // 'mean' or 'sum'
    let playing  = false;
    let lastStep = 0;
    const STEP_MS = 1100;

    function reset() {
        features = new Float32Array(N);
        features[source] = 1;
        round = 0;
    }
    reset();

    // ----- Message passing step -----
    function step() {
        const next = new Float32Array(N);
        for (let v = 0; v < N; v++) {
            let agg = features[v];          // include self
            let count = 1;
            for (const u of adj[v]) {
                agg += features[u];
                count++;
            }
            next[v] = (aggKind === 'mean') ? agg / count : agg;
        }
        // Re-pin the source to 1 so it doesn't decay — easier to grasp the diffusion
        if (aggKind === 'mean') {
            // Cap to [0, 1]
            for (let i = 0; i < N; i++) next[i] = Math.min(1, next[i]);
        } else {
            // For sum, normalise so max stays at 1
            let mx = 0;
            for (const v of next) if (v > mx) mx = v;
            if (mx > 0) for (let i = 0; i < N; i++) next[i] /= mx;
        }
        features = next;
        round++;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(320, cssW * 0.62)));
        W = cssW;
        H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    // Map node logical (cx, cy) ∈ [0, 1]² → canvas pixels
    function nodePos(i) {
        const pad = 50;
        return {
            x: pad + NODES[i].cx * (W - 2 * pad),
            y: pad + NODES[i].cy * (H - 2 * pad),
        };
    }

    const NODE_R = 26;

    function nodeColour(t) {
        // Cream → indigo
        const r = 251 + (79  - 251) * t;
        const g = 250 + (70  - 250) * t;
        const b = 247 + (229 - 247) * t;
        return `rgb(${r|0}, ${g|0}, ${b|0})`;
    }
    function nodeTextColour(t) {
        return t > 0.55 ? '#ffffff' : '#1a1a1a';
    }

    // ----- Draw -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Edges
        for (const [a, b] of EDGES) {
            const pa = nodePos(a);
            const pb = nodePos(b);
            // Active edges (both ends are warm) get a stronger stroke
            const va = features[a], vb = features[b];
            const activity = Math.max(va, vb);
            const alpha = 0.18 + activity * 0.5;
            ctx.strokeStyle = `rgba(79, 70, 229, ${alpha})`;
            ctx.lineWidth = 1.2 + activity * 1.5;
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
        }

        // Nodes
        for (let i = 0; i < N; i++) {
            const p = nodePos(i);
            const t = features[i];
            const isSource = (i === source);

            // Source halo
            if (isSource) {
                ctx.fillStyle = 'rgba(234, 121, 89, 0.18)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, NODE_R + 8, 0, Math.PI * 2);
                ctx.fill();
            }

            ctx.fillStyle = nodeColour(t);
            ctx.strokeStyle = isSource ? '#ea7959' : '#5f5f5f';
            ctx.lineWidth   = isSource ? 2.5 : 1.4;
            ctx.beginPath();
            ctx.arc(p.x, p.y, NODE_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Label
            ctx.fillStyle = nodeTextColour(t);
            ctx.font      = '600 14px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(NODES[i].label, p.x, p.y - 4);

            // Value (small mono below)
            ctx.font      = '10px "JetBrains Mono", monospace';
            ctx.fillStyle = t > 0.55 ? 'rgba(255, 255, 255, 0.75)' : 'rgba(26, 26, 26, 0.55)';
            ctx.fillText(t.toFixed(2), p.x, p.y + 10);
        }

        if (roundEl) roundEl.textContent = `Round ${round}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (round === 0) {
            captionEl.innerHTML =
                `Only the <strong>source</strong> (${NODES[source].label}) has signal. ` +
                `Click <strong>Step</strong> — every node averages its own value with its neighbours'. ` +
                `After ${aggKind === 'mean' ? 'mean' : 'sum'}-aggregation, signal flows along the edges.`;
        } else {
            // Count nodes that have meaningfully received signal
            const reached = features.reduce((n, v) => n + (v > 0.05 ? 1 : 0), 0);
            captionEl.innerHTML =
                `After <strong>${round} round${round === 1 ? '' : 's'}</strong>, ` +
                `the signal from <strong>${NODES[source].label}</strong> has reached ` +
                `<strong>${reached} of ${N} nodes</strong>. ` +
                `Real GNNs do this with learned weight matrices instead of plain averaging — ` +
                `but the propagation pattern is identical.`;
        }
    }

    // ----- Interactions -----
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }
    function nodeAt(x, y) {
        for (let i = 0; i < N; i++) {
            const p = nodePos(i);
            if (Math.hypot(x - p.x, y - p.y) < NODE_R + 4) return i;
        }
        return -1;
    }
    function onClick(e) {
        const [x, y] = getPos(e);
        const i = nodeAt(x, y);
        if (i < 0) return;
        source = i;
        reset();
        playing = false;
        updatePlayBtn();
        draw();
    }
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); onClick(e); });

    stepBtn?.addEventListener('click',  () => { step(); draw(); });
    resetBtn?.addEventListener('click', () => { reset(); draw(); });
    function updatePlayBtn() {
        if (playBtn) playBtn.textContent = playing ? 'Pause' : 'Auto';
    }
    playBtn?.addEventListener('click', () => {
        playing = !playing;
        updatePlayBtn();
        if (playing) lastStep = performance.now();
    });
    if (aggSelect) {
        aggSelect.innerHTML = `
            <option value="mean">Mean aggregation</option>
            <option value="sum">Sum aggregation</option>
        `;
        aggSelect.addEventListener('change', () => {
            aggKind = aggSelect.value;
            reset();
            draw();
        });
    }

    // ----- Animation loop -----
    function loop(now) {
        if (playing && now - lastStep >= STEP_MS) {
            // Stop auto-play once the graph is saturated (or after many rounds)
            const reached = features.reduce((n, v) => n + (v > 0.95 ? 1 : 0), 0);
            if (round > 12 || reached === N) {
                playing = false;
                updatePlayBtn();
            } else {
                step();
                draw();
            }
            lastStep = now;
        }
        requestAnimationFrame(loop);
    }

    // ----- Init -----
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
