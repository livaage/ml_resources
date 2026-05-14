/* Interactive inductive-bias viz.
 * Same 2D classification dataset shown to four model families with completely
 * different built-in assumptions:
 *   linear  → can only draw straight boundaries
 *   tree    → axis-aligned rectangles
 *   k-NN    → jagged regions following neighbours
 *   RBF     → smooth curves (Gaussian votes)
 * Each panel paints its decision regions; the same points are overlaid so
 * the visual difference is purely about each model's hypothesis space. */

(function () {
    const canvas    = document.getElementById('viz-ib-canvas');
    const dataSel   = document.getElementById('viz-ib-data');
    const resetBtn  = document.getElementById('viz-ib-reset');
    const captionEl = document.getElementById('viz-ib-caption');
    if (!canvas) return;

    let dataset = 'moons';
    let points = [];
    const PALETTE = ['#4f46e5', '#ea7959'];
    const SOFT    = ['#eef2ff', '#fdf0eb'];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function buildDataset() {
        const pts = [];
        if (dataset === 'moons') {
            for (let i = 0; i < 30; i++) {
                const a = Math.PI * i / 29;
                pts.push({ x:  Math.cos(a) * 0.55 - 0.2 + randn() * 0.04,
                           y:  Math.sin(a) * 0.4 + 0.05 + randn() * 0.04, c: 0 });
                pts.push({ x:  Math.cos(a) * 0.55 + 0.2 + randn() * 0.04,
                           y: -Math.sin(a) * 0.4 + 0.15 + randn() * 0.04, c: 1 });
            }
        } else if (dataset === 'linear') {
            for (let i = 0; i < 30; i++) pts.push({ x: -0.45 + randn() * 0.18, y:  0.25 + randn() * 0.18, c: 0 });
            for (let i = 0; i < 30; i++) pts.push({ x:  0.45 + randn() * 0.18, y: -0.25 + randn() * 0.18, c: 1 });
        } else if (dataset === 'xor') {
            for (let i = 0; i < 60; i++) {
                const x = -0.8 + Math.random() * 1.6, y = -0.8 + Math.random() * 1.6;
                pts.push({ x, y, c: (x * y > 0) ? 0 : 1 });
            }
        } else if (dataset === 'circles') {
            for (let i = 0; i < 30; i++) {
                const a1 = Math.random() * Math.PI * 2, r1 = 0.3 + randn() * 0.05;
                pts.push({ x: r1 * Math.cos(a1), y: r1 * Math.sin(a1), c: 0 });
                const a2 = Math.random() * Math.PI * 2, r2 = 0.72 + randn() * 0.05;
                pts.push({ x: r2 * Math.cos(a2), y: r2 * Math.sin(a2), c: 1 });
            }
        }
        return pts;
    }

    // ----- Models (lightweight implementations) -----

    // Linear: logistic regression via batch gradient descent
    function fitLinear(pts) {
        let w = [0.01, 0.01], b = 0;
        for (let it = 0; it < 300; it++) {
            const lr = 0.15 / (1 + it / 80);
            let gw0 = 0, gw1 = 0, gb = 0;
            for (const p of pts) {
                const z = w[0] * p.x + w[1] * p.y + b;
                const s = 1 / (1 + Math.exp(-z));
                const y = p.c, e = s - y;
                gw0 += e * p.x; gw1 += e * p.y; gb += e;
            }
            const N = pts.length;
            w[0] -= lr * gw0 / N; w[1] -= lr * gw1 / N; b -= lr * gb / N;
        }
        return (x, y) => (w[0] * x + w[1] * y + b > 0) ? 1 : 0;
    }

    // Tree: greedy CART, max depth 5
    function gini(arr) {
        if (!arr.length) return 0;
        let n0 = 0, n1 = 0;
        for (const p of arr) (p.c === 0 ? n0++ : n1++);
        const p0 = n0 / arr.length, p1 = n1 / arr.length;
        return 1 - p0 * p0 - p1 * p1;
    }
    function majority(arr) {
        let n0 = 0, n1 = 0;
        for (const p of arr) (p.c === 0 ? n0++ : n1++);
        return n0 >= n1 ? 0 : 1;
    }
    function buildTree(pts, depth) {
        if (depth === 0 || pts.length < 3) return { leaf: true, c: majority(pts) };
        let best = null;
        for (const feat of ['x', 'y']) {
            const sorted = [...pts].sort((a, b) => a[feat] - b[feat]);
            for (let i = 1; i < sorted.length; i++) {
                if (sorted[i][feat] === sorted[i - 1][feat]) continue;
                const t = (sorted[i][feat] + sorted[i - 1][feat]) / 2;
                const l = sorted.slice(0, i), r = sorted.slice(i);
                const wG = (l.length * gini(l) + r.length * gini(r)) / pts.length;
                if (!best || wG < best.wG) best = { feat, t, wG, l, r };
            }
        }
        if (!best || gini(pts) - best.wG < 0.005) return { leaf: true, c: majority(pts) };
        return { leaf: false, feat: best.feat, t: best.t,
                 left: buildTree(best.l, depth - 1),
                 right: buildTree(best.r, depth - 1) };
    }
    function fitTree(pts) {
        const root = buildTree(pts, 5);
        return (x, y) => {
            let n = root;
            while (!n.leaf) {
                const v = n.feat === 'x' ? x : y;
                n = v <= n.t ? n.left : n.right;
            }
            return n.c;
        };
    }

    // k-NN (k=5)
    function fitKNN(pts, k = 5) {
        return (x, y) => {
            const dists = pts.map(p => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, c: p.c }));
            dists.sort((a, b) => a.d - b.d);
            let n0 = 0, n1 = 0;
            for (let i = 0; i < k && i < dists.length; i++) {
                if (dists[i].c === 0) n0++; else n1++;
            }
            return n0 >= n1 ? 0 : 1;
        };
    }

    // RBF kernel density classifier
    function fitRBF(pts, gamma = 4) {
        return (x, y) => {
            let s = 0;
            for (const p of pts) {
                const d2 = (x - p.x) ** 2 + (y - p.y) ** 2;
                const w = Math.exp(-gamma * d2);
                s += (p.c === 1 ? w : -w);
            }
            return s > 0 ? 1 : 0;
        };
    }

    const MODELS = [
        { name: 'LINEAR',     bias: 'straight lines',         fit: fitLinear },
        { name: 'TREE',       bias: 'axis-aligned splits',    fit: fitTree },
        { name: 'k-NN',       bias: 'local neighbourhood',    fit: fitKNN },
        { name: 'RBF KERNEL', bias: 'smooth Gaussian votes',  fit: fitRBF },
    ];

    let fits = [];        // [{name, bias, predict}]
    let cache = [];       // per-panel pre-rendered heatmap canvases

    function refit() {
        fits = MODELS.map(m => ({ name: m.name, bias: m.bias, predict: m.fit(points) }));
        cache = MODELS.map(() => null);
        draw();
    }
    function reset() {
        points = buildDataset();
        refit();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(560, Math.max(440, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        cache = MODELS.map(() => null);
        draw();
    }
    function layout() {
        const pad = 14;
        const gap = 14;
        const usableW = W - 2 * pad - gap;
        const usableH = H - 2 * pad - gap - 28;
        const cellW = usableW / 2;
        const cellH = usableH / 2;
        const cells = [];
        for (let i = 0; i < 4; i++) {
            const r = Math.floor(i / 2), c = i % 2;
            cells.push({
                x: pad + c * (cellW + gap),
                y: pad + 14 + r * (cellH + gap + 14),
                w: cellW,
                h: cellH,
            });
        }
        return cells;
    }
    function toPx(x, y, box) {
        const m = Math.min(box.w, box.h) / 2 - 6;
        return [box.x + box.w / 2 + x * m, box.y + box.h / 2 - y * m];
    }

    function renderCellHeatmap(i, box) {
        if (cache[i]) return cache[i];
        const G = 48;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        const predict = fits[i].predict;
        for (let j = 0; j < G; j++) {
            for (let k = 0; k < G; k++) {
                const dx =  ((k + 0.5) / G * 2 - 1) * 1.05;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.05;
                const cls = predict(dx, dy);
                const col = SOFT[cls];
                const m = col.match(/#?(.{2})(.{2})(.{2})/);
                const idx = (j * G + k) * 4;
                img.data[idx]     = parseInt(m[1], 16);
                img.data[idx + 1] = parseInt(m[2], 16);
                img.data[idx + 2] = parseInt(m[3], 16);
                img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        cache[i] = off;
        return off;
    }

    function drawCell(i, box) {
        const m = Math.min(box.w, box.h) / 2 - 6;
        const sx = box.x + box.w / 2 - m * 1.05;
        const sy = box.y + box.h / 2 - m * 1.05;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(renderCellHeatmap(i, box), sx, sy, m * 2.1, m * 2.1);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(sx - 0.5, sy - 0.5, m * 2.1 + 1, m * 2.1 + 1);

        // Points
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y, box);
            ctx.fillStyle = PALETTE[p.c];
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 2.6, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Train accuracy
        let correct = 0;
        for (const p of points) if (fits[i].predict(p.x, p.y) === p.c) correct++;
        const acc = (correct / points.length * 100).toFixed(0);

        // Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(fits[i].name, box.x, box.y - 4);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText(`bias: ${fits[i].bias}`, box.x + 80, box.y - 4);
        // Accuracy badge
        ctx.font = '600 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillStyle = '#4f46e5';
        ctx.fillText(`${acc}%`, box.x + box.w, box.y - 4);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        const cells = layout();
        for (let i = 0; i < 4; i++) drawCell(i, cells[i]);
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            moons:   `The moons aren't linearly separable — the linear panel can only manage ~75% accuracy. The tree carves rectangles that follow each moon roughly; k-NN traces the actual curve; RBF gives a clean smooth boundary.`,
            linear:  `Two well-separated blobs are easy for everyone — but notice the linear model is at its <em>most natural</em> here, where the optimal boundary is a line.`,
            xor:     `XOR breaks the linear model entirely (~50% accuracy, the worst possible). The tree needs at least depth 2; k-NN copes naturally; RBF carves the four quadrants smoothly.`,
            circles: `Concentric classes are linearly hopeless. RBF and k-NN both nail the circular boundary — RBF more smoothly. The tree cuts the circle into a staircase.`,
        };
        captionEl.innerHTML = notes[dataset] +
            ` Each panel uses the <em>same</em> data — only the model's hypothesis space differs.`;
    }

    // ----- Controls -----
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="moons">Two moons</option>
            <option value="linear">Linear (easy)</option>
            <option value="xor">XOR</option>
            <option value="circles">Concentric</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }
    resetBtn?.addEventListener('click', reset);

    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
