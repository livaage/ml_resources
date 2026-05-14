/* Interactive decision tree viz.
 * 2D points belong to one of 2 or 3 classes; a CART-flavoured tree is grown
 * by greedy Gini-impurity reduction. The left panel shows feature space with
 * rectangular leaf regions tinted by their majority class; the right panel
 * shows the tree structure with split labels.
 *
 * Slide the max-depth slider to watch the tree carve more refined regions —
 * and inevitably overfit at deep settings on noisy data. */

(function () {
    const canvas    = document.getElementById('viz-tree-canvas');
    const depthSel  = document.getElementById('viz-tree-depth');
    const depthLbl  = document.getElementById('viz-tree-depth-lbl');
    const dataSel   = document.getElementById('viz-tree-data');
    const resetBtn  = document.getElementById('viz-tree-reset');
    const captionEl = document.getElementById('viz-tree-caption');
    if (!canvas) return;

    // ----- State -----
    let maxDepth = 3;
    let dataset = 'linear';
    let points = [];     // each: { x, y, c }
    let tree = null;
    const PALETTE = ['#4f46e5', '#ea7959', '#10847e'];
    const SOFT    = ['#eef2ff', '#fdf0eb', '#e5f3f1'];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'linear') {
            for (let i = 0; i < 70; i++) {
                const x = -0.85 + Math.random() * 1.7;
                const y = -0.85 + Math.random() * 1.7;
                pts.push({ x, y, c: (y > 0.2 * x + 0.05 + randn() * 0.06) ? 0 : 1 });
            }
        } else if (dataset === 'xor') {
            for (let i = 0; i < 80; i++) {
                const x = -0.85 + Math.random() * 1.7;
                const y = -0.85 + Math.random() * 1.7;
                pts.push({ x, y, c: (x * y > 0) ? 0 : 1 });
            }
        } else if (dataset === 'stripes') {
            for (let i = 0; i < 90; i++) {
                const x = -0.85 + Math.random() * 1.7;
                const y = -0.85 + Math.random() * 1.7;
                const band = Math.floor((y + 0.85) / 0.45);
                pts.push({ x, y, c: band % 2 });
            }
        } else if (dataset === 'rings') {
            for (let i = 0; i < 90; i++) {
                const r = Math.random() * 0.95;
                const a = Math.random() * Math.PI * 2;
                const x = r * Math.cos(a), y = r * Math.sin(a);
                pts.push({ x, y, c: (r < 0.35) ? 0 : (r < 0.7 ? 1 : 2) });
            }
        }
        return pts;
    }

    // ----- Tree construction -----
    function gini(labels) {
        if (!labels.length) return 0;
        const counts = new Map();
        for (const l of labels) counts.set(l, (counts.get(l) || 0) + 1);
        let g = 1;
        for (const c of counts.values()) {
            const p = c / labels.length;
            g -= p * p;
        }
        return g;
    }

    function bestSplit(pts, depthRem) {
        if (depthRem === 0 || pts.length < 4) return null;
        const labels = pts.map(p => p.c);
        if (new Set(labels).size === 1) return null;

        let best = null;
        for (const feat of ['x', 'y']) {
            const sorted = [...pts].sort((a, b) => a[feat] - b[feat]);
            for (let i = 1; i < sorted.length; i++) {
                if (sorted[i][feat] === sorted[i - 1][feat]) continue;
                const threshold = (sorted[i][feat] + sorted[i - 1][feat]) / 2;
                const left = sorted.slice(0, i);
                const right = sorted.slice(i);
                if (left.length < 1 || right.length < 1) continue;
                const wG = (left.length * gini(left.map(p => p.c))
                          + right.length * gini(right.map(p => p.c))) / sorted.length;
                if (!best || wG < best.gini) best = { feat, threshold, gini: wG, left, right };
            }
        }
        return best;
    }

    function majorityClass(pts) {
        const counts = new Map();
        for (const p of pts) counts.set(p.c, (counts.get(p.c) || 0) + 1);
        let best = -1, bestC = -1;
        for (const [k, v] of counts.entries()) if (v > bestC) { bestC = v; best = k; }
        return best;
    }

    function buildTree(pts, depthRem, region) {
        const split = bestSplit(pts, depthRem);
        if (!split) {
            return { leaf: true, class: majorityClass(pts), n: pts.length, region };
        }
        const { feat, threshold, left, right } = split;
        const lReg = { ...region }, rReg = { ...region };
        if (feat === 'x') { lReg.x1 = threshold; rReg.x0 = threshold; }
        else              { lReg.y0 = threshold; rReg.y1 = threshold; }
        return {
            leaf: false, feat, threshold,
            left:  buildTree(left,  depthRem - 1, lReg),
            right: buildTree(right, depthRem - 1, rReg),
            region, n: pts.length,
        };
    }

    function rebuild() {
        const fullRegion = { x0: -1, x1: 1, y0: -1, y1: 1 };
        tree = buildTree(points, maxDepth, fullRegion);
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(560, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(330, cssW * 0.50)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function layout() {
        const pad = 14;
        const totalW = W - 2 * pad;
        const scatterW = Math.min(H - 2 * pad - 18, totalW * 0.55);
        const scatter = { x: pad, y: pad + 16, w: scatterW, h: H - 2 * pad - 16 };
        const tree = {
            x: scatter.x + scatter.w + 18,
            y: pad + 16,
            w: W - (scatter.x + scatter.w + 18) - pad,
            h: H - 2 * pad - 16,
        };
        return { scatter, tree };
    }

    function dataToPx(x, y, box) {
        return [
            box.x + ((x + 1) / 2) * box.w,
            box.y + ((1 - y) / 2) * box.h,
        ];
    }

    // ----- Draw scatter + region tiling -----
    function drawRegions(node, box) {
        if (node.leaf) {
            const [x0, y1] = dataToPx(node.region.x0, node.region.y0, box);
            const [x1, y0] = dataToPx(node.region.x1, node.region.y1, box);
            ctx.fillStyle = SOFT[node.class % SOFT.length];
            ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
            return;
        }
        drawRegions(node.left,  box);
        drawRegions(node.right, box);
    }
    function drawSplits(node, box) {
        if (node.leaf) return;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.lineWidth = 1.2;
        if (node.feat === 'x') {
            const [px,] = dataToPx(node.threshold, 0, box);
            const [, py0] = dataToPx(0, node.region.y1, box);
            const [, py1] = dataToPx(0, node.region.y0, box);
            ctx.beginPath();
            ctx.moveTo(px, py0); ctx.lineTo(px, py1);
            ctx.stroke();
        } else {
            const [, py] = dataToPx(0, node.threshold, box);
            const [px0,] = dataToPx(node.region.x0, 0, box);
            const [px1,] = dataToPx(node.region.x1, 0, box);
            ctx.beginPath();
            ctx.moveTo(px0, py); ctx.lineTo(px1, py);
            ctx.stroke();
        }
        drawSplits(node.left,  box);
        drawSplits(node.right, box);
    }

    function drawScatter(box) {
        // Background
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        drawRegions(tree, box);
        drawSplits(tree, box);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Points
        for (const p of points) {
            const [px, py] = dataToPx(p.x, p.y, box);
            ctx.fillStyle = PALETTE[p.c % PALETTE.length];
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 3.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('FEATURE SPACE', box.x, box.y - 5);
    }

    // ----- Draw tree diagram (recursive layout) -----
    // Assign each leaf an x slot, internal nodes get midpoint of children.
    function layoutTree(node, depth, slot) {
        node._depth = depth;
        if (node.leaf) {
            node._slotL = slot.i; node._slotR = slot.i;
            slot.i += 1;
            return;
        }
        layoutTree(node.left,  depth + 1, slot);
        layoutTree(node.right, depth + 1, slot);
        node._slotL = node.left._slotL;
        node._slotR = node.right._slotR;
    }
    function nLeaves(node) {
        return node.leaf ? 1 : nLeaves(node.left) + nLeaves(node.right);
    }
    function maxDepthOf(node, d = 0) {
        return node.leaf ? d : Math.max(maxDepthOf(node.left, d + 1), maxDepthOf(node.right, d + 1));
    }

    function drawTree(box) {
        layoutTree(tree, 0, { i: 0 });
        const leaves = nLeaves(tree);
        const depth = Math.max(1, maxDepthOf(tree));
        const yStep = box.h / (depth + 1.2);
        const xStep = box.w / Math.max(1, leaves);

        function nodeXY(node) {
            const x = box.x + (node._slotL + node._slotR + 1) / 2 * xStep;
            const y = box.y + (node._depth + 0.55) * yStep;
            return [x, y];
        }

        // Edges first
        function drawEdges(node) {
            if (node.leaf) return;
            const [px, py] = nodeXY(node);
            const [lx, ly] = nodeXY(node.left);
            const [rx, ry] = nodeXY(node.right);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(px, py); ctx.lineTo(lx, ly);
            ctx.moveTo(px, py); ctx.lineTo(rx, ry);
            ctx.stroke();
            drawEdges(node.left);
            drawEdges(node.right);
        }
        drawEdges(tree);

        // Nodes
        function drawNode(node) {
            const [x, y] = nodeXY(node);
            if (node.leaf) {
                ctx.fillStyle = PALETTE[node.class % PALETTE.length];
                ctx.beginPath();
                ctx.arc(x, y, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
                ctx.font = '500 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`n=${node.n}`, x, y + 18);
            } else {
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                ctx.arc(x, y, 9, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.font = '500 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`${node.feat} ≤ ${node.threshold.toFixed(2)}`, x, y + 22);
                drawNode(node.left);
                drawNode(node.right);
            }
        }
        drawNode(tree);

        // Frame + title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('TREE', box.x, box.y - 5);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        if (!tree) return;
        const lay = layout();
        drawScatter(lay.scatter);
        drawTree(lay.tree);
        updateCaption();
        if (depthLbl) depthLbl.textContent = `max depth = ${maxDepth}`;
    }

    function updateCaption() {
        if (!captionEl) return;
        const leaves = nLeaves(tree);
        // Training error
        let wrong = 0;
        function predict(node, p) {
            if (node.leaf) return node.class;
            if (p[node.feat] <= node.threshold) return predict(node.left,  p);
            return predict(node.right, p);
        }
        for (const p of points) if (predict(tree, p) !== p.c) wrong++;
        const err = (wrong / points.length * 100).toFixed(0);
        captionEl.innerHTML =
            `<strong>depth ${maxDepth}</strong> → ${leaves} leaves, ` +
            `training error <strong>${err}%</strong>. ` +
            (maxDepth >= 6
                ? `At this depth a single tree memorises noise — this is why we ensemble (see Random Forests).`
                : `Each leaf is one rectangular region of feature space; the tree's prediction is whichever class the majority of training points in that leaf had.`);
    }

    function reset() {
        points = buildDataset();
        rebuild();
        draw();
    }

    // ----- Controls -----
    if (depthSel) {
        depthSel.min = 1; depthSel.max = 8; depthSel.step = 1; depthSel.value = 3;
        depthSel.addEventListener('input', () => {
            maxDepth = parseInt(depthSel.value, 10);
            rebuild();
            draw();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="linear">Linear boundary</option>
            <option value="xor">XOR</option>
            <option value="stripes">Stripes</option>
            <option value="rings">Concentric rings</option>
        `;
        dataSel.addEventListener('change', () => {
            dataset = dataSel.value;
            reset();
        });
    }
    resetBtn?.addEventListener('click', reset);

    // ----- Init -----
    reset();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
