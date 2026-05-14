/* Interactive random-forests viz.
 * Big panel: the ensemble's decision surface (heatmap from vote fraction)
 * and the f = 0.5 boundary. The thumbnail strip shows individual trees —
 * each trained on a bootstrap sample with a random feature subset at every
 * split. Slide N up and watch the ensemble boundary smooth out compared to
 * any single jagged tree. */

(function () {
    const canvas    = document.getElementById('viz-rf-canvas');
    const nSlider   = document.getElementById('viz-rf-n');
    const nLbl      = document.getElementById('viz-rf-n-lbl');
    const depthSel  = document.getElementById('viz-rf-depth');
    const dataSel   = document.getElementById('viz-rf-data');
    const resetBtn  = document.getElementById('viz-rf-reset');
    const captionEl = document.getElementById('viz-rf-caption');
    if (!canvas) return;

    let nTrees = 20;
    let maxDepth = 5;
    let dataset = 'moons';
    let points = [];
    let forest = [];
    let heatCache = null;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'moons') {
            for (let i = 0; i < 60; i++) {
                const a = Math.PI * i / 59;
                pts.push({ x: Math.cos(a) * 0.55 - 0.2 + randn() * 0.04,
                           y: Math.sin(a) * 0.4 + 0.05 + randn() * 0.04, c: 0 });
                pts.push({ x:  Math.cos(a) * 0.55 + 0.2 + randn() * 0.04,
                           y: -Math.sin(a) * 0.4 + 0.15 + randn() * 0.04, c: 1 });
            }
        } else if (dataset === 'circles') {
            for (let i = 0; i < 60; i++) {
                const a1 = Math.random() * Math.PI * 2;
                const r1 = 0.3 + randn() * 0.05;
                pts.push({ x: r1 * Math.cos(a1), y: r1 * Math.sin(a1), c: 0 });
                const a2 = Math.random() * Math.PI * 2;
                const r2 = 0.75 + randn() * 0.05;
                pts.push({ x: r2 * Math.cos(a2), y: r2 * Math.sin(a2), c: 1 });
            }
        } else if (dataset === 'blobs') {
            for (let i = 0; i < 60; i++) {
                pts.push({ x: -0.4 + randn() * 0.12, y:  0.3 + randn() * 0.12, c: 0 });
                pts.push({ x:  0.4 + randn() * 0.12, y: -0.3 + randn() * 0.12, c: 1 });
            }
        } else if (dataset === 'spiral') {
            for (let i = 0; i < 60; i++) {
                const t = 0.1 + (i / 59) * 1.8;
                const r = t * 0.42;
                pts.push({ x: r * Math.cos(t * 3),     y: r * Math.sin(t * 3),     c: 0 });
                pts.push({ x: r * Math.cos(t * 3 + Math.PI),
                           y: r * Math.sin(t * 3 + Math.PI), c: 1 });
            }
        }
        return pts;
    }

    // ----- Single tree built with bootstrap + random feature subsetting -----
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
    function bestSplitOnFeature(pts, feat) {
        const sorted = [...pts].sort((a, b) => a[feat] - b[feat]);
        let best = null;
        for (let i = 1; i < sorted.length; i++) {
            if (sorted[i][feat] === sorted[i - 1][feat]) continue;
            const t = (sorted[i][feat] + sorted[i - 1][feat]) / 2;
            const left = sorted.slice(0, i);
            const right = sorted.slice(i);
            const wG = (left.length * gini(left) + right.length * gini(right)) / sorted.length;
            if (!best || wG < best.gini) best = { feat, t, gini: wG, left, right };
        }
        return best;
    }
    function buildTree(pts, depthRem) {
        if (depthRem === 0 || pts.length < 3) return { leaf: true, class: majority(pts), n: pts.length };
        const features = Math.random() < 0.5 ? ['x'] : ['y'];  // mtry = 1 of 2
        let best = null;
        for (const feat of features) {
            const s = bestSplitOnFeature(pts, feat);
            if (s && (!best || s.gini < best.gini)) best = s;
        }
        if (!best || gini(pts) - best.gini < 0.005) return { leaf: true, class: majority(pts), n: pts.length };
        return {
            leaf: false, feat: best.feat, threshold: best.t,
            left:  buildTree(best.left,  depthRem - 1),
            right: buildTree(best.right, depthRem - 1),
        };
    }
    function predictTree(node, x, y) {
        if (node.leaf) return node.class;
        const v = node.feat === 'x' ? x : y;
        return v <= node.threshold ? predictTree(node.left, x, y) : predictTree(node.right, x, y);
    }

    function rebuildForest() {
        forest = [];
        for (let t = 0; t < nTrees; t++) {
            const bag = [];
            for (let i = 0; i < points.length; i++) {
                bag.push(points[Math.floor(Math.random() * points.length)]);
            }
            forest.push(buildTree(bag, maxDepth));
        }
        heatCache = null;
    }
    function reset() {
        points = buildDataset();
        rebuildForest();
        draw();
    }

    function voteFraction(x, y) {
        let n1 = 0;
        for (const t of forest) n1 += predictTree(t, x, y);
        return n1 / forest.length;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(560, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(360, cssW * 0.58)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        heatCache = null;
        draw();
    }

    function layout() {
        const pad = 14;
        const thumbsH = 88;
        const mainH = H - thumbsH - 3 * pad;
        const main = { x: pad + 18, y: pad + 14, w: H - 2 * pad - 14, h: mainH };
        // Centre the square main panel; right side is for legend + caption space
        main.x = pad + 18;
        main.w = main.h;
        const side = { x: main.x + main.w + 14, y: main.y, w: W - main.x - main.w - 14 - pad, h: main.h };
        const thumbs = { x: pad + 18, y: main.y + main.h + 24, w: W - 2 * pad - 18, h: thumbsH - 18 };
        return { main, side, thumbs };
    }

    function dataToPx(x, y, box) {
        const m = Math.min(box.w, box.h) / 2 - 6;
        return [box.x + box.w / 2 + x * m, box.y + box.h / 2 - y * m];
    }
    function voteColour(p) {
        // p ∈ [0, 1].  0 → indigo wash, 1 → terracotta wash
        if (p < 0.5) {
            const t = (0.5 - p) * 1.1;
            return `rgb(${251 + (79  - 251) * t | 0}, ${250 + (70  - 250) * t | 0}, ${247 + (229 - 247) * t | 0})`;
        }
        const t = (p - 0.5) * 1.1;
        return `rgb(${251 + (234 - 251) * t | 0}, ${250 + (121 - 250) * t | 0}, ${247 + (89 - 247) * t | 0})`;
    }
    function buildHeatmap(box) {
        if (heatCache && heatCache.w === box.w) return heatCache.canvas;
        const G = 60;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.05;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.05;
                const v = voteFraction(dx, dy);
                const col = voteColour(v);
                const m = col.match(/rgb\((\d+), (\d+), (\d+)\)/);
                const idx = (j * G + i) * 4;
                img.data[idx]     = +m[1];
                img.data[idx + 1] = +m[2];
                img.data[idx + 2] = +m[3];
                img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatCache = { w: box.w, canvas: off };
        return off;
    }

    function drawMain(box) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(buildHeatmap(box), box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Points
        for (const p of points) {
            const [px, py] = dataToPx(p.x, p.y, box);
            ctx.fillStyle = p.c === 0 ? '#4f46e5' : '#ea7959';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(px, py, 3.2, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`ENSEMBLE — ${nTrees} tree${nTrees === 1 ? '' : 's'}, max depth ${maxDepth}`,
            box.x, box.y - 5);
    }

    function drawThumb(box, tree) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        const G = 18;
        const cw = box.w / G, ch = box.h / G;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.05;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.05;
                const cls = predictTree(tree, dx, dy);
                ctx.fillStyle = cls === 0 ? '#eef2ff' : '#fdf0eb';
                ctx.fillRect(box.x + i * cw, box.y + j * ch, cw + 0.6, ch + 0.6);
            }
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
        ctx.lineWidth = 0.8;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);
    }
    function drawThumbs(box) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('SAMPLED TREES — each is biased + brittle on its own', box.x, box.y - 5);

        const max = Math.min(8, forest.length);
        const slotW = (box.w - (max - 1) * 6) / max;
        for (let i = 0; i < max; i++) {
            const slot = {
                x: box.x + i * (slotW + 6),
                y: box.y,
                w: slotW,
                h: box.h,
            };
            // Spread thumbnails across the forest (not just first N)
            const idx = Math.floor(i * forest.length / max);
            drawThumb(slot, forest[idx]);
        }
    }

    function drawLegend(box) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('VOTE FRACTION', box.x, box.y + 10);
        // Vertical colour bar
        const barX = box.x + 6, barY = box.y + 26, barW = 18, barH = box.h - 60;
        for (let i = 0; i < barH; i++) {
            ctx.fillStyle = voteColour(1 - i / barH);
            ctx.fillRect(barX, barY + i, barW, 1.5);
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.strokeRect(barX - 0.5, barY - 0.5, barW + 1, barH + 1);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.fillText('1.00', barX + barW + 4, barY + 6);
        ctx.fillText('0.50', barX + barW + 4, barY + barH / 2 + 3);
        ctx.fillText('0.00', barX + barW + 4, barY + barH);

        // OOB accuracy summary
        let correct = 0, total = 0;
        for (const p of points) {
            const f = voteFraction(p.x, p.y);
            const pred = f >= 0.5 ? 1 : 0;
            if (pred === p.c) correct++;
            total++;
        }
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('train acc', box.x, box.y + box.h - 18);
        ctx.font = '600 13px "JetBrains Mono", monospace';
        ctx.fillStyle = '#4f46e5';
        ctx.fillText(`${(correct / total * 100).toFixed(0)}%`, box.x, box.y + box.h - 2);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        if (forest.length === 0) return;
        const lay = layout();
        drawMain(lay.main);
        drawLegend(lay.side);
        drawThumbs(lay.thumbs);
        if (nLbl) nLbl.textContent = `N = ${nTrees}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        captionEl.innerHTML =
            `<strong>${nTrees} tree${nTrees === 1 ? '' : 's'}.</strong> ` +
            `Each tree saw a bootstrap sample (with replacement) and only one feature at every split. ` +
            (nTrees < 4
                ? `With this few trees, the ensemble boundary inherits the brittleness of any single tree — look at the jagged thumbnails below.`
                : `Notice how the ensemble's boundary (smooth-ish curve where the colour flips) is much cleaner than any individual tree's. That's bagging in action.`);
    }

    // ----- Controls -----
    if (nSlider) {
        nSlider.min = 1; nSlider.max = 80; nSlider.step = 1; nSlider.value = 20;
        nSlider.addEventListener('input', () => {
            nTrees = parseInt(nSlider.value, 10);
            rebuildForest();
            draw();
        });
    }
    if (depthSel) {
        depthSel.innerHTML = `
            <option value="3">depth 3</option>
            <option value="5" selected>depth 5</option>
            <option value="7">depth 7</option>
        `;
        depthSel.addEventListener('change', () => {
            maxDepth = parseInt(depthSel.value, 10);
            rebuildForest();
            draw();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="moons">Two moons</option>
            <option value="circles">Concentric</option>
            <option value="blobs">Blobs</option>
            <option value="spiral">Spiral</option>
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
