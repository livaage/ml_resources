/* Interactive semi-supervised viz.
 * Two natural clusters (classes 0 / 1) with only 8 labelled points; the rest
 * are grey. Toggle between strategies:
 *   sup    — supervised-only LR on the 8 labels
 *   pseudo — single round of pseudo-labelling
 *   graph  — label-propagation through a k-NN graph
 *   self   — iterated self-training (3 rounds, threshold 0.85)  */

(function () {
    const canvas    = document.getElementById('viz-ssl2-canvas');
    const supBtn    = document.getElementById('viz-ssl2-sup');
    const pseudoBtn = document.getElementById('viz-ssl2-pseudo');
    const graphBtn  = document.getElementById('viz-ssl2-graph');
    const selfBtn   = document.getElementById('viz-ssl2-self');
    const resetBtn  = document.getElementById('viz-ssl2-reset');
    const captionEl = document.getElementById('viz-ssl2-caption');
    if (!canvas) return;

    let mode = 'sup';
    let points = [];   // {x, y, c_true, label}    label: 0, 1, or -1 (unlabelled)

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function regenerate() {
        points = [];
        for (let i = 0; i < 40; i++) {
            const a = Math.PI * i / 39;
            points.push({ x: Math.cos(a) * 0.55 - 0.2 + randn() * 0.05,
                          y: Math.sin(a) * 0.4 + 0.05 + randn() * 0.05,
                          c: 0, label: -1 });
            points.push({ x: Math.cos(a) * 0.55 + 0.2 + randn() * 0.05,
                          y: -Math.sin(a) * 0.4 + 0.15 + randn() * 0.05,
                          c: 1, label: -1 });
        }
        // Label 4 points from each class — chosen to be spread out
        const class0 = points.filter(p => p.c === 0);
        const class1 = points.filter(p => p.c === 1);
        for (const lst of [class0, class1]) {
            // Pick 4 widely-separated indices
            const idxs = [0, Math.floor(lst.length * 0.33),
                          Math.floor(lst.length * 0.67), lst.length - 1];
            for (const i of idxs) lst[i].label = lst[i].c;
        }
    }
    regenerate();

    function fitLR(pts, weights) {
        let w = [0, 0], b = 0;
        for (let it = 0; it < 250; it++) {
            const lr = 0.4 / (1 + it / 80);
            let gw0 = 0, gw1 = 0, gb = 0;
            let denom = 0;
            for (let i = 0; i < pts.length; i++) {
                const p = pts[i];
                const z = w[0] * p.x + w[1] * p.y + b;
                const s = 1 / (1 + Math.exp(-z));
                const wgt = weights ? weights[i] : 1;
                const e = wgt * (s - p.label);
                gw0 += e * p.x; gw1 += e * p.y; gb += e;
                denom += wgt;
            }
            w[0] -= lr * gw0 / denom;
            w[1] -= lr * gw1 / denom;
            b    -= lr * gb / denom;
        }
        return { w, b };
    }
    function predictBound(model, x, y) {
        return (model.w[0] * x + model.w[1] * y + model.b > 0) ? 1 : 0;
    }
    function predictProb(model, x, y) {
        const z = model.w[0] * x + model.w[1] * y + model.b;
        return 1 / (1 + Math.exp(-z));
    }

    function runStrategy() {
        let labelled = points.filter(p => p.label >= 0);
        let assignedLabels = points.map(p => p.label);    // -1 if still unlabelled

        if (mode === 'sup') {
            return { model: fitLR(labelled), assigned: assignedLabels };
        }
        if (mode === 'pseudo') {
            const sup = fitLR(labelled);
            // One round of pseudo-labelling: mark unlabelled with the prediction
            for (let i = 0; i < points.length; i++) {
                if (points[i].label === -1) assignedLabels[i] = predictBound(sup, points[i].x, points[i].y);
            }
            const trainPts = points.map((p, i) => ({ ...p, label: assignedLabels[i] }));
            return { model: fitLR(trainPts), assigned: assignedLabels };
        }
        if (mode === 'self') {
            // 3 rounds of confidence-thresholded self-training
            for (let r = 0; r < 3; r++) {
                const lp = points.map((p, i) => ({ ...p, label: assignedLabels[i] }))
                                 .filter(p => p.label >= 0);
                const model = fitLR(lp);
                for (let i = 0; i < points.length; i++) {
                    if (assignedLabels[i] === -1) {
                        const prob = predictProb(model, points[i].x, points[i].y);
                        if (prob > 0.85) assignedLabels[i] = 1;
                        else if (prob < 0.15) assignedLabels[i] = 0;
                    }
                }
            }
            // Final fit on everything assigned
            const trainPts = points.map((p, i) => ({ ...p, label: assignedLabels[i] }))
                                  .filter(p => p.label >= 0);
            return { model: fitLR(trainPts), assigned: assignedLabels };
        }
        if (mode === 'graph') {
            // Label propagation via k-NN graph
            const K = 6;
            const N = points.length;
            const adj = Array.from({length: N}, () => []);
            for (let i = 0; i < N; i++) {
                const ds = [];
                for (let j = 0; j < N; j++) {
                    if (j === i) continue;
                    ds.push({ j, d: (points[i].x - points[j].x) ** 2 +
                                    (points[i].y - points[j].y) ** 2 });
                }
                ds.sort((a, b) => a.d - b.d);
                for (let k = 0; k < K; k++) adj[i].push(ds[k].j);
            }
            // Soft labels in [0,1], 0.5 = unknown
            const f = points.map(p => p.label === -1 ? 0.5 : p.label);
            for (let r = 0; r < 30; r++) {
                const fnew = f.slice();
                for (let i = 0; i < N; i++) {
                    if (points[i].label !== -1) continue;
                    let sum = 0;
                    for (const j of adj[i]) sum += f[j];
                    fnew[i] = sum / adj[i].length;
                }
                for (let i = 0; i < N; i++) f[i] = fnew[i];
            }
            for (let i = 0; i < N; i++) {
                assignedLabels[i] = f[i] > 0.5 ? 1 : 0;
            }
            const trainPts = points.map((p, i) => ({ ...p, label: assignedLabels[i] }));
            return { model: fitLR(trainPts), assigned: assignedLabels };
        }
        return { model: fitLR(labelled), assigned: assignedLabels };
    }

    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.55)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 20;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const r = runStrategy();
        const m = Math.min(W, H) / 2 - 20;
        ctx.fillStyle = '#fff';
        ctx.fillRect(W / 2 - m, H / 2 - m, 2 * m, 2 * m);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m - 0.5, H / 2 - m - 0.5, 2 * m + 1, 2 * m + 1);

        // Background decision regions
        const G = 50;
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx = -1.05 + (i + 0.5) / G * 2.1;
                const dy = -1.05 + (j + 0.5) / G * 2.1;
                const cls = predictBound(r.model, dx, dy);
                ctx.fillStyle = cls === 0 ? '#eef2ff' : '#fdf0eb';
                const [px, py] = toPx(dx, dy);
                ctx.fillRect(px - m * 1.05 / G - 1, py - m * 1.05 / G - 1,
                             m * 2.1 / G + 2, m * 2.1 / G + 2);
            }
        }

        // Boundary line
        if (Math.abs(r.model.w[1]) > 1e-6) {
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            const xL = -1.05, xR = 1.05;
            const yL = (-r.model.b - r.model.w[0] * xL) / r.model.w[1];
            const yR = (-r.model.b - r.model.w[0] * xR) / r.model.w[1];
            const [a, b] = toPx(xL, yL), [c, d] = toPx(xR, yR);
            ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
        }

        // Points
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const [px, py] = toPx(p.x, p.y);
            const isLabelled = (p.label !== -1);
            const assigned = r.assigned[i];
            if (isLabelled) {
                ctx.fillStyle = p.label === 0 ? '#4f46e5' : '#ea7959';
                ctx.strokeStyle = '#1a1a1a';
                ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
            } else if (mode === 'sup' || assigned === -1) {
                ctx.fillStyle = 'rgba(80, 80, 80, 0.4)';
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
            } else {
                // Soft assignment from the strategy
                ctx.fillStyle = assigned === 0 ? 'rgba(79, 70, 229, 0.65)'
                                               : 'rgba(234, 121, 89, 0.65)';
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.arc(px, py, 3.2, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
            }
        }

        // Accuracy on true labels
        let correct = 0;
        for (const p of points) if (predictBound(r.model, p.x, p.y) === p.c) correct++;
        const acc = (correct / points.length * 100).toFixed(0);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { sup: 'SUPERVISED — 8 labels only', pseudo: 'PSEUDO-LABELLING — 1 round',
                         graph: 'GRAPH PROPAGATION — k-NN',  self: 'SELF-TRAINING — 3 rounds' };
        ctx.fillText(titles[mode], W / 2 - m, H / 2 - m - 6);
        ctx.fillStyle = '#4f46e5';
        ctx.font = '600 11px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`acc ${acc}%`, W / 2 + m, H / 2 - m - 6);

        updateCaption(acc);
    }

    function updateCaption(acc) {
        if (!captionEl) return;
        const notes = {
            sup:    `<strong>Supervised only.</strong> Just 8 labels. The boundary is determined entirely by where those 8 points sit — accuracy ${acc}%. The grey points were ignored.`,
            pseudo: `<strong>Pseudo-labelling.</strong> The supervised model labelled every grey point; the final boundary used all of them. Accuracy ${acc}%. Errors from the initial fit get baked in.`,
            graph:  `<strong>Graph propagation.</strong> Built a k=6 nearest-neighbour graph; labels spread along the edges. Accuracy ${acc}% — graph methods exploit cluster structure directly.`,
            self:   `<strong>Self-training, 3 rounds.</strong> Each round, only the most confident pseudo-labels (prob &gt; 0.85) are added. Accuracy ${acc}% — confidence-thresholding limits error compounding.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [supBtn, pseudoBtn, graphBtn, selfBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    supBtn?.addEventListener('click',    () => setMode('sup', supBtn));
    pseudoBtn?.addEventListener('click', () => setMode('pseudo', pseudoBtn));
    graphBtn?.addEventListener('click',  () => setMode('graph', graphBtn));
    selfBtn?.addEventListener('click',   () => setMode('self', selfBtn));
    resetBtn?.addEventListener('click', () => { regenerate(); draw(); });

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
