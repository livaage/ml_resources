/* Interactive anomaly-detection viz.
 * A Gaussian kernel density estimate over the 2D scatter; the background
 * heatmap shows local density. A threshold slider sets the cutoff — any
 * point whose KDE score is below the cutoff is flagged as an anomaly
 * (outlined in terracotta). Slide it down to flag only the most extreme
 * points; up to flag more. */

(function () {
    const canvas    = document.getElementById('viz-anom-canvas');
    const tSlider   = document.getElementById('viz-anom-threshold');
    const tLbl      = document.getElementById('viz-anom-threshold-lbl');
    const bSlider   = document.getElementById('viz-anom-bw');
    const bLbl      = document.getElementById('viz-anom-bw-lbl');
    const dataSel   = document.getElementById('viz-anom-data');
    const resetBtn  = document.getElementById('viz-anom-reset');
    const captionEl = document.getElementById('viz-anom-caption');
    if (!canvas) return;

    let dataset = 'cluster_outliers';
    let bandwidth = 0.12;
    let threshold = 0.25;   // fraction of max density
    let points = [];
    let scores = [];
    let heatCache = null;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function buildDataset() {
        const pts = [];
        if (dataset === 'cluster_outliers') {
            for (let i = 0; i < 120; i++) pts.push({ x: 0.0 + randn() * 0.18, y: 0.05 + randn() * 0.18 });
            // Outliers
            pts.push({ x:  0.7, y:  0.7 });
            pts.push({ x: -0.8, y: -0.6 });
            pts.push({ x:  0.85, y: -0.5 });
            pts.push({ x: -0.6, y:  0.75 });
        } else if (dataset === 'two_clusters') {
            for (let i = 0; i < 70; i++) pts.push({ x: -0.45 + randn() * 0.12, y:  0.30 + randn() * 0.12 });
            for (let i = 0; i < 70; i++) pts.push({ x:  0.45 + randn() * 0.12, y: -0.30 + randn() * 0.12 });
            // Outliers between
            pts.push({ x: 0, y: 0 });
            pts.push({ x: 0.0, y: 0.8 });
            pts.push({ x: -0.8, y: -0.2 });
        } else if (dataset === 'ring') {
            for (let i = 0; i < 120; i++) {
                const a = Math.random() * Math.PI * 2;
                const r = 0.55 + randn() * 0.04;
                pts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
            }
            pts.push({ x: 0, y: 0 });
            pts.push({ x: 0.1, y: -0.1 });
            pts.push({ x: -0.05, y: 0.05 });
        }
        return pts;
    }

    function kde(x, y) {
        let s = 0;
        const bw2 = 2 * bandwidth * bandwidth;
        for (const p of points) {
            const d2 = (x - p.x) ** 2 + (y - p.y) ** 2;
            s += Math.exp(-d2 / bw2);
        }
        return s;  // unnormalised — we compare relative to max in scores
    }

    function recomputeScores() {
        scores = points.map(p => kde(p.x, p.y));
        heatCache = null;
    }
    function reset() {
        points = buildDataset();
        recomputeScores();
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        heatCache = null;
        draw();
    }
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function densityColour(t) {
        // t ∈ [0, 1] — cream → indigo
        const r = 251 + (79 - 251) * t * 0.85;
        const g = 250 + (70 - 250) * t * 0.85;
        const b = 247 + (229 - 247) * t * 0.85;
        return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    }

    function buildHeatmap() {
        if (heatCache) return heatCache;
        const G = 64;
        const off = document.createElement('canvas');
        off.width = G; off.height = G;
        const ictx = off.getContext('2d');
        const img = ictx.createImageData(G, G);
        let mx = 0;
        const vals = new Float32Array(G * G);
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const dx =  ((i + 0.5) / G * 2 - 1) * 1.1;
                const dy = -((j + 0.5) / G * 2 - 1) * 1.1;
                const v = kde(dx, dy);
                vals[j * G + i] = v;
                if (v > mx) mx = v;
            }
        }
        for (let j = 0; j < G; j++) {
            for (let i = 0; i < G; i++) {
                const v = vals[j * G + i] / (mx || 1);
                const col = densityColour(v);
                const m = col.match(/rgb\((\d+), (\d+), (\d+)\)/);
                const idx = (j * G + i) * 4;
                img.data[idx]     = +m[1];
                img.data[idx + 1] = +m[2];
                img.data[idx + 2] = +m[3];
                img.data[idx + 3] = 255;
            }
        }
        ictx.putImageData(img, 0, 0);
        heatCache = { canvas: off, mx };
        return heatCache;
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const m = Math.min(W, H) / 2 - 16;
        const hm = buildHeatmap();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(hm.canvas, W / 2 - m * 1.1, H / 2 - m * 1.1, m * 2.2, m * 2.2);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m * 1.1 - 0.5, H / 2 - m * 1.1 - 0.5, m * 2.2 + 1, m * 2.2 + 1);

        // Determine anomalies: score < threshold * mx
        const cutoff = threshold * (hm.mx || 1);
        let anomalies = 0;
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const isAnom = scores[i] < cutoff;
            if (isAnom) anomalies++;
            const [px, py] = toPx(p.x, p.y);
            if (isAnom) {
                ctx.fillStyle = 'rgba(234, 121, 89, 0.25)';
                ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill();
            }
            ctx.fillStyle = isAnom ? '#ea7959' : '#4f46e5';
            ctx.strokeStyle = isAnom ? '#1a1a1a' : 'rgba(255, 255, 255, 0.85)';
            ctx.lineWidth = isAnom ? 1.6 : 1;
            ctx.beginPath();
            ctx.arc(px, py, isAnom ? 4.5 : 3.2, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`KDE — bandwidth ${bandwidth.toFixed(2)}, threshold ${threshold.toFixed(2)}`,
            W / 2 - m * 1.1, H / 2 - m * 1.1 - 6);

        updateCaption(anomalies);
    }

    function updateCaption(nAnom) {
        if (!captionEl) return;
        captionEl.innerHTML =
            `<strong>${nAnom} point${nAnom === 1 ? '' : 's'} flagged.</strong> The background heatmap is the kernel ` +
            `density estimate; each training point contributes a Gaussian of width <em>${bandwidth.toFixed(2)}</em>. ` +
            `Anomalies are the points that landed where the model thinks density is low — slide the threshold up to ` +
            `flag more, down to be more conservative. Drop the bandwidth to make the density "lumpier"; raise it for ` +
            `a smoother estimate.`;
    }

    // ----- Controls -----
    if (tSlider) {
        tSlider.min = 0.01; tSlider.max = 0.95; tSlider.step = 0.01; tSlider.value = 0.25;
        tSlider.addEventListener('input', () => {
            threshold = parseFloat(tSlider.value);
            if (tLbl) tLbl.textContent = `τ = ${threshold.toFixed(2)}`;
            draw();
        });
    }
    if (bSlider) {
        bSlider.min = 0.04; bSlider.max = 0.40; bSlider.step = 0.02; bSlider.value = 0.12;
        bSlider.addEventListener('input', () => {
            bandwidth = parseFloat(bSlider.value);
            if (bLbl) bLbl.textContent = `h = ${bandwidth.toFixed(2)}`;
            recomputeScores();
            draw();
        });
    }
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="cluster_outliers">Cluster + outliers</option>
            <option value="two_clusters">Two clusters</option>
            <option value="ring">Ring + inside</option>
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
