/* Interactive PCA viz.
 * Left panel: 2D scatter with a draggable projection axis. As you spin the
 * axis, the right panel updates a 1D histogram of the projected coordinates
 * and shows the resulting variance.
 *
 * PCA's first principal component is, by definition, the axis that maximises
 * that variance. Hit "Find PC1" to snap to it — and notice the captured
 * variance always equals the larger eigenvalue of the data covariance. */

(function () {
    const canvas    = document.getElementById('viz-dr-canvas');
    const angleSli  = document.getElementById('viz-dr-angle');
    const angleLbl  = document.getElementById('viz-dr-angle-lbl');
    const findBtn   = document.getElementById('viz-dr-find');
    const dataSel   = document.getElementById('viz-dr-data');
    const resetBtn  = document.getElementById('viz-dr-reset');
    const captionEl = document.getElementById('viz-dr-caption');
    if (!canvas) return;

    let dataset = 'tilted';
    let points = [];          // [{x, y}]
    let theta = 0;            // projection angle in radians
    let pc1Theta = 0;
    let eigs = { l1: 1, l2: 1 };
    let animating = false;
    let animFrom = 0, animTo = 0, animStart = 0;
    const ANIM_MS = 700;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function buildDataset() {
        const pts = [];
        if (dataset === 'tilted') {
            // 60° tilted, elongated cluster
            const c = Math.cos(Math.PI / 3), s = Math.sin(Math.PI / 3);
            for (let i = 0; i < 140; i++) {
                const a = randn() * 0.55, b = randn() * 0.10;
                pts.push({ x: c * a - s * b, y: s * a + c * b });
            }
        } else if (dataset === 'flat') {
            // Horizontal elongated
            for (let i = 0; i < 140; i++) pts.push({ x: randn() * 0.6, y: randn() * 0.08 });
        } else if (dataset === 'circle') {
            // Isotropic — PCA is degenerate
            for (let i = 0; i < 140; i++) pts.push({ x: randn() * 0.35, y: randn() * 0.35 });
        } else if (dataset === 'twoblobs') {
            for (let i = 0; i < 70; i++) pts.push({ x: -0.4 + randn() * 0.08, y:  0.25 + randn() * 0.08 });
            for (let i = 0; i < 70; i++) pts.push({ x:  0.4 + randn() * 0.08, y: -0.25 + randn() * 0.08 });
        }
        return pts;
    }

    function fitPCA() {
        // 2x2 covariance + eigendecomposition
        const N = points.length;
        let mx = 0, my = 0;
        for (const p of points) { mx += p.x; my += p.y; }
        mx /= N; my /= N;
        let cxx = 0, cyy = 0, cxy = 0;
        for (const p of points) {
            const dx = p.x - mx, dy = p.y - my;
            cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
        }
        cxx /= N; cyy /= N; cxy /= N;
        const tr = cxx + cyy;
        const det = cxx * cyy - cxy * cxy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        let ang;
        if (Math.abs(cxy) > 1e-9) ang = Math.atan2(l1 - cxx, cxy);
        else if (cxx >= cyy)      ang = 0;
        else                      ang = Math.PI / 2;
        return { l1: Math.max(0, l1), l2: Math.max(0, l2), angle: ang };
    }

    function projectedVariance(angle) {
        // Variance of {<p, (cos a, sin a)>}
        const N = points.length;
        const ca = Math.cos(angle), sa = Math.sin(angle);
        let m = 0;
        for (const p of points) m += ca * p.x + sa * p.y;
        m /= N;
        let v = 0;
        for (const p of points) {
            const z = ca * p.x + sa * p.y - m;
            v += z * z;
        }
        return v / N;
    }

    function reset() {
        points = buildDataset();
        const fit = fitPCA();
        eigs = { l1: fit.l1, l2: fit.l2 };
        pc1Theta = fit.angle;
        theta = 0;
        if (angleSli) angleSli.value = 0;
        if (angleLbl) angleLbl.textContent = `θ = 0°`;
        animating = false;
        draw();
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(340, cssW * 0.50)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function layout() {
        const pad = 14;
        const half = Math.min(H - 2 * pad, (W - 3 * pad) / 2);
        const scat = { x: pad, y: pad + 14, w: half, h: H - 2 * pad - 14 };
        const hist = { x: scat.x + scat.w + pad, y: scat.y, w: W - scat.x - scat.w - 2 * pad, h: scat.h };
        return { scat, hist };
    }
    function toPx(x, y, box) {
        const m = Math.min(box.w, box.h) / 2 - 8;
        return [box.x + box.w / 2 + x * m, box.y + box.h / 2 - y * m];
    }

    function drawScatter(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Axes
        const [cx, cy] = toPx(0, 0, box);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(box.x + 6, cy); ctx.lineTo(box.x + box.w - 6, cy);
        ctx.moveTo(cx, box.y + 6); ctx.lineTo(cx, box.y + box.h - 6);
        ctx.stroke();
        ctx.setLineDash([]);

        // PC1 / PC2 ghost arrows (faint)
        const m = Math.min(box.w, box.h) / 2 - 8;
        const r1 = Math.sqrt(eigs.l1) * m * 2.2;
        const r2 = Math.sqrt(eigs.l2) * m * 2.2;
        ctx.strokeStyle = 'rgba(79, 70, 229, 0.18)';
        ctx.lineWidth = 1;
        function ghostLine(angle, r) {
            const ca = Math.cos(angle), sa = Math.sin(angle);
            ctx.beginPath();
            ctx.moveTo(cx - ca * r, cy + sa * r);
            ctx.lineTo(cx + ca * r, cy - sa * r);
            ctx.stroke();
        }
        ghostLine(pc1Theta, r1);
        ghostLine(pc1Theta + Math.PI / 2, r2);

        // Points + their projection onto the axis
        const ca = Math.cos(theta), sa = Math.sin(theta);
        // First draw projection segments (so points overlay them)
        ctx.strokeStyle = 'rgba(234, 121, 89, 0.18)';
        for (const p of points) {
            const proj = ca * p.x + sa * p.y;  // scalar projection
            const px = proj * ca, py = proj * sa;
            const [a1, b1] = toPx(p.x, p.y, box);
            const [a2, b2] = toPx(px, py, box);
            ctx.beginPath(); ctx.moveTo(a1, b1); ctx.lineTo(a2, b2); ctx.stroke();
        }

        // Current axis line
        const r = m * 2.2;
        ctx.strokeStyle = '#ea7959';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(cx - ca * r, cy + sa * r);
        ctx.lineTo(cx + ca * r, cy - sa * r);
        ctx.stroke();

        // Points
        ctx.fillStyle = '#4f46e5';
        for (const p of points) {
            const [px, py] = toPx(p.x, p.y, box);
            ctx.beginPath(); ctx.arc(px, py, 2.6, 0, Math.PI * 2); ctx.fill();
        }
        // Projected points on the axis
        ctx.fillStyle = '#ea7959';
        for (const p of points) {
            const proj = ca * p.x + sa * p.y;
            const px = proj * ca, py = proj * sa;
            const [pxx, pyy] = toPx(px, py, box);
            ctx.beginPath(); ctx.arc(pxx, pyy, 2.2, 0, Math.PI * 2); ctx.fill();
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('DATA + projection axis', box.x, box.y - 5);
    }

    function drawHist(box) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Compute 1D projection
        const ca = Math.cos(theta), sa = Math.sin(theta);
        const projs = points.map(p => ca * p.x + sa * p.y);
        let lo = Infinity, hi = -Infinity;
        for (const v of projs) { if (v < lo) lo = v; if (v > hi) hi = v; }
        // Pin the range to [-1.2, 1.2] for stability when comparing angles
        lo = -1.2; hi = 1.2;
        const nb = 28;
        const bins = new Float32Array(nb);
        for (const v of projs) {
            const t = (v - lo) / (hi - lo);
            const idx = Math.max(0, Math.min(nb - 1, Math.floor(t * nb)));
            bins[idx]++;
        }
        let mx = 0;
        for (const c of bins) if (c > mx) mx = c;

        const barW = (box.w - 24) / nb;
        for (let i = 0; i < nb; i++) {
            const h = (bins[i] / (mx || 1)) * (box.h - 90);
            ctx.fillStyle = 'rgba(234, 121, 89, 0.75)';
            ctx.fillRect(box.x + 12 + i * barW, box.y + box.h - 50 - h, barW - 1, h);
        }

        // Variance
        const v = projectedVariance(theta);
        const vRatio = v / Math.max(1e-9, eigs.l1 + eigs.l2);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('PROJECTED — 1D histogram + variance', box.x, box.y - 5);

        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText(`variance(projection) = ${v.toFixed(3)}`, box.x + 14, box.y + box.h - 28);
        ctx.fillText(`captured variance ratio = ${(vRatio * 100).toFixed(0)}%`, box.x + 14, box.y + box.h - 12);

        // Max-variance hint (small ▲ at the PC1 angle on the histogram footer)
        const fillBarW = box.w - 24;
        // Visualise: paint a small marker at the best variance
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const cur = box.x + 14 + (v / eigs.l1) * (fillBarW - 14);
        ctx.moveTo(box.x + 14, box.y + 16);
        ctx.lineTo(box.x + 14 + fillBarW - 14, box.y + 16);
        ctx.stroke();
        // current marker
        ctx.fillStyle = '#ea7959';
        ctx.beginPath();
        ctx.arc(cur, box.y + 16, 5, 0, Math.PI * 2);
        ctx.fill();
        // λ1 marker
        ctx.fillStyle = '#4f46e5';
        ctx.fillText(`λ₁ = ${eigs.l1.toFixed(3)}`, box.x + 14, box.y + 36);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);
        const lay = layout();
        drawScatter(lay.scat);
        drawHist(lay.hist);
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const v = projectedVariance(theta);
        const ratio = v / Math.max(1e-9, eigs.l1 + eigs.l2);
        const captured = (ratio * 100).toFixed(0);
        const optDeg = (pc1Theta * 180 / Math.PI).toFixed(0);
        captionEl.innerHTML =
            `<strong>${captured}%</strong> of total variance is captured by the current axis. ` +
            `PC1 — the variance-maximising direction — is at <strong>${optDeg}°</strong>. ` +
            `When θ matches PC1, the histogram is at its widest; perpendicular to PC1, it's at its narrowest. ` +
            `That single number, λ₁ / (λ₁ + λ₂), is the fraction of information you'd keep by reducing 2D → 1D.`;
    }

    // ----- Animation -----
    function loop(now) {
        if (animating) {
            const p = Math.min(1, (now - animStart) / ANIM_MS);
            theta = animFrom + (animTo - animFrom) * (p * p * (3 - 2 * p));   // smoothstep
            if (angleSli) angleSli.value = ((theta % Math.PI) + Math.PI) % Math.PI;
            if (angleLbl) angleLbl.textContent = `θ = ${(theta * 180 / Math.PI).toFixed(0)}°`;
            draw();
            if (p >= 1) animating = false;
        }
        requestAnimationFrame(loop);
    }

    // ----- Controls -----
    if (angleSli) {
        angleSli.min = 0; angleSli.max = Math.PI.toFixed(4);
        angleSli.step = 0.02; angleSli.value = 0;
        angleSli.addEventListener('input', () => {
            animating = false;
            theta = parseFloat(angleSli.value);
            if (angleLbl) angleLbl.textContent = `θ = ${(theta * 180 / Math.PI).toFixed(0)}°`;
            draw();
        });
    }
    findBtn?.addEventListener('click', () => {
        // Animate from current theta toward pc1Theta (mod π)
        animFrom = theta;
        animTo = ((pc1Theta % Math.PI) + Math.PI) % Math.PI;
        // Pick the shorter direction
        if (Math.abs(animTo - animFrom) > Math.PI / 2) {
            if (animTo > animFrom) animTo -= Math.PI;
            else                    animTo += Math.PI;
        }
        animStart = performance.now();
        animating = true;
    });
    if (dataSel) {
        dataSel.innerHTML = `
            <option value="tilted">Tilted cloud</option>
            <option value="flat">Horizontal stretch</option>
            <option value="circle">Isotropic (PCA degenerate)</option>
            <option value="twoblobs">Two blobs</option>
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
    requestAnimationFrame(loop);
})();
