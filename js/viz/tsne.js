/* Interactive t-SNE-flavoured viz.
 * 5 colour-coded clusters of 2D points; each Step applies an attractive force
 * between same-cluster points and a repulsive force between different-cluster
 * points (a simplified version of the t-SNE gradient). Watch the clusters
 * pull apart as the optimiser runs.
 *
 * This is a teaching cartoon — real t-SNE computes pairwise probabilities
 * in high-D and minimises KL — but the dynamics are recognisably similar. */

(function () {
    const canvas    = document.getElementById('viz-tsne-canvas');
    const stepBtn   = document.getElementById('viz-tsne-step');
    const playBtn   = document.getElementById('viz-tsne-play');
    const resetBtn  = document.getElementById('viz-tsne-reset');
    const stepLbl   = document.getElementById('viz-tsne-step-lbl');
    const captionEl = document.getElementById('viz-tsne-caption');
    if (!canvas) return;

    const COLOURS = ['#4f46e5', '#ea7959', '#10847e', '#d4a13c', '#a05bb0'];
    const N_CLUSTERS = 5, N_PER_CLUSTER = 18;
    let pts = [];      // {x, y, c}
    let step = 0;
    let playing = false, lastStep = 0;
    let ctx;
    let W = 0, H = 0;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function reset() {
        pts = [];
        for (let c = 0; c < N_CLUSTERS; c++) {
            for (let i = 0; i < N_PER_CLUSTER; i++) {
                pts.push({ x: randn() * 0.5, y: randn() * 0.5, c });
            }
        }
        step = 0;
        playing = false;
        if (playBtn) playBtn.textContent = 'Play';
        draw();
    }
    reset();

    function trainStep() {
        // Compute pairwise forces
        const N = pts.length;
        const fx = new Float32Array(N), fy = new Float32Array(N);
        const eps = 0.0001;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                if (i === j) continue;
                const dx = pts[i].x - pts[j].x;
                const dy = pts[i].y - pts[j].y;
                const d2 = dx * dx + dy * dy + eps;
                const inv = 1 / (1 + d2);
                if (pts[i].c === pts[j].c) {
                    // Attraction (same cluster) — force ~ -(target_attraction) * d / (1+d²)
                    fx[i] -= 0.04 * dx * inv;
                    fy[i] -= 0.04 * dy * inv;
                } else {
                    // Repulsion (different cluster) — force ~ +inv² * d
                    fx[i] += 0.6 * dx * inv * inv;
                    fy[i] += 0.6 * dy * inv * inv;
                }
            }
        }
        // Apply with small step
        const lr = 0.02;
        for (let i = 0; i < N; i++) {
            pts[i].x += lr * fx[i];
            pts[i].y += lr * fy[i];
            // Soft confinement
            const r = Math.hypot(pts[i].x, pts[i].y);
            if (r > 1.7) {
                pts[i].x *= 1.7 / r;
                pts[i].y *= 1.7 / r;
            }
        }
        step++;
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(440, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.62)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m / 1.8, H / 2 - y * m / 1.8];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const m = Math.min(W, H) / 2 - 16;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(W / 2 - m - 4, H / 2 - m - 4, 2 * m + 8, 2 * m + 8);

        for (const p of pts) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = COLOURS[p.c];
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`SIMPLIFIED t-SNE — step ${step}`, 14, 18);

        if (stepLbl) stepLbl.textContent = `step ${step}`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (step === 0) {
            captionEl.innerHTML =
                `<strong>Step 0.</strong> Points are randomly scattered. The colours encode "high-D cluster membership" — but spatially they're all mixed up. Click <strong>Step</strong> or <strong>Play</strong> — attractive forces pull same-colour pairs together; repulsive forces push different-colour pairs apart.`;
        } else if (step < 80) {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> Clusters are starting to form. The t-SNE force balance trades local pull-together against long-range push-apart. The Student-t tail in the denominator is what stops far-apart pairs from being crushed together (the "crowding problem" the original SNE had).`;
        } else {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> Clusters separated. Note that the <em>positions</em> of the clusters in this 2D plane are not meaningful — only that they're separated. That's why Wattenberg et al. (Distill 2016) say "don't read distances in t-SNE plots".`;
        }
    }

    function loop(now) {
        if (playing && now - lastStep > 50) {
            trainStep();
            draw();
            lastStep = now;
            if (step >= 400) { playing = false; if (playBtn) playBtn.textContent = 'Play'; }
        }
        requestAnimationFrame(loop);
    }

    stepBtn?.addEventListener('click', () => { trainStep(); draw(); });
    playBtn?.addEventListener('click', () => {
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
        lastStep = performance.now();
    });
    resetBtn?.addEventListener('click', reset);

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
    requestAnimationFrame(loop);
})();
