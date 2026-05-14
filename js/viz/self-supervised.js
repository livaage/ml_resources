/* Interactive contrastive learning viz.
 * 8 "source images" each have 2 augmented views (16 points in embedding space).
 * Same-source pairs are connected by faint dashed lines. Each Step applies one
 * gradient on the contrastive loss: positive pairs attract; negatives repel.
 * Over time, the 8 source clusters pull together and spread away from each
 * other on the unit circle. */

(function () {
    const canvas    = document.getElementById('viz-ssl-canvas');
    const stepBtn   = document.getElementById('viz-ssl-step');
    const playBtn   = document.getElementById('viz-ssl-play');
    const resetBtn  = document.getElementById('viz-ssl-reset');
    const stepLbl   = document.getElementById('viz-ssl-step-lbl');
    const captionEl = document.getElementById('viz-ssl-caption');
    if (!canvas) return;

    const NUM_SOURCES = 8;
    const VIEWS = 2;
    const TEMP = 0.2;
    const LR = 0.18;
    const COLOURS = [
        '#4f46e5', '#ea7959', '#10847e', '#d4a13c',
        '#a05bb0', '#688f5b', '#c25b86', '#3b8da6',
    ];

    let pts = [];      // {x, y, source}
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
        for (let s = 0; s < NUM_SOURCES; s++) {
            for (let v = 0; v < VIEWS; v++) {
                pts.push({ x: 0.6 * randn(), y: 0.6 * randn(), source: s });
            }
        }
        step = 0;
        playing = false;
        if (playBtn) playBtn.textContent = 'Play';
        draw();
    }
    reset();

    function normalize(p) {
        const r = Math.hypot(p.x, p.y);
        if (r > 1e-9) { p.x /= r; p.y /= r; }
    }

    function trainStep() {
        // Project everything onto the unit circle first
        for (const p of pts) normalize(p);
        const N = pts.length;
        const grad = pts.map(() => ({ x: 0, y: 0 }));
        // For each anchor, compute InfoNCE-style gradient
        for (let i = 0; i < N; i++) {
            const a = pts[i];
            // positive: the other view of the same source
            const j_pos = pts.findIndex((q, k) => k !== i && q.source === a.source);
            // Softmax similarities over all other points
            const sims = [], idxs = [];
            for (let j = 0; j < N; j++) {
                if (j === i) continue;
                sims.push((a.x * pts[j].x + a.y * pts[j].y) / TEMP);
                idxs.push(j);
            }
            const mxSim = Math.max(...sims);
            const exps = sims.map(s => Math.exp(s - mxSim));
            const Z = exps.reduce((s, v) => s + v, 0);
            // InfoNCE loss: -log( exp(s_pos)/Z ).
            // Gradient on similarity to each point:
            //   ∂L/∂s_j = -1[j=pos] + p_j      (softmax probabilities)
            for (let k = 0; k < idxs.length; k++) {
                const j = idxs[k];
                const p_j = exps[k] / Z;
                const ind = (j === j_pos) ? 1 : 0;
                const coeff = (p_j - ind) / TEMP;
                // ∂sim/∂a = pts[j] / TEMP (because sim = (a · pts[j]) / TEMP)
                grad[i].x += coeff * pts[j].x;
                grad[i].y += coeff * pts[j].y;
            }
        }
        // Apply gradient and re-normalise
        for (let i = 0; i < N; i++) {
            pts[i].x -= LR * grad[i].x;
            pts[i].y -= LR * grad[i].y;
            normalize(pts[i]);
        }
        step++;
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
    function toPx(x, y) {
        const m = Math.min(W, H) / 2 - 24;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const m = Math.min(W, H) / 2 - 24;
        // Unit circle
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(W / 2, H / 2, m, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);

        // Positive-pair connectors
        for (let s = 0; s < NUM_SOURCES; s++) {
            const ps = pts.filter(p => p.source === s);
            if (ps.length < 2) continue;
            const [a, b] = ps;
            const [ax, ay] = toPx(a.x, a.y);
            const [bx, by] = toPx(b.x, b.y);
            ctx.strokeStyle = COLOURS[s] + '55';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 3]);
            ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
            ctx.setLineDash([]);
        }

        // Points
        for (const p of pts) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = COLOURS[p.source];
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`EMBEDDING SPACE (unit circle) — 8 sources × 2 views`, 16, 18);

        // Compute alignment & uniformity rough metrics
        let align = 0;
        for (let s = 0; s < NUM_SOURCES; s++) {
            const ps = pts.filter(p => p.source === s);
            if (ps.length < 2) continue;
            const [a, b] = ps;
            align += 2 - 2 * (a.x * b.x + a.y * b.y);  // sq-dist on circle
        }
        align /= NUM_SOURCES;

        ctx.fillStyle = '#4f46e5';
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`align = ${align.toFixed(2)}`, W - 16, 18);

        if (stepLbl) stepLbl.textContent = `step ${step}`;
        updateCaption(align);
    }

    function updateCaption(align) {
        if (!captionEl) return;
        if (step === 0) {
            captionEl.innerHTML =
                `<strong>Step 0.</strong> 16 random points on the unit circle — same-coloured dashed pairs are <em>positives</em> ` +
                `(augmented views of one source). Click <strong>Step</strong> — the InfoNCE loss pulls positives together and ` +
                `pushes everything else apart.`;
        } else if (align > 0.5) {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> Positives are getting closer (alignment ${align.toFixed(2)}). ` +
                `Keep stepping — eventually each colour collapses to a point and the 8 points spread maximally on the circle.`;
        } else {
            captionEl.innerHTML =
                `<strong>Step ${step}.</strong> Each colour has essentially collapsed to a single point (alignment ${align.toFixed(2)}); ` +
                `the 8 collapsed points spread out across the circle. That's alignment + uniformity — exactly what Wang &amp; Isola (2020) ` +
                `argued contrastive learning optimises.`;
        }
    }

    function loop(now) {
        if (playing && now - lastStep > 70) {
            trainStep();
            draw();
            lastStep = now;
            if (step >= 200) { playing = false; if (playBtn) playBtn.textContent = 'Play'; }
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
