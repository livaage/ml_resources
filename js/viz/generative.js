/* Interactive generative-models comparison viz.
 * Same 2D target (a ring) shown four ways, each illustrating that family's
 * characteristic behaviour:
 *   VAE   — blurry Gaussian blob centred where the mean is (mode coverage but smoothed)
 *   GAN   — sharp samples concentrated at a small portion of the ring (mode collapse)
 *   flow  — exact match to the ring (when the architecture allows it)
 *   diff  — high-quality coverage from iterative denoising
 *   AR    — sequential generation; show samples drawn one coordinate at a time */

(function () {
    const canvas    = document.getElementById('viz-gen-canvas');
    const vaeBtn    = document.getElementById('viz-gen-vae');
    const ganBtn    = document.getElementById('viz-gen-gan');
    const flowBtn   = document.getElementById('viz-gen-flow');
    const diffBtn   = document.getElementById('viz-gen-diff');
    const arBtn     = document.getElementById('viz-gen-ar');
    const captionEl = document.getElementById('viz-gen-caption');
    if (!canvas) return;

    let mode = 'vae';
    let ctx;
    let W = 0, H = 0;

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // True samples — a ring
    function trueSamples(n) {
        const pts = [];
        for (let i = 0; i < n; i++) {
            const a = Math.random() * Math.PI * 2;
            const r = 0.6 + randn() * 0.04;
            pts.push([r * Math.cos(a), r * Math.sin(a)]);
        }
        return pts;
    }

    // Each model's characteristic samples (hand-crafted for the viz)
    function modelSamples(n) {
        const pts = [];
        if (mode === 'vae') {
            // Blurry — ring with too much variance (mode coverage but smoothed)
            for (let i = 0; i < n; i++) {
                const a = Math.random() * Math.PI * 2;
                const r = 0.6 + randn() * 0.18;     // big variance
                pts.push([r * Math.cos(a), r * Math.sin(a)]);
            }
        } else if (mode === 'gan') {
            // Mode collapse — concentrated in one arc of the ring
            for (let i = 0; i < n; i++) {
                const a = 0.4 + Math.random() * 1.2;     // narrow arc
                const r = 0.6 + randn() * 0.03;          // tight on the ring
                pts.push([r * Math.cos(a), r * Math.sin(a)]);
            }
        } else if (mode === 'flow') {
            // Exact — tight match to the ring
            for (let i = 0; i < n; i++) {
                const a = Math.random() * Math.PI * 2;
                const r = 0.6 + randn() * 0.03;
                pts.push([r * Math.cos(a), r * Math.sin(a)]);
            }
        } else if (mode === 'diff') {
            // High quality — close to true distribution
            for (let i = 0; i < n; i++) {
                const a = Math.random() * Math.PI * 2;
                const r = 0.6 + randn() * 0.045;
                pts.push([r * Math.cos(a), r * Math.sin(a)]);
            }
        } else if (mode === 'ar') {
            // Autoregressive — quantised, samples on a grid (each coord drawn sequentially)
            for (let i = 0; i < n; i++) {
                const a = Math.random() * Math.PI * 2;
                let x = 0.6 * Math.cos(a) + randn() * 0.05;
                let y = 0.6 * Math.sin(a) + randn() * 0.05;
                x = Math.round(x * 10) / 10;       // discretise x first
                y = Math.round(y * 10) / 10;       // then y, given x
                pts.push([x, y]);
            }
        }
        return pts;
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
        const m = Math.min(W, H) / 2 - 20;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const m = Math.min(W, H) / 2 - 20;
        ctx.fillStyle = '#fff';
        ctx.fillRect(W / 2 - m, H / 2 - m, 2 * m, 2 * m);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(W / 2 - m - 0.5, H / 2 - m - 0.5, 2 * m + 1, 2 * m + 1);

        // True distribution as a faint dashed ring
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const [cx, cy] = toPx(0, 0);
        ctx.arc(cx, cy, 0.6 * m, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Model samples
        const samples = modelSamples(200);
        ctx.fillStyle = '#ea7959';
        for (const [x, y] of samples) {
            const [px, py] = toPx(x, y);
            ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2); ctx.fill();
        }

        // Legend & title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { vae: 'VAE', gan: 'GAN', flow: 'NORMALISING FLOW',
                         diff: 'DIFFUSION', ar: 'AUTOREGRESSIVE' };
        ctx.fillText(`${titles[mode]} — samples (orange) vs true distribution (dashed)`,
                     W / 2 - m, H / 2 - m - 6);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            vae:  `<strong>VAE.</strong> Samples cover the ring but are spread too widely — the KL-to-prior + reconstruction loss smooths out the true distribution. Fast (one decode), no mode collapse, but blurry.`,
            gan:  `<strong>GAN.</strong> Samples are tight on the ring — but only in one arc. Classic mode collapse: the generator found a region the discriminator can't distinguish, and stopped exploring. Sharp samples, no likelihood.`,
            flow: `<strong>Normalising flow.</strong> Exact match to the ring — invertible architecture means we can map noise → data and back. Computable likelihood; architecture constraints limit expressive power.`,
            diff: `<strong>Diffusion.</strong> Samples closely track the ring with the right spread. High-quality coverage; the down side is iterative sampling — 50–1000 forward passes per generation. Distillation (consistency models, flow matching) helps.`,
            ar:   `<strong>Autoregressive.</strong> Sample x first, then y conditional on x. Notice the grid pattern — the model factorises through discrete steps. Exact likelihood; sequential sampling is the cost.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [vaeBtn, ganBtn, flowBtn, diffBtn, arBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    vaeBtn?.addEventListener('click',  () => setMode('vae', vaeBtn));
    ganBtn?.addEventListener('click',  () => setMode('gan', ganBtn));
    flowBtn?.addEventListener('click', () => setMode('flow', flowBtn));
    diffBtn?.addEventListener('click', () => setMode('diff', diffBtn));
    arBtn?.addEventListener('click',   () => setMode('ar', arBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
