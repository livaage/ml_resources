/* Interactive 2D gradient descent viz.
 * Click anywhere on the loss surface to start a new trajectory.
 * Adjust the learning rate to see how step size affects convergence. */

(function () {
    const canvas = document.getElementById('viz-gd-canvas');
    if (!canvas) return;

    // ----- Coordinate system -----
    const MIN_X = -2.0, MAX_X = 2.0;
    const MIN_Y = -1.0, MAX_Y = 3.0;
    const PAD = 0;   // heatmap covers the whole canvas

    let ctx, W, H;
    let heatmap = null;

    function toPx(x, y) {
        return [
            PAD + (x - MIN_X) / (MAX_X - MIN_X) * (W - 2 * PAD),
            H - PAD - (y - MIN_Y) / (MAX_Y - MIN_Y) * (H - 2 * PAD),
        ];
    }
    function toData(px, py) {
        return [
            MIN_X + (px - PAD) / (W - 2 * PAD) * (MAX_X - MIN_X),
            MIN_Y + (H - PAD - py) / (H - 2 * PAD) * (MAX_Y - MIN_Y),
        ];
    }

    // ----- Loss landscape (scaled Rosenbrock — banana valley) -----
    const A = 10;
    const MIN_POS = { x: 1, y: 1 };

    function loss(x, y) {
        return (1 - x) ** 2 + A * (y - x * x) ** 2;
    }
    function grad(x, y) {
        return [
            -2 * (1 - x) - 4 * A * x * (y - x * x),
             2 * A * (y - x * x),
        ];
    }

    // ----- State -----
    let trail = [];           // [{x, y}, ...]
    let learningRate = 0.005;
    let playing = true;
    let lastStepTime = 0;
    const STEP_MS = 60;       // ms between gradient steps

    // ----- Colour ramp (loss → colour) -----
    // Cream → light indigo → deep indigo, perceptually monotone-ish
    function lerp(a, b, t) { return a + (b - a) * t; }
    function lerpRgb(a, b, t) {
        return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
    }
    const STOPS = [
        { t: 0.00, rgb: [251, 250, 247] },  // cream
        { t: 0.40, rgb: [199, 195, 240] },  // light indigo
        { t: 0.75, rgb: [110,  97, 220] },  // mid indigo
        { t: 1.00, rgb: [ 49,  46, 129] },  // deep indigo
    ];
    function colourFor(t) {
        for (let i = 1; i < STOPS.length; i++) {
            if (t <= STOPS[i].t) {
                const u = (t - STOPS[i-1].t) / (STOPS[i].t - STOPS[i-1].t);
                return lerpRgb(STOPS[i-1].rgb, STOPS[i].rgb, u);
            }
        }
        return STOPS[STOPS.length - 1].rgb;
    }

    function precomputeHeatmap() {
        // Find min/max log-loss for normalisation by sampling on a coarse grid
        let lMin = Infinity, lMax = -Infinity;
        const N = 50;
        for (let i = 0; i <= N; i++) {
            for (let j = 0; j <= N; j++) {
                const x = MIN_X + i * (MAX_X - MIN_X) / N;
                const y = MIN_Y + j * (MAX_Y - MIN_Y) / N;
                const l = Math.log(loss(x, y) + 0.01);
                if (l < lMin) lMin = l;
                if (l > lMax) lMax = l;
            }
        }

        const img = ctx.createImageData(W, H);
        for (let py = 0; py < H; py++) {
            for (let px = 0; px < W; px++) {
                const [x, y] = toData(px, py);
                const t = (Math.log(loss(x, y) + 0.01) - lMin) / (lMax - lMin);
                const c = colourFor(Math.max(0, Math.min(1, t)));
                const k = (py * W + px) * 4;
                img.data[k    ] = c[0];
                img.data[k + 1] = c[1];
                img.data[k + 2] = c[2];
                img.data[k + 3] = 255;
            }
        }
        heatmap = img;
    }

    // ----- Optimization step -----
    function step() {
        if (trail.length === 0) return;
        const p = trail[trail.length - 1];
        const [gx, gy] = grad(p.x, p.y);
        // Clamp gradient magnitude to prevent explosion at high learning rates
        const gnorm = Math.hypot(gx, gy);
        const MAX_G = 200;
        const scale = gnorm > MAX_G ? MAX_G / gnorm : 1;
        const nx = p.x - learningRate * gx * scale;
        const ny = p.y - learningRate * gy * scale;
        // Stop if we've left the visualisation box (diverged)
        if (nx < MIN_X || nx > MAX_X || ny < MIN_Y || ny > MAX_Y) {
            playing = false;
            return;
        }
        // Stop if we're effectively at the minimum
        if (Math.hypot(nx - p.x, ny - p.y) < 1e-4) {
            playing = false;
            return;
        }
        trail.push({ x: nx, y: ny });
    }

    // ----- Draw -----
    function draw() {
        if (!heatmap) return;
        ctx.putImageData(heatmap, 0, 0);

        // Mark global minimum at (1, 1)
        const [mx, my] = toPx(MIN_POS.x, MIN_POS.y);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(mx, my, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx - 3, my); ctx.lineTo(mx + 3, my);
        ctx.moveTo(mx, my - 3); ctx.lineTo(mx, my + 3);
        ctx.stroke();

        // Trail line
        if (trail.length > 1) {
            ctx.strokeStyle = '#ea7959';
            ctx.lineWidth = 2.25;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            for (let i = 0; i < trail.length; i++) {
                const [px, py] = toPx(trail[i].x, trail[i].y);
                if (i === 0) ctx.moveTo(px, py);
                else        ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Trail dots — fade earlier ones
        for (let i = 0; i < trail.length; i++) {
            const [px, py] = toPx(trail[i].x, trail[i].y);
            const isLast = (i === trail.length - 1);
            const alpha = isLast ? 1 : Math.max(0.25, 1 - (trail.length - 1 - i) * 0.03);
            ctx.fillStyle = `rgba(234, 121, 89, ${alpha})`;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(px, py, isLast ? 6 : 2.5, 0, Math.PI * 2);
            ctx.fill();
            if (isLast) ctx.stroke();
        }

        // Update stats
        const stats = document.getElementById('viz-gd-stats');
        if (stats) {
            const cur  = trail[trail.length - 1] || { x: 0, y: 0 };
            const dist = trail.length ? Math.hypot(cur.x - MIN_POS.x, cur.y - MIN_POS.y) : 0;
            stats.innerHTML = `
                <span><span class="label">iter</span>${trail.length ? trail.length - 1 : 0}</span>
                <span><span class="label">loss</span>${trail.length ? loss(cur.x, cur.y).toFixed(4) : '—'}</span>
                <span><span class="label">‖θ − θ*‖</span>${dist.toFixed(3)}</span>
                <span><span class="label">lr η</span>${learningRate.toFixed(4)}</span>
            `;
        }
    }

    // ----- Animation loop -----
    function loop(now) {
        if (playing && now - lastStepTime >= STEP_MS && trail.length > 0) {
            step();
            lastStepTime = now;
        }
        draw();
        requestAnimationFrame(loop);
    }

    // ----- Interaction -----
    function startTrail(x, y) {
        trail = [{ x, y }];
        playing = true;
        lastStepTime = performance.now();
        const playBtn = document.getElementById('viz-gd-play');
        if (playBtn) playBtn.textContent = 'Pause';
    }

    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const [x, y] = toData(px, py);
        if (x >= MIN_X && x <= MAX_X && y >= MIN_Y && y <= MAX_Y) {
            startTrail(x, y);
        }
    });

    const lrSlider = document.getElementById('viz-gd-lr');
    if (lrSlider) {
        lrSlider.addEventListener('input', (e) => {
            // Slider value 0..100 → log range 0.0005 .. 0.05
            const t = e.target.value / 100;
            learningRate = 0.0005 * Math.pow(100, t);
        });
    }

    const playBtn = document.getElementById('viz-gd-play');
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            playing = !playing;
            playBtn.textContent = playing ? 'Pause' : 'Play';
            if (playing) lastStepTime = performance.now();
        });
    }

    const resetBtn = document.getElementById('viz-gd-reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            startTrail(-1.5, 2.5);   // top-left start — long banana traverse
        });
    }

    // ----- Sizing (DPR-aware) -----
    // The canvas may live inside a hidden tab when this script runs.
    // Use ResizeObserver so we re-size + re-render once it becomes visible.
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const W_css = Math.round(rect.width);
        if (W_css <= 0) return;                       // not laid out yet
        const H_css = Math.round(W_css * 9 / 16);     // fixed 16:9
        canvas.style.height = H_css + 'px';

        const dpr = window.devicePixelRatio || 1;
        W = W_css;
        H = H_css;
        canvas.width  = W_css * dpr;
        canvas.height = H_css * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        precomputeHeatmap();
    }

    if (typeof ResizeObserver !== 'undefined') {
        let debounce = null;
        const ro = new ResizeObserver(() => {
            clearTimeout(debounce);
            debounce = setTimeout(resize, 60);
        });
        ro.observe(canvas);
    } else {
        // Fallback for very old browsers
        window.addEventListener('resize', resize);
    }
    resize();

    // Start with a default trajectory so the page isn't visually empty
    startTrail(-1.5, 2.5);
    requestAnimationFrame(loop);
})();
