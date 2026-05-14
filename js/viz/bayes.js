/* Interactive Bayesian inference viz.
 * The unknown is a coin's bias θ ∈ [0, 1]. The prior is a Beta(α₀, β₀).
 * Each "Flip H" / "Flip T" click adds one observation; the posterior
 * is Beta(α₀ + H, β₀ + T). Likelihood is θ^H · (1-θ)^T. All three
 * densities are plotted on the same axis, so you can watch the
 * posterior emerge as prior × likelihood (normalised). */

(function () {
    const canvas    = document.getElementById('viz-bayes-canvas');
    const aSlider   = document.getElementById('viz-bayes-alpha');
    const bSlider   = document.getElementById('viz-bayes-beta');
    const aLbl      = document.getElementById('viz-bayes-alpha-lbl');
    const bLbl      = document.getElementById('viz-bayes-beta-lbl');
    const flipH     = document.getElementById('viz-bayes-flip-h');
    const flipT     = document.getElementById('viz-bayes-flip-t');
    const flip10Btn = document.getElementById('viz-bayes-flip-10');
    const resetBtn  = document.getElementById('viz-bayes-reset');
    const captionEl = document.getElementById('viz-bayes-caption');
    if (!canvas) return;

    let alpha0 = 2;
    let beta0  = 2;
    let trueTheta = 0.65;
    let history = [];   // sequence of 1 (H) / 0 (T)

    function logBeta(a, b) {
        // log B(a, b) using lgamma
        return lgamma(a) + lgamma(b) - lgamma(a + b);
    }
    function lgamma(x) {
        // Lanczos approximation
        const g = 7;
        const c = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        ];
        if (x < 0.5) {
            return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
        }
        x -= 1;
        let a = c[0];
        const t = x + g + 0.5;
        for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
        return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
    }
    function betaPDF(theta, a, b) {
        if (theta <= 0 || theta >= 1) return 0;
        return Math.exp((a - 1) * Math.log(theta)
                      + (b - 1) * Math.log(1 - theta)
                      - logBeta(a, b));
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(420, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(320, cssW * 0.50)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function flip(value) {
        history.push(value);
        draw();
    }
    function flipMany() {
        // Sample 10 outcomes from the (hidden) trueTheta
        for (let i = 0; i < 10; i++) history.push(Math.random() < trueTheta ? 1 : 0);
        draw();
    }
    function reset() {
        history = [];
        draw();
    }

    // ----- Draw -----
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Layout
        const pad = 14;
        const tickY = H - pad - 22;
        const plotTop = pad + 18;
        const plotBot = tickY - 8;
        const plotL = pad + 30;
        const plotR = W - pad - 6;
        const plotW = plotR - plotL;

        // Axes / frame
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plotL, plotTop); ctx.lineTo(plotL, plotBot);
        ctx.moveTo(plotL, plotBot); ctx.lineTo(plotR, plotBot);
        ctx.stroke();
        // 0, 0.5, 1 ticks
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const t of [0, 0.5, 1]) {
            const x = plotL + t * plotW;
            ctx.fillText(t.toFixed(1), x, plotBot + 14);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
            ctx.beginPath(); ctx.moveTo(x, plotBot); ctx.lineTo(x, plotBot + 3); ctx.stroke();
        }
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('θ — unknown coin bias', plotL, plotTop - 6);

        const H_ = history.reduce((s, v) => s + v, 0);
        const T_ = history.length - H_;
        const aPost = alpha0 + H_;
        const bPost = beta0 + T_;

        // Densities on grid
        const G = 250;
        const xs = new Float64Array(G + 1);
        const prior = new Float64Array(G + 1);
        const like = new Float64Array(G + 1);
        const post = new Float64Array(G + 1);
        let likeMax = 0;
        for (let i = 0; i <= G; i++) {
            const t = i / G;
            xs[i] = t;
            prior[i] = betaPDF(t, alpha0, beta0);
            const lk = Math.pow(t, H_) * Math.pow(1 - t, T_);
            like[i]  = lk;
            if (lk > likeMax) likeMax = lk;
            post[i]  = betaPDF(t, aPost, bPost);
        }
        // Normalise likelihood to area 1 over the grid for plotting
        let likeSum = 0;
        for (let i = 0; i <= G; i++) likeSum += like[i];
        if (likeSum > 0) for (let i = 0; i <= G; i++) like[i] /= (likeSum / G);

        // Max y across all curves
        let yMax = 0;
        for (let i = 0; i <= G; i++) {
            if (prior[i] > yMax) yMax = prior[i];
            if (like[i]  > yMax) yMax = like[i];
            if (post[i]  > yMax) yMax = post[i];
        }
        yMax = Math.max(1, yMax * 1.05);

        function drawCurve(arr, colour, w, dash) {
            ctx.strokeStyle = colour;
            ctx.lineWidth = w;
            if (dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
            ctx.beginPath();
            for (let i = 0; i <= G; i++) {
                const x = plotL + xs[i] * plotW;
                const y = plotBot - (arr[i] / yMax) * (plotBot - plotTop);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
        // Posterior fill
        ctx.fillStyle = 'rgba(234, 121, 89, 0.15)';
        ctx.beginPath();
        for (let i = 0; i <= G; i++) {
            const x = plotL + xs[i] * plotW;
            const y = plotBot - (post[i] / yMax) * (plotBot - plotTop);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.lineTo(plotR, plotBot); ctx.lineTo(plotL, plotBot);
        ctx.closePath();
        ctx.fill();

        // Curves
        drawCurve(prior, 'rgba(79, 70, 229, 0.7)', 1.7, [4, 4]);
        drawCurve(like,  'rgba(0, 0, 0, 0.35)', 1.5);
        drawCurve(post,  '#ea7959', 2.6);

        // Posterior mean (vertical line)
        const postMean = aPost / (aPost + bPost);
        const meanX = plotL + postMean * plotW;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(meanX, plotTop); ctx.lineTo(meanX, plotBot); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#ea7959';
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`μ = ${postMean.toFixed(2)}`, meanX, plotTop + 11);

        // History ticks
        const tickX0 = plotL + 4;
        const tickX1 = plotR - 4;
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'left';
        ctx.fillText(`H = ${H_}, T = ${T_}, n = ${history.length}`, plotL, tickY + 9);
        const maxTicks = Math.min(history.length, 80);
        const step = (tickX1 - tickX0) / Math.max(1, maxTicks);
        for (let i = 0; i < maxTicks; i++) {
            const v = history[history.length - maxTicks + i];
            ctx.fillStyle = v === 1 ? '#ea7959' : '#4f46e5';
            ctx.fillRect(tickX0 + i * step, tickY + 13, Math.max(2, step - 1), 6);
        }

        // Legend
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(79, 70, 229, 0.85)';
        ctx.fillText('— prior',       plotR - 220, plotTop - 6);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('— likelihood',  plotR - 160, plotTop - 6);
        ctx.fillStyle = '#ea7959';
        ctx.fillText('— posterior',   plotR - 80,  plotTop - 6);

        updateCaption(aPost, bPost, postMean, H_, T_);
    }

    function updateCaption(a, b, mean, H_, T_) {
        if (!captionEl) return;
        if (history.length === 0) {
            captionEl.innerHTML =
                `<strong>No data yet</strong> — the posterior is the prior. Adjust α₀ and β₀ to encode your belief ` +
                `before seeing any coin flips. A Beta(1, 1) is uniform (no preference); higher α biases toward heads.`;
        } else if (history.length < 5) {
            captionEl.innerHTML =
                `<strong>${history.length} flip${history.length === 1 ? '' : 's'}.</strong> The posterior moves toward ` +
                `H/(H+T), but the prior still has a strong say. Posterior mean is ${mean.toFixed(2)}; ` +
                `posterior is Beta(${a.toFixed(0)}, ${b.toFixed(0)}).`;
        } else {
            const variance = a * b / ((a + b) ** 2 * (a + b + 1));
            const ci = Math.sqrt(variance) * 2;
            captionEl.innerHTML =
                `<strong>${history.length} flips.</strong> Posterior mean ${mean.toFixed(2)}, ±${ci.toFixed(2)} (2σ). ` +
                `The posterior is concentrating — that's Bayesian learning. Notice how the prior's dashed curve ` +
                `gets ignored once you have enough data.`;
        }
    }

    // ----- Controls -----
    if (aSlider) {
        aSlider.min = 0.5; aSlider.max = 20; aSlider.step = 0.1; aSlider.value = 2;
        aSlider.addEventListener('input', () => {
            alpha0 = parseFloat(aSlider.value);
            if (aLbl) aLbl.textContent = `α₀ = ${alpha0.toFixed(1)}`;
            draw();
        });
    }
    if (bSlider) {
        bSlider.min = 0.5; bSlider.max = 20; bSlider.step = 0.1; bSlider.value = 2;
        bSlider.addEventListener('input', () => {
            beta0 = parseFloat(bSlider.value);
            if (bLbl) bLbl.textContent = `β₀ = ${beta0.toFixed(1)}`;
            draw();
        });
    }
    flipH?.addEventListener('click', () => flip(1));
    flipT?.addEventListener('click', () => flip(0));
    flip10Btn?.addEventListener('click', flipMany);
    resetBtn?.addEventListener('click', reset);

    // ----- Init -----
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
