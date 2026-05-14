/* Interactive "what each paradigm sees" viz.
 * Same 2D dataset (two natural clusters) shown five ways:
 *   Supervised      — all points labelled (indigo / terracotta)
 *   Semi-supervised — most points grey, a few labelled
 *   Self-supervised — half each point's "context" hidden, asked to predict from the other
 *   Unsupervised    — all points grey
 *   Reinforcement   — no points; a tiny grid world with a reward signal */

(function () {
    const canvas    = document.getElementById('viz-lt-canvas');
    const btnSup    = document.getElementById('viz-lt-sup');
    const btnSemi   = document.getElementById('viz-lt-semi');
    const btnSelf   = document.getElementById('viz-lt-self');
    const btnUnsup  = document.getElementById('viz-lt-unsup');
    const btnRL     = document.getElementById('viz-lt-rl');
    const captionEl = document.getElementById('viz-lt-caption');
    if (!canvas) return;

    let mode = 'supervised';
    let points = [];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    (function init() {
        for (let i = 0; i < 60; i++) points.push({ x: -0.4 + randn() * 0.16, y:  0.3 + randn() * 0.16, c: 0 });
        for (let i = 0; i < 60; i++) points.push({ x:  0.4 + randn() * 0.16, y: -0.3 + randn() * 0.16, c: 1 });
    })();

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
        const m = Math.min(W, H) / 2 - 16;
        return [W / 2 + x * m, H / 2 - y * m];
    }

    function drawScatter(showLabels) {
        const m = Math.min(W, H) / 2 - 16;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        ctx.strokeRect(W / 2 - m - 4, H / 2 - m - 4, 2 * m + 8, 2 * m + 8);

        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            let col;
            if (showLabels === 'all')                col = p.c === 0 ? '#4f46e5' : '#ea7959';
            else if (showLabels === 'none')          col = 'rgba(60, 60, 60, 0.55)';
            else if (showLabels === 'partial') {
                // First 6 of each class show labels, rest grey
                col = (p._labelled) ? (p.c === 0 ? '#4f46e5' : '#ea7959')
                                    : 'rgba(60, 60, 60, 0.35)';
            }
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(px, py, showLabels === 'all' ? 3.4 : 3.0, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function drawSelfSupervised() {
        // Same scatter, but each point has a "context" (left half is visible, right is hidden)
        // Visualise by drawing each point twice: left-half region as a circle, right-half as faded ?
        const m = Math.min(W, H) / 2 - 16;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.strokeRect(W / 2 - m - 4, H / 2 - m - 4, 2 * m + 8, 2 * m + 8);

        for (const p of points) {
            const [px, py] = toPx(p.x, p.y);
            ctx.fillStyle = 'rgba(79, 70, 229, 0.55)';
            // Left half — "visible context"
            ctx.beginPath();
            ctx.arc(px, py, 4, Math.PI / 2, 3 * Math.PI / 2);
            ctx.fill();
            // Right half — "hidden, predict me"
            ctx.fillStyle = 'rgba(60, 60, 60, 0.25)';
            ctx.beginPath();
            ctx.arc(px, py, 4, -Math.PI / 2, Math.PI / 2);
            ctx.fill();
        }
        // Annotation
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('predict the grey half from the indigo half',
                     W / 2, H / 2 + m + 24);
    }

    function drawRL() {
        // Small grid world with an agent and a goal
        const cols = 7, rows = 5;
        const cw = Math.min((W - 40) / cols, (H - 80) / rows);
        const ox = (W - cw * cols) / 2;
        const oy = (H - cw * rows) / 2 - 4;

        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                ctx.strokeRect(ox + c * cw, oy + r * cw, cw, cw);
            }
        }
        // Agent
        const ax = 1, ay = 3;
        ctx.fillStyle = '#4f46e5';
        ctx.beginPath();
        ctx.arc(ox + (ax + 0.5) * cw, oy + (ay + 0.5) * cw, cw * 0.3, 0, Math.PI * 2);
        ctx.fill();
        // Goal
        const gx = 5, gy = 1;
        ctx.fillStyle = '#ea7959';
        const s = cw * 0.35;
        ctx.fillRect(ox + (gx + 0.5) * cw - s / 2, oy + (gy + 0.5) * cw - s / 2, s, s);
        ctx.fillStyle = '#fff';
        ctx.font = '600 12px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('+1', ox + (gx + 0.5) * cw, oy + (gy + 0.5) * cw);
        // Arrow showing planned action
        ctx.strokeStyle = '#4f46e5';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(ox + (ax + 0.5) * cw + cw * 0.35, oy + (ay + 0.5) * cw);
        ctx.lineTo(ox + (ax + 1) * cw + cw * 0.05,   oy + (ay + 0.5) * cw);
        ctx.stroke();
        // Arrow head
        ctx.beginPath();
        ctx.moveTo(ox + (ax + 1) * cw + cw * 0.05,   oy + (ay + 0.5) * cw);
        ctx.lineTo(ox + (ax + 1) * cw - cw * 0.04,   oy + (ay + 0.5) * cw - 5);
        ctx.lineTo(ox + (ax + 1) * cw - cw * 0.04,   oy + (ay + 0.5) * cw + 5);
        ctx.closePath();
        ctx.fillStyle = '#4f46e5';
        ctx.fill();
        // Reset baseline
        ctx.textBaseline = 'alphabetic';

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('agent picks an action, receives a reward, repeats',
                     W / 2, oy + cw * rows + 28);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        // Top label
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`WHAT THE LEARNER SEES — ${mode}`, 14, 18);

        if (mode === 'supervised')        drawScatter('all');
        else if (mode === 'unsupervised') drawScatter('none');
        else if (mode === 'semi-supervised') {
            // Label first 6 of each class
            const counts = [0, 0];
            for (const p of points) {
                const want = counts[p.c] < 6;
                if (want) { p._labelled = true; counts[p.c]++; } else p._labelled = false;
            }
            drawScatter('partial');
        }
        else if (mode === 'self-supervised') drawSelfSupervised();
        else if (mode === 'reinforcement')   drawRL();

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            'supervised':      `<strong>Supervised.</strong> Every point comes with its label. The model learns a function from <em>x</em> to <em>y</em>. The most common setting and the easiest to evaluate — but you have to pay for the labels.`,
            'semi-supervised': `<strong>Semi-supervised.</strong> Only a small handful (12 here) of points are labelled; the rest are grey. The model must use the structure of the grey points to extrapolate from the few labelled ones — pseudo-labelling, consistency, contrastive losses all build on this.`,
            'self-supervised': `<strong>Self-supervised.</strong> No external labels — but the input itself gives them. Predict the masked half from the visible half; predict the next token from the past. The model learns rich representations cheaply, with effectively-infinite "labels".`,
            'unsupervised':    `<strong>Unsupervised.</strong> Just points. No labels, no objective beyond "find structure". Clustering, density estimation, dimensionality reduction. Useful when you don't yet know what to predict.`,
            'reinforcement':   `<strong>Reinforcement.</strong> No labels and no static dataset. The agent acts in an environment and receives a numerical reward. The learning signal is sparse and delayed. Used for control, games, and (increasingly) aligning LLMs.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [btnSup, btnSemi, btnSelf, btnUnsup, btnRL]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    btnSup?.addEventListener('click',   () => setMode('supervised', btnSup));
    btnSemi?.addEventListener('click',  () => setMode('semi-supervised', btnSemi));
    btnSelf?.addEventListener('click',  () => setMode('self-supervised', btnSelf));
    btnUnsup?.addEventListener('click', () => setMode('unsupervised', btnUnsup));
    btnRL?.addEventListener('click',    () => setMode('reinforcement', btnRL));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
