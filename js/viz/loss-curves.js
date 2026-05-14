/* Interactive loss-curve forensics viz.
 * Six canonical training curves — pick one, see the pattern + diagnosis. */

(function () {
    const canvas    = document.getElementById('viz-lc-canvas');
    const captionEl = document.getElementById('viz-lc-caption');
    if (!canvas) return;
    const buttons = {
        healthy:   document.getElementById('viz-lc-healthy'),
        diverge:   document.getElementById('viz-lc-diverge'),
        plateau:   document.getElementById('viz-lc-plateau'),
        oscillate: document.getElementById('viz-lc-oscillate'),
        overfit:   document.getElementById('viz-lc-overfit'),
        schedule:  document.getElementById('viz-lc-schedule'),
    };

    let mode = 'healthy';
    let ctx;
    let W = 0, H = 0;
    const N = 200;

    function smooth(s) {
        const out = []; let m = 0; const α = 0.85;
        for (let i = 0; i < s.length; i++) {
            m = i === 0 ? s[i] : α * m + (1 - α) * s[i];
            out.push(m);
        }
        return out;
    }
    function rand() { return Math.random() * 2 - 1; }

    function curves() {
        const train = [], val = [];
        for (let i = 0; i < N; i++) {
            const t = i / N;
            let trn, va;
            if (mode === 'healthy') {
                trn = 1.5 * Math.exp(-t * 3) + 0.18 + 0.04 * rand();
                va  = 1.5 * Math.exp(-t * 2.5) + 0.22 + 0.04 * rand();
            } else if (mode === 'diverge') {
                trn = i < 50 ? 1.5 - t * 1.0 + 0.04 * rand()
                              : Math.min(20, 0.5 * Math.exp((i - 50) * 0.05)) + 0.04 * rand();
                va  = trn + 0.05;
            } else if (mode === 'plateau') {
                trn = 1.0 - 0.02 * t + 0.03 * rand();
                va  = trn + 0.05;
            } else if (mode === 'oscillate') {
                trn = 0.8 + 0.5 * Math.sin(i * 0.4) + 0.05 * rand();
                va  = trn + 0.07 + 0.04 * rand();
            } else if (mode === 'overfit') {
                trn = 0.05 + 1.2 * Math.exp(-t * 4) + 0.02 * rand();
                va  = 0.4  + 1.0 * Math.exp(-t * 4) + 0.07 * t + 0.04 * rand();
            } else if (mode === 'schedule') {
                let base = 0.6 + 0.8 * Math.exp(-t * 1.5) + 0.04 * rand();
                if (i > 100) base -= 0.2 * (1 - Math.exp(-(i - 100) * 0.04));
                if (i > 150) base -= 0.05 * (1 - Math.exp(-(i - 150) * 0.05));
                trn = base; va = base + 0.05;
            }
            train.push(trn); val.push(va);
        }
        return { train: smooth(train), val: smooth(val) };
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(280, cssW * 0.42)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 40;
        const box = { x: pad, y: 30, w: W - 2 * pad, h: H - 80 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        const { train, val } = curves();
        const yMax = Math.min(4, Math.max(2, Math.max(...train, ...val) * 1.05));

        // Y gridlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.setLineDash([2, 3]);
        for (let v = 0; v <= yMax; v++) {
            const y = box.y + box.h - (v / yMax) * box.h;
            ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
        }
        ctx.setLineDash([]);

        // Curves
        function drawCurve(data, col) {
            ctx.strokeStyle = col;
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {
                const x = box.x + (i / (data.length - 1)) * box.w;
                const y = box.y + box.h - (Math.min(yMax, data[i]) / yMax) * box.h;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        drawCurve(train, '#4f46e5');
        drawCurve(val,   '#ea7959');

        // Axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const v of [0, N / 2, N]) {
            ctx.fillText(v, box.x + (v / N) * box.w, box.y + box.h + 12);
        }
        ctx.textAlign = 'right';
        for (const v of [0, yMax / 2, yMax]) {
            ctx.fillText(v.toFixed(1), box.x - 4,
                          box.y + box.h - (v / yMax) * box.h + 3);
        }
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('step', box.x + box.w / 2, box.y + box.h + 24);
        ctx.save();
        ctx.translate(box.x - 26, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('loss', 0, 0);
        ctx.restore();

        // Title + legend
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = {
            healthy: 'HEALTHY — train and val both decrease, val gap stable',
            diverge: 'DIVERGES — loss explodes, then NaN',
            plateau: 'PLATEAU — loss never decreases',
            oscillate: 'OSCILLATES — loss bounces, no convergence',
            overfit: 'OVERFITS — train drops, val rises',
            schedule: 'SCHEDULE KICK — visible drop where lr decreases',
        };
        ctx.fillText(titles[mode], box.x, box.y - 8);
        // Legend
        ctx.fillStyle = '#4f46e5'; ctx.fillRect(box.x + box.w - 90, box.y - 14, 10, 2.4);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)'; ctx.font = '500 9px "Inter", system-ui, sans-serif';
        ctx.fillText('train', box.x + box.w - 76, box.y - 8);
        ctx.fillStyle = '#ea7959'; ctx.fillRect(box.x + box.w - 50, box.y - 14, 10, 2.4);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('val', box.x + box.w - 36, box.y - 8);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            healthy:   `<strong>Healthy.</strong> Train loss decreases smoothly; val tracks it with a small gap that stabilises. This is the target shape — every other panel here is a failure mode to avoid.`,
            diverge:   `<strong>Diverges.</strong> Loss rises and then explodes — usually NaN soon after. Causes: learning rate too high, exploding gradient, fp16 underflow, bad init. Fix: drop lr 10×, clip gradients, try bf16.`,
            plateau:   `<strong>Plateau.</strong> Loss never decreases. Gradients aren't flowing — detached graph, frozen layer, wrong optimizer setup. Run a 1-batch overfit; if it fails, you have a loop bug.`,
            oscillate: `<strong>Oscillates.</strong> Loss bounces up and down. Almost always learning rate too high (especially with a small batch). Halve the lr, try again.`,
            overfit:   `<strong>Overfits.</strong> Train loss continues to drop but val rises. Need regularization (dropout, weight decay), more data, or earlier stopping.`,
            schedule:  `<strong>Schedule kick.</strong> Smooth curve with a sudden drop when the LR scheduler kicks in. Often a good sign — the model finally settled at a lower lr. Make sure the kick is intentional.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const k in buttons) buttons[k]?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    for (const k in buttons) {
        buttons[k]?.addEventListener('click', () => setMode(k, buttons[k]));
    }

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
