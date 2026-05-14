/* Interactive loss-function viz.
 * Regression mode: x = residual (y − ŷ), curves are MSE / MAE / Huber.
 * Classification mode: x = margin (y · f(x)), curves are 0/1, hinge,
 * log-loss, and exponential. Hovering tracks each loss's value at the
 * cursor; sliding Huber δ shows the knee between quadratic and linear. */

(function () {
    const canvas    = document.getElementById('viz-loss-canvas');
    const regBtn    = document.getElementById('viz-loss-reg');
    const clsBtn    = document.getElementById('viz-loss-cls');
    const dSlider   = document.getElementById('viz-loss-delta');
    const dLbl      = document.getElementById('viz-loss-delta-lbl');
    const captionEl = document.getElementById('viz-loss-caption');
    if (!canvas) return;

    let mode = 'regression';   // 'regression' | 'classification'
    let delta = 0.5;
    let hoverX = null;

    const REG_CURVES = [
        {
            name: 'MSE',
            colour: '#4f46e5',
            f: (x) => x * x,
            note: 'quadratic — outliers dominate',
        },
        {
            name: 'MAE',
            colour: '#ea7959',
            f: (x) => Math.abs(x),
            note: 'linear — robust, but non-smooth at 0',
        },
        {
            name: 'Huber',
            colour: '#10847e',
            f: (x) => {
                const a = Math.abs(x);
                return a <= delta ? 0.5 * x * x : delta * (a - 0.5 * delta);
            },
            note: 'quadratic near 0, linear far away',
        },
    ];
    const CLS_CURVES = [
        {
            name: '0/1',
            colour: 'rgba(0, 0, 0, 0.5)',
            f: (x) => x > 0 ? 0 : 1,
            note: 'what you actually want — but flat → no gradient',
        },
        {
            name: 'log-loss',
            colour: '#4f46e5',
            f: (x) => Math.log(1 + Math.exp(-x)),
            note: 'cross-entropy — penalises even confident correct',
        },
        {
            name: 'hinge',
            colour: '#ea7959',
            f: (x) => Math.max(0, 1 - x),
            note: 'SVM choice — zero penalty past the margin',
        },
        {
            name: 'exponential',
            colour: '#10847e',
            f: (x) => Math.exp(-x),
            note: 'AdaBoost choice — penalises very strongly',
        },
    ];

    function curves() { return mode === 'regression' ? REG_CURVES : CLS_CURVES; }
    function xRange() { return mode === 'regression' ? [-2, 2] : [-2, 3]; }
    function xLabel() { return mode === 'regression' ? 'residual  y − ŷ' : 'margin  y · f(x)'; }
    function yMax()   { return 3.0; }

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

    function plotBox() {
        return { x: 50, y: 26, w: W - 220, h: H - 60 };
    }
    function toPx(xv, yv) {
        const box = plotBox();
        const [xmin, xmax] = xRange();
        return [
            box.x + ((xv - xmin) / (xmax - xmin)) * box.w,
            box.y + box.h - (yv / yMax()) * box.h,
        ];
    }
    function fromPxX(px) {
        const box = plotBox();
        const [xmin, xmax] = xRange();
        return xmin + ((px - box.x) / box.w) * (xmax - xmin);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const box = plotBox();
        // Frame
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Grid + axis labels
        const [xmin, xmax] = xRange();
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.setLineDash([2, 3]);
        for (let v = Math.ceil(xmin); v <= Math.floor(xmax); v++) {
            const [px] = toPx(v, 0);
            ctx.beginPath(); ctx.moveTo(px, box.y); ctx.lineTo(px, box.y + box.h); ctx.stroke();
        }
        for (let v = 0; v <= yMax(); v++) {
            const [, py] = toPx(0, v);
            ctx.beginPath(); ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.stroke();
        }
        ctx.setLineDash([]);

        // Zero line
        const [zx] = toPx(0, 0);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(zx, box.y); ctx.lineTo(zx, box.y + box.h); ctx.stroke();

        // Axis labels / ticks
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (let v = Math.ceil(xmin); v <= Math.floor(xmax); v++) {
            const [px] = toPx(v, 0);
            ctx.fillText(v, px, box.y + box.h + 14);
        }
        ctx.textAlign = 'right';
        for (let v = 0; v <= yMax(); v++) {
            const [, py] = toPx(0, v);
            ctx.fillText(v.toFixed(1), box.x - 6, py + 3);
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(xLabel(), box.x + box.w / 2, box.y + box.h + 30);
        ctx.save();
        ctx.translate(16, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('loss', 0, 0);
        ctx.restore();

        // Curves
        const G = 200;
        const cs = curves();
        for (const c of cs) {
            ctx.strokeStyle = c.colour;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (let i = 0; i <= G; i++) {
                const xv = xmin + (i / G) * (xmax - xmin);
                const yv = Math.min(yMax() + 0.5, c.f(xv));
                const [px, py] = toPx(xv, yv);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Hover line + readouts
        if (hoverX !== null) {
            const [hxpx] = toPx(hoverX, 0);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(hxpx, box.y); ctx.lineTo(hxpx, box.y + box.h); ctx.stroke();
            ctx.setLineDash([]);
            for (const c of cs) {
                const yv = c.f(hoverX);
                if (yv > yMax() + 0.2) continue;
                const [px, py] = toPx(hoverX, yv);
                ctx.fillStyle = c.colour;
                ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
            }
        }

        // Legend
        let lx = box.x + box.w + 14;
        let ly = box.y + 4;
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        for (const c of cs) {
            ctx.fillStyle = c.colour;
            ctx.fillRect(lx, ly + 4, 12, 2.6);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.font = '600 10px "Inter", system-ui, sans-serif';
            ctx.fillText(c.name, lx + 18, ly + 8);
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.fillText(c.note, lx + 18, ly + 20);
            // hover value
            if (hoverX !== null) {
                const yv = c.f(hoverX);
                ctx.font = '500 9px "JetBrains Mono", monospace';
                ctx.fillStyle = c.colour;
                ctx.fillText(`= ${yv.toFixed(2)}`, lx + 18, ly + 32);
            }
            ly += hoverX !== null ? 46 : 34;
        }

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (mode === 'regression') {
            captionEl.innerHTML =
                `<strong>Regression losses.</strong> At residual 2, MSE pays 4× the price of residual 1; MAE only pays 2×. ` +
                `That's why MSE chases outliers and MAE shrugs at them. Huber's δ knob controls where the switch happens — ` +
                `small δ behaves like MAE everywhere; large δ behaves like MSE everywhere.`;
        } else {
            captionEl.innerHTML =
                `<strong>Classification losses.</strong> All four agree the answer is "be on the positive-margin side". ` +
                `But log-loss keeps gently pulling even at margin 5 (the model can always be more confident); hinge stops ` +
                `caring past margin 1; exponential punishes negative margins extremely. The 0/1 curve is what we actually ` +
                `want, but its zero gradient makes it un-trainable by SGD — every other curve is a smooth surrogate for it.`;
        }
    }

    // ----- Interactions -----
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        hoverX = fromPxX(e.clientX - rect.left);
        draw();
    });
    canvas.addEventListener('mouseleave', () => { hoverX = null; draw(); });

    regBtn?.addEventListener('click', () => {
        mode = 'regression';
        regBtn.classList.add('active'); clsBtn.classList.remove('active');
        draw();
    });
    clsBtn?.addEventListener('click', () => {
        mode = 'classification';
        clsBtn.classList.add('active'); regBtn.classList.remove('active');
        draw();
    });
    if (dSlider) {
        dSlider.min = 0.1; dSlider.max = 1.5; dSlider.step = 0.05; dSlider.value = 0.5;
        dSlider.addEventListener('input', () => {
            delta = parseFloat(dSlider.value);
            if (dLbl) dLbl.textContent = `δ = ${delta.toFixed(2)}`;
            draw();
        });
    }

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
