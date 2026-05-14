/* Interactive transfer-learning viz.
 * Three accuracy-vs-#labels curves on a synthetic learning-curves problem:
 *   from scratch     — random init, gradient descent on labels alone
 *   linear probe     — frozen embedding, train only linear head
 *   fine-tune        — start from pre-trained, update everything
 * Each curve is a smooth fit to canonical empirical shapes:
 *   from scratch     ≈ a / (a + b/n)   with low ceiling at small n
 *   linear probe     ≈ tops out at moderate ceiling; quick to plateau
 *   fine-tune        ≈ best at all but the very largest n  */

(function () {
    const canvas    = document.getElementById('viz-tl-canvas');
    const stepBtn   = document.getElementById('viz-tl-step');
    const nSlider   = document.getElementById('viz-tl-n');
    const nLbl      = document.getElementById('viz-tl-n-lbl');
    const captionEl = document.getElementById('viz-tl-caption');
    if (!canvas) return;

    let nLabels = 20;
    let seed = 0;

    // Curves (acc as a function of log10(n))
    function scratchAcc(n) {
        // From scratch — needs lots of data
        return 0.5 + 0.45 * (1 - Math.exp(-n / 800));
    }
    function probeAcc(n) {
        // Linear probe — strong floor (pretrained features), modest ceiling
        return 0.78 + 0.13 * (1 - Math.exp(-n / 200));
    }
    function fineTuneAcc(n) {
        // Fine-tune — high at all data sizes, edge over probe with more data
        return 0.80 + 0.18 * (1 - Math.exp(-n / 300));
    }
    function jitter(seed_v) {
        // Deterministic noise per seed
        const t = Math.sin(seed_v * 12.9898 + 78.233) * 43758.5453;
        return (t - Math.floor(t)) * 2 - 1;
    }

    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(340, cssW * 0.50)));
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

        const pad = 50;
        const box = { x: pad, y: 30, w: W - pad - 200, h: H - 80 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // X axis is log10(n) from 1 (10) to 4 (10000)
        function xToPx(logN) {
            return box.x + (logN - 1) / 3 * box.w;
        }
        function yToPx(acc) {
            return box.y + box.h - (acc - 0.4) / 0.6 * box.h;
        }

        // Gridlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.07)';
        ctx.setLineDash([2, 3]);
        for (const v of [10, 30, 100, 300, 1000, 3000, 10000]) {
            const x = xToPx(Math.log10(v));
            ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
        }
        for (const v of [0.5, 0.7, 0.9]) {
            const y = yToPx(v);
            ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
        }
        ctx.setLineDash([]);

        // Tick labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const v of [10, 100, 1000, 10000]) {
            ctx.fillText(v, xToPx(Math.log10(v)), box.y + box.h + 12);
        }
        ctx.textAlign = 'right';
        for (const v of [0.5, 0.7, 0.9]) {
            ctx.fillText(v.toFixed(1), box.x - 4, yToPx(v) + 3);
        }
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('# labelled training examples', box.x + box.w / 2, box.y + box.h + 26);
        ctx.save();
        ctx.translate(box.x - 32, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('test accuracy', 0, 0);
        ctx.restore();

        // Curves with noise scaled to "how many runs averaged" intuition
        const curves = [
            { name: 'fine-tune',    fn: fineTuneAcc, col: '#10847e' },
            { name: 'linear probe', fn: probeAcc,    col: '#ea7959' },
            { name: 'from scratch', fn: scratchAcc,  col: '#4f46e5' },
        ];
        const ns = [];
        for (let lN = 1; lN <= 4; lN += 0.05) ns.push(Math.round(10 ** lN));
        for (const c of curves) {
            ctx.strokeStyle = c.col;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (let i = 0; i < ns.length; i++) {
                const acc = c.fn(ns[i]);
                const [px, py] = [xToPx(Math.log10(ns[i])), yToPx(acc)];
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
            // Point at current n
            const acc = c.fn(nLabels) + jitter(seed + nLabels) * 0.015;
            const [px, py] = [xToPx(Math.log10(nLabels)), yToPx(acc)];
            ctx.fillStyle = c.col;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
            c._acc = acc;
        }

        // Vertical line at current n
        const nx = xToPx(Math.log10(nLabels));
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(nx, box.y); ctx.lineTo(nx, box.y + box.h); ctx.stroke();
        ctx.setLineDash([]);

        // Legend / values
        let lx = box.x + box.w + 20, ly = box.y + 16;
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'left';
        ctx.fillText(`n = ${nLabels}`, lx, ly);
        ly += 18;
        for (const c of curves) {
            ctx.fillStyle = c.col;
            ctx.fillRect(lx, ly - 8, 12, 12);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.font = '600 10px "Inter", system-ui, sans-serif';
            ctx.fillText(c.name, lx + 18, ly);
            ctx.font = '600 14px "JetBrains Mono", monospace';
            ctx.fillStyle = c.col;
            ctx.fillText(`${(c._acc * 100).toFixed(1)}%`, lx + 18, ly + 18);
            ly += 38;
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('LEARNING CURVES — accuracy vs labels', box.x, box.y - 8);

        if (nLbl) nLbl.textContent = `n = ${nLabels}`;
        updateCaption(curves);
    }

    function updateCaption(curves) {
        if (!captionEl) return;
        const ft = curves[0]._acc, lp = curves[1]._acc, sc = curves[2]._acc;
        const gap = ((ft - sc) * 100).toFixed(0);
        if (nLabels < 50) {
            captionEl.innerHTML =
                `<strong>n = ${nLabels} labels.</strong> Fine-tune wins by <strong>${gap} points</strong> over from-scratch. ` +
                `At this scale you simply cannot train a useful model from random init — but a pre-trained encoder gives you a flying start.`;
        } else if (nLabels < 1000) {
            captionEl.innerHTML =
                `<strong>n = ${nLabels} labels.</strong> The gap is shrinking (still ${gap} points). Fine-tune is still the best ` +
                `choice; linear probe and fine-tune are converging. From-scratch is still well behind.`;
        } else {
            captionEl.innerHTML =
                `<strong>n = ${nLabels} labels.</strong> With enough data, from-scratch catches up. Fine-tune still has a small edge ` +
                `(${gap} points) but the gap closes as labels grow. Linear probe is now slightly behind fine-tune — it can't adapt features.`;
        }
    }

    stepBtn?.addEventListener('click', () => { seed = Math.random() * 1000; draw(); });
    if (nSlider) {
        nSlider.min = 5; nSlider.max = 10000; nSlider.step = 1; nSlider.value = 20;
        nSlider.addEventListener('input', () => {
            nLabels = parseInt(nSlider.value, 10);
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
