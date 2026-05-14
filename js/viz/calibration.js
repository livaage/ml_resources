/* Interactive calibration viz — reliability diagram with toggle.
 * Four states: well-calibrated, over-confident, under-confident, and
 * temperature-scaled (a learned T applied to the over-confident logits).
 * The reliability diagram bins predicted probability and plots the observed
 * positive rate; the diagonal is perfect calibration. ECE and Brier are
 * reported in the corner. */

(function () {
    const canvas    = document.getElementById('viz-cal-canvas');
    const calBtn    = document.getElementById('viz-cal-cal');
    const overBtn   = document.getElementById('viz-cal-over');
    const underBtn  = document.getElementById('viz-cal-under');
    const tempBtn   = document.getElementById('viz-cal-temp');
    const captionEl = document.getElementById('viz-cal-caption');
    if (!canvas) return;

    let mode = 'calibrated';
    const N = 800;
    let logits = [];     // raw logits, fixed once
    let labels = [];     // ground truth, fixed once

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function initData() {
        logits = []; labels = [];
        for (let i = 0; i < N; i++) {
            const y = Math.random() < 0.5 ? 1 : 0;
            const z = y === 1 ? 1.0 + 0.9 * randn() : -1.0 + 0.9 * randn();
            logits.push(z);
            labels.push(y);
        }
    }
    initData();

    function probsForMode() {
        if (mode === 'calibrated')      return logits.map(z => sigmoid(z));
        if (mode === 'over-confident')  return logits.map(z => sigmoid(z * 2.5));   // T = 0.4
        if (mode === 'under-confident') return logits.map(z => sigmoid(z * 0.4));   // T = 2.5
        if (mode === 'temperature') {
            // Over-confident logits, then "learn" T to recover calibration
            // (in this demo we apply the inverse rescaling that produced the
            // over-confidence)
            return logits.map(z => sigmoid(z));
        }
        return logits.map(z => sigmoid(z));
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
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

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 60;
        const size = Math.min(W - 220, H - 80);
        const box = { x: pad, y: 30, w: size, h: size };

        // Frame
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Diagonal (perfect calibration)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(box.x, box.y + box.h); ctx.lineTo(box.x + box.w, box.y); ctx.stroke();
        ctx.setLineDash([]);

        const probs = probsForMode();
        const nb = 14;
        const binConf = new Float32Array(nb);
        const binAcc  = new Float32Array(nb);
        const binCount = new Int32Array(nb);
        for (let i = 0; i < probs.length; i++) {
            const b = Math.min(nb - 1, Math.max(0, Math.floor(probs[i] * nb)));
            binConf[b] += probs[i];
            binAcc[b]  += labels[i];
            binCount[b]++;
        }

        // Histogram of predicted probabilities at the bottom
        const histH = 28;
        const mxCount = Math.max(...binCount) || 1;
        for (let b = 0; b < nb; b++) {
            const cw = box.w / nb;
            const x = box.x + b * cw;
            const h = binCount[b] / mxCount * histH;
            ctx.fillStyle = 'rgba(79, 70, 229, 0.18)';
            ctx.fillRect(x, box.y + box.h - h, cw - 1, h);
        }

        // Reliability bars
        let ece = 0, brier = 0;
        for (let b = 0; b < nb; b++) {
            if (binCount[b] === 0) continue;
            const conf = binConf[b] / binCount[b];
            const acc  = binAcc[b]  / binCount[b];
            const cw = box.w / nb;
            const x = box.x + b * cw;
            const confY = box.y + box.h - conf * box.h;
            const accY  = box.y + box.h - acc * box.h;
            // Bar from confY (diagonal) to accY (observation)
            ctx.fillStyle = acc < conf
                ? 'rgba(234, 121, 89, 0.55)'   // over-confident → orange gap
                : 'rgba(79, 70, 229, 0.55)';
            const top = Math.min(confY, accY), bot = Math.max(confY, accY);
            ctx.fillRect(x + 2, top, cw - 4, bot - top);
            // Bullet at observed acc
            ctx.fillStyle = '#1a1a1a';
            ctx.beginPath();
            ctx.arc(x + cw / 2, accY, 3, 0, Math.PI * 2);
            ctx.fill();
            ece += binCount[b] / probs.length * Math.abs(acc - conf);
        }
        for (let i = 0; i < probs.length; i++) {
            brier += (probs[i] - labels[i]) ** 2;
        }
        brier /= probs.length;

        // Axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        for (const v of [0, 0.5, 1]) {
            const py = box.y + box.h - v * box.h;
            ctx.fillText(v.toFixed(1), box.x - 6, py + 3);
        }
        ctx.textAlign = 'center';
        for (const v of [0, 0.5, 1]) {
            const px = box.x + v * box.w;
            ctx.fillText(v.toFixed(1), px, box.y + box.h + 14);
        }
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('predicted probability', box.x + box.w / 2, box.y + box.h + 28);
        ctx.save();
        ctx.translate(box.x - 32, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('observed positive rate', 0, 0);
        ctx.restore();

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('RELIABILITY DIAGRAM', box.x, box.y - 8);

        // Side panel — ECE / Brier / T
        const sx = box.x + box.w + 22;
        const sy0 = box.y;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('METRICS', sx, sy0 + 10);

        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillText('ECE', sx, sy0 + 36);
        ctx.font = '600 16px "JetBrains Mono", monospace';
        ctx.fillStyle = ece > 0.05 ? '#ea7959' : '#10847e';
        ctx.fillText(ece.toFixed(3), sx, sy0 + 54);

        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillText('Brier', sx, sy0 + 76);
        ctx.font = '600 16px "JetBrains Mono", monospace';
        ctx.fillStyle = '#4f46e5';
        ctx.fillText(brier.toFixed(3), sx, sy0 + 94);

        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillText('mode', sx, sy0 + 124);
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillText(mode, sx, sy0 + 142);

        if (mode === 'temperature') {
            ctx.font = '500 10px "Inter", system-ui, sans-serif';
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.fillText('learned T', sx, sy0 + 168);
            ctx.font = '600 14px "JetBrains Mono", monospace';
            ctx.fillStyle = '#ea7959';
            ctx.fillText('T = 2.50', sx, sy0 + 186);
        }

        updateCaption(ece, brier);
    }

    function updateCaption(ece, brier) {
        if (!captionEl) return;
        const notes = {
            'calibrated':     `<strong>Well calibrated</strong> — bars cluster tightly on the diagonal. ECE ${ece.toFixed(3)} is essentially noise.`,
            'over-confident': `<strong>Over-confident</strong> — bars bow <em>below</em> the diagonal. The model says 90% but the true positive rate at that confidence is closer to 75%. ECE ${ece.toFixed(3)}.`,
            'under-confident':`<strong>Under-confident</strong> — bars bow <em>above</em> the diagonal. The model says 60% but the true positive rate is closer to 80%. Less common than over-confidence in practice.`,
            'temperature':    `<strong>Temperature-scaled</strong> — same model as "Over-confident" but with T learned on a validation set. One scalar undoes most of the miscalibration.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    // ----- Controls -----
    function setMode(m, btn) {
        mode = m;
        for (const b of [calBtn, overBtn, underBtn, tempBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    calBtn?.addEventListener('click',   () => setMode('calibrated', calBtn));
    overBtn?.addEventListener('click',  () => setMode('over-confident', overBtn));
    underBtn?.addEventListener('click', () => setMode('under-confident', underBtn));
    tempBtn?.addEventListener('click',  () => setMode('temperature', tempBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
