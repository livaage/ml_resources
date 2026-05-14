/* Interactive data-leakage viz.
 * Side-by-side comparison of two pipelines on the same data:
 *   "naive"   — StandardScaler.fit_transform(X) then split → leakage
 *   "correct" — split first, scaler fit only on train
 * Report both the val-set accuracy (the headline number) and the
 * "deployment-true" accuracy (apply a model trained the naive way to genuinely
 * fresh data) — they differ. */

(function () {
    const canvas    = document.getElementById('viz-leak-canvas');
    const wrongBtn  = document.getElementById('viz-leak-wrong');
    const rightBtn  = document.getElementById('viz-leak-right');
    const captionEl = document.getElementById('viz-leak-caption');
    if (!canvas) return;

    let mode = 'wrong';
    let trainData = [], valData = [], freshData = [];

    function randn() {
        const u1 = Math.max(Math.random(), 1e-9), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    (function init() {
        // The "true" model is f(x) > 0 → class 1, with x ~ N(c*0.6, 0.5)
        function gen(n) {
            const out = [];
            for (let i = 0; i < n; i++) {
                const c = Math.random() < 0.5 ? 0 : 1;
                const x = (c === 0 ? -0.6 : 0.6) + randn() * 0.5;
                const y = (c === 0 ? -0.3 : 0.3) + randn() * 0.5;
                out.push({ x, y, c });
            }
            return out;
        }
        trainData = gen(60);
        valData   = gen(60);
        freshData = gen(200);   // hypothetical production data
    })();

    function mean(arr, fn) { return arr.reduce((s, p) => s + fn(p), 0) / arr.length; }
    function std(arr, fn, m) {
        const v = arr.reduce((s, p) => s + (fn(p) - m) ** 2, 0) / arr.length;
        return Math.sqrt(v) || 1;
    }

    function pipeline(modeName) {
        // Combined data the "wrong" pipeline can see
        const combined = [...trainData, ...valData];
        let mx, sx, my, sy;
        if (modeName === 'wrong') {
            mx = mean(combined, p => p.x);
            sx = std (combined, p => p.x, mx);
            my = mean(combined, p => p.y);
            sy = std (combined, p => p.y, my);
        } else {
            mx = mean(trainData, p => p.x);
            sx = std (trainData, p => p.x, mx);
            my = mean(trainData, p => p.y);
            sy = std (trainData, p => p.y, my);
        }
        const scale = (p) => ({ x: (p.x - mx) / sx, y: (p.y - my) / sy, c: p.c });
        const trS = trainData.map(scale);
        const vaS = valData.map(scale);
        const frS = freshData.map(scale);
        // Fit LR on trS
        let w = [0, 0], b = 0;
        for (let it = 0; it < 250; it++) {
            const lr = 0.4 / (1 + it / 80);
            let gw0 = 0, gw1 = 0, gb = 0;
            for (const p of trS) {
                const z = w[0] * p.x + w[1] * p.y + b;
                const s = 1 / (1 + Math.exp(-z));
                const e = s - p.c;
                gw0 += e * p.x; gw1 += e * p.y; gb += e;
            }
            w[0] -= lr * gw0 / trS.length;
            w[1] -= lr * gw1 / trS.length;
            b    -= lr * gb / trS.length;
        }
        const predict = (p) => (w[0] * p.x + w[1] * p.y + b > 0) ? 1 : 0;
        function acc(arr) {
            let correct = 0;
            for (const p of arr) if (predict(p) === p.c) correct++;
            return correct / arr.length;
        }
        return { trainAcc: acc(trS), valAcc: acc(vaS), freshAcc: acc(frS),
                 model: { w, b, mx, sx, my, sy }, trS, vaS };
    }

    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(420, Math.max(320, cssW * 0.42)));
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

        const r = pipeline(mode);

        // Big number panel
        const pad = 20;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(mode === 'wrong' ? 'NAIVE PIPELINE — scaler fit on the entire dataset'
                                       : 'CORRECT PIPELINE — scaler fit on train only',
                     pad, pad + 4);

        const colW = (W - 2 * pad) / 3;
        const cy = H / 2 - 16;
        const items = [
            ['TRAIN', r.trainAcc, '#4f46e5', 'fit on this'],
            ['VAL',   r.valAcc,   '#ea7959', 'the headline number'],
            ['PRODUCTION (truly unseen)', r.freshAcc, '#10847e', 'what users experience'],
        ];
        for (let i = 0; i < 3; i++) {
            const [lbl, v, col, sub] = items[i];
            const x = pad + i * colW;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.font = '500 10px "Inter", system-ui, sans-serif';
            ctx.fillText(lbl, x + 20, cy - 30);
            ctx.fillStyle = col;
            ctx.font = '700 30px "JetBrains Mono", monospace';
            ctx.fillText(`${(v * 100).toFixed(1)}%`, x + 20, cy + 10);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.font = '500 10px "Inter", system-ui, sans-serif';
            ctx.fillText(sub, x + 20, cy + 28);
        }

        // Gap callout
        const gap = (r.valAcc - r.freshAcc) * 100;
        ctx.fillStyle = mode === 'wrong' ? '#ea7959' : 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
            mode === 'wrong'
                ? `↘ val − production = ${gap > 0 ? '+' : ''}${gap.toFixed(1)} percentage points  (the leak)`
                : `val − production = ${gap > 0 ? '+' : ''}${gap.toFixed(1)} percentage points  (within noise)`,
            W / 2, H - 24);

        updateCaption(r);
    }

    function updateCaption(r) {
        if (!captionEl) return;
        const gap = (r.valAcc - r.freshAcc) * 100;
        if (mode === 'wrong') {
            captionEl.innerHTML =
                `<strong>Naive pipeline.</strong> The scaler saw the validation rows when fitting — so the model's standardised inputs ` +
                `have val-set means baked in. Val accuracy reports ${(r.valAcc * 100).toFixed(1)}%, but truly fresh data ` +
                `only reaches ${(r.freshAcc * 100).toFixed(1)}%. The ${gap.toFixed(1)}-point gap is fictional performance.`;
        } else {
            captionEl.innerHTML =
                `<strong>Correct pipeline.</strong> The scaler only saw the training rows. Val and production accuracy are now ` +
                `essentially the same — what you measure on val is what you get in production.`;
        }
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [wrongBtn, rightBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    wrongBtn?.addEventListener('click', () => setMode('wrong', wrongBtn));
    rightBtn?.addEventListener('click', () => setMode('right', rightBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
