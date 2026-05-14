/* Interactive cross-validation viz.
 * Each row is one fold of training. Indigo cells = train, terracotta = validate.
 * Four modes:
 *   K-Fold       — standard, contiguous chunks
 *   Stratified   — same but preserving class colour bands
 *   Time-Series  — train on past, validate on next chunk
 *   LOO          — leave-one-out, K = N */

(function () {
    const canvas    = document.getElementById('viz-cv-canvas');
    const kfoldBtn  = document.getElementById('viz-cv-kfold');
    const stratBtn  = document.getElementById('viz-cv-strat');
    const tssBtn    = document.getElementById('viz-cv-tss');
    const loocvBtn  = document.getElementById('viz-cv-loocv');
    const kSlider   = document.getElementById('viz-cv-k');
    const kLbl      = document.getElementById('viz-cv-k-lbl');
    const captionEl = document.getElementById('viz-cv-caption');
    if (!canvas) return;

    let mode = 'kfold';
    let K = 5;
    const N = 30;
    // Class labels for stratified mode — pretend classes 0/1 in a 2:1 ratio
    const classes = Array.from({length: N}, (_, i) => i % 3 === 2 ? 1 : 0);

    function folds() {
        const list = [];
        if (mode === 'loocv') {
            for (let v = 0; v < N; v++) {
                list.push({ train: Array.from({length: N}, (_, i) => i !== v),
                            valid: Array.from({length: N}, (_, i) => i === v) });
            }
            return list;
        }
        if (mode === 'tss') {
            // Increasing-window time-series: train on [0..t), validate on [t..t+chunk)
            const chunk = Math.max(2, Math.floor(N / (K + 1)));
            for (let k = 0; k < K; k++) {
                const trainEnd = chunk * (k + 1);
                const validEnd = Math.min(N, trainEnd + chunk);
                const tr = Array(N).fill(false), va = Array(N).fill(false);
                for (let i = 0; i < trainEnd; i++) tr[i] = true;
                for (let i = trainEnd; i < validEnd; i++) va[i] = true;
                list.push({ train: tr, valid: va });
            }
            return list;
        }
        if (mode === 'stratified') {
            // Pseudo-stratify: round-robin through classes, then chunk into K
            const order = [];
            for (let c = 0; c <= 1; c++) {
                for (let i = 0; i < N; i++) if (classes[i] === c) order.push(i);
            }
            // Now order has all 0s then all 1s; interleave so each chunk hits both
            const interleaved = [];
            const n0 = order.filter(i => classes[i] === 0).length;
            for (let i = 0; i < N; i++) {
                if (i % 3 === 2) interleaved.push(order[n0 + Math.floor(i / 3)]);
                else interleaved.push(order[Math.floor(i / 3) * 2 + (i % 3)]);
            }
            // Build chunks
            for (let k = 0; k < K; k++) {
                const tr = Array(N).fill(false), va = Array(N).fill(false);
                for (let i = 0; i < N; i++) {
                    const pos = interleaved[i];
                    const chunk = Math.floor(i * K / N);
                    if (chunk === k) va[pos] = true; else tr[pos] = true;
                }
                list.push({ train: tr, valid: va });
            }
            return list;
        }
        // Plain k-fold
        const chunk = N / K;
        for (let k = 0; k < K; k++) {
            const lo = Math.floor(k * chunk), hi = Math.floor((k + 1) * chunk);
            const tr = Array(N).fill(true), va = Array(N).fill(false);
            for (let i = lo; i < hi; i++) { tr[i] = false; va[i] = true; }
            list.push({ train: tr, valid: va });
        }
        return list;
    }

    // ----- Canvas -----
    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(280, cssW * 0.42)));
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

        const f = folds();
        const rows = f.length;
        const pad = 14;
        const leftLabel = 70;
        const topLabel = 40;
        const cellW = (W - pad * 2 - leftLabel) / N;
        const maxRows = Math.min(rows, mode === 'loocv' ? 15 : rows);
        const availH = H - pad - topLabel - 28;
        const cellH = Math.max(8, Math.min(22, availH / maxRows));

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = {
            kfold:      `${K}-FOLD CV — each row is one training run`,
            stratified: `STRATIFIED ${K}-FOLD — preserves class proportions per fold`,
            tss:        `TIME-SERIES CV — validation always after train in time`,
            loocv:      `LEAVE-ONE-OUT (showing first ${maxRows} of ${rows})`,
        };
        ctx.fillText(titles[mode], pad + leftLabel, pad + 14);

        // Column header: stratified shows class bands
        if (mode === 'stratified') {
            for (let i = 0; i < N; i++) {
                ctx.fillStyle = classes[i] === 0 ? 'rgba(79, 70, 229, 0.15)'
                                                 : 'rgba(234, 121, 89, 0.15)';
                ctx.fillRect(pad + leftLabel + i * cellW, pad + topLabel - 10,
                             cellW - 1, 6);
            }
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('class', pad + leftLabel - 4, pad + topLabel - 4);
        }

        // Rows
        for (let r = 0; r < Math.min(rows, maxRows); r++) {
            const fold = f[r];
            const y = pad + topLabel + r * cellH;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
            ctx.font = '500 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`fold ${r + 1}`, pad + leftLabel - 6, y + cellH * 0.7);

            for (let i = 0; i < N; i++) {
                const x = pad + leftLabel + i * cellW;
                let colour;
                if (fold.valid[i]) colour = '#ea7959';
                else if (fold.train[i]) colour = '#4f46e5';
                else colour = 'rgba(0, 0, 0, 0.08)';
                ctx.fillStyle = colour;
                ctx.fillRect(x, y, cellW - 1, cellH - 1);
            }
        }
        // ... last row indicator if truncated
        if (rows > maxRows) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.font = '500 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`+ ${rows - maxRows} more`, pad + leftLabel - 6,
                         pad + topLabel + maxRows * cellH + 12);
        }

        // Legend
        const ly = H - pad - 6;
        ctx.fillStyle = '#4f46e5';
        ctx.fillRect(pad + leftLabel, ly - 10, 12, 12);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('train', pad + leftLabel + 18, ly);
        ctx.fillStyle = '#ea7959';
        ctx.fillRect(pad + leftLabel + 60, ly - 10, 12, 12);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillText('validate', pad + leftLabel + 78, ly);

        if (kLbl) {
            kLbl.textContent = mode === 'loocv' ? `K = ${rows}` : `K = ${K}`;
        }
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            kfold:      `Standard k-fold. ${K} chunks, ${K} training runs. Each example serves as validation in exactly one run. Bigger K → less validation data per fold, more runs, lower-variance estimate.`,
            stratified: `Stratified ${K}-fold. Class proportions are preserved in each fold — critical for imbalanced data, otherwise a fold might get zero examples from a rare class.`,
            tss:        `Time-series CV. Validation always comes after the training window. Never shuffle — the model would peek at the future and CV would lie about generalization.`,
            loocv:      `Leave-one-out. Every example is validated by a model trained on all the rest. Lowest bias, highest variance estimate. Expensive — N models.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    // ----- Controls -----
    function setMode(m, btn) {
        mode = m;
        for (const b of [kfoldBtn, stratBtn, tssBtn, loocvBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    kfoldBtn?.addEventListener('click', () => setMode('kfold', kfoldBtn));
    stratBtn?.addEventListener('click', () => setMode('stratified', stratBtn));
    tssBtn?.addEventListener('click',   () => setMode('tss', tssBtn));
    loocvBtn?.addEventListener('click', () => setMode('loocv', loocvBtn));
    if (kSlider) {
        kSlider.min = 2; kSlider.max = 10; kSlider.step = 1; kSlider.value = 5;
        kSlider.addEventListener('input', () => {
            K = parseInt(kSlider.value, 10);
            if (kLbl) kLbl.textContent = `K = ${K}`;
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
