/* Interactive ICA cocktail-party viz.
 * Two ground-truth sources (sine + sawtooth). Two linear mixtures (microphones).
 * Toggle between source / mixed / PCA-recovery / ICA-recovery views to see
 * which method actually unscrambles the signals. */

(function () {
    const canvas    = document.getElementById('viz-ica-canvas');
    const sourcesBtn = document.getElementById('viz-ica-sources');
    const mixedBtn   = document.getElementById('viz-ica-mixed');
    const pcaBtn     = document.getElementById('viz-ica-pca');
    const icaBtn     = document.getElementById('viz-ica-ica');
    const captionEl = document.getElementById('viz-ica-caption');
    if (!canvas) return;

    let mode = 'sources';
    let ctx;
    let W = 0, H = 0;
    const N = 400;

    // Generate ground-truth sources
    function buildSources() {
        const s1 = new Float64Array(N), s2 = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            const t = i / N * 8;
            s1[i] = Math.sin(2 * t);
            // Sawtooth
            const x = (3 * t) % (2 * Math.PI);
            s2[i] = (x / Math.PI - 1);
        }
        return [s1, s2];
    }
    // Mix linearly: x = A * s
    function mix(s1, s2) {
        const a11 = 1.0, a12 = 0.7, a21 = 0.5, a22 = 1.4;
        const x1 = new Float64Array(N), x2 = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            x1[i] = a11 * s1[i] + a12 * s2[i];
            x2[i] = a21 * s1[i] + a22 * s2[i];
        }
        return [x1, x2];
    }
    function centre(a) {
        let m = 0;
        for (const v of a) m += v;
        m /= a.length;
        return a.map(v => v - m);
    }
    function dot(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s / a.length;
    }
    // PCA via 2×2 eigendecomposition
    function pcaRecover(x1, x2) {
        const cxx = dot(x1, x1), cxy = dot(x1, x2), cyy = dot(x2, x2);
        const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        // Eigenvectors
        let ang;
        if (Math.abs(cxy) > 1e-9) ang = Math.atan2(l1 - cxx, cxy);
        else                       ang = (cxx >= cyy) ? 0 : Math.PI / 2;
        const c = Math.cos(ang), s = Math.sin(ang);
        const p1 = new Float64Array(N), p2 = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            p1[i] = ( c * x1[i] + s * x2[i]) / Math.sqrt(l1 || 1);
            p2[i] = (-s * x1[i] + c * x2[i]) / Math.sqrt(l2 || 1);
        }
        return [p1, p2];
    }
    // FastICA on 2 components — whiten then rotate to maximise kurtosis
    function icaRecover(x1, x2) {
        // Whiten using PCA
        const [w1, w2] = pcaRecover(x1, x2);
        // Search rotation θ that maximises sum of |kurtosis| of [c·w1 + s·w2, ...]
        function kurt(arr) {
            let m2 = 0, m4 = 0;
            for (const v of arr) { m2 += v * v; m4 += v * v * v * v; }
            m2 /= arr.length; m4 /= arr.length;
            return m4 / (m2 * m2) - 3;
        }
        let bestAng = 0, bestScore = -Infinity;
        for (let i = 0; i < 360; i++) {
            const ang = (i / 360) * Math.PI;
            const c = Math.cos(ang), s = Math.sin(ang);
            const r1 = w1.map((v, j) => c * v + s * w2[j]);
            const r2 = w1.map((v, j) => -s * v + c * w2[j]);
            const score = Math.abs(kurt(r1)) + Math.abs(kurt(r2));
            if (score > bestScore) { bestScore = score; bestAng = ang; }
        }
        const c = Math.cos(bestAng), s = Math.sin(bestAng);
        const r1 = new Float64Array(N), r2 = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            r1[i] =  c * w1[i] + s * w2[i];
            r2[i] = -s * w1[i] + c * w2[i];
        }
        return [r1, r2];
    }

    // Pre-compute all
    let [s1, s2] = buildSources();
    let [x1, x2] = mix(s1, s2);
    x1 = centre(x1); x2 = centre(x2);
    let [p1, p2] = pcaRecover(x1, x2);
    let [i1, i2] = icaRecover(x1, x2);

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(320, cssW * 0.46)));
        W = cssW; H = cssH;
        canvas.style.height = cssH + 'px';
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
    }

    function drawSignal(arr, box, col, label) {
        const pad = 4;
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // Y range
        let mx = 0;
        for (const v of arr) if (Math.abs(v) > mx) mx = Math.abs(v);
        if (mx < 1e-9) mx = 1;
        const midY = box.y + box.h / 2;
        // Zero line
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.beginPath(); ctx.moveTo(box.x + pad, midY); ctx.lineTo(box.x + box.w - pad, midY); ctx.stroke();

        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const x = box.x + pad + (i / (arr.length - 1)) * (box.w - 2 * pad);
            const y = midY - (arr[i] / mx) * (box.h / 2 - pad - 2);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, box.x + 4, box.y - 6);
    }

    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const pad = 14;
        const labelGap = 18;
        const sigH = (H - 3 * pad - 2 * labelGap) / 2;
        const top = { x: pad, y: pad + labelGap, w: W - 2 * pad, h: sigH };
        const bot = { x: pad, y: pad + labelGap + sigH + labelGap + pad, w: W - 2 * pad, h: sigH };

        let a, b, labelA, labelB, colA, colB;
        if (mode === 'sources') {
            a = s1; b = s2; labelA = 'SOURCE 1 (sine)'; labelB = 'SOURCE 2 (sawtooth)';
            colA = '#4f46e5'; colB = '#ea7959';
        } else if (mode === 'mixed') {
            a = x1; b = x2; labelA = 'MIXTURE 1 (microphone A)'; labelB = 'MIXTURE 2 (microphone B)';
            colA = '#1a1a1a'; colB = '#1a1a1a';
        } else if (mode === 'pca') {
            a = p1; b = p2; labelA = 'PCA COMPONENT 1'; labelB = 'PCA COMPONENT 2';
            colA = '#10847e'; colB = '#10847e';
        } else if (mode === 'ica') {
            a = i1; b = i2; labelA = 'ICA COMPONENT 1'; labelB = 'ICA COMPONENT 2';
            colA = '#4f46e5'; colB = '#ea7959';
        }

        drawSignal(a, top, colA, labelA);
        drawSignal(b, bot, colB, labelB);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        const notes = {
            sources: `<strong>Ground truth.</strong> Two statistically-independent sources — a sine wave and a sawtooth. The "speakers" we want to recover, in their original form.`,
            mixed:   `<strong>Observed mixtures.</strong> Each microphone hears a different linear combination of both sources. Neither signal looks like either source.`,
            pca:     `<strong>PCA recovery.</strong> PCA decorrelates the mixtures. The resulting components are orthogonal but still look like mixtures — PCA only finds <em>uncorrelated</em> directions, not <em>independent</em> ones.`,
            ica:     `<strong>ICA recovery.</strong> ICA recovers the original sine and sawtooth (up to sign and scale). It searches for the rotation of the whitened data that maximises non-Gaussianity — independence reveals itself in higher-order statistics.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [sourcesBtn, mixedBtn, pcaBtn, icaBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    sourcesBtn?.addEventListener('click', () => setMode('sources', sourcesBtn));
    mixedBtn?.addEventListener('click',   () => setMode('mixed',   mixedBtn));
    pcaBtn?.addEventListener('click',     () => setMode('pca',     pcaBtn));
    icaBtn?.addEventListener('click',     () => setMode('ica',     icaBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
