/* Interactive HMM viz.
 * A 3-state weather model emits one of three activities each timestep.
 * The user sees the activity sequence and the inferred hidden-state posterior
 * (forward-backward) plus the Viterbi most-likely path.
 *
 * Click any observation to cycle it through Walk / Shop / Clean and watch
 * the inferred states (which we never see directly) re-shuffle to explain
 * the new evidence.
 *
 * STATES: 0 = Sunny, 1 = Cloudy, 2 = Rainy
 * OBSERVATIONS: 0 = Walk, 1 = Shop, 2 = Clean */

(function () {
    const canvas    = document.getElementById('viz-hmm-canvas');
    const presetSel = document.getElementById('viz-hmm-preset');
    const resetBtn  = document.getElementById('viz-hmm-reset');
    const captionEl = document.getElementById('viz-hmm-caption');
    if (!canvas) return;

    const STATES = ['Sunny', 'Cloudy', 'Rainy'];
    const OBS    = ['Walk', 'Shop', 'Clean'];
    const OBS_ICONS = ['🚶', '🛍️', '🧹'];

    // Default model parameters
    const PI = [0.5, 0.3, 0.2];
    let A = [
        [0.70, 0.20, 0.10],  // Sunny → ...
        [0.30, 0.45, 0.25],  // Cloudy → ...
        [0.10, 0.35, 0.55],  // Rainy → ...
    ];
    let B = [
        [0.60, 0.30, 0.10],  // Sunny: Walk likely
        [0.25, 0.45, 0.30],  // Cloudy: Shop likely
        [0.10, 0.35, 0.55],  // Rainy: Clean likely
    ];

    let obs = [0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2];
    let posterior = null;
    let viterbi   = null;

    // ----- Forward-backward -----
    function forwardBackward(obs) {
        const T = obs.length, K = 3;
        const alpha = Array.from({length: T}, () => new Float64Array(K));
        const scale = new Float64Array(T);
        // Forward
        for (let s = 0; s < K; s++) alpha[0][s] = PI[s] * B[s][obs[0]];
        scale[0] = alpha[0].reduce((a, b) => a + b, 0);
        for (let s = 0; s < K; s++) alpha[0][s] /= scale[0];
        for (let t = 1; t < T; t++) {
            for (let s = 0; s < K; s++) {
                let sum = 0;
                for (let p = 0; p < K; p++) sum += alpha[t - 1][p] * A[p][s];
                alpha[t][s] = sum * B[s][obs[t]];
            }
            scale[t] = alpha[t].reduce((a, b) => a + b, 0);
            for (let s = 0; s < K; s++) alpha[t][s] /= scale[t];
        }
        // Backward
        const beta = Array.from({length: T}, () => new Float64Array(K));
        for (let s = 0; s < K; s++) beta[T - 1][s] = 1;
        for (let t = T - 2; t >= 0; t--) {
            for (let s = 0; s < K; s++) {
                let sum = 0;
                for (let n = 0; n < K; n++) sum += A[s][n] * B[n][obs[t + 1]] * beta[t + 1][n];
                beta[t][s] = sum;
            }
            // Normalise to avoid underflow
            const z = beta[t].reduce((a, b) => a + b, 0) || 1;
            for (let s = 0; s < K; s++) beta[t][s] /= z;
        }
        // Posterior γ[t][s]
        const gamma = Array.from({length: T}, () => new Float64Array(K));
        for (let t = 0; t < T; t++) {
            let z = 0;
            for (let s = 0; s < K; s++) {
                gamma[t][s] = alpha[t][s] * beta[t][s];
                z += gamma[t][s];
            }
            for (let s = 0; s < K; s++) gamma[t][s] /= z || 1;
        }
        return gamma;
    }

    // ----- Viterbi -----
    function viterbiPath(obs) {
        const T = obs.length, K = 3;
        const V = Array.from({length: T}, () => new Float64Array(K));
        const back = Array.from({length: T}, () => new Int8Array(K));
        for (let s = 0; s < K; s++) V[0][s] = Math.log(PI[s] + 1e-12) + Math.log(B[s][obs[0]] + 1e-12);
        for (let t = 1; t < T; t++) {
            for (let s = 0; s < K; s++) {
                let bestP = -Infinity, bestPrev = 0;
                for (let p = 0; p < K; p++) {
                    const prob = V[t - 1][p] + Math.log(A[p][s] + 1e-12);
                    if (prob > bestP) { bestP = prob; bestPrev = p; }
                }
                V[t][s] = bestP + Math.log(B[s][obs[t]] + 1e-12);
                back[t][s] = bestPrev;
            }
        }
        // Backtrack
        let last = 0, bestF = -Infinity;
        for (let s = 0; s < K; s++) if (V[T - 1][s] > bestF) { bestF = V[T - 1][s]; last = s; }
        const path = new Int8Array(T);
        path[T - 1] = last;
        for (let t = T - 2; t >= 0; t--) path[t] = back[t + 1][path[t + 1]];
        return path;
    }

    function recompute() {
        posterior = forwardBackward(obs);
        viterbi   = viterbiPath(obs);
    }

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

    function stateColour(idx, alpha = 1) {
        const palette = [
            `rgba(234, 121, 89, ${alpha})`,   // Sunny — terracotta
            `rgba(212, 161, 60, ${alpha})`,   // Cloudy — mustard
            `rgba(79, 70, 229, ${alpha})`,    // Rainy — indigo
        ];
        return palette[idx];
    }

    function layout() {
        const pad = 16;
        const T = obs.length;
        const obsH = 56;
        const stateH = 60;
        const totalH = obsH + stateH * 3 + pad * 3 + 30;
        const cellW = (W - 2 * pad - 80) / T;
        return {
            pad, cellW,
            labelX: pad + 4,
            obsY: pad + 30,
            obsH,
            stateY: pad + 30 + obsH + pad,
            stateH,
        };
    }

    function draw() {
        if (!ctx || !posterior) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fbfaf7';
        ctx.fillRect(0, 0, W, H);

        const lay = layout();
        const T = obs.length;

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('OBSERVATIONS — click to cycle activity', lay.labelX + 70, lay.obsY - 7);
        ctx.fillText('INFERRED STATE POSTERIOR — colour fraction = P(state | observations)',
            lay.labelX + 70, lay.stateY - 7);

        // Observation row
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('observe', lay.labelX + 64, lay.obsY + lay.obsH / 2 + 4);

        for (let t = 0; t < T; t++) {
            const x = lay.labelX + 70 + t * lay.cellW;
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(x + 1, lay.obsY, lay.cellW - 2, lay.obsH);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
            ctx.strokeRect(x + 1, lay.obsY, lay.cellW - 2, lay.obsH);
            // Icon
            ctx.fillStyle = '#1a1a1a';
            ctx.font = '22px "Apple Color Emoji", "Segoe UI Emoji", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(OBS_ICONS[obs[t]], x + lay.cellW / 2, lay.obsY + 28);
            // Label
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.font = '500 9px "Inter", system-ui, sans-serif';
            ctx.fillText(OBS[obs[t]], x + lay.cellW / 2, lay.obsY + 46);
        }

        // State rows
        for (let s = 0; s < 3; s++) {
            const rowY = lay.stateY + s * lay.stateH;
            // Label
            ctx.fillStyle = stateColour(s);
            ctx.font = '600 11px "Inter", system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(STATES[s], lay.labelX + 64, rowY + lay.stateH / 2 + 4);

            for (let t = 0; t < T; t++) {
                const x = lay.labelX + 70 + t * lay.cellW;
                const p = posterior[t][s];
                // Cell background tint
                ctx.fillStyle = stateColour(s, p * 0.85);
                ctx.fillRect(x + 1, rowY + 4, lay.cellW - 2, lay.stateH - 8);
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
                ctx.strokeRect(x + 1, rowY + 4, lay.cellW - 2, lay.stateH - 8);
                // Probability number
                ctx.fillStyle = p > 0.55 ? '#ffffff' : 'rgba(0, 0, 0, 0.6)';
                ctx.font = '500 10px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(p.toFixed(2), x + lay.cellW / 2, rowY + lay.stateH / 2 + 3);
            }
        }

        // Viterbi path — line through the chosen state per timestep
        const path = viterbi;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.lineWidth = 2.4;
        ctx.setLineDash([]);
        ctx.beginPath();
        for (let t = 0; t < T; t++) {
            const x = lay.labelX + 70 + t * lay.cellW + lay.cellW / 2;
            const y = lay.stateY + path[t] * lay.stateH + lay.stateH / 2;
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        // Dots on path
        for (let t = 0; t < T; t++) {
            const x = lay.labelX + 70 + t * lay.cellW + lay.cellW / 2;
            const y = lay.stateY + path[t] * lay.stateH + lay.stateH / 2;
            ctx.fillStyle = '#ffffff';
            ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = stateColour(path[t]);
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Legend for Viterbi line
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('— Viterbi most-likely path',
            lay.labelX + 70, lay.stateY + 3 * lay.stateH + 18);

        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        // Find which state Viterbi visits most
        const counts = [0, 0, 0];
        for (const s of viterbi) counts[s]++;
        const dominant = counts.indexOf(Math.max(...counts));
        captionEl.innerHTML =
            `Each cell shows <em>P(state | every observation)</em> after running forward-backward. ` +
            `The dark line is the <strong>Viterbi</strong> path — the single most likely state sequence. ` +
            `It spends most of its time in <strong>${STATES[dominant]}</strong> here. ` +
            `Click any observation icon above to flip the activity — watch the state posterior re-shuffle ` +
            `as the evidence changes.`;
    }

    // ----- Interactions -----
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches ? e.touches[0] : e;
        return [t.clientX - rect.left, t.clientY - rect.top];
    }
    canvas.addEventListener('click', (e) => {
        const [px, py] = getPos(e);
        const lay = layout();
        if (py < lay.obsY || py > lay.obsY + lay.obsH) return;
        const tIdx = Math.floor((px - lay.labelX - 70) / lay.cellW);
        if (tIdx < 0 || tIdx >= obs.length) return;
        obs[tIdx] = (obs[tIdx] + 1) % 3;
        recompute();
        draw();
    });

    // ----- Presets -----
    const PRESETS = {
        normal:    [0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2],
        rainy:     [2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2],
        flip:      [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
        anomaly:   [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    };
    if (presetSel) {
        presetSel.innerHTML = `
            <option value="normal">Mixed week</option>
            <option value="rainy">Rainy stretch</option>
            <option value="flip">Sunny → Rainy flip</option>
            <option value="anomaly">Single weird day</option>
        `;
        presetSel.addEventListener('change', () => {
            obs = [...PRESETS[presetSel.value]];
            recompute();
            draw();
        });
    }
    resetBtn?.addEventListener('click', () => {
        obs = [...PRESETS.normal];
        recompute();
        draw();
    });

    // ----- Init -----
    recompute();
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
