/* Interactive LLM scaling viz.
 * Stylised capability-vs-scale curves on a log x-axis. As the user slides
 * model size, a vertical line moves; each capability's value at that scale
 * is read off and shown. Several abilities show sharp "emergent" jumps
 * around specific scales. */

(function () {
    const canvas    = document.getElementById('viz-llm-canvas');
    const sSlider   = document.getElementById('viz-llm-scale');
    const sLbl      = document.getElementById('viz-llm-scale-lbl');
    const captionEl = document.getElementById('viz-llm-caption');
    if (!canvas) return;

    let scale = 9;     // log10 of params, so 9 = 10^9 (1B)
    let ctx;
    let W = 0, H = 0;

    // Capabilities — log-x mid-points where the sigmoid reaches 0.5
    const CAPABILITIES = [
        { name: 'memorise facts',          mid: 7.5,  steep: 1.2, ceil: 0.95, col: '#4f46e5' },
        { name: 'basic grammar',           mid: 8.0,  steep: 1.5, ceil: 0.98, col: '#688f5b' },
        { name: 'multi-step arithmetic',   mid: 10.2, steep: 2.0, ceil: 0.85, col: '#ea7959' },
        { name: 'chain-of-thought',        mid: 10.8, steep: 2.5, ceil: 0.80, col: '#10847e' },
        { name: 'few-shot in-context',     mid: 9.6,  steep: 1.8, ceil: 0.88, col: '#d4a13c' },
        { name: 'tool use',                mid: 11.3, steep: 2.0, ceil: 0.75, col: '#a05bb0' },
    ];

    function ability(cap, logN) {
        return cap.ceil / (1 + Math.exp(-cap.steep * (logN - cap.mid)));
    }

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
        const box = { x: pad, y: 30, w: W - pad - 220, h: H - 80 };
        ctx.fillStyle = '#fff';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
        ctx.strokeRect(box.x - 0.5, box.y - 0.5, box.w + 1, box.h + 1);

        // X axis: log10 from 7 to 13 (10M to 10T)
        function xToPx(logN) { return box.x + (logN - 7) / 6 * box.w; }
        function yToPx(v) { return box.y + box.h - v * (box.h - 16); }

        // Gridlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.setLineDash([2, 3]);
        for (const v of [7, 8, 9, 10, 11, 12, 13]) {
            const x = xToPx(v);
            ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
        }
        for (const v of [0.25, 0.5, 0.75, 1]) {
            const y = yToPx(v);
            ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
        }
        ctx.setLineDash([]);

        // X-axis labels
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        for (const v of [7, 9, 11, 13]) {
            ctx.fillText(`10^${v}`, xToPx(v), box.y + box.h + 12);
        }
        ctx.textAlign = 'right';
        for (const v of [0, 0.5, 1]) {
            ctx.fillText(v.toFixed(1), box.x - 4, yToPx(v) + 3);
        }
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'center';
        ctx.fillText('parameters (log scale)', box.x + box.w / 2, box.y + box.h + 26);
        ctx.save();
        ctx.translate(box.x - 32, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('capability score', 0, 0);
        ctx.restore();

        // Curves
        for (const cap of CAPABILITIES) {
            ctx.strokeStyle = cap.col;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (let lN = 7; lN <= 13; lN += 0.05) {
                const v = ability(cap, lN);
                const [px, py] = [xToPx(lN), yToPx(v)];
                if (lN === 7) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
            cap._cur = ability(cap, scale);
            // Marker at current
            const [px, py] = [xToPx(scale), yToPx(cap._cur)];
            ctx.fillStyle = cap.col;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }

        // Current-scale vertical line
        const sx = xToPx(scale);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(sx, box.y); ctx.lineTo(sx, box.y + box.h); ctx.stroke();
        ctx.setLineDash([]);

        // Legend / values
        let lx = box.x + box.w + 18, ly = box.y + 12;
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'left';
        const params = (10 ** scale);
        const fmt = params >= 1e12 ? `${(params / 1e12).toFixed(1)}T`
                  : params >= 1e9  ? `${(params / 1e9).toFixed(1)}B`
                  : params >= 1e6  ? `${(params / 1e6).toFixed(0)}M`
                  : `${params.toExponential(0)}`;
        ctx.fillText(`scale: ${fmt} params`, lx, ly);
        ly += 16;
        for (const cap of CAPABILITIES) {
            ctx.fillStyle = cap.col;
            ctx.fillRect(lx, ly - 8, 10, 10);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.font = '500 10px "Inter", system-ui, sans-serif';
            ctx.fillText(cap.name, lx + 14, ly);
            ctx.font = '600 11px "JetBrains Mono", monospace';
            ctx.fillStyle = cap.col;
            ctx.fillText(`${(cap._cur * 100).toFixed(0)}%`, lx + 160, ly);
            ly += 18;
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('CAPABILITY vs SCALE (stylised)', box.x, box.y - 8);

        if (sLbl) sLbl.textContent = `10^${scale.toFixed(1)} params`;
        updateCaption();
    }

    function updateCaption() {
        if (!captionEl) return;
        if (scale < 8) {
            captionEl.innerHTML =
                `<strong>10⁷–10⁸ (10–100M params).</strong> Pre-LLM-era models. Memorise facts; basic grammar starts to emerge. Multi-step reasoning is essentially zero. Examples: BERT, GPT-2.`;
        } else if (scale < 10) {
            captionEl.innerHTML =
                `<strong>10⁹–10¹⁰ (1–10B params).</strong> Modern small/medium open-source LLMs. Few-shot in-context learning works. Chain-of-thought starts to help. Examples: GPT-3, Llama 2 7B.`;
        } else if (scale < 12) {
            captionEl.innerHTML =
                `<strong>10¹⁰–10¹² (10–1000B params).</strong> Frontier models. Multi-step arithmetic, code, tool use all reach useful levels. Examples: GPT-3.5/4, Claude 3 Opus, Llama 3 70B.`;
        } else {
            captionEl.innerHTML =
                `<strong>10¹²+ (1T+ params, possibly MoE).</strong> The largest frontier models. Most capabilities near their asymptote. Marginal gains require fundamentally new data or training procedures.`;
        }
    }

    if (sSlider) {
        sSlider.min = 7; sSlider.max = 13; sSlider.step = 0.1; sSlider.value = 9;
        sSlider.addEventListener('input', () => {
            scale = parseFloat(sSlider.value);
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
