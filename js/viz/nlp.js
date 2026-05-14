/* Interactive NLP tokenisation viz.
 * Toy BPE-style tokeniser: split on common sub-word units; rare words break
 * down further. Each token gets a deterministic 8-dim "embedding" displayed
 * as a coloured bar pattern so the same token always looks the same.
 * Educational, not a real BPE algorithm. */

(function () {
    const canvas    = document.getElementById('viz-nlp-canvas');
    const textIn    = document.getElementById('viz-nlp-text');
    const resetBtn  = document.getElementById('viz-nlp-reset');
    const captionEl = document.getElementById('viz-nlp-caption');
    if (!canvas) return;

    let text = textIn?.value ?? 'the quick brown fox jumps over the lazy dog';
    let ctx;
    let W = 0, H = 0;

    // Toy "vocabulary" of common sub-word pieces, longest first
    const VOCAB = [
        'jumping', 'jumps', 'quick', 'brown', 'lazy', 'dog', 'fox', 'over',
        'the', 'a', 'an', 'of', 'and', 'is', 'are', 'in', 'to', 'for',
        'ed', 'ing', 'ly', 'er', 'est', 'tion', 'ness', 'ment', 'able',
    ].sort((a, b) => b.length - a.length);

    function tokenise(s) {
        const tokens = [];
        const words = s.toLowerCase().split(/\s+/).filter(Boolean);
        for (const word of words) {
            let i = 0;
            while (i < word.length) {
                let matched = null;
                for (const v of VOCAB) {
                    if (word.slice(i).startsWith(v)) { matched = v; break; }
                }
                if (matched) {
                    tokens.push({ token: matched, start: i === 0 });
                    i += matched.length;
                } else {
                    tokens.push({ token: word[i], start: i === 0 });
                    i++;
                }
            }
        }
        return tokens;
    }

    function tokenHash(s) {
        let h = 5381;
        for (const c of s) h = ((h * 33) ^ c.charCodeAt(0)) >>> 0;
        return h;
    }
    function tokenEmbed(s) {
        // 8 deterministic floats in [-1, 1] keyed on the token string
        const h = tokenHash(s);
        const emb = [];
        for (let i = 0; i < 8; i++) {
            const v = ((h >>> (i * 4)) & 0xff) / 255;
            emb.push(v * 2 - 1);
        }
        return emb;
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(440, Math.max(280, cssW * 0.36)));
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

        const tokens = tokenise(text);
        if (tokens.length === 0) return;

        const pad = 24;
        const rowY1 = 50;     // raw text row
        const rowY2 = 100;    // tokens row
        const embY  = 140;    // embeddings row
        const embH  = H - embY - 40;

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('TEXT → TOKENS → EMBEDDINGS', pad, 22);

        // Compute layout widths per token
        ctx.font = '600 14px "JetBrains Mono", monospace';
        const widths = tokens.map(t => Math.max(40, ctx.measureText(t.token).width + 18));
        const totalW = widths.reduce((s, w) => s + w + 6, 0);
        let scale = 1;
        if (totalW > W - 2 * pad) scale = (W - 2 * pad) / totalW;

        // Draw raw text (with subtle highlighting per word)
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.fillText('text', pad - 4, rowY1 + 4);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
        ctx.font = '500 13px "Inter", system-ui, sans-serif';
        ctx.fillText(text, pad + 30, rowY1 + 4);

        // Token chips
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.fillText('tokens', pad - 4, rowY2 + 4);

        let x = pad + 50;
        for (let i = 0; i < tokens.length; i++) {
            const w = widths[i] * scale;
            const h = 26;
            const isFullWord = tokens[i].start;
            ctx.fillStyle = isFullWord ? 'rgba(79, 70, 229, 0.18)' : 'rgba(234, 121, 89, 0.18)';
            ctx.strokeStyle = isFullWord ? 'rgba(79, 70, 229, 0.65)' : 'rgba(234, 121, 89, 0.65)';
            ctx.lineWidth = 1.4;
            ctx.fillRect(x, rowY2 - h / 2 - 2, w - 4, h);
            ctx.strokeRect(x - 0.5, rowY2 - h / 2 - 2.5, w - 4 + 1, h + 1);
            ctx.fillStyle = '#1a1a1a';
            ctx.font = '600 13px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(tokens[i].token, x + (w - 4) / 2, rowY2 + 4);
            // Index below
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.font = '500 9px "JetBrains Mono", monospace';
            ctx.fillText(`#${tokens[i].token.length}`, x + (w - 4) / 2, rowY2 + 22);
            x += w + 6;
        }

        // Embeddings — 8-dim coloured bar pattern per token
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '500 11px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('embedding', pad - 4, embY + 12);

        x = pad + 50;
        const D = 8;
        const cellH = (embH - 30) / D;
        for (let i = 0; i < tokens.length; i++) {
            const w = widths[i] * scale - 4;
            const emb = tokenEmbed(tokens[i].token);
            for (let d = 0; d < D; d++) {
                const v = emb[d];
                const t = (v + 1) / 2;
                // Diverging: indigo (negative) ↔ cream ↔ terracotta (positive)
                let r, g, b;
                if (v < 0) {
                    const a = -v;
                    r = 251 + (79 - 251) * a;
                    g = 250 + (70 - 250) * a;
                    b = 247 + (229 - 247) * a;
                } else {
                    r = 251 + (234 - 251) * v;
                    g = 250 + (121 - 250) * v;
                    b = 247 + (89 - 247) * v;
                }
                ctx.fillStyle = `rgb(${r|0}, ${g|0}, ${b|0})`;
                ctx.fillRect(x, embY + 16 + d * cellH, w, cellH + 0.5);
            }
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
            ctx.strokeRect(x - 0.5, embY + 16 - 0.5, w + 1, D * cellH + 1);
            x += widths[i] * scale + 6 - 4 + 4;
        }

        updateCaption(tokens);
    }

    function updateCaption(tokens) {
        if (!captionEl) return;
        const numTokens = tokens.length;
        const wordCount = text.split(/\s+/).filter(Boolean).length;
        const ratio = (numTokens / wordCount).toFixed(2);
        captionEl.innerHTML =
            `<strong>${numTokens} tokens</strong> from ${wordCount} words (ratio ${ratio}). ` +
            `Indigo chips are full-word tokens; terracotta are sub-word pieces (suffixes, fragments). ` +
            `Each token gets a deterministic embedding sketch — real models learn these from data, ` +
            `with 768 to 4096 dimensions instead of our 8.`;
    }

    textIn?.addEventListener('input', () => {
        text = textIn.value;
        draw();
    });
    resetBtn?.addEventListener('click', () => {
        textIn.value = 'the quick brown fox jumps over the lazy dog';
        text = textIn.value;
        draw();
    });

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
