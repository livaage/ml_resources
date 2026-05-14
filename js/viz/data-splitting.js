/* Interactive data-splitting viz.
 * 60 examples arranged in a row, coloured by class (80/20 imbalanced),
 * with a "group" stripe above. Toggle the splitting strategy and watch
 * which examples land in train (indigo bar) vs val (orange bar) vs test (teal). */

(function () {
    const canvas    = document.getElementById('viz-ds-canvas');
    const randBtn   = document.getElementById('viz-ds-random');
    const stratBtn  = document.getElementById('viz-ds-strat');
    const timeBtn   = document.getElementById('viz-ds-time');
    const groupBtn  = document.getElementById('viz-ds-group');
    const captionEl = document.getElementById('viz-ds-caption');
    if (!canvas) return;

    let mode = 'random';
    const N = 60;
    let examples = [];

    function mulberry32(seed) {
        return function () {
            let t = (seed = (seed + 0x6D2B79F5) | 0);
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    const rng = mulberry32(7);
    for (let i = 0; i < N; i++) {
        const c = rng() < 0.8 ? 0 : 1;
        const group = Math.floor(i / 3);
        examples.push({ idx: i, c, group });
    }

    function assign() {
        // Returns array of "train" | "val" | "test"
        const out = Array(N).fill('train');
        if (mode === 'random') {
            const r = mulberry32(11);
            const perm = examples.map((_, i) => i).sort(() => r() - 0.5);
            for (let i = 0; i < N; i++) {
                if (i < N * 0.6) out[perm[i]] = 'train';
                else if (i < N * 0.8) out[perm[i]] = 'val';
                else out[perm[i]] = 'test';
            }
        } else if (mode === 'stratified') {
            const cls = [[], []];
            for (const e of examples) cls[e.c].push(e.idx);
            const r = mulberry32(11);
            for (const lst of cls) lst.sort(() => r() - 0.5);
            for (const lst of cls) {
                for (let i = 0; i < lst.length; i++) {
                    if (i < lst.length * 0.6) out[lst[i]] = 'train';
                    else if (i < lst.length * 0.8) out[lst[i]] = 'val';
                    else out[lst[i]] = 'test';
                }
            }
        } else if (mode === 'time') {
            for (let i = 0; i < N; i++) {
                if (i < N * 0.6) out[i] = 'train';
                else if (i < N * 0.8) out[i] = 'val';
                else out[i] = 'test';
            }
        } else if (mode === 'group') {
            const numGroups = Math.max(...examples.map(e => e.group)) + 1;
            const r = mulberry32(11);
            const order = [];
            for (let g = 0; g < numGroups; g++) order.push(g);
            order.sort(() => r() - 0.5);
            for (let i = 0; i < numGroups; i++) {
                const role = i < numGroups * 0.6 ? 'train' :
                             i < numGroups * 0.8 ? 'val' : 'test';
                for (const e of examples) if (e.group === order[i]) out[e.idx] = role;
            }
        }
        return out;
    }

    let ctx;
    let W = 0, H = 0;
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;
        const cssW = Math.max(540, Math.round(rect.width));
        const cssH = Math.round(Math.min(360, Math.max(260, cssW * 0.32)));
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

        const roles = assign();
        const pad = 30;
        const cellW = (W - 2 * pad) / N;
        const rowY = pad + 60;
        const rowH = 26;
        const groupY = rowY - rowH - 12;
        const classY = rowY + rowH + 12;
        const splitY = classY + 28;
        const splitH = rowH;

        // Group stripe (only meaningful for group mode but always shown faintly)
        const colours = ['#eef2ff', '#fdf0eb', '#e5f3f1', '#fff3e0', '#f3e5f5'];
        for (let i = 0; i < N; i++) {
            const x = pad + i * cellW;
            const g = examples[i].group;
            ctx.fillStyle = colours[g % colours.length];
            ctx.fillRect(x, groupY, cellW - 0.5, rowH * 0.6);
        }
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 9px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('group', pad - 4, groupY + 11);

        // Example boxes coloured by class (faint)
        for (let i = 0; i < N; i++) {
            const x = pad + i * cellW;
            ctx.fillStyle = examples[i].c === 0 ? 'rgba(79, 70, 229, 0.6)' : 'rgba(234, 121, 89, 0.85)';
            ctx.fillRect(x, rowY, cellW - 0.5, rowH);
        }
        ctx.fillText('class', pad - 4, rowY + rowH / 2 + 3);

        // Split row
        for (let i = 0; i < N; i++) {
            const x = pad + i * cellW;
            const role = roles[i];
            const col = role === 'train' ? '#4f46e5'
                       : role === 'val'   ? '#ea7959' : '#10847e';
            ctx.fillStyle = col;
            ctx.fillRect(x, splitY, cellW - 0.5, splitH);
        }
        ctx.fillText('split', pad - 4, splitY + splitH / 2 + 3);

        // Class proportions per split
        const stats = { train: [0, 0], val: [0, 0], test: [0, 0] };
        for (let i = 0; i < N; i++) stats[roles[i]][examples[i].c]++;
        const ly = splitY + splitH + 38;
        ctx.font = '500 10px "Inter", system-ui, sans-serif';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.textAlign = 'left';
        const formatStat = (role, col) => {
            const t = stats[role];
            const total = t[0] + t[1];
            return [`${role}`, `n=${total}`, `${t[0]}/${t[1]}`, col];
        };
        const labels = [
            formatStat('train', '#4f46e5'),
            formatStat('val',   '#ea7959'),
            formatStat('test',  '#10847e'),
        ];
        let lx = pad;
        for (const [role, count, classCount, col] of labels) {
            ctx.fillStyle = col;
            ctx.fillRect(lx, ly - 9, 10, 10);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
            ctx.fillText(`${role}  ${count}  (cls0/cls1 ${classCount})`, lx + 14, ly);
            lx += 180;
        }

        // Title
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.font = '600 10px "Inter", system-ui, sans-serif';
        ctx.textAlign = 'left';
        const titles = { random: 'RANDOM SPLIT', stratified: 'STRATIFIED SPLIT',
                         time: 'TIME-BASED SPLIT', group: 'GROUP SPLIT' };
        ctx.fillText(titles[mode], pad, 16);

        updateCaption(stats);
    }

    function updateCaption(stats) {
        if (!captionEl) return;
        const notes = {
            random:     `<strong>Random split.</strong> Each example assigned independently — fine for iid data, but the class ratio in val/test can drift, and groups get split across folds.`,
            stratified: `<strong>Stratified split.</strong> Class proportions preserved exactly in each split. The right default for any classification task with imbalance.`,
            time:       `<strong>Time-based split.</strong> First 60% to train, next 20% to val, last 20% to test. The only correct choice for sequential data — validation always comes after train in time.`,
            group:      `<strong>Group split.</strong> All examples from each group end up in the same fold. Required when independence breaks (multiple records per patient, per user, per session) — otherwise you measure within-group memorisation.`,
        };
        captionEl.innerHTML = notes[mode];
    }

    function setMode(m, btn) {
        mode = m;
        for (const b of [randBtn, stratBtn, timeBtn, groupBtn]) b?.classList.remove('active');
        btn?.classList.add('active');
        draw();
    }
    randBtn?.addEventListener('click',  () => setMode('random', randBtn));
    stratBtn?.addEventListener('click', () => setMode('stratified', stratBtn));
    timeBtn?.addEventListener('click',  () => setMode('time', timeBtn));
    groupBtn?.addEventListener('click', () => setMode('group', groupBtn));

    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => resize());
        ro.observe(canvas);
    } else {
        window.addEventListener('resize', resize);
    }
    resize();
})();
