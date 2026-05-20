/* Topic-page interactivity: tier selector + notation toggle + code copy buttons.
 * Loaded by every page built from templates/topic.html. */

(function () {
    const body = document.body;

    // ----- Tier selector -----
    const tabs   = document.querySelectorAll('.level-tab');
    const levels = document.querySelectorAll('.topic-level');
    const KEY    = 'ml-hub-topic-level';

    // Hide tabs whose content section doesn't exist on this page.
    // Lets sources opt into a subset of tiers (e.g. just Intuition).
    const availableLevels = new Set(
        Array.from(levels).map(l => l.dataset.level)
    );
    tabs.forEach(t => {
        if (!availableLevels.has(t.dataset.level)) {
            t.style.display = 'none';
        }
    });

    function setLevel(level) {
        if (!availableLevels.has(level)) level = 'intuition';
        tabs.forEach(t => t.classList.toggle('active', t.dataset.level === level));
        levels.forEach(l => l.classList.toggle('active', l.dataset.level === level));
        body.setAttribute('data-tier', level);
        try { localStorage.setItem(KEY, level); } catch (e) {}
    }

    tabs.forEach(t => t.addEventListener('click', () => setLevel(t.dataset.level)));

    // Inline "switch to X" buttons at the bottom of each level
    document.querySelectorAll('[data-go-to]').forEach(btn => {
        btn.addEventListener('click', () => {
            setLevel(btn.dataset.goTo);
            const tabBar = document.querySelector('.level-tabs');
            if (tabBar) tabBar.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    let savedLevel = 'intuition';
    try { savedLevel = localStorage.getItem(KEY) || 'intuition'; } catch (e) {}
    setLevel(savedLevel);


    // ----- Notation toggle (Plain vs Math) -----
    const NKEY = 'ml-hub-notation';
    const notationButtons = document.querySelectorAll('.notation-toggle .seg button');

    function setNotation(mode) {
        if (mode !== 'plain') mode = 'standard';
        body.setAttribute('data-notation', mode);
        notationButtons.forEach(b =>
            b.classList.toggle('active', b.dataset.notation === mode)
        );
        try { localStorage.setItem(NKEY, mode); } catch (e) {}
    }

    notationButtons.forEach(btn => {
        btn.addEventListener('click', () => setNotation(btn.dataset.notation));
    });

    let savedNotation = 'standard';
    try { savedNotation = localStorage.getItem(NKEY) || 'standard'; } catch (e) {}
    setNotation(savedNotation);


    // ----- Copy button on each code block -----
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const pre = btn.parentElement.querySelector('pre');
            if (!pre) return;
            try {
                await navigator.clipboard.writeText(pre.innerText);
                btn.textContent = 'Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 1500);
            } catch (err) {
                btn.textContent = 'Failed';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            }
        });
    });
})();
