/* Topic-page interactivity: difficulty selector + code-block copy buttons.
 * Loaded by every page built from templates/topic.html. */

(function () {
    // ----- Difficulty selector -----
    const tabs   = document.querySelectorAll('.level-tab');
    const levels = document.querySelectorAll('.topic-level');
    const KEY    = 'ml-hub-topic-level';

    // Hide tabs whose content section doesn't exist on this page.
    // This lets sources opt into the "Visual" tab (or any future one)
    // without forcing every topic to define it.
    const availableLevels = new Set(
        Array.from(levels).map(l => l.dataset.level)
    );
    tabs.forEach(t => {
        if (!availableLevels.has(t.dataset.level)) {
            t.style.display = 'none';
        }
    });

    function setLevel(level) {
        // Fall back to "intuition" if a saved level isn't available on this page
        if (!availableLevels.has(level)) level = 'intuition';
        tabs.forEach(t => t.classList.toggle('active', t.dataset.level === level));
        levels.forEach(l => l.classList.toggle('active', l.dataset.level === level));
        localStorage.setItem(KEY, level);
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

    // Initialise with saved preference, falling back to "intuition" (the new default)
    setLevel(localStorage.getItem(KEY) || 'intuition');


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
