/* Topic-page interactivity: difficulty selector + code-block copy buttons.
 * Loaded by every page built from templates/topic.html. */

(function () {
    // ----- Difficulty selector -----
    const tabs   = document.querySelectorAll('.level-tab');
    const levels = document.querySelectorAll('.topic-level');
    const KEY    = 'ml-hub-topic-level';

    function setLevel(level) {
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

    // Initialise with saved preference, falling back to "standard"
    setLevel(localStorage.getItem(KEY) || 'standard');


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
