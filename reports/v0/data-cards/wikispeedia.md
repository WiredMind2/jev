# Data card — Wikispeedia next-click

- **Task:** variable-N Choice. State = current article + target; criteria = outgoing titles.
- **Source:** West & Leskovec, WWW 2012. SNAP `wikispeedia_paths-and-graph`. Raw dump is **not** vendored in git.
- **License:** SNAP distribution + Wikipedia CC BY-SA. Read https://snap.stanford.edu/data/wikispeedia.html before redistributing converted files.
- **Criteria file:** `assets/criteria/wikispeedia.json` freezes instruction text and `title_template`, not a global menu.
- **`group_id`:** **target article** (target-disjoint test).
- **Back-clicks:** tokens equal to `<` pop a navigation stack; only forward (current, next) pairs are examples.
- **Cap 255:** always keep gold; sample other outgoing links down to 255.
- **This run (SNAP):** train 159134 / val 26900 / calib 48951 / test 39786, target-disjoint. Manifest `reports/v0/manifests/wikispeedia-snap.json`. skipped_rows=128 (menus with fewer than 2 options).
- **Fixture:** `tests/fixtures/data/wikispeedia.json` (260-degree hub, a `<` back-click path, extra Target_* paths so all four splits fill). Converts to train 24 / val 3 / calib 5 / test 4, target-disjoint. Manifest `reports/v0/manifests/wikispeedia-fixture.json`. Hub menus cap at 255 while keeping gold.
- Shuffled-context is mandatory: hub priors otherwise look like intelligence.
