# Data card — Wikispeedia next-click

- **Task:** variable-N Choice. State = current article + target; criteria = outgoing titles.
- **Source:** West & Leskovec, WWW 2012. SNAP `wikispeedia_paths-and-graph`. Raw dump is **not** vendored.
- **License:** SNAP distribution + Wikipedia CC BY-SA. Read https://snap.stanford.edu/data/wikispeedia.html before redistributing converted files.
- **Criteria file:** `assets/criteria/wikispeedia.json` freezes instruction text and `title_template`, not a global menu.
- **`group_id`:** **target article** (target-disjoint test).
- **Back-clicks:** tokens equal to `<` pop a navigation stack; only forward (current, next) pairs are examples.
- **Cap 255:** always keep gold; sample other outgoing links down to 255.
- **This run:** fixture `tests/fixtures/data/wikispeedia.json` (includes a 260-degree hub and a `<` path). Manifest `reports/v0/manifests/wikispeedia-fixture.json`. Shuffled-context is mandatory: hub priors otherwise look like intelligence.
