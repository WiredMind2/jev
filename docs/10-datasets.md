# Datasets

TypeSafe has not published a decision-training corpus. An open engine has
to be trained and measured on **public labeled tasks that match the three
primitives**, plus one variable-menu task so the head is not just a fixed
classifier.

Do not vendor raw datasets in git. Download them locally, convert to
[`../schemas/training-example.schema.json`](../schemas/training-example.schema.json),
and keep converted JSONL under `data/` (gitignored except README). Cite
the source. Honor each dataset's license; those terms are stricter than
this repo's MIT license.

Retrieved 19 September 2026.

## What a useful row looks like

A Jev-like example is not "text → class index" in isolation. It is
**state + the full option set the model will see + one gold key**.

| Primitive | Gold | Option set |
|---|---|---|
| Choice | option key | 2–255 named criteria; include `other` / OOS when the source has it |
| Score | level index `0 … K-1` | 2–10 **ordered** situation descriptions, not "low/medium/high" |
| Noul | `true` / `false` | optional `{true, false}` criteria |

The compact `jevlike` row `{context, options, label}` is a projection of
the same thing. Use it for trainers; keep the full record for eval and
data cards.

## Recommended v0 mix

Train and debug in this order. Do not mix all of them into one head until
each task reports on its own.

| Order | Dataset | Primitive | Why first |
|---|---|---|---|
| 0 | `jevlike` synthetic menus | Choice | Proves the trainer. ~98% is expected; if you cannot hit that, the code is wrong. |
| 1 | **BANKING77** | Choice (77) | Fine-grained routing, CC BY 4.0, already used to measure hosted Jev. |
| 2 | **SST-5** | Score (5) | Ordered rubric without inventing levels. |
| 3 | **BoolQ** | Noul | Passage + yes/no; CC BY-SA 3.0; better license than IMDB. |
| 4 | **CLINC150** (with OOS) | Choice + defer | 150 intents plus an explicit out-of-scope class. |
| 5 | **Wikispeedia next-click** | Choice, variable `N` | The only widely used *variable-menu* public task in the Jev-like literature. |

After those five work, add When2Call (agent abstention) and Amazon ESCI
(query–product menus up to 40). Skip AG News and IMDB as *training* sets:
they are easy, often contaminated in encoder pretraining, and
license-awkward.

## Community measurements on hosted Jev

These are **measurements** from other people, n often 300–500, not
reproduced here. They tell you which tasks are already comparable.

| Source | Task | Hosted Jev | Note |
|---|---|---|---|
| [jev-eval](https://github.com/4esv/jev-eval) (300 items, 17 Sep 2026) | BANKING77 Choice | acc 0.78, ECE 0.11 | GPT-5.6 Terra 0.85 / 0.08; Jev ~5× faster, ~50× cheaper per call |
| same | SST-5 Score | acc 0.57, ECE 0.20 | Terra 0.59 / 0.30 — both weak; ordinal is the hard primitive |
| same | IMDB Noul | acc 0.97, ECE 0.04 | Tied with Terra; too easy as a training target |
| [janus-decide](https://pypi.org/project/janus-decide/) (500 items) | BANKING77 | 77.8% | Cascade to DeepSeek at confidence 0.67 → 80.2% |
| [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) (100/condition) | AG News / BANKING77-BTZSC / DAIR Emotion | 0.91 / 0.87 / 0.48 | Emotion: Jev badly calibrated (Brier 0.846) |
| [jevlike](https://github.com/vinnylarouge/jevlike) | Wikispeedia next-click | n/a (open model) | Frozen Qwen2.5-0.5B + head ~26% vs ~8% shuffled |
| [jev-decision-benchmarks](https://github.com/baibizhe/jev-decision-benchmarks) | MetaTool / When2Call / BFCL V4 | see that repo | Tool selection and abstention, adapter-dependent |

Takeaway: a first open model that is *interesting* must beat chance on
BANKING77, produce a usable ordinal distribution on SST-5, and collapse
under shuffled context on Wikispeedia. Matching hosted Jev's 78% on
BANKING77 is a later bar, not week-one.

## Catalog

### Choice — fixed label set (routing / intent)

#### BANKING77 — **start here for real Choice**

- **What:** 13,083 banking support queries, 77 intents. Train 10,003 /
  test 3,080. Single domain, fine-grained.
- **Map:** `state` = utterance. `criteria` = all 77 intent names with
  one-line descriptions you write once and freeze. `gold` = intent key.
  `group_id` is optional (utterances are short and mostly independent).
- **License:** CC BY 4.0 (Casanueva et al., EMNLP 2020).
- **Get:** `PolyAI/banking77` or `mteb/banking77` on Hugging Face.
- **Pitfall:** 77-way softmax over *names* without descriptions becomes a
  lexical prior test. Write criteria. Do not drop rare intents from the
  menu at train time if they appear at test time.

#### CLINC150 / CLINC-OOS

- **What:** 150 in-scope intents across 10 domains, plus out-of-scope
  queries. Full split: 100 train / 20 val / 30 test per intent; OOS has
  100 train, 100 val, **1,000** test.
- **Map:** in-scope as Choice over 150 (+ `out_of_scope`). This is the
  public dataset that most cleanly teaches `insufficient_information`.
- **License:** CC BY 3.0.
- **Get:** `clinc/oos-eval` or `clinc/clinc_oos`.
- **Pitfall:** if you train without OOS, the model cannot say "none of
  these." That is exactly the type-safe-but-wrong failure mode.

#### HWU64 / MASSIVE

- **What:** HWU64 is 25,716 utterances, 64 intents (CC BY-SA 3.0).
  MASSIVE is ~1M parallel utterances, 60 intents, 51 languages
  (CC BY 4.0). English MASSIVE is enough for v0; multilingual is a
  later stress test (hosted Jev is English-first).
- **Map:** same as BANKING77.
- **Use:** after BANKING77, to check that the head is not banking-only.

### Choice — variable menu (the actual Jev-like problem)

Fixed-label intent data can be solved by a 77-way linear head on pooled
embeddings. That is **not** the architecture we want. You need rows where
`N` and the option *text* change.

#### Wikispeedia next-click — **first variable-menu task**

- **What:** Human paths on a 4,604-article Wikipedia subset. 51,318
  finished paths, 24,875 unfinished, 119,882 links. Median out-degree
  ~19, max 294.
- **Map:** For each click: `state` = current article plaintext + target
  title (and optionally the path so far). `criteria` = outgoing article
  titles (and a short snippet if you have plaintext). `gold` = next
  article. `group_id` = **target article**, so the test set is
  target-disjoint (`jevlike` does this).
- **License:** SNAP distribution + Wikipedia article licenses (typically
  CC BY-SA). Cite West & Leskovec, WWW 2012. Read
  https://snap.stanford.edu/data/wikispeedia.html before redistributing
  converted files.
- **Get:** `jevlike` `scripts/get_wikispeedia.sh` downloads
  `wikispeedia_paths-and-graph.tar.gz` and plaintext, then
  `jevlike-data wikispeedia`.
- **Pitfalls:**
  - Out-degree can exceed 255. Always include the gold; sample other
    outgoing links down to 255, or run the two-stage shortlist that
    TypeSafe describes for high cardinality.
  - Strip back-click tokens (`<`) if you model forward navigation.
  - Shuffled-context control is mandatory. Option priors (hubs like
    "Water") will otherwise look like intelligence.

#### Amazon ESCI (Shopping Queries)

- **What:** Query + up to 40 products, each labeled Exact / Substitute /
  Complement / Irrelevant. Large split: 130,652 queries, ~2.62M
  judgments (EN/ES/JA). Reduced "hard" split: 48,300 queries.
- **Map (pick one and freeze it):**
  - Score: levels `[Irrelevant, Complement, Substitute, Exact]` on each
    pair.
  - Choice: among products for a query, pick one `Exact` (skip queries
    with none).
  - Noul: "is this product an Exact match?"
- **License:** Apache 2.0 (amazon-science/esci-data).
- **Use:** after Wikispeedia, for menus that look like search/rerank
  (TypeSafe's rerank cookbook). English-only first. Split by `query_id`.

#### When2Call

- **What:** 15k SFT / 9k preference train; 3,652 MCQ test. Four actions:
  call a tool, ask for information, answer directly, cannot complete.
  Includes 258 no-tool items used for hallucination rate.
- **Map:** Choice over those four actions; tool specs go in `state`.
- **License:** check the NVIDIA Hugging Face card (`nvidia/When2Call`)
  before commercial use.
- **Open project:** hosted Jev Choice adapter reported 74.84% accuracy
  and **76% tool hallucination** on the no-tool slice. A replica that
  cannot abstain is not a decision model.

#### MetaTool / ToolE

- **What:** 21,127 queries with tool names and descriptions; similar-tool
  and abstention splits (~995 public each).
- **Map:** Choice over a candidate tool list plus `none`. Descriptions
  are the criteria values.
- **Use:** agent-routing eval, not the first training set (LLM-generated
  queries; treat labels as noisy).

### Score — ordered levels

#### SST-5 (Stanford Sentiment Treebank, fine-grained)

- **What:** Movie-review sentences, 5 ordered labels: very negative →
  very positive. Sentence-level HF mirrors have ~11.8k rows
  (`SetFit/sst5`).
- **Map:** Score with five *situational* level texts, e.g. "clearly
  negative throughout" … "clearly positive throughout". Gold = 0..4.
  Report expected score MAE as well as accuracy.
- **License:** original SST terms (Penn Treebank / Stanford). Fine for
  research; do not assume MIT-like redistribution.
- **Why:** jev-eval already scored hosted Jev here (0.57). If your head
  only learns Choice, this is where it will show.

Do **not** describe levels as "0, 1, 2, 3, 4". TypeSafe's Score page
shows numeric-only levels collapse.

#### SemEval-2018 Task 1 V-oc / EI-oc

- **What:** Tweets with 7-point valence or 4-point emotion intensity
  (anger, fear, joy, sadness). True ordinal classification.
- **Map:** one Score question per emotion; keep K ≤ 10.
- **Use:** second Score task, once SST-5 conversion is frozen.

GoEmotions (58k Reddit comments, 27 emotions + neutral, Apache 2.0) is
**multi-label Choice**, not Score. Use it later as 27 independent Nouls
or a multi-label eval — do not smash it into one 28-way Choice and call
it ordinal.

### Noul — binary propositions

#### BoolQ — **start here for Noul**

- **What:** 15,942 naturally occurring yes/no questions with a Wikipedia
  passage. Triplet `(question, passage, yes/no)`.
- **Map:** `state` = `{title, passage, question}`. `instructions` = the
  question. Optional criteria for yes/no. `noul` gold = 1 if yes.
- **License:** CC BY-SA 3.0.
- **Pitfall:** this is reading-comprehension, not "is this urgent?"
  Still the cleanest public Noul. Split so the same Wikipedia page does
  not appear in train and test (`group_id` = title).

#### FEVER

- **What:** 185k claims labeled Supported / Refuted / NotEnoughInfo,
  with evidence sentences for the first two.
- **Map:** either 3-way Choice, or Noul "is this claim supported?" after
  dropping NEI (worse). Putting evidence in `state` is the Jev-like
  version; hiding it tests parametric knowledge (not the goal).
- **License:** Wikipedia-derived; CC BY-SA 3.0 fallback.

#### MultiNLI

- **What:** ~433k premise–hypothesis pairs; entailment / neutral /
  contradiction; 10 genres.
- **Map:** 3-way Choice, or Noul "does the premise entail the
  hypothesis?" with contradiction+neutral as false.
- **License:** mostly OANC (permissive); fiction section is mixed CC.
  Use `genre` as `group_id` for mismatched eval.
- **Why:** cheap volume for a binary/3-way head. Genre-mismatch is a
  real OOD test.

ANLI is harder NLI but **CC BY-NC 4.0** — skip if you want a
commercially reusable training mix.

### Datasets to treat carefully

| Dataset | Issue |
|---|---|
| AG News | 4 easy classes; original corpus is non-commercial / no-redistribute; likely in every encoder |
| IMDB Maas reviews | Easy Noul; scraped reviews, no clean license on the Stanford page |
| DAIR Emotion | Hosted Jev already fails calibration here; 6 overlapping labels |
| TypeSafe workflow evals | Reference labels are other LLMs, not human gold; workflows not a public train set |
| LLM-generated ToolE queries | Fine for eval, noisy as the only train signal |
| Customer tickets you do not own | Do not commit |

## Conversion recipes

BANKING77 → Choice:

```json
{
  "id": "banking77_0042",
  "state": {"text": "I was charged twice for the same ATM withdrawal."},
  "question": {
    "type": "choice",
    "instructions": "Which banking intent best matches `text`?",
    "criteria": {
      "card_payment_wrong_exchange_rate": "The customer reports a bad FX rate on a card payment.",
      "reverted_card_payment": "A card payment was reversed or bounced.",
      "insufficient_information": "The utterance does not match any listed intent."
    }
  },
  "gold": "card_payment_wrong_exchange_rate",
  "metadata": {"domain": "banking77", "group_id": "intent:card_payment_wrong_exchange_rate"}
}
```

(Expand `criteria` to the full 77 + optional OOS. Freeze the description
strings in a file and hash it into `metadata.schema_version`.)

SST-5 → Score:

```json
{
  "id": "sst5_0088",
  "state": {"text": "A sometimes tedious film."},
  "question": {
    "type": "score",
    "instructions": "What is the sentiment of `text`?",
    "criteria": [
      "Very negative: hostile or dismissive throughout.",
      "Negative: clearly unfavorable, not mixed.",
      "Neutral: mixed, factual, or no clear valence.",
      "Positive: clearly favorable, not ecstatic.",
      "Very positive: enthusiastic praise throughout."
    ]
  },
  "gold": 1,
  "metadata": {"domain": "sst5"}
}
```

BoolQ → Noul:

```json
{
  "id": "boolq_1201",
  "state": {
    "title": "Greece",
    "passage": "...",
    "question": "Is Greece a member of the European Union?"
  },
  "question": {
    "type": "noul",
    "instructions": "Given `passage`, is the answer to `question` yes?",
    "criteria": {
      "true": "The passage supports a yes answer.",
      "false": "The passage supports a no answer, or does not decide it."
    }
  },
  "gold": true,
  "metadata": {"domain": "boolq", "group_id": "Greece"}
}
```

Wikispeedia → variable Choice:

```json
{
  "id": "wiki_path318_step4",
  "state": {
    "target": "Albert Einstein",
    "current": "Quantum mechanics",
    "current_text": "... first paragraphs ..."
  },
  "question": {
    "type": "choice",
    "instructions": "Which outgoing article should a player click next to reach `target` from `current`?",
    "criteria": {
      "Photon": "Outgoing link: Photon",
      "Albert_Einstein": "Outgoing link: Albert Einstein",
      "Wave–particle_duality": "Outgoing link: Wave–particle duality"
    }
  },
  "gold": "Albert_Einstein",
  "metadata": {"domain": "wikispeedia", "group_id": "Albert Einstein"}
}
```

## Split and leakage rules (dataset-specific)

| Dataset | Split by |
|---|---|
| BANKING77 | Official train/test; carve calibration from train, stratified by intent |
| CLINC150 | Official splits; keep OOS test intact |
| SST-5 | Official sentence splits; never train on phrase-level trees if you test on sentences |
| BoolQ / FEVER | Wikipedia page title |
| Wikispeedia | Target article (not random clicks) |
| ESCI | Query id |
| When2Call | Official MCQ test; do not train on it |

Always keep a **calibration** slice that is not used for early stopping.
Always run shuffled-state and shuffled-option-order on the test set.

## Contamination

Public intent and sentiment sets have been in LM pretraining data for
years. That inflates zero-shot logprob scores. Mitigations:

- Prefer target-disjoint Wikispeedia and query-disjoint ESCI as the
  "did we learn compatibility?" tests.
- Report shuffled-context controls.
- Do not tune instructions on the test set (jev-eval's BANKING77 numbers
  are a comparison protocol, not a training recipe).
- When you later scrape your own tickets, that is the only uncontaminated
  deployment set.

## What we still do not have

- TypeSafe's RLCD data.
- A large public corpus of *multi-question* requests against one long
  structured state (the actual product workload). Approximate it later by
  attaching several BANKING77-style questions to one synthetic ticket
  object.
- Human-labeled support-triage with policy outcomes (refund issued, etc.).
  That is what you collect after the public mix works.

## Sources (datasets)

- Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders",
  BANKING77, 2020. https://huggingface.co/datasets/PolyAI/banking77
- Larson et al., "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction", EMNLP 2019. https://github.com/clinc/oos-eval
- West & Leskovec, "Human Wayfinding in Information Networks", WWW 2012.
  https://snap.stanford.edu/data/wikispeedia.html
- Reddy, Chen, Manning, BoolQ, 2019.
  https://github.com/google-research-datasets/boolean-questions
- Socher et al., Stanford Sentiment Treebank, 2013.
- Amazon, Shopping Queries / ESCI, KDD Cup 2022.
  https://github.com/amazon-science/esci-data
- Ross et al., When2Call, NAACL 2025.
  https://huggingface.co/datasets/nvidia/When2Call
- Huang et al., MetaTool / ToolE, ICLR 2024.
- Thorne et al., FEVER, 2018.
- Williams, Nangia, Bowman, MultiNLI, 2018.
- Aarab, BTZSC zero-shot classification suite.
  https://huggingface.co/datasets/btzsc/btzsc
