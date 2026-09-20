# Bibliography

Retrieved 19 September 2026 unless noted. Inclusion is not endorsement.

## TypeSafe primary sources

- Diogo Almeida, "Introducing System One Models and Jev", TypeSafe,
  15 September 2026.
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe docs index.
  https://docs.typesafe.ai/llms.txt
- System One concept.
  https://docs.typesafe.ai/concepts/system-one.md
- State.
  https://docs.typesafe.ai/concepts/state.md
- Primitives, Choice, Score, Noul, API, models, confidence, AI primer,
  quickstart, jaggedness.
  https://docs.typesafe.ai/
- Python / JS SDKs.
  https://github.com/typesafe-ai/typesafe-sdk-python
  https://github.com/typesafe-ai/typesafe-sdk-js

## Independent technical writeups

- Anthony Maio, "Jev: The Language Model That Won't Talk", 16 September
  2026. Distinguishes type-safety from correctness; notes RLCD is
  unpublished as an algorithm.
  https://anthonymaio.substack.com/p/jev-the-language-model-that-wont
- Flavio Copes, "A deep dive into Jev, TypeSafe's System One model",
  updated 18 September 2026.
  https://flaviocopes.com/jev/
- Laurie Voss, "TypeSafe Jev: Can Decision Models Replace LLM Judges?",
  Arize, September 2026.
  https://arize.com/blog/typesafe-jev-llm-judge/
- Sydney Runkle and Hunter Lovell, "Building a Harness with Jev",
  LangChain, 17 September 2026.
  https://www.langchain.com/blog/building-a-harness-with-jev
- SeeAPI, "TypeSafe Jev Explained", 18 September 2026 (model limits and
  pricing snapshot).
  https://www.seeapi.com/blogs/news/typesafe-jev-system-one-model/
- Michał Chromiak, "Jev: Typed decisions for enterprise AI", 17 September
  2026.
  https://mchromiak.github.io/articles/2026/Sep/17/Jev-Typed-Decisions-for-Enterprise-AI/

## Open reconstructions

- daseinlabs/open-jev — cached batched option scoring + System One API.
  https://github.com/daseinlabs/open-jev
- vinnylarouge/jevlike — option-query cross-attention trainer.
  https://github.com/vinnylarouge/jevlike
- cobanov/awesome-jev — ecosystem catalog, review 19 September 2026.
  https://github.com/cobanov/awesome-jev

## Datasets and community evals

Full mapping to Choice / Score / Noul is in [Datasets](10-datasets.md).

- BANKING77 — https://huggingface.co/datasets/PolyAI/banking77
- CLINC150 — https://github.com/clinc/oos-eval
- Wikispeedia — https://snap.stanford.edu/data/wikispeedia.html
- BoolQ — https://github.com/google-research-datasets/boolean-questions
- SST-5 / Stanford Sentiment Treebank
- Amazon ESCI — https://github.com/amazon-science/esci-data
- When2Call — https://huggingface.co/datasets/nvidia/When2Call
- MetaTool — https://github.com/howiehwong/metatool
- FEVER, MultiNLI
- BTZSC suite — https://huggingface.co/datasets/btzsc/btzsc
- jev-eval — https://github.com/4esv/jev-eval
- jev-benchmarks — https://github.com/AbdelStark/jev-benchmarks
- jev-decision-benchmarks — https://github.com/baibizhe/jev-decision-benchmarks

## Compute notes

- Google Colab FAQ. Resource limits, free-notebook cap of at most 12
  hours, GPU type not guaranteed. Retrieved 20 September 2026.
  https://research.google.com/colaboratory/faq.html
  How this repo uses it: [Hosted GPUs](11-colab.md).

## Related prior art (not TypeSafe)

These are listed because they occupy the same *problem*, not because
TypeSafe uses them:

- Proper scoring rules, temperature scaling, vector scaling, isotonic
  regression, Dirichlet calibration.
- Listwise learning-to-rank / cross-encoders.
- Selective prediction and risk–coverage curves.
- Research on rewarding calibrated confidence / "doubt."
- Facebook's *Reinforcement Learning from AI Feedback* paper, also
  abbreviated RLCD in some literature — **a different method**. Do not
  conflate it with TypeSafe's term.

## How to update this file

When TypeSafe publishes a paper, model card, or architecture note, add it
under primary sources and mark which unknowns in
[Public record](01-public-record.md) were closed. When an open project
reports a grouped, leakage-controlled eval with ECE and shuffled-context
controls, add it under reconstructions with the dataset name and date.
