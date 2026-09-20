# API contract

Reconstructed from TypeSafe's public HTTP reference, primitive pages, and
SDK types as of 19 September 2026. An open engine that wants drop-in
clients should match this shape. It should not copy TypeSafe branding or
claim compatibility beyond the schema.

Endpoint (hosted):

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

## Request

```json
{
  "model": "jev-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {
        "true": "Explicitly time-sensitive",
        "false": "No urgency expressed"
      }
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

### Fields

| Field | Required | Notes |
|---|---|---|
| `state` | Yes | String, object, or array of text. Canonicalize object key order before hashing or caching. |
| `model` | Yes in HTTP examples | Hosted default is `jev-latest`. An open server can ignore or echo this. |
| `questions` | Yes | Map of id → question. Ids are **not** model inputs. |

`instructions` and each criteria entry may be a string, object, or array.
Field names inside structured criteria are not reserved; the model sees
those names as labels. Start with strings.

Question ids should be treated as opaque handles. Write the complete
question in `instructions` even if the id looks self-explanatory.

When state is an object, TypeSafe recommends pointing at fields with
backticked JSON-path style names: `` `ticket.messages[0].text` ``.

## Choice

Hard constraints for an open implementation:

- At least 2 options; at most 255.
- Option keys are the values the model may return.
- Descriptions (or `null` if the key is already clear) go to the model
  along with the keys.
- Always consider an `other`, `unknown`, or `insufficient_information`
  option. Without one, probability is forced onto the closest listed key.

Desired output:

```json
{
  "type": "choice",
  "choice": "technical",
  "probabilities": {
    "billing": 0.08,
    "technical": 0.85,
    "sales": 0.07
  },
  "confidence": 0.82
}
```

`choice` is `argmax` of `probabilities`. The distribution must sum to 1
(within ordinary float tolerance). `confidence` ∈ [0, 1] is derived from
the shape of that distribution.

**Inference (open default).** TypeSafe has not published the confidence
formula. `open-jev` uses normalized entropy:

```text
H(p) = -Σ p_j log p_j
confidence(p) = 1 - H(p) / log K
```

This is 0 for uniform and 1 for one-hot. It measures **decisiveness**, not
calibration. Document the formula you ship; do not pretend it is TypeSafe's.

## Noul

A binary proposition. Internally, treat it as a two-class choice over
`{false, true}`:

```text
p_true = softmax(s_true, s_false)_true
```

Request:

```json
{
  "type": "noul",
  "instructions": "Does this request require human approval before execution?",
  "criteria": {
    "true": "Irreversible, high-impact, privileged, or financially consequential.",
    "false": "Read-only, reversible, or low-impact."
  }
}
```

`criteria` is optional. Official Noul answers carry only:

```json
{
  "type": "noul",
  "noul": 0.972
}
```

There is **no** separate `confidence`. Near 1 is a strong yes, near 0 a
strong no, near 0.5 uncertain. The Elixir client derives
`max(noul, 1 - noul)` locally so a shared `gate()` helper works; that
derived value never drops below 0.5.

Do not treat a high Noul as authorization. Policy gates for spend,
privacy, and irreversible writes stay in deterministic code.

## Score

Ordered categorical distribution over levels `k ∈ {0, …, K-1}`, with
`K ∈ [2, 10]`.

**Fact.** TypeSafe states that each level is evaluated separately and that
the model does not see the level number or its neighbours. The continuous
score is the expectation of the index:

```text
score = Σ_{k=0}^{K-1} k · p_k
```

If you need a 0–100 scale, that is application code:

```text
score_0_100 = 100 · score / (K - 1)
```

Response:

```json
{
  "type": "score",
  "score": 1.6,
  "legend": {
    "0": "Calm",
    "1": "Frustrated",
    "2": "Very angry"
  },
  "probabilities": {
    "0": 0.05,
    "1": 0.30,
    "2": 0.65
  },
  "confidence": 0.78
}
```

HTTP JSON keys `legend` and `probabilities` by stringified indices. The
Python SDK exposes integer keys.

The same expected score can come from very different distributions
(all mass on 1 vs 50/50 on 0 and 2). Operational policy should threshold
`P(y ≥ k)`, not only `E[y]`.

Write levels as situations, not as "low / medium / high." Because levels
are scored independently, relative phrasing ("worse than the previous
level") does not work.

## Full response envelope

```json
{
  "model": "jev-1.13.0",
  "answers": { "...": "..." },
  "usage": {
    "input_tokens": 312,
    "output_tokens": 48
  }
}
```

Hosted `usage.output_tokens` is still reported even though output is
unmetered. An open logprob scorer can count candidate-label tokens here.

## Errors to implement

| Status | Meaning |
|---|---|
| 401 | Missing or invalid key |
| 422 | Schema failure (missing field, <2 options, >255 choices, >10 score levels) |
| 429 | Rate limit |
| 529 | Overloaded (TypeSafe-specific; optional for a local server) |

## TypeScript contract for an open server

```ts
type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue };

type ChoiceQuestion = {
  type: "choice";
  instructions: JsonValue;
  criteria: Record<string, JsonValue | null>;
};

type ScoreQuestion = {
  type: "score";
  instructions: JsonValue;
  criteria: JsonValue[]; // length 2..10
};

type NoulQuestion = {
  type: "noul";
  instructions: JsonValue;
  criteria?: { true?: JsonValue; false?: JsonValue };
};

type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

type SystemOneRequest = {
  model?: string;
  state: JsonValue;
  questions: Record<string, Question>;
};
```

Machine-readable copies: [`../schemas/systemone.request.schema.json`](../schemas/systemone.request.schema.json)
and [`../schemas/systemone.response.schema.json`](../schemas/systemone.response.schema.json).

## Programming-model implications

1. Require at least two choices; enforce the 255 / 10 caps.
2. Canonicalize serialization before prefix caching.
3. Keep long descriptions separate from stable option IDs.
4. Support `unknown` / `defer` where semantically appropriate.
5. Never let model output determine authorization boundaries.
6. Prefer one request with many independent questions over a chain of
   calls. TypeSafe's parallel-questions cookbook reports ~10× faster and
   ~12× cheaper for 13 questions batched versus 13 separate calls, with
   unchanged answers—**measurement**, vendor cookbook.
7. Make a second request only when the option set or state cannot be
   known until the first answer exists (hierarchical classification,
   skill shortlist rerank, structure recovery).
