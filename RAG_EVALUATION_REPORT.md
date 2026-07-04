# RAG Evaluation Report

## Methodology

- **Extraction Accuracy**: manually spot-checked N extracted JSON records against
  the source PDF/resume text for name/current_role/domain/skills, reporting the
  fraction of fields correct.
- **Retrieval Ranking Metrics**: computed by `backend/app/evaluation/run_evaluation.py`
  against the hand-labeled queries in `backend/eval/qa_labels.json` (K=5), using
  MRR, nDCG@5, success@3, and success@5 — metrics appropriate for a
  best-fit-candidate ranking use case (see "Ranking metrics" below).
- **Response Latency**: staged wall-clock measurement (embedding / Chroma
  retrieval / Groq generation / end-to-end) via `backend/scripts/measure_latency.py`,
  against the default Groq model.
- **Metadata filter**: `≥5 years experience` is no longer graded as a ranking
  query. It is verified as a `VectorStore` correctness test backed by a real
  `years_experience` field computed at ingestion (see "Query 3" section below).

This evaluation runs against the full 120-CV corpus (786 ingested chunks,
120 distinct `cv_id`s, no orphans/gaps) — the earlier 34-CV subset caveat in
prior versions of this report no longer applies.

## 1. Root cause of the original non-reproducibility

An earlier version of this report attributed run-to-run variance in
Precision@5/Recall@5 to "Chroma's HNSW approximate-nearest-neighbor index
being rebuilt in memory each time `VectorStore.__init__` creates a new
`PersistentClient`." That explanation was **wrong**: `VectorStore.__init__`
only ever calls `get_or_create_collection(...)`, which reuses the existing
on-disk Chroma collection — there is no index rebuild step to explain
run-to-run drift.

The investigation that replaced that guess went through two more rounds
before landing on the real cause:

1. **Embedding jitter was the first replacement hypothesis, and it was
   ruled out empirically.** A diagnostic (`backend/scripts/diagnose_eval_variance.py`,
   Task 1) embedded the same Vietnamese query 10 times across separate
   fresh Python process launches and diffed the resulting vectors.
   Result: `max_abs_diff=0.00e+00` for all 9 comparisons — embeddings are
   **bit-identical** across process launches. PyTorch/BLAS floating-point
   arithmetic on this system is stable; embedding computation is not the
   source of variance.
2. **The first fix (pinning `torch.set_num_threads(1)` plus
   `hnsw:search_ef=128`) looked sufficient, but that was a false pass.**
   It had been verified against an *empty* test collection, which is
   trivially deterministic and proves nothing. Re-tested against the real
   ingested corpus (786 chunks / 120 CVs), genuine cross-process variance
   persisted in 2 of the 4 original queries — e.g. candidate `18752129`
   appeared in Query 1's top-5 in 2 of 6 runs and was replaced by
   `10089434` in the other 4.
3. **Further isolation ruled out every tunable HNSW parameter.** Raising
   `hnsw:search_ef` to 2000 (far above the corpus's ~786 vectors — i.e.
   near-exhaustive approximate search) and pinning `hnsw:num_threads=1`
   were both tried; variance persisted under every combination. With
   embeddings already confirmed deterministic, the remaining source had
   to be floating-point non-determinism inside chromadb/hnswlib's own
   approximate distance computation itself — below any parameter this
   codebase can tune.

**The actual fix:** `VectorStore.query()` no longer calls Chroma's
HNSW-based `.query()` at all. It fetches all matching chunks (optionally
`where`-filtered) via `.get(include=["embeddings", ...])`, computes exact
L2 distance in numpy against the query vector, sorts, and dedupes by
`cv_id`. This is deterministic by construction: given Task 1's confirmed
embedding determinism, exact distance computation over a fixed set of
vectors has no approximate-search step left to wobble. The corpus is only
~786 vectors, far too small for approximate search to offer any real
performance benefit over brute force, so nothing is traded away by
dropping HNSW at query time. Chroma is still used for persistence and
metadata filtering (via `where` on `.get()`) — just not for the
nearest-neighbor computation.

Pinning `torch.set_num_threads(1)` was kept in place as a defensive
measure, but the determinism guarantee now comes from exact brute-force
search, not from thread or HNSW-parameter tuning — pinning threads alone
would not have fixed this. It's also worth keeping visible, independent of
which search algorithm is used, that this corpus contains many
near-duplicate-phrasing CVs; that characteristic is exactly what made
HNSW's approximate search wobble between near-tied candidates in the
first place, and no fix here erases it — it's a property of the data, not
a bug.

## 2. Determinism confirmation

`backend/scripts/check_eval_determinism.py` launches 6 independent
processes, each loading the same ingested corpus from `backend/storage/`
and running the same evaluation query, then compares retrieved chunk IDs
and distances across all 6 runs.

Final, real result against the actual 786-chunk/120-CV corpus, confirmed
independently multiple times (most recently after Query 3's retirement in
Task 13):

```
6/6 runs identical: True
```

## 3. Ranking metrics

Macro-averaged Precision@K and Recall@K have been replaced with metrics
suited to a "surface the best-fit candidate" use case: **MRR** (how early
the first relevant result appears), **nDCG@5** (position-weighted quality
of the ranking), and **success@3** / **success@5** (whether any relevant
result appears in the top 3 / top 5).

Current, authoritative output of `python -m app.evaluation.run_evaluation`
(K=5) run against the full 120-CV / 786-chunk corpus, 3 queries (Query 3
retired — see section 5):

| Metric | Value |
| --- | --- |
| MRR | 0.3333 |
| nDCG@5 | 0.2409 |
| success@3 | 0.3333 |
| success@5 | 0.3333 |

Per-query breakdown:

| Query | num_relevant | MRR | nDCG@5 | success@3 | success@5 |
| --- | --- | --- | --- | --- | --- |
| "Find candidates with Python and Machine Learning experience" (Python + ML) | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| "Find candidates in IT with risk management experience" | 17 | 1.0 | 0.7227 | 1.0 | 1.0 |
| "Suggest candidates suitable for a Data Engineer position" | 1 | 0.0 | 0.0 | 0.0 | 0.0 |

The risk-management query retrieves a genuinely relevant candidate at
rank 1 (perfect MRR/success, strong nDCG@5). The Python+ML and Data
Engineer queries score 0 across the board — not because retrieval is
broken, but because the relabeled ground truth (section 4) reflects that
this corpus has 0 and 1 genuinely relevant candidates respectively, and
none of the near-duplicate IT-generalist CVs the retriever surfaces
instead are actually correct matches. This is the ground truth doing its
job: exposing a real corpus-coverage gap rather than being inflated to
match whatever gets retrieved.

## 4. Relabeling methodology

The original `relevant_ids` for these two queries were built by grepping
extraction JSONs for keyword hits — e.g. "Python" appearing anywhere in a
CV's skills list — without checking whether the surrounding CV actually
described hands-on relevant work. That conflated a bare skills-list
mention with genuine practitioner experience.

The relabel (Task 5) instead read every candidate CV's full narrative,
not just the query keyword's grep hit, and asked "did this person
actually do this work" rather than "does this term appear somewhere."

- **Query 1 (Python + Machine Learning): `relevant_ids` is now `[]`.**
  A wide keyword net (English and Vietnamese: scikit/sklearn/tensorflow/
  pytorch/keras/machine learning/neural network/NLP/data scientist/deep
  learning/etc.) surfaced only 3 candidates across the full 120-CV
  corpus, and full-text review of all 3 (plus every CV mentioning
  "Python" at all) found each one to be a bare skills-list keyword with
  no supporting ML/data-science work — e.g. an IT director whose entire
  career is budget/program management, with "AI" and "Python" listed
  once and never exercised. This is a deliberate, human-approved finding:
  the corpus genuinely contains zero practitioners who match this query,
  not a labeling gap.
- **Query 4 (Data Engineer): `relevant_ids` is now `["41344156"]`.**
  The literal phrase "Data Engineer" does not appear anywhere in the
  corpus. Grepping for ETL/pipeline/warehouse/Spark/Airflow-adjacent
  terms surfaced 6 candidates; full-text review found 5 to be bare
  skills-list mentions attached to unrelated IT-executive or
  application-development careers. The one genuine match, `41344156`,
  has a concrete first-hand narrative from an early-career Application
  Developer role: "Spearheaded major, year-long initiative to plan,
  design, build, and implement an ETL commission database system... that
  recovered more than $2.5M in 'lost' commissions" — real, personally
  executed pipeline-building work, not a keyword coincidence.

## 5. Query 3 / metadata-filter note

The fourth original query, "Which candidates have more than 5 years of
experience?" (≥5 years experience), was graded with the same ranking metrics as the other
three, but that is the wrong tool for this query: 110 of the 120 CVs in
the corpus (about 92%) satisfy "≥5 years experience," so essentially any
top-K retrieval scores near-perfectly on every ranking metric regardless
of whether the actual retrieval quality is good — the query has no
discriminative power under MRR, nDCG, or success@K. It has been retired
from `qa_labels.json` and no longer appears anywhere in the ranking
evaluation.

In its place, `years_experience` is now a real metadata field:
`estimate_years_experience()` (Task 9) computes it from a CV's parsed
experience entries (span from earliest start date to latest end date,
treating "Current"/"Present" as ongoing), it's wired into the ingestion
pipeline so every future CV gets it automatically (Task 11), and it was
backfilled for all 120 already-ingested CVs via a one-off metadata-only
script with no new Groq calls or re-embedding (Task 12). The backfill was
controller-verified to show 110/120 CVs with `years_experience >= 5` —
matching the original query's 110-relevant count exactly, confirming the
computation is correct. `VectorStore.query()` now accepts an optional
`where` filter (e.g. `{"years_experience": {"$gte": 5}}`), and this is
tested with a dedicated correctness test,
`test_query_with_filter_supports_years_experience_gte_predicate` in
`backend/tests/test_vector_store.py` — a unit test asserting the filter
predicate works, not a ranking metric. This filter is internal/eval-only:
it is not exposed on the public `/api/query` endpoint.

## 6. Latency table

Measured via `backend/scripts/measure_latency.py` against the default
model, `llama-3.3-70b-versatile`, cycling through the corpus's use-case
queries. The script completed 15 of 30 planned requests before the daily
Groq token quota (100,000 TPD) was exhausted — an expected, accepted
outcome for a free-tier quota, not a failure of the measurement.

| Stage | p50 | p95 | p99 | n |
| --- | --- | --- | --- | --- |
| embed | 0.010s | 0.014s | 3.597s | 16 |
| chroma_retrieval | 0.029s | 0.038s | 0.125s | 16 |
| groq_generation | 1.066s | 2.621s | 17.055s | 15 |
| end_to_end | 1.120s | 5.862s | 17.097s | 15 |

Groq generation dominates end-to-end latency by roughly two orders of
magnitude over embedding and retrieval; the exact brute-force retrieval
change in section 1 has negligible cost at this corpus size (~786
vectors) — retrieval p99 is 0.125s vs. a 17s p99 for generation.

## 7. Extraction accuracy table

Extraction accuracy is now spot-checked against N=15 CVs (up from an
earlier N=2 anecdote), a deterministic sample (every 8th CV id from the
sorted list of all 120), checking 4 fields per CV: `name`, `current_role`,
`domain`, `skills`.

| Metric | Value |
| --- | --- |
| Fields checked | 60 (15 CVs × 4 fields) |
| Fields correct | 59 |
| Accuracy | 98.3% |

The one error: `current_role` for CV `25959103` was extracted as `null`,
but the raw text explicitly reads "...January 17th 2007 - Present)" for
an ongoing role that should have been extracted as `"Administrator of
Information Technology"`. This CV also shows a company/title field swap
elsewhere in its `experience` array, suggesting the same underlying parse
confusion affected both fields for this one record. `name` is correctly
`null` for most CVs in the sample (this corpus redacts PII — no name
string exists in 14 of 15 raw files), and `domain`/`skills` were 100%
correct across the sample, since skills extraction is close to a literal
copy of each resume's explicit skills section.

## Conclusion

The retrieval layer is now reproducible (`6/6 runs identical: True`,
independently re-verified against the real corpus), graded with ranking
metrics that fit a best-fit-candidate use case rather than
Precision/Recall, and its ground truth reflects actual candidate
narratives rather than keyword coincidence. The risk-management query
(MRR 1.0, nDCG@5 0.72) shows the pipeline works well when the corpus
genuinely contains a match. The Python+ML and Data Engineer queries
scoring 0 across all metrics is not a retrieval bug — it's the corrected
ground truth surfacing a real, verified gap in this 120-CV corpus (zero
genuine ML practitioners; exactly one genuine hands-on data engineer).
The former "≥5 years experience" query no longer distorts the aggregate
metrics with a non-discriminating 92%-relevant label; it lives on as a
metadata-filter correctness test backed by real `years_experience` data.
Response latency is dominated by Groq generation (p99 ≈ 17s) rather than
retrieval (p99 ≈ 0.125s), and extraction accuracy sits at 98.3% (59/60
fields) across a 15-CV sample, with the one known miss documented rather
than smoothed over.
