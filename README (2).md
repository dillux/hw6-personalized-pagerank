# HW6 — Graph-Based Recommendations with Personalized PageRank

LLM4Rec, HSE University · Week 6 · Dilara Faritova (dafaritova@edu.hse.ru)

Implements the seven `TODO(hw6)` functions over a user–movie bipartite graph built from
MovieLens 100K positive training ratings, and compares popularity, global PageRank, Personalized
PageRank and the previous matrix-factorization model on one shared retrieval cohort.

## Run

The data layer uses `fetch`, so the page must be served over HTTP:

```bash
python3 -m http.server 8001
# then open http://localhost:8001/
```

Order of operations in the UI:

1. **Build graph** — confirm the Graph summary shows 943 users / 1 611 movies / 2 554 nodes
   before anything else. The graph is not built automatically, and PageRank and PPR return zero
   without it.
2. **Run tests** — expect 12 passed / 0 failed / 0 pending.
3. Set **alpha** to 0.50 and **Method** to `ppr`.
4. **Recommend Top-10** for users 615 and 897.
5. **Run two-user demo (alpha / seed)**.
6. **Evaluate Recall@10 — all four methods** (slow; see Known issues).

## Dependencies as actually used

| Component | Version | Source |
|---|---|---|
| TensorFlow.js | 4.22.0 | CDN, loaded by `index.html` |
| d3 | vendored | `vendor/d3.min.js` |
| Backend | WebGL | reported by the page; falls back to CPU |
| Python (cross-check only) | 3.x + NumPy | not required to run the app |

## What was implemented

`graph.js` — `buildGraph`, `rowNormalize`, `pageRank`, `personalizedPageRank`
`script.js` — `movieScores`, `recommendTopK`, `recallAtK`

`data.js`, `index.html` and `style.css` are unmodified from the starter code. The Week 5 directory
and the original dataset files were not touched.

## Graph

| Quantity | Value |
|---|---:|
| User nodes / movie nodes | 943 / 1 611 |
| Nodes (no padding node) | 2 554 |
| Positive training edges (rating ≥ 4.0) | 45 602 |
| Zero-degree users / movies | 1 / 253 |

Dangling mass is redistributed through the restart vector (`alpha * (Q^T p + d * v) + (1-alpha) v`)
rather than dropped, so the distribution keeps summing to one.

## Results

Selected restart parameter: **alpha = 0.50**, chosen on validation Recall@10 only.

| Method | test Recall@10 | mean Top-10 popularity |
|---|---:|---:|
| Popularity | 0.0760 | 272.0 |
| Global PageRank | 0.0797 | 257.8 |
| Personalized PageRank | **0.1122** | 244.2 |
| Matrix factorization | 0.0211 | 96.8 |

Shared cohort: 861 users. Exclusions: 13 relevant test items absent from the training candidate
pool, 81 users without an eligible relevant item, 1 user with no positive training edge.

## Known issues

- **The evaluation path returns zero for `pagerank` and `ppr`.** `recommendTopK` produces correct
  Top-10 lists, but `hw6ScoreVector` → `hw6RunPPR` fails to resolve a raw user id to its graph node
  and yields an empty score vector, which the harness turns into 0.0000. The PPR and PageRank rows
  above therefore come from an independent Python reimplementation whose Top-10 output is identical
  to this application's, position for position. Not fixed before the deadline.
- **The in-browser alpha sweep does not finish in reasonable time.** PPR is computed once per
  (user, alpha) pair — 861 runs per evaluation, ~3 400 for the sweep — and the UI thread does not
  yield. A batched formulation runs the same sweep in seconds, but batching would require editing
  the adapters and caches, which the assignment lists as protected scaffolding.
- **The Top-10 table leaves Title and Genres blank**, and the exported `recommendations.csv` from
  the application is empty. The `recommendations.csv` included here was generated from the
  independent implementation with titles and genres filled in; its ids, order and popularity values
  match the application's output exactly.
- **Two exclusion counters disagree with their own data**: the "unseen excluded" field reports 12
  while the exported id list contains 13 entries, and "no eligible user" reports 0 while 81 users
  are in fact excluded (943 − 1 − 81 = 861, the cohort size the application itself reports).

## Files in this repository

```
graph.js      buildGraph, rowNormalize, pageRank, personalizedPageRank  (implemented)
script.js     movieScores, recommendTopK, recallAtK                     (implemented)
data.js       provided, unmodified
index.html    provided, unmodified
style.css     provided, unmodified
readme.md     the assignment specification, as provided
vendor/       d3, vendored by the starter
u.data        MovieLens 100K ratings
u.item        MovieLens 100K item metadata
```

## Data

MovieLens 100K, GroupLens, University of Minnesota.
F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," *ACM Trans.
Interact. Intell. Syst.*, vol. 5, no. 4, art. 19, Dec. 2015. doi: 10.1145/2827872
