# ResonanceIndex3D — Edge-Native Graph-Aware Sparse Memory Engine

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![NumPy](https://img.shields.io/badge/numpy-only-lightgrey) ![Latency](https://img.shields.io/badge/p50-~0.4ms%20@100k-green) ![License](https://img.shields.io/badge/license-MIT-yellow)

**ResonanceIndex3D** is a lightweight, zero-GPU retrieval engine for resource-constrained edge devices. It was built as the long-term associative memory core for **Jarvis** (a local cyber-physical assistant running next to quantized LLMs on a CPU-only laptop). It combines **lexical matching** (tokens, prefixes, character trigrams) and **knowledge-graph neighborhood blending** in a single sparse pass, with **bounded per-query work** and an index of roughly **0.4–0.65 KB per node**.

It is a **classical, quantum-inspired** design (complex-wave language, Born-rule scoring on a sparse real feature space). It does not use or simulate a quantum computer, and it makes no quantum-speedup claim.

---

## What it is good at, and what it is not

| | |
| --- | --- |
| **Strength** | Predictable sub-millisecond query latency on head-heavy multi-word queries, even at 100k nodes, with a small in-RAM numpy index. |
| **Strength** | One pass handles exact tokens, prefixes and character-level fuzziness, plus a precomputed neighbor blend (no BFS/DFS at query time). |
| **Weakness** | Ranking quality is below SQLite FTS5 / BM25 and **degrades as the corpus grows** (see Benchmark Suite v2). |
| **Weakness** | Index memory is linear in the number of nodes; only per-query compute is bounded. |

Use it when you want a tiny, dependency-light, latency-bounded associative lookup tied to a graph. Use SQLite FTS5 or Lucene when ranking quality over a large text corpus matters most.

---

## How it works

1. **Feature encoding.** Each node is a sparse TF-IDF weighted vector of atoms: full tokens (`t:`, weight 1), 4-character prefixes (`p:`, 0.8) and padded character trigrams (`g:`, 0.5), built from its name, subject and the file names in its SSD folder.
2. **Neighborhood superposition (startup).** Each node blends in a fraction of its neighbors' token atoms, so no graph walk happens at query time:

$$
|\Psi_i\rangle = \frac{\mathbf{v}_i + \alpha \sum_{j \in \mathcal{N}(i)} \mathbf{v}_j^{(\text{tokens})}}{\left\| \mathbf{v}_i + \alpha \sum_{j \in \mathcal{N}(i)} \mathbf{v}_j^{(\text{tokens})} \right\|_2}, \qquad \alpha = 0.35
$$

3. **Impact-ordered postings.** Posting lists live in contiguous numpy arrays, sorted by descending weight. A query reads at most `MAX_POSTINGS = 256` entries per atom, so the work per query is bounded.
4. **Born-rule scoring (query time).** With a normalized query wave $|q\rangle$:

$$
P_i = \left|\langle q \mid \Psi_i \rangle\right|^2
$$

5. **Lazy SSD loading.** Only node ids, metadata and the numpy index stay in RAM. Documents are read from disk only for matched or clicked nodes.

---

## Benchmark Suite v1 (self-comparison, `bench_search.py`)

Measured on the author's machine (Intel i7-12th Gen U, CPU only), each size in a fresh process, 300 random single-word queries. Recall is measured against the engine's own uncapped ("exact") mode, and precision against a trivial ground truth, so it says little about relevance quality.

| Nodes | Index size | Build | p50 | p95 | Linear scan p50 | Speedup vs scan | Recall@10 vs own exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 0.4 MB | 0.1 s | 0.078 ms | 0.209 ms | 0.048 ms | 0.6× | 1.00 |
| 10,000 | 3.8 MB | 1.0 s | 0.174 ms | 0.377 ms | 0.461 ms | 2.6× | 0.95 |
| 100,000 | 38.2 MB | 11.6 s | 0.267 ms | 0.479 ms | 5.631 ms | 21× | 0.993 |

Latency grew about 3.4× for a 100× increase in nodes: bounded, not strictly constant. The linear scan only loses to the index above roughly 1,000 nodes.

---

## Benchmark Suite v2 (head-to-head, `bench_search_v2.py`)

Compared against **SQLite FTS5** (OR and AND queries, `bm25()` ranking, prefix indexes, `optimize`, on-disk) and **`rank_bm25`** (`BM25Okapi`, whole-corpus `get_scores`) on a **Zipf-distributed corpus** (s = 1.07, Heaps-law vocabulary). All engines index the same text per node. Each engine is built in its own process. Top-10 results, `top_k = 10`, `min_resonance = 0` for Resonance.

> **Provenance:** the numbers below were produced in a Linux sandbox, one run per size, 200 queries per class, using a local line-for-line replica of `BM25Okapi` because the package was not installable there. They are **not** from the i7 above. Re-run `python bench_search_v2.py 1000 10000 100000` after `pip install rank_bm25` and publish your own numbers.

**Query classes.** `zipf`: 2–4 words drawn from the corpus Zipf distribution (latency, and top-10 overlap with FTS5-OR). `exact`, `typo`, `prefix`: 2–3 words taken from one target node, with one word either intact, missing one character, or cut to 4–5 letters; quality is success@10 of finding the target.

### Latency on Zipf multi-word queries, p50 / p95 (ms)

| Nodes | ResonanceIndex3D | FTS5 (OR) | FTS5 (AND) | rank_bm25 |
| --- | --- | --- | --- | --- |
| 1,000 | 0.20 / 0.30 | 0.30 / 0.69 | 0.018 / 0.084 | 0.28 / 0.40 |
| 10,000 | 0.31 / 0.54 | 2.9 / 8.1 | 0.041 / 0.45 | 3.0 / 5.0 |
| 100,000 | 0.37 / 0.62 | 21.8 / 85 | 0.18 / 4.1 | 37.6 / 66 |

### Known-item quality, success@10

| Nodes | Class | ResonanceIndex3D | FTS5 (OR) | FTS5 (AND) | rank_bm25 |
| --- | --- | --- | --- | --- | --- |
| 1,000 | exact | 0.905 | 0.930 | 0.935 | 0.940 |
| 1,000 | typo | 0.865 | 0.770 | 0.000 | 0.765 |
| 1,000 | prefix | 0.925 | 0.980 | 0.980 | 0.825 |
| 10,000 | exact | 0.795 | 0.890 | 0.895 | 0.875 |
| 10,000 | typo | 0.625 | 0.580 | 0.000 | 0.580 |
| 10,000 | prefix | 0.725 | 0.840 | 0.860 | 0.610 |
| 100,000 | exact | 0.535 | 0.750 | 0.780 | 0.745 |
| 100,000 | typo | 0.365 | 0.470 | 0.000 | 0.480 |
| 100,000 | prefix | 0.485 | 0.825 | 0.835 | 0.470 |

Top-10 overlap with FTS5-OR on Zipf queries (1k / 10k / 100k): Resonance 0.58 / 0.41 / 0.30, `rank_bm25` 0.82 / 0.77 / 0.75.

### Cost at 100,000 nodes

| Engine | Build | Index | RAM (RSS growth around build) |
| --- | --- | --- | --- |
| ResonanceIndex3D | 10.4 s | 63.5 MB numpy | ~248 MB (includes build-time transients) |
| FTS5 | 1.3 s | 19.7 MB on disk | ~3 MB |
| rank_bm25 | 0.7 s | in RAM | ~60 MB |

### What the results say

- **Latency:** at 100k nodes Resonance is roughly 58× faster than FTS5-OR and 100× faster than `rank_bm25` at the median, and its p95 stays under 1 ms. FTS5-AND is faster still, but it returns nothing when any word is mistyped.
- **Quality:** Resonance ranks worse than FTS5/BM25 on exact and prefix queries, and the gap widens with scale (exact success@10: 0.905 at 1k, 0.795 at 10k, 0.535 at 100k, against 0.75–0.94 for the baselines).
- **Typos:** Resonance has a small edge at 1k–10k nodes and falls behind at 100k. FTS5-OR still finds the target on typo queries when the other words are intact, which flatters it too.
- **Posting cap:** raising `MAX_POSTINGS` at 100k nodes to 1,024 and 4,096 lifted exact success@10 only from 0.535 to 0.555 and 0.575, while p50 latency rose from 0.37 ms to 0.88 ms and 2.6 ms. The cap alone does not explain the quality loss; the cause is still under investigation.
- **Neighbor blending:** on this corpus the random graph links unrelated nodes, so v2 cannot show a benefit from blending (with vs without neighbors differs by at most a few points). A test on a graph whose links reflect real topical relatedness is still needed.

---

## Limitations

- **Not O(1) space.** Index memory is O(N). Only per-query compute is bounded by the posting cap.
- **Approximate ranking.** The cap trades quality for latency. On the Zipf-corpus known-item tests success@10 falls to about 0.5 at 100k nodes, well below FTS5 and BM25.
- **Quantum-inspired only.** Wave vectors and Born-rule scoring are a scoring metaphor on a classical sparse index.
- **Not a full-text search replacement.** For keyword search over large prose corpora without graph structure, SQLite FTS5 or Apache Lucene (Block-Max WAND) are better.
- **Synthetic benchmarks.** All numbers come from generated names and files, not from a real knowledge base.

---

## Quickstart & Interactive 3D Brain UI

The repository includes a lightweight **Flask + Three.js** visualizer that renders your knowledge graph as an interactive 3D cortex connected to local SSD folders.

```bash
# 1. Clone the repository
git clone https://github.com/Anas-khemakhem/ResonanceIndex3D
cd ResonanceIndex3D

# 2. Install dependencies (Flask, NumPy, NetworkX)
pip install -r requirements.txt

# 3. Launch the memory server & 3D Brain UI
python app.py
# Open http://127.0.0.1:5000 in your browser
```

On first launch, `app.py` creates `./jarvis_ssd_memory/` with sample technical notes for 14 interconnected subject nodes, so keyword search, neighborhood resonance and lazy SSD retrieval work immediately.

### Reproduce the benchmarks

```bash
python bench_search.py 1000 10000 100000        # v1: vs linear scan
pip install rank_bm25                            # optional but recommended
python bench_search_v2.py 1000 10000 100000     # v2: vs SQLite FTS5 and rank_bm25
```

v2 writes `bench_v2_results.json` and `bench_v2_report.md`. Options: `--nq` (queries per class), `--cap` (postings cap), `--engines`.

---

## Roadmap

- [x] **v1.0 — Sparse graph-superposed index & 3D cortex UI**
- [x] **v1.1 — Benchmark Suite v2:** head-to-head against SQLite FTS5 and `rank_bm25` on Zipf-distributed multi-word queries (results above; to be re-run on the target i7)
- [ ] **v1.1.1 — Quality at scale:** find why known-item success@10 falls from 0.9 to 0.5 between 1k and 100k nodes, and evaluate on a graph with real topical links
- [ ] **v1.2 — Document segmentation & chunking pipeline:** markdown/PDF ingestion into SSD-backed sub-nodes
- [ ] **Integration:** context provider for **Jarvis Local RAG** (Qwen / DeepSeek via Ollama) and **3-DOF robotic arm** spatial memory

---

## License

MIT License — free to use, benchmark and integrate into your own edge AI or robotics projects.
