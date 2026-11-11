Writing

# ResonanceIndex3D — Edge-Native Graph-Aware Sparse Memory Engine

\!Python \!NumPy \!Memory \!Latency

**ResonanceIndex3D** is an ultra-lightweight, zero-GPU semantic & topological retrieval engine designed for resource-constrained edge devices. Built originally as the long-term associative memory core for **Jarvis** (a local cyber-physical assistant running alongside quantized LLMs on a CPU-only laptop), it unifies **lexical fuzzy matching** and **3D knowledge-graph neighborhood proximity** into a single sparse vector pass—achieving **sub-millisecond bounded latency** with a **\~0.4 KB/node** RAM footprint.

---

## Why ResonanceIndex3D? (Architectural Trade-offs)

When running local LLMs (e.g., Qwen, DeepSeek via Ollama) and computer vision pipelines on a CPU-only laptop (`Intel i7-12th Gen U, 24GB RAM, no discrete GPU`), standard retrieval stacks create severe bottlenecks:

1. **Vector Databases (FAISS / ChromaDB) + Neural Embeddings:** Require hundreds of megabytes to gigabytes of RAM and heavy matrix multiplications that starve local LLMs of memory bandwidth.
2. **Pure Text Search (SQLite FTS5 / BM25):** Fast for isolated text, but blind to knowledge-graph topology—requiring a separate graph traversal (`BFS/DFS`) or relational `JOIN` to pull connected concepts.
3. **Exact Hash Maps (****`dict`****):** Fast for exact keys, but fail completely on prefixes, morphological variants, or minor typos without bloating storage.

**ResonanceIndex3D** bridges this gap by pre-computing **graph neighborhood superposition** over a compact, impact-ordered sparse inverted index backed by lazy SSD payload loading.

### Core Advantages

- **Zero-Hop Graph + Text Fusion:** Connected nodes blend a weighted fraction (`NEIGHBOR_WEIGHT = 0.35`) of their neighbors' lexical atoms at index build time. A single dot-product query retrieves both direct matches and topologically adjacent concepts with **zero graph traversal at query time**.
- **Bounded Soft Real-Time Latency:** Posting lists are sorted by descending TF-IDF impact weight at startup and capped at `MAX_POSTINGS = 256` per query atom, guaranteeing predictable sub-millisecond response times.
- **Built-in Prefix & Typo Resilience:** Simultaneously indexes full tokens (`t:`), 4-character prefixes (`p:`), and padded character trigrams (`g:`) inside contiguous NumPy arrays.
- **Strict RAM / SSD Separation:** Only node IDs, metadata, and compact NumPy posting arrays stay in RAM (\~38 MB for 100,000 nodes). Heavy documents remain on SSD and are read lazily only for top-ranked matches.

---


## Mathematical Formulation

### 1. Feature Encoding & Neighborhood Superposition (Startup)

Each node $i$ is represented as a sparse TF-IDF weighted atom vector $\mathbf{v}_i$.

To encode graph topology without runtime traversal, each node state $|\Psi_i\rangle$ is constructed as a normalized superposition of its own features and its direct neighbors $\mathcal{N}(i)$:

$$
|\Psi_i\rangle =
\frac{
\mathbf{v}_i + \alpha \sum_{j \in \mathcal{N}(i)}
\mathbf{v}_j^{(\text{tokens})}
}{
\left\|
\mathbf{v}_i + \alpha \sum_{j \in \mathcal{N}(i)}
\mathbf{v}_j^{(\text{tokens})}
\right\|_2
}
$$

### 2. Born-Rule Relevance Scoring (Query Time)

Given a normalized query wave $|q\rangle$, candidate postings are accumulated up to `MAX_POSTINGS` per atom, and nodes are ranked via squared amplitude overlap:

$$
P_i = \left|\langle q \mid \Psi_i \rangle\right|^2
$$

## Benchmarks

Evaluated in isolated fresh processes on an **Intel i7-12th Gen U (CPU only, no dGPU)** using `bench_search.py`:

| Graph Scale (Nodes) | Index RAM (`index_MB`) | Index Size / Node | Build Time | p50 Latency | p95 Latency | Linear Scan p50 | Speedup vs. Scan | Recall@10 (vs. Exact) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1,000** | 0.4 MB | \~0.40 KB | 0.1 s | **0.078 ms** | 0.112 ms | 0.056 ms | 0.7× | 0.998 |
| **10,000** | 3.8 MB | \~0.39 KB | 1.1 s | **0.145 ms** | 0.210 ms | 0.560 ms | **3.9×** | 0.954 |
| **100,000** | **38.0 MB** | **\~0.39 KB** | 11.6 s | **0.267 ms** | 0.389 ms | 5.610 ms | **21.0×** | 0.993 |

### Honest Engineering Limitations

- **Not****$\mathcal{O}(1)$****Asymptotic Space:** Index memory scales linearly at $\mathcal{O}(N)$ (\~0.4 KB/node). Only per-query compute is bounded via top-$K$ posting truncation (`MAX_POSTINGS`).
- **Recall vs. Cap Trade-off:** On ultra-frequent atoms exceeding `MAX_POSTINGS = 256`, early termination trades a small fraction of recall (\~1–5%) for deterministic sub-millisecond latency.
- **Not a Replacement for Full-Text Search Engines:** For pure keyword search over massive prose corpora without graph topology, established engines like **SQLite FTS5** or **Apache Lucene (Block-Max WAND)** remain superior.

---

## Quickstart & Interactive 3D Brain UI

The repository includes a lightweight **Flask + Three.js** visualizer that renders your knowledge graph as an interactive 3D neural cortex connected to local SSD folders.

bash

```bash
# 1. Clone the repository
git clone https://github.com/Anas-khemakhem/ResonanceIndex3D
cd ResonanceIndex3D

# 2. Install minimal dependencies (Flask, NumPy, NetworkX)
pip install -r requirements.txt

# 3. Launch the memory server & 3D Brain UI
python app.py
# Open http://127.0.0.1:5000 in your browser
```

On first launch, `app.py` automatically initializes `./jarvis_ssd_memory/` with sample technical notes mapped to 14 interconnected subject nodes so you can test keyword search, neighborhood resonance, and lazy SSD file retrieval immediately.

---

## Roadmap (Modular Jarvis Ecosystem)

This repository is the **Memory Core** of a modular, edge-native AI architecture:

- [x]**v1.0 — Sparse Graph-Superposed Index & 3D Cortex UI** _(Current)_
- [ ]**v1.1 — Benchmark Suite v2:** Direct head-to-head comparison against `SQLite FTS5` and `rank_bm25` on Zipf-distributed multi-word queries.
- [ ]**v1.2 — Document Segmentation & Chunking Pipeline:** Automatic markdown/PDF ingestion into SSD-backed semantic sub-nodes.
- [ ]**Integration:** Plug-and-play context provider for **Jarvis Local RAG** (`Qwen` / `DeepSeek` via Ollama) and **3-DOF Robotic Arm** spatial memory.

---

## License

MIT License — free to use, benchmark, and integrate into your own edge AI or robotics projects.
