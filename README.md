# Patent Prior-Art Agent

Conversational patent search and prior-art analysis built on PANORAMA. The project supports:

- benchmark evaluation on PANORAMA `PAR4PC`
- free-text patent search over a local persistent index
- grounded QA over retrieved patent evidence
- multi-turn follow-up handling with working-set reuse
- claim decomposition, evidence extraction, and verification for benchmark cases

This repo is set up for local development and GitHub sharing. Secrets are not committed.

## What It Does

There are two main modes:

1. **Free-text Search**
   - enter a claim, invention description, or vague query
   - retrieve related patents
   - answer from retrieved evidence
   - handle follow-up questions like:
     - `Which of those includes access control?`
     - `Compare the top two`
     - `If I combine that with smart invitations, what should I inspect next?`

2. **Benchmark Analysis**
   - run on labeled PANORAMA `PAR4PC` cases
   - rank A-H candidate patents
   - decompose the claim
   - extract evidence
   - render a structured report

## Repo Layout

```text
app.py                     Streamlit UI
src/
  build_patent_index.py    Build persistent FAISS index
  run_free_text_demo.py    Single-turn CLI QA demo
  run_conversation_demo.py Multi-turn CLI conversation demo
  evaluate_par4pc.py       Benchmark evaluation
  graph.py                 Benchmark workflow
  llm_tools.py             OpenAI-backed grounded answer / verification helpers
  query_planner.py         Follow-up intent + retrieval policy
  persistent_index.py      Persistent FAISS index I/O
scripts/
  run_app.sh               Streamlit launcher
environment.yml
requirements.txt
.env.example
```

## Setup

### 1. Create or activate the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate patent-agent
```

Or install manually:

```bash
pip install -r requirements.txt
```

### 2. Optional OpenAI setup

Copy the example env file:

```bash
cp .env.example .env
```

Then fill in:

```bash
OPENAI_API_KEY=your_key_here
PATENT_AGENT_MODEL=gpt-4o-mini
```

`.env` is gitignored.

### 3. Clone PANORAMA separately

This repo expects the PANORAMA benchmark samples to be available locally, for example as a sibling folder:

```text
your-workspace/
  PANORAMA/
  patent-agent/
```

Default local benchmark path used by the code:

```text
../PANORAMA/data/benchmark/par4pc
```

## Quick Start

### Build a persistent local demo index

This builds a small combined demo index from local PANORAMA sample patents plus a small Hub-backed PAR4PC slice.

```bash
python -m src.build_patent_index \
  --pool-source combined \
  --hub-rows-per-split 50 \
  --index-dir data/indexes/par4pc_patentsberta_demo
```

### Run the UI

```bash
./scripts/run_app.sh
```

Recommended demo settings:

- `Mode = Free-text Search`
- `Retrieval method = local-embedding`
- `Free-text patent pool = Persistent local index`
- `Persistent index directory = data/indexes/par4pc_patentsberta_demo`

Then click `Use example query`.

## CLI Demos

### Single-turn free-text QA

```bash
python -m src.run_free_text_demo
```

### Multi-turn conversation demo

```bash
python -m src.run_conversation_demo
```

### Benchmark evaluation

```bash
python -m src.evaluate_par4pc --retrieval-method local-embedding
```

Expected local pilot result on the bundled sample set:

```text
hit@1: 0.800
hit@3: 1.000
recall@3: 1.000
exact@|gold|: 0.700
```

### Retrieval comparison

```bash
python -m src.compare_retrieval --output outputs/retrieval_comparison.csv
```

## Input Guidance

Best input for retrieval:

- patent claim text, e.g. `1. A method comprising: ...`

Also supported:

- invention description
- vague technical search query
- follow-up question over prior retrieved results

## Conversation Behavior

Free-text mode uses a simple planner:

- **new search** -> retrieve over the selected corpus
- **follow-up / aspect filter** -> rerank the current working patent set
- **comparison** -> compare within current results
- **combination exploration** -> retrieve again with prior context included

The planner decision is shown in the UI.

## Notes

- Without `OPENAI_API_KEY`, the system still works using heuristic grounded answers.
- With `OPENAI_API_KEY`, free-text mode can generate LLM-grounded answers from retrieved evidence only.
- This tool is for technical prior-art exploration and evidence-grounded search, not legal advice.

## Preparing for GitHub

Secrets and generated artifacts are excluded:

- `.env`
- `outputs/`
- `data/indexes/`
- local caches and virtual environments

To publish:

```bash
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```
