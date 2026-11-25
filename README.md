# HrudayAI

An AI-powered rural healthcare voice intake, disease prediction, and voice response system built with free, open-source Python tools and models.

## Overview

HrudayAI is organized into three modular phases connected through reusable Python interfaces in `shared/`:

1. **Phase 1 (`phase1_voice/`)** — Voice input capture, audio preprocessing, and open-source speech-to-text / acoustic feature extraction.
2. **Phase 2 (`phase2_prediction/`)** — Clinical feature structuring and open-source disease prediction / risk assessment models.
3. **Phase 3 (`phase3_voice_output/`)** — Accessible response generation and open-source text-to-speech (TTS) synthesis.

## Repository Structure

```text
HrudayAI/
├── phase1_voice/           # Phase 1: Voice intake & transcription/audio features
│   ├── src/                # Reusable Python modules for Phase 1
│   ├── notebooks/          # Google Colab notebooks for Phase 1 experiments
│   ├── tests/              # Unit & integration tests for Phase 1
│   └── configs/            # Non-sensitive configuration files for Phase 1
├── phase2_prediction/      # Phase 2: Symptom/feature processing & disease prediction
│   ├── src/                # Reusable Python modules for Phase 2
│   ├── notebooks/          # Google Colab notebooks for Phase 2 experiments
│   └── tests/              # Unit & integration tests for Phase 2
├── phase3_voice_output/    # Phase 3: Response formatting & open-source TTS output
│   ├── src/                # Reusable Python modules for Phase 3
│   ├── notebooks/          # Google Colab notebooks for Phase 3 experiments
│   └── tests/              # Unit & integration tests for Phase 3
├── shared/                 # Cross-phase Python utilities, schemas & data contracts
├── docs/                   # Architecture notes, clinical workflows & documentation
├── .gitignore              # Blocks secrets, patient data, audio, weights & datasets
└── README.md               # Project overview and development workflow
```

## Core Principles

- **Free & Open-Source Only:** Prioritize open-source libraries and open-weight models; do not rely on paid APIs.
- **Modular Python Design:** Each phase exposes clean, reusable Python functions and shared data contracts via `shared/` so phases can be developed, tested, and composed independently.
- **Privacy & Security First:** Never commit credentials, private patient data, audio recordings, model weights, or generated datasets to Git.

## Development Workflow

- **GitHub (Source of Truth):** All source code, configuration templates, documentation, tests, and clean notebooks are versioned in this repository.
- **Google Antigravity (Code & Architecture):** Write, refactor, and test modular Python code in `phase*/src/`, `shared/`, and `tests/`, and commit changes to GitHub.
- **Google Colab (AI/ML Execution):** Clone or pull the repository from GitHub inside Google Colab to run GPU/TPU workloads using the notebooks in `phase*/notebooks/`. Store datasets, audio samples, and trained weights in private external storage (e.g., mounted Drive)—never push them to GitHub.

## Current Status

- Initial modular repository structure, documentation, and `.gitignore` are set up.
- Phase 1 dependencies (`phase1_voice/requirements.txt`) will be added once the Phase 1 open-source stack is selected.
- Model downloading and training have not started yet.
