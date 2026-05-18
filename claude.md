# NVIDIA Nemotron Model Reasoning Challenge Instructions

Welcome! This instructions file (`claude.md`) guides all AI assistant interactions in this workspace. The primary goal of this project is to help Gaurab build and submit a winning entry to the **NVIDIA Nemotron Model Reasoning Challenge**.

---

## 🎯 Mission & Core Objective
Achieve the highest possible reasoning accuracy on a dataset of logical reasoning puzzles (bit manipulation, algebraic equations, transformations) using the **NVIDIA Nemotron-3-Nano-30B** model.

---

## 📋 Rules, Specs & Constraints
Ensure all proposed solutions, fine-tuning jobs, and formatting pipelines strictly adhere to these official competition constraints:

*   **Base Model:** `NVIDIA Nemotron-3-Nano-30B`
*   **Submission Format:** A ZIP archive `submission.zip` containing a **LoRA adapter** (including `adapter_config.json`).
*   **LoRA Constraints:**
    *   Rank ($r$) must be **at most 32**.
    *   Must be compatible with the vLLM inference engine used for scoring.
*   **Token & Length Limits:**
    *   Maximum generation tokens: **7680**.
    *   Maximum model sequence length: **8192**.
*   **Evaluation Output Format:**
    *   The model must output its final target answer wrapped inside a LaTeX `\boxed{}` command (e.g., `\boxed{42}` or `\boxed{True}`).
    *   Evaluation extracts the answer from this box, with exact string matching or a relative numerical tolerance of $10^{-2}$.
*   **Key Dates:**
    *   **Entry & Team Merger Deadline:** June 8, 2026.
    *   **Competition End Date:** June 15, 2026.

---

## 🗺️ Strategic Roadmap & Execution Plan

### Phase 1: Setup & Local Evaluation (Baseline)
1.  **Download Dataset:** Pull `train.csv` and `test.csv` from Kaggle.
2.  **Data Exploration:** Analyze puzzle distributions, input formats, and reasoning lengths.
3.  **Local Evaluation Pipeline:**
    *   Set up a local evaluation script using the same exact extraction regex/parser as Kaggle to extract answers from `\boxed{}`.
    *   Establish a fast, cheap validation set (e.g., a stratified split or cross-validation).
4.  **Baseline Prompting:** Create a simple zero-shot/few-shot baseline with the base model and evaluate local accuracy.

### Phase 2: Advanced Prompting & Inference Strategy
1.  **Chain of Thought (CoT):** Design structured reasoning steps (e.g., step-by-step deduction, Scratchpad) to force the model to decompose the logical puzzles.
2.  **In-Context Learning (ICL):** Select diverse few-shot exemplars dynamically based on puzzle similarity.
3.  **Self-Consistency / Voting:** Explore majority voting across multiple temperature-sampled chains if inference time limits allow.

### Phase 3: Data Curation & Synthetic Data Generation
1.  **Augmentation:** Generate synthetic logic puzzles with known answers to augment training data.
2.  **Reasoning Demos:** Use stronger frontier models (e.g., Gemini 1.5 Pro, Claude 3.5 Sonnet) to generate high-quality Chain of Thought reasoning paths for the `train.csv` puzzles.
3.  **Filtering:** Clean and curate training inputs, filtering out incorrect reasoning paths.

### Phase 4: LoRA Fine-Tuning (Rank <= 32)
1.  **Supervised Fine-Tuning (SFT):** Fine-tune the Nemotron base model on the curated logical reasoning dataset with target outputs formatted as `<Reasoning Chain> ... \boxed{Answer}`.
2.  **Hyperparameter Tuning:**
    *   Experiment with LoRA ranks ($r \in \{8, 16, 32\}$), alpha values, and target modules (e.g., Q, K, V, O, Gate, Up, Down projections).
    *   Monitor learning rates, batch sizes, and epoch count to prevent overfitting on simple puzzle structures.

### Phase 5: Packaging & Verification
1.  **LoRA Adapter Export:** Ensure the adapter is correctly saved with `adapter_config.json`.
2.  **Packaging Script:** Create a robust script to generate `submission.zip` exactly as required.
3.  **Sanity Check:** Validate the output structure and run a dry-run evaluation using a local vLLM container.

---

## 🧠 Assistant Principles & Best Practices
When working in this workspace, follow these principles:

1.  **Iterate Rapidly & Log Results:** Always log hyperparameters, local validation accuracy, and Kaggle submission scores in a centralized log/registry (e.g., `experiments.md`).
2.  **Strict Compliance:** Verify that any LoRA adapter configuration is explicitly $\le 32$ in rank.
3.  **Clean Code:** Build modular, readable Python pipelines for data processing, local evaluation, and model training.
4.  **LaTeX Box Assertion:** Write assertions or test suites to ensure generated responses reliably contain a match for `\boxed{...}`.
