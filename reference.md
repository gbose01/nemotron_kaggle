# 🏆 NVIDIA Nemotron Model Reasoning Challenge: Comprehensive Reference Guide

This reference guide compiles critical insights, training recipes, evaluation configurations, synthetic data strategies, and bug fixes for the **NVIDIA Nemotron Model Reasoning Challenge** (NVIDIA Nemotron-3-Nano-30B). 

---

## 1. 🐛 Data Bugs & Quality Issues

Analysis of the training data (`train.csv`) reveals significant noise and structural flaws. To achieve high reasoning accuracy, these issues must be addressed prior to training.

### A. Bit Manipulation Tasks: Result vs. Boxed Mismatch (50.5% Error Rate)
In the provided bit manipulation training examples, the intermediate reasoning steps frequently contradict the final boxed answer.
*   **The Issue:** In approximately **50.5%** of generated/provided bit manipulation traces, the final line `Result: X` does not match the `\boxed{Y}` answer.
*   **The Fix:** Filter out mismatching examples from your SFT dataset or regenerate the reasoning chains using a frontier model (e.g., Gemini 1.5 Pro) while asserting exact matches.

Here is the validation pattern from the community `check_bugs.py` script to identify these errors:
```python
import re

def check_bit_manipulation_mismatch(reasoning_text):
    result_match = re.search(r'Result: (\d+)', reasoning_text)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', reasoning_text)
    
    if result_match and boxed_match:
        computed = result_match.group(1)
        answer = boxed_match.group(1)
        if computed != answer:
            return False  # Mismatch found!
    return True  # Valid example
```

---

### B. Equation Transformation Tasks: Invalid 1:1 Mapping Assumption
Many standard baseline models use a simple character-to-character mapping approach (charmap) to solve equation transformations. This approach is fundamentally flawed due to data bugs in the dataset.
*   **The Issue:** Approximately **49%** of equation transformation training examples have input/output string length mismatches (`len(lhs) != len(rhs)`), meaning a 1:1 mapping is mathematically impossible.
*   **Uncovered Queries:** The example character mappings frequently fail to cover the characters present in the query string, resulting in unknown `?` characters in predictions.
*   **The Fix:** Detect length mismatches and filter out examples where the examples do not cover the query alphabet, or train a sequence-to-sequence transformer model rather than a rigid charmap parser.

Here is the detection logic from `check_bugs.py`:
```python
def check_equation_transformation_quality(examples, query):
    # 1. Check input/output length mismatches
    for lhs, rhs in examples:
        if len(lhs) != len(rhs):
            return "length_mismatch"
            
    # 2. Check for unknown characters in the query
    char_map = {}
    for lhs, rhs in examples:
        for j in range(min(len(lhs), len(rhs))):
            if lhs[j] not in char_map:
                char_map[lhs[j]] = rhs[j]
                
    result = "".join(char_map.get(c, "?") for c in query)
    if "?" in result:
        return "unknown_query_chars"
        
    # 3. Verify the charmap can successfully reconstruct all example targets
    for lhs, rhs in examples:
        predicted = "".join(char_map.get(c, "?") for c in lhs[:len(rhs)])
        if predicted != rhs:
            return "charmap_reconstruction_fail"
            
    return "valid"
```

---

## 2. 🍳 Training Recipes & GRPO Configurations

The reasoning capabilities of Nemotron-3-Nano are aligned using **GRPO (Group Relative Policy Optimization)**. The model utilizes a Hybrid Mamba-Transformer sparse MoE architecture.

### A. Core GRPO Hyperparameters
GRPO optimizes policy outputs by evaluating groups of generations per prompt to calculate relative advantage, avoiding the need for a separate critic model.

| Hyperparameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `grpo.num_prompts_per_step` | `128` | Number of unique prompts in a single step |
| `grpo.num_generations_per_prompt` | `16` | Group size ($G$) for relative advantage calculation |
| `grpo.normalize_rewards` | `true` | Normalizes rewards across the group of 16 generations |
| `grpo.use_leave_one_out_baseline`| `true` | Calculates baseline by excluding the current sample |
| `policy.train_global_batch_size` | `2048` | Total global batch size for the policy model |
| `policy.train_micro_batch_size` | `1` | Micro-batch size per GPU device |
| `policy.precision` | `"bfloat16"` | High-precision standard training format |
| `optimizer.lr` | `3e-6` | Adam learning rate (keep constant) |
| `loss_fn.ratio_clip_min` | `0.2` | Policy ratio lower clip bound |
| `loss_fn.ratio_clip_max` | `0.28` | Policy ratio upper clip bound |
| `loss_fn.token_level_loss` | `true` | Computes loss at the individual token level |

---

### B. Megatron-Parallelism Settings (Model Scale-Up)
Because the 30B A3B is a sparse MoE model, expert parallelism and model tensor parallelism must be configured correctly to prevent memory overflow:
```yaml
policy:
  megatron_cfg:
    enabled: true
    activation_checkpointing: true
    tensor_model_parallel_size: 2
    expert_tensor_parallel_size: 1
    expert_model_parallel_size: 8
    pipeline_model_parallel_size: 2
    context_parallel_size: 4
    sequence_parallel: true
    freeze_moe_router: true
    moe_router_dtype: "fp32"
    moe_router_load_balancing_type: "none" # CRITICAL: "seq_aux_loss" causes logprob divergence in GRPO
    moe_router_enable_expert_bias: true
    apply_rope_fusion: true
    defer_fp32_logits: true
```

---

### C. GRPO Generation & Reasoning Parsers
To enforce the model to produce structured reasoning before outputting the final boxed answer, use a specialized **reasoning parser** in the generation config:
```yaml
policy:
  generation:
    backend: "vllm"
    temperature: 1.0
    top_p: 1.0
    vllm_cfg:
      tensor_parallel_size: 4
      expose_http_server: true
      http_server_serving_chat_kwargs:
        enable_auto_tools: true
        tool_parser: "qwen3_coder"
        reasoning_parser: "deepseek_r1"  # CRITICAL: Uses DeepSeek-R1 <think> parsing tags
```

---

### D. Reward Environments (NeMo-Gym)
Configure the reward functions to provide high-quality feedback signals to the GRPO policy. NeMo-Gym supports blending multiple reward servers:
```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - resources_servers/math_with_judge/configs/math_with_judge.yaml
      - resources_servers/code_gen/configs/code_gen.yaml
      - resources_servers/instruction_following/configs/instruction_following.yaml
      - resources_servers/structured_outputs/configs/structured_outputs_json.yaml
```

---

## 3. 📊 Evaluation Parameters

To guarantee that local validation results are fully aligned and reproducible with the competition's scoring engine, utilize the settings from `nano-v3-reproducibility.md`.

### A. Default Inference Settings
```yaml
inference:
  max_new_tokens: 131072
  temperature: 0.99999
  top_p: 0.99999
  parallelism: 512
  request_timeout: 3600  # Reasoning chains require high timeout limits
  max_retries: 10
```

### B. Benchmark-Specific Settings (Consensus Voting)
*   **Berkeley Function Calling (BFCL):** Disable client parsing; use `temperature: 0.6` and `top_p: 0.95`.
*   **AIME (Mathematics):** Use a math-specific prompt template and run `num_repeats: 64`. **Apply majority voting/consensus selection across these 64 samples** to maximize accuracy.
*   **GPQA / LiveCodeBench:** Set `num_repeats: 8` to evaluate `pass@1` via consensus scoring.
*   **RULER (Long-Context):** Set `temperature: 0.00001`, disable thinking/reasoning tags, and run with a local HuggingFace tokenizer.

---

## 4. 🧬 Synthetic Data & DataDesigner Pipelines

Augmenting the dataset with clean, verified reasoning chains is vital. NVIDIA provides two toolkits for this purpose:

### A. NeMo Data Designer
Used for **Synthetic Data Generation (SDG)**. You can define dependency-aware schema columns where logical puzzles are generated and validated programmatically:
```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder()

# 1. Sample logic puzzle types
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="puzzle_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["bit_manipulation", "algebraic_equations", "character_transformations"],
        ),
    )
)

# 2. Generate a reasoning query using an LLM
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="puzzle_prompt",
        model_alias="nvidia-text",
        prompt="Generate a complex logical reasoning puzzle of type {{ puzzle_type }}. Include step-by-step mathematical rules.",
    )
)
```
*   **Python Validators:** Wrap a validator column using a script similar to `check_bugs.py` to automatically filter out puzzles whose generated reasoning doesn't produce the target answer.

---

### B. NeMo Curator
Used for **Data Curation and Quality Filtering**. It is GPU-accelerated via RAPIDS and Ray, making it highly efficient for multi-node datasets:
*   **Deduplication:** Run exact or fuzzy MinHash LSH deduplication to remove near-identical reasoning puzzles.
*   **Quality Filtering:** Run modular quality classification stages (e.g., fastText, GPU quality classifiers) to filter out scrambled or poorly formatted text.

---

## 5. 💡 Winning Strategies & General Tips

To secure a top position in the Kaggle leaderboard, implement these expert strategies:

1.  **Establish strict SFT format compliance:** Ensure the base model is completely adjusted to wrapping its final answer inside a LaTeX `\boxed{}` command before applying GRPO.
2.  **Clean training data aggressively:** Remove all 50.5% bit manipulation mismatches and 49% equation transformation mismatches. Quality of CoT training signals far outweighs quantity.
3.  **Leverage DeepSeek-R1 parser:** Force the model to separate thinking from final responses by embedding `<think> ... </think>` tags inside the GRPO generation environment.
4.  **Use MCP Sandboxed Python Tool:** In tasks where mathematical computation or script execution is required, allow the model to evaluate Python code during the reasoning phase (as seen in the AIME/GPQA tools config).
5.  **Consensus Voting (Self-Consistency):** Generate multiple samples at a higher temperature (e.g., 16 or 64 samples) during test inference and use majority voting to select the most common boxed value as the final submission answer.

---
TAG=agy
CONV=353f1f23-1b80-41ff-9831-45578d89d34c
