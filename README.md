# 📝 SOP Generator — Instruction Fine-Tuning for Statement of Purpose Drafting

Fine-tuning **Llama 3.2 1B Instruct** with **LoRA** to generate structured Statement of Purpose (SOP) drafts for graduate applications. Covers the full pipeline: dataset construction → instruction formatting → parameter-efficient fine-tuning → inference.

---

## 📊 Results

| Metric | Value |
|---|---|
| Base model | Llama 3.2 1B Instruct |
| Method | LoRA (r=16, α=32, dropout=0.05) |
| Trainable parameters | **3.4M / 1.24B → 0.275%** |
| Training steps | 250 (5 epochs) |
| **Train loss** | 2.177 → **1.864** |
| **Validation loss** | 2.328 → **2.082** |
| Training time | ~12 min on a single Tesla T4 |
| Adapter size | 14 MB |

**Loss trajectory:**

| Step | Train Loss | Val Loss |
|---|---|---|
| 50 | 2.177 | 2.328 |
| 100 | 2.029 | 2.237 |
| 150 | 2.055 | 2.147 |
| 200 | 1.945 | 2.089 |
| 250 | 1.864 | **2.082** |

Validation loss decreased monotonically and had **not plateaued** at step 250 — the model was still improving when training stopped. See [Future Work](#-future-work).

---

## 🔍 Did it actually learn the task?

Loss says the model predicts the reference tokens better. It does not say the *behaviour* changed — and behaviour is what the fine-tune was for. So both models were run on the same 20 validation prompts, with identical generation settings and identical per-example seeds, and the outputs compared directly.

| | Base | Fine-tuned |
|---|---|---|
| Opens with conversational preamble | **100%** | **0%** |
| Bullet markers per output | 2.05 | **0.0** |
| First-person sentence ratio | 0.816 | **0.878** |
| Mean length (words) | 604 | 674 |
| Unfilled `[Placeholder]` slots | 90% | 70% |
| ROUGE-L vs reference | 0.181 | 0.189 |

**The headline is the first row.** Every base output addresses the reader before writing anything. None of the fine-tuned ones do.

### Same prompt, both models

> *Write a Statement of Purpose for a Master's program in Computational Biology.*

**Base — Llama 3.2 1B Instruct**
> Here's a sample Statement of Purpose for a Master's program in Computational Biology:
>
> **Introduction**
>
> As a lifelong enthusiast of the intersection of life sciences and technology, I am excited to pursue a Master's degree in Computational Biology…

**Fine-tuned — + SOP LoRA adapter**
> STATEMENT OF PURPOSE:
>
> My interest in computational biology was sparked during my undergraduate studies in biochemistry, where I was introduced to the intricate mechanisms of biological processes. My curiosity about how these processes translate into computational models fueled my desire to delve deeper into the field…

The base model *presents a sample* to a user: a preamble, then an `**Introduction**` heading, then generic enthusiasm. The fine-tuned model *writes the document*: a title, then a specific first-person narrative opening. Both are fluent — the difference is entirely in what the model thinks the task is.

### Why this needed measuring separately

**ROUGE-L moved 0.008.** If similarity to the reference were the only reported number, this fine-tune would look like it did almost nothing, when in fact it changed the output format on every single example. There are many valid SOPs for one prompt, so n-gram overlap with a single reference is close to blind to the thing that actually changed.

This is the same lesson as the loss curve, one level up: a metric can improve, or fail to improve, for reasons unrelated to whether the model does the job.

### What it did not fix

**70% of fine-tuned outputs still contain unfilled template slots** like `[University Name]`, down from 90%. That is the least flattering number here and the clearest target for the next iteration — most likely a data problem, since the training SOPs contain such placeholders themselves.

### Reproducing

```bash
python -m evaluation.build_dataset --source-dir <dir with the three source .zip files>
python -m evaluation.run_comparison --n 20
```

Outputs land in `evaluation/results/` — `summary.json` for the metrics, `generations.json` for all 20 pairs. Roughly 20 minutes on a 4 GB laptop GPU.

### Caveats

- **These prompts cannot be proven held out.** The original split ran in Colab and ordered files by upload, so `build_dataset.py` rebuilds *a* seed-42 split, not necessarily *the* one the adapter trained on. The structural comparison does not depend on held-out status — the base model's preamble habit is visible on any prompt — but the ROUGE-L figures should be read as indicative only.
- **n = 20.** Enough for a 100%-vs-0% split to be meaningful; not enough to resolve the smaller differences, such as the length gap.
- **Base weights come from `unsloth/Llama-3.2-1B-Instruct`**, an ungated mirror of `meta-llama/Llama-3.2-1B-Instruct` (which the adapter config still names). Identical weights, no licence gate, so the script runs without credentials.
- **Generation quality is not scored.** These metrics say the output has the shape of an SOP, not that it is a good one.

---

## 🧠 Technical Highlights

### Assistant-only loss masking
Loss is computed **only on the assistant's response**, not on the system/user prompt. Prompt tokens are masked with `-100` so the model learns to *generate* SOPs rather than to *reproduce instructions*:

```python
labels = input_ids.copy()
prompt_len = min(len(prompt_ids), len(labels))
labels[:prompt_len] = [-100] * prompt_len   # mask the prompt
```

This is a meaningful correctness detail — training on the full sequence dilutes the learning signal and degrades instruction-following.

### Custom causal-LM collator
A custom data collator pads variable-length sequences while padding labels with `-100`, so padding tokens are excluded from the loss.

### Parameter-efficient fine-tuning
LoRA adapters applied to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) — training 0.275% of parameters instead of all 1.24B, making the run feasible on a free-tier T4.

---

## 📁 Dataset

500 SOP examples, converted to chat-format instruction data:

| Source | Count |
|---|---|
| Real-world SOPs | 60 |
| Augmented variations | 300 |
| Synthetic SOPs | 140 |
| **Total** | **500** |

Split 80/20 → **400 train / 100 validation** (seed=42).

Format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert at writing compelling Statements of Purpose..."},
    {"role": "user", "content": "Write a Statement of Purpose for a Master's program in Computer Science."},
    {"role": "assistant", "content": "...SOP text..."}
  ]
}
```

---

## 🛠 Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `PEFT (LoRA)` · `Datasets` · `Kaggle/Colab GPU`

---

## 📓 Notebooks

| Notebook | Description | Status |
|---|---|---|
| [`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) | Loads SOPs, parses field metadata, formats to chat schema, splits, exports JSONL | ✅ Complete |
| [`02_model_training.ipynb`](notebooks/02_model_training.ipynb) | LoRA fine-tuning with assistant-only loss + inference test | ✅ Complete |
| [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | Base-vs-LoRA evaluation with ROUGE-L and BERTScore over the full 100-example split, for a GPU notebook environment | 🧪 Ready to run, not yet run |
| `04_demo.ipynb` | Gradio inference demo | 🚧 Open |

| Script | Description | Status |
|---|---|---|
| [`evaluation/build_dataset.py`](evaluation/build_dataset.py) | Rebuilds the 500-example set and 80/20 split from the source archives | ✅ Run |
| [`evaluation/run_comparison.py`](evaluation/run_comparison.py) | Base vs fine-tuned generation on identical prompts, structural metrics + ROUGE-L | ✅ Run — [results](#-did-it-actually-learn-the-task) |

---

## ⚠️ Known Limitations

- **Similarity metrics are thin.** ROUGE-L is reported over 20 examples against a single reference each; BERTScore over the full 100-example split is still open, and `03_evaluation.ipynb` covers it. See [Did it actually learn the task?](#-did-it-actually-learn-the-task) for what is measured today and why the structural metrics carry more weight here than ROUGE.
- **Undertrained.** Val loss was still falling at step 250; more epochs or a larger LoRA rank would likely help.
- **Small base model.** Llama 3.2 1B is a size compromise for a free T4. Outputs are coherent but 70% still contain template placeholders (`[University Name]`, `[Your City]`) — measured, down from 90% for the base model.
- **Dataset skew.** 440 of 500 examples are augmented or synthetic; real-world diversity is limited.
- **Outputs are drafts, not final essays.** Not intended to replace an applicant's own writing.

---

## 🤝 Contributing

This project has clear, self-contained open pieces. Contributions welcome — see [Issues](../../issues).

**Good first contributions:**

1. **Run the evaluation (`03_evaluation.ipynb`)** — Execute the prepared base-vs-LoRA comparison on all 100 validation SOPs and add the resulting ROUGE-L/BERTScore table to this README.
2. **Gradio demo (`04_demo.ipynb`)** — Load the LoRA adapter, expose a field-selection dropdown + generate button, deploy to Hugging Face Spaces.
3. **Extended training run** — Val loss hadn't plateaued. Re-run with more epochs / higher LoRA rank and report the new loss curve.
4. **Dataset cleaning** — Detect and remove near-duplicate or template-heavy augmented samples.
5. **Placeholder handling** — The model emits `[University Name]`-style placeholders. Either template them properly at inference or filter them from training data.

To contribute: fork → branch → PR. Please include your loss curves / metrics in the PR description.

---

## 🔮 Future Work

- Run the prepared quantitative evaluation and report ROUGE-L/BERTScore results for base vs. LoRA
- Longer training run — val loss had not converged
- Scale to Llama 3.1 8B with QLoRA (4-bit) on a larger GPU
- Deployed Gradio demo on HF Spaces
- Field-specific prompt templates (CS, Biology, Business, …)

---

## 👥 Authors

- **Ali Alavi** — M.Sc. Computer Science, University of Messina · [LinkedIn](https://www.linkedin.com/in/ali-alavi-cs/) · [GitHub](https://github.com/salavii)
- **Mohaddeseh Jahanabadi** — [GitHub](https://github.com/jahanabadi-n)
