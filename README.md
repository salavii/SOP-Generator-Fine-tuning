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
| [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | Reproducible base-vs-LoRA evaluation with ROUGE-L and BERTScore | 🧪 Ready to run |
| `04_demo.ipynb` | Gradio inference demo | 🚧 Open |

---

## ⚠️ Known Limitations

- **Quantitative results are not reported yet.** A reproducible base-vs-LoRA evaluation notebook is included, but the full 100-example run and its ROUGE-L/BERTScore results still need to be completed and summarized here.
- **Undertrained.** Val loss was still falling at step 250; more epochs or a larger LoRA rank would likely help.
- **Small base model.** Llama 3.2 1B is a size compromise for a free T4. Outputs are coherent but contain template placeholders (`[University Name]`, `[Your City]`).
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
