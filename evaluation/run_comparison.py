"""Generate from the base model and the LoRA model on identical prompts.

    python -m evaluation.run_comparison --n 20

Both models see the same prompts, the same generation settings, and the same
per-example random seed, so differences in output are attributable to the
adapter rather than to sampling luck.

WHY the structural metrics and not just ROUGE:

Validation loss fell from 2.33 to 2.08, which says the model got better at
predicting the reference tokens. It does not say whether the *behaviour*
changed, and behaviour is what the fine-tune was for. A 1B instruct model asked
to write a Statement of Purpose tends to answer like a chatbot -- a preamble
("Sure! Here's a draft..."), bullet-pointed advice, a closing offer to revise --
rather than producing the document itself.

So alongside ROUGE-L this measures things that separate "wrote an SOP" from
"talked about writing an SOP":

  meta_preamble   -- opens by addressing the user instead of starting the essay
  first_person    -- share of sentences using I/my, as SOP prose does
  bullet_markers  -- list formatting, which an SOP should not contain
  hit_token_cap   -- ran to max_new_tokens without emitting a stop token

ROUGE-L is reported but should be read with care: there are many good SOPs for
one prompt, and it is scored against a single reference.

WHY each model runs in its own process:

A 4 GB laptop GPU cannot hold both models, and releasing the first one is not
enough -- the CUDA context and allocator fragmentation keep roughly 0.8 GB even
after empty_cache(), which is enough to make the second load fail. Running the
stages as separate processes tears the context down completely. It also makes
the run resumable, which matters when a full run takes half an hour.

Environment notes for reproducing:
  HF_HUB_DISABLE_XET=1   huggingface_hub's Xet backend stalled at an identical
                         byte offset on every retry here; plain HTTP works.
  transformers==4.46.3   transformers 5.x segfaults with torch 2.5.1.
  safetensors==0.4.5     safetensors 0.8.0 segfaults inside torch.storage
                         with torch 2.5.1.
"""

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

# NOTE: torch and transformers are imported inside the generation functions, not
# at module scope. The driver process spawns the two generation stages as
# subprocesses; if the driver imported torch and touched CUDA it would hold a
# context worth several hundred MB for the whole run, which on a 4 GB card is
# enough to make the child's model load fail with an out-of-memory error that
# looks like it has nothing to do with the parent.

CACHE_DIR = Path(__file__).parent / ".cache"
RESULTS_DIR = Path(__file__).parent / "results"

# Ungated mirror of meta-llama/Llama-3.2-1B-Instruct (identical weights). The
# official repo is gated behind a manual licence acceptance, which would make
# this script unrunnable without credentials. The adapter's own config still
# records the official repo as its base model.
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_ADAPTER = Path("D:/my_project/kaggle/working/sop_lora_adapter")

SEED = 42
GENERATION_CONFIG = {
    # The notebook proposed 850. Reference SOPs in this dataset run 610-700
    # words, which is roughly 800-900 tokens, so a *correctly* written SOP would
    # hit an 850-token cap and be indistinguishable from a model that never
    # learned to stop. 1024 leaves room to finish, which is what makes
    # `hit_token_cap` a real signal rather than a property of the cap.
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

META_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|here(?:'s| is)|absolutely|below is|i'd be happy"
    r"|great choice|okay|ok\b|as an ai|note:|\*\*|#)",
    re.IGNORECASE,
)
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+\.)\s+", re.MULTILINE)
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|my|me)\b")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_validation(n: int) -> list[dict]:
    path = CACHE_DIR / "val.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python -m evaluation.build_dataset --source-dir <dir>"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return random.Random(SEED).sample(rows, min(n, len(rows)))


def split_example(example: dict) -> tuple[list[dict], str]:
    messages = example["messages"]
    prompt = [m for m in messages if m["role"] != "assistant"]
    reference = next(m["content"] for m in messages if m["role"] == "assistant")
    return prompt, reference


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def load_base(dtype, device: str):
    """Load the base model straight onto the target device.

    low_cpu_mem_usage + device_map builds on the meta device and streams weights
    in at `dtype`. The default path materialises the model in fp32 in RAM first,
    which on a 15 GB machine fails while allocating the 128256 x 2048 lm_head --
    as an access violation rather than a clean MemoryError.
    """
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device if device == "cuda" else None,
    )


def generate_all(model, tokenizer, prompts: list[list[dict]], device: str) -> list[dict]:
    import torch

    outputs = []
    for index, messages in enumerate(prompts):
        # Same per-example seed for both models, so any difference in the output
        # comes from the weights rather than from the sampling stream.
        torch.manual_seed(SEED + index)

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs, pad_token_id=tokenizer.eos_token_id, **GENERATION_CONFIG
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        outputs.append(
            {
                "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                "n_new_tokens": int(new_tokens.shape[0]),
                "hit_token_cap": int(new_tokens.shape[0]) >= GENERATION_CONFIG["max_new_tokens"],
            }
        )
        print(f"    [{index + 1}/{len(prompts)}] {outputs[-1]['n_new_tokens']:4d} tokens",
              flush=True)
    return outputs


def run_stage(stage: str, n: int, adapter: Path) -> None:
    """Generate for one model and write a partial result file, then exit."""
    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  device={device} dtype={dtype}", flush=True)

    examples = load_validation(n)
    prompts = [split_example(e)[0] for e in examples]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_base(dtype, device)
    if stage == "tuned":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    generations = generate_all(model, tokenizer, prompts, device)

    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  vram after release: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
          if torch.cuda.is_available() else "", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"_{stage}.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def structural_metrics(text: str) -> dict:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    first_person = sum(bool(FIRST_PERSON.search(s)) for s in sentences)
    return {
        "words": len(text.split()),
        "meta_preamble": bool(META_PREAMBLE.match(text)),
        "bullet_markers": len(BULLET.findall(text)),
        "first_person_ratio": (first_person / len(sentences)) if sentences else 0.0,
    }


def aggregate(generations: list[dict]) -> dict:
    n = len(generations) or 1
    metrics = [structural_metrics(g["text"]) for g in generations]
    return {
        "mean_words": round(sum(m["words"] for m in metrics) / n, 1),
        "pct_meta_preamble": round(100 * sum(m["meta_preamble"] for m in metrics) / n, 1),
        "mean_bullet_markers": round(sum(m["bullet_markers"] for m in metrics) / n, 2),
        "mean_first_person_ratio": round(sum(m["first_person_ratio"] for m in metrics) / n, 3),
        "pct_hit_token_cap": round(100 * sum(g["hit_token_cap"] for g in generations) / n, 1),
    }


def rouge_l(predictions: list[str], references: list[str]) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("  (rouge-score not installed; skipping ROUGE-L)")
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores), 4) if scores else None


def report(n: int, adapter: Path) -> None:
    base = json.loads((RESULTS_DIR / "_base.json").read_text(encoding="utf-8"))
    tuned = json.loads((RESULTS_DIR / "_tuned.json").read_text(encoding="utf-8"))

    examples = load_validation(n)
    prompts, references = zip(*(split_example(e) for e in examples))

    summary = {
        "base_model": BASE_MODEL,
        "adapter": str(adapter),
        "n_examples": len(base),
        "generation_config": GENERATION_CONFIG,
        "seed": SEED,
        "base": aggregate(base),
        "tuned": aggregate(tuned),
    }
    summary["base"]["rouge_l"] = rouge_l([g["text"] for g in base], list(references))
    summary["tuned"]["rouge_l"] = rouge_l([g["text"] for g in tuned], list(references))

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "generations.json").write_text(
        json.dumps(
            [
                {
                    "prompt": prompt[-1]["content"],
                    "base": b["text"],
                    "tuned": t["text"],
                    "reference_excerpt": reference[:400],
                }
                for prompt, b, t, reference in zip(prompts, base, tuned, references)
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 66)
    print(f"{'metric':28s} {'base':>16s} {'fine-tuned':>16s}")
    for key in summary["base"]:
        print(f"{key:28s} {str(summary['base'][key]):>16s} {str(summary['tuned'][key]):>16s}")
    print(f"\nwrote {RESULTS_DIR/'summary.json'} and generations.json")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage", choices=["base", "tuned", "report"],
                        help="internal: run one stage in this process")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"Adapter not found: {args.adapter}")

    if args.stage in ("base", "tuned"):
        run_stage(args.stage, args.n, args.adapter)
        return
    if args.stage == "report":
        report(args.n, args.adapter)
        return

    # Driver: each generation stage gets a fresh process so the CUDA context is
    # torn down between models.
    for stage in ("base", "tuned"):
        print(f"\n=== {stage} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_comparison",
             "--stage", stage, "--n", str(args.n), "--adapter", str(args.adapter)],
            env={**os.environ},
        )
        if result.returncode != 0:
            raise SystemExit(f"stage {stage} failed with code {result.returncode}")

    report(args.n, args.adapter)


if __name__ == "__main__":
    main()
