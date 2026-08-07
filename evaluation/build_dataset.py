"""Rebuild the train/validation split from the three source archives.

Mirrors `notebooks/01_data_preparation.ipynb` exactly: same metadata parsing,
same chat format, same `random.seed(42)` 80/20 split.

IMPORTANT CAVEAT -- this does not necessarily reproduce the *original* split.
The notebook ran in Colab and collected files with `Path(folder).glob('**/*.txt')`
after a `files.upload()`, so the pre-shuffle ordering depended on upload and
filesystem order. Shuffling a differently-ordered list with the same seed gives
a different partition. This script sorts filenames so that IT is reproducible,
but the boundary between train and validation almost certainly differs from the
run that produced the adapter.

The consequence, stated plainly: examples in the validation split here cannot be
assumed unseen by the adapter. Similarity metrics computed against them are
indicative of style, not held-out generalisation. See README.

    python -m evaluation.build_dataset --source-dir D:/my_project/sop_data
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
ARCHIVES = ["SOP_Dataset.zip", "augmented_sops.zip", "synthetic_sops.zip"]
SEED = 42
TRAIN_FRACTION = 0.8

SYSTEM_PROMPT = (
    "You are an expert at writing compelling Statements of Purpose "
    "for graduate school applications."
)


def parse_sop(content: str, filename: str) -> dict:
    """Split a source file into metadata and SOP body, and read its Field."""
    if "---" in content:
        metadata, sop_text = content.split("---", 1)
        metadata, sop_text = metadata.strip(), sop_text.strip()
    else:
        metadata, sop_text = "", content

    field = "Computer Science"  # same default as the notebook
    for line in metadata.split("\n"):
        if "Field:" in line:
            field = line.split("Field:")[1].strip()
            break

    return {"field": field, "text": sop_text, "filename": filename}


def format_for_training(sop: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Write a Statement of Purpose for a Master's program in {sop['field']}.",
            },
            {"role": "assistant", "content": sop["text"]},
        ]
    }


def build(source_dir: Path) -> tuple[list[dict], list[dict]]:
    sops: list[dict] = []
    for archive_name in ARCHIVES:
        archive = source_dir / archive_name
        if not archive.exists():
            raise FileNotFoundError(f"Missing source archive: {archive}")

        # Read straight out of the archive rather than extracting: some source
        # filenames are long enough that extracting under a deep path trips
        # Windows' 260-character MAX_PATH limit.
        with zipfile.ZipFile(archive) as zf:
            # sorted() for determinism -- see the caveat in the module docstring.
            names = sorted(
                name for name in zf.namelist()
                if name.endswith(".txt") and not name.startswith("__MACOSX/")
            )
            print(f"  {archive_name:24s} {len(names):4d} files")
            for name in names:
                content = zf.read(name).decode("utf-8", errors="ignore")
                sops.append(parse_sop(content, Path(name).name))

    examples = [format_for_training(sop) for sop in sops]

    random.seed(SEED)
    random.shuffle(examples)
    split_index = int(len(examples) * TRAIN_FRACTION)
    return examples[:split_index], examples[split_index:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory holding the three source .zip archives")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train, validation = build(args.source_dir)

    write_jsonl(CACHE_DIR / "train.jsonl", train)
    write_jsonl(CACHE_DIR / "val.jsonl", validation)

    print(f"\ntotal {len(train) + len(validation)} examples "
          f"-> train {len(train)} / val {len(validation)}")
    fields = {ex["messages"][1]["content"] for ex in validation}
    print(f"{len(fields)} distinct prompts in the validation split")


if __name__ == "__main__":
    main()
