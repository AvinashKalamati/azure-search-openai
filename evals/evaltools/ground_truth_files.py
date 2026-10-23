import json
from pathlib import Path

GROUND_TRUTH_BY_FILE_DIR = "ground_truth_by_file"


def questions_path(by_file_dir: Path, source_file: str) -> Path:
    return by_file_dir / f"{source_file}.jsonl"


def knowledge_graph_path(by_file_dir: Path, source_file: str) -> Path:
    return by_file_dir / f"{source_file}.kg.json"


def write_qa_pairs(path: Path, qa_pairs: list[dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_testdata(path: Path) -> list[dict]:
    """Load questions from a single .jsonl file, or from every .jsonl file in a folder."""
    if path.is_dir():
        rows: list[dict] = []
        for file_path in sorted(path.glob("*.jsonl")):
            rows.extend(load_jsonl(file_path))
        return rows
    return load_jsonl(path)


def cites_source_file(truth: str, source_file: str) -> bool:
    return f"[{source_file}#" in truth or f"[{source_file}]" in truth


def remove_source_file(by_file_dir: Path, combined_path: Path, source_file: str) -> tuple[list[Path], int]:
    """Delete a document's question file and knowledge graph, and drop its questions from the combined file.

    Returns the deleted paths and how many questions were removed from the combined file.
    """
    deleted_paths = []
    for path in (questions_path(by_file_dir, source_file), knowledge_graph_path(by_file_dir, source_file)):
        if path.exists():
            path.unlink()
            deleted_paths.append(path)

    removed_count = 0
    if combined_path.exists():
        rows = load_jsonl(combined_path)
        kept_rows = [row for row in rows if not cites_source_file(row.get("truth", ""), source_file)]
        removed_count = len(rows) - len(kept_rows)
        if removed_count:
            write_qa_pairs(combined_path, kept_rows, append=False)

    return deleted_paths, removed_count
