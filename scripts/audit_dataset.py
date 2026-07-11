"""Print CourtVision YOLO label counts by split and class."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    class_names = {0: "ball", 1: "net"}

    for split in ("train", "valid", "test"):
        counts: Counter[int] = Counter()
        label_files = 0
        empty_files = 0
        label_dir = root / split / "labels"

        for label_path in label_dir.glob("*.txt"):
            if label_path.name == "classes.txt":
                continue

            label_files += 1
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                empty_files += 1

            for line in lines:
                class_id = int(line.split()[0])
                counts[class_id] += 1

        readable = {class_names.get(class_id, str(class_id)): count for class_id, count in sorted(counts.items())}
        print(f"{split}: files={label_files}, empty={empty_files}, labels={readable}")


if __name__ == "__main__":
    main()
