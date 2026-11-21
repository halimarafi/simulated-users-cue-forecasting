"""Remove lines that show your computer's file paths from the saved outputs of every notebook.
Run once from the project folder, before publishing:  python strip_local_paths.py
Only text outputs are changed; code, figures and tables are left as they are."""
import json, re
from pathlib import Path

PATTERN = re.compile(r"(/Users/|/var/folders/|/private/var/|C:\\\\Users\\\\)")
for nb_path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text())
    removed = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                lines = out["text"] if isinstance(out["text"], list) else out["text"].splitlines(keepends=True)
                kept = [l for l in lines if not PATTERN.search(l)]
                removed += len(lines) - len(kept)
                out["text"] = kept
            if out.get("output_type") == "error":
                tb = out.get("traceback", [])
                kept = [l for l in tb if not PATTERN.search(l)]
                removed += len(tb) - len(kept)
                out["traceback"] = kept
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{nb_path.name}: removed {removed} lines containing local paths")
