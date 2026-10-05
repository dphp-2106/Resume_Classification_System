"""Demo on the provided PDF resumes (the 4 category folders in the dataset).

Extracts text with pypdf, runs the full pipeline, and reports accuracy
against the folder labels. These PDFs were NOT part of the CSV training
data, so this is a genuine unseen-data test.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PDF_DIRS, REPORT_DIR  # noqa: E402
from src.pipeline import ResumeClassifier  # noqa: E402


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    r = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in r.pages)


def main():
    clf = ResumeClassifier()
    rows, correct, total = [], 0, 0
    for cat, folder in PDF_DIRS.items():
        for pdf in sorted(folder.glob("*.pdf")):
            raw = pdf_text(pdf)
            if len(raw.split()) < 3:
                continue
            out = clf.predict(raw, model="best")
            ok = out["predicted"] == cat
            correct += int(ok)
            total += 1
            rows.append({"file": pdf.name, "folder_label": cat,
                         "predicted": out["predicted"],
                         "confidence": out["confidence"],
                         "correct": ok})
            print(f"{pdf.name:15s} true={cat:24s} "
                  f"pred={out['predicted']:24s} "
                  f"conf={out['confidence']:.2f} {'OK' if ok else 'X'}")
    acc = correct / total if total else 0
    summary = {"n_pdfs": total, "accuracy_on_pdfs": round(acc, 4),
               "rows": rows}
    (REPORT_DIR / "pdf_demo.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nPDF demo accuracy: {correct}/{total} = {acc:.1%}")
    return summary


if __name__ == "__main__":
    main()
