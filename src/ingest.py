"""
Batch ingestion over a whole dataset directory.

Routes each file to the right adapter by extension:
  .csv         -> csv_adapter.read_csv_statement            (your real file)
  .xlsx / .xls -> excel_adapter.read_excel_statement          (YOUR real file,
                   if you drop it into src/) or, failing that,
                   excel_adapter_generic.read_excel_statement (bundled stand-in)

PDFs and any file that raises an exception are logged and skipped rather
than crashing the batch — mirrors the "Could not find transaction header"
skip-and-continue behavior already used across the existing adapters.
"""

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transaction_schema import Transaction  # noqa: E402
from csv_adapter import read_csv_statement  # noqa: E402

try:
    from excel_adapter import read_excel_statement as read_excel_statement_impl
    EXCEL_ADAPTER_SOURCE = "excel_adapter (your real file)"
except ImportError:
    from excel_adapter_generic import read_excel_statement as read_excel_statement_impl
    EXCEL_ADAPTER_SOURCE = "excel_adapter_generic (bundled stand-in)"


@dataclass
class IngestReport:
    ok_files: List[str] = field(default_factory=list)
    failed_files: List[tuple] = field(default_factory=list)
    skipped_extensions: List[str] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"Excel adapter in use : {EXCEL_ADAPTER_SOURCE}",
            f"Files parsed OK      : {len(self.ok_files)}",
            f"Files failed/skipped : {len(self.failed_files)}",
            f"Total transactions   : {len(self.transactions)}",
        ]
        return "\n".join(lines)


SUPPORTED = {".csv", ".xlsx", ".xls"}


def ingest_directory(root: str, verbose: bool = True) -> IngestReport:
    root_path = Path(root)
    report = IngestReport()

    files = sorted(p for p in root_path.rglob("*") if p.is_file())

    for path in files:
        ext = path.suffix.lower()

        if ext not in SUPPORTED:
            report.skipped_extensions.append(path.name)
            continue

        try:
            if ext == ".csv":
                txns = read_csv_statement(str(path))
            else:
                txns = read_excel_statement_impl(str(path))

            report.transactions.extend(txns)
            report.ok_files.append(path.name)

            if verbose:
                print(f"OK    {path.name[:65]:65s} -> {len(txns):6d} txns")

        except Exception as e:
            report.failed_files.append((path.name, type(e).__name__, str(e)[:80]))

            if verbose:
                print(f"FAIL  {path.name[:65]:65s} -> {type(e).__name__}: {str(e)[:60]}")

    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py <dataset_root_dir>")
        raise SystemExit(1)

    t0 = time.time()
    report = ingest_directory(sys.argv[1])
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(report.summary)
    print(f"Skipped (unsupported extension, e.g. PDFs): {len(report.skipped_extensions)}")
    print(f"Elapsed: {elapsed:.1f}s")
