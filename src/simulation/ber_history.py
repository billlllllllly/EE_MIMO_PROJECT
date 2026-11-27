"""Save per-SNR Rayleigh simulation results to a mounted Drive folder."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from filelock import FileLock


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    staged = path.with_suffix(".csv.tmp")
    with staged.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def _run_record(run_id, timestamp, rows, antenna_columns):
    bits, = {int(row["total_bits_transmitted"]) for row in rows}
    errors = {
        row["detector"]: [int(row[column]) for column in antenna_columns]
        for row in rows
    }
    if len(errors) != len(rows):
        raise ValueError("Repeated detector rows for the same SNR")
    return dict(run_id=run_id, timestamp=timestamp, bits_simulated=bits, errors_by_detector=errors)


def _sum_records(records, snr, antenna_columns):
    totals = {}
    for record in records:
        for name, errors in record["errors_by_detector"].items():
            row = totals.setdefault(name, dict.fromkeys(antenna_columns, 0))
            for column, count in zip(antenna_columns, errors, strict=True):
                row[column] += count
            row["total_bits_transmitted"] = row.get("total_bits_transmitted", 0) + record["bits_simulated"]

    return [
        {"snr_db": snr, "detector": name, **row,
         "ber": sum(row[col] for col in antenna_columns) / row["total_bits_transmitted"]}
        for name, row in totals.items()
    ]


def save_ber_history(error_csv, output_dir, run_id, completed_at):
    """Publish layer CSVs and pool bit counts. Re-exporting a run is idempotent."""
    error_csv = Path(error_csv)
    rows_by_snr = defaultdict(list)
    for row in _read_csv(error_csv):
        rows_by_snr[float(row["snr_db"])].append(row)

    summary = []
    for snr, rows in rows_by_snr.items():
        antenna_columns = [col for col in rows[0] if col.endswith("_bit_errors")]
        folder = f"{str(snr).removesuffix('.0')}dB"
        dest = Path(output_dir) / folder
        dest.mkdir(parents=True, exist_ok=True)
        counts_csv = dest / "rayleigh_bit_errors.csv"
        history = dest / "simulation_log.txt"
        record = _run_record(run_id, completed_at, rows, antenna_columns)

        with FileLock(str(dest / ".ber.lock")):
            records = []
            if history.exists():
                with history.open(encoding="utf-8") as stream:
                    records = [json.loads(line) for line in stream if line.strip()]

            additions = []
            if not records and counts_csv.exists():
                # Preserve a pre-existing per-SNR error CSV on the first export.
                prior = _read_csv(counts_csv)
                if any(float(row["snr_db"]) != snr for row in prior):
                    raise ValueError(f"Unexpected SNR in {counts_csv}")
                additions.append(_run_record("prior-csv", None, prior, antenna_columns))

            if not any(item["run_id"] == run_id for item in records):
                for layer in range(len(antenna_columns)):
                    name = f"layer_{layer}.csv"
                    staged = dest / f".{name}.tmp"
                    shutil.copyfile(error_csv.parent / folder / name, staged)
                    staged.replace(dest / name)
                additions.append(record)

            # Commit run counts to the append-only log first. A retry can always
            # rebuild the cumulative CSV, even if the previous export stopped here.
            if additions:
                with history.open("a", encoding="utf-8") as stream:
                    for addition in additions:
                        stream.write(json.dumps(addition, ensure_ascii=False) + "\n")
            cumulative = _sum_records(records + additions, snr, antenna_columns)
            _write_csv(counts_csv, cumulative)
            summary.extend(cumulative)

    _write_csv(error_csv.with_name("cumulative_rayleigh_bit_errors.csv"), summary)
    return {(row["snr_db"], row["detector"]): row for row in summary}
