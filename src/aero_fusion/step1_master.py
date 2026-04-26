from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FLIGHT_FILES = (
    "adsc.parquet",
    "adsb_before.parquet",
    "adsb_after.parquet",
    "stitched_minimal.parquet",
    "metadata.json",
)


@dataclass(frozen=True)
class RawRun:
    name: str
    path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_raw_runs(repo_root: Path | None = None) -> list[RawRun]:
    root = repo_root or _repo_root()
    artifacts_root = root / "artifacts"
    runs = [
        RawRun(name=path.name, path=path)
        for path in sorted(artifacts_root.glob("step1_raw_*"), key=lambda item: item.name)
        if path.is_dir()
    ]
    return runs


def default_master_root(repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root()
    return root / "artifacts" / "step1_master"


def _to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert(None)
    return ts


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _safe_read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_reason_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _safe_row_count(path: Path) -> tuple[int | None, str | None]:
    try:
        return int(len(pd.read_parquet(path))), None
    except Exception as exc:  # pragma: no cover - audit path
        return None, f"{type(exc).__name__}: {exc}"


def _summary_path(run: RawRun) -> Path:
    return run.path / "validated_fusion_candidates.parquet"


def _load_summary(run: RawRun) -> pd.DataFrame:
    path = _summary_path(run)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return frame
    for column in ("segment_start_time", "segment_end_time", "flight_start_time", "flight_end_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["source_run"] = run.name
    frame["source_root"] = str(run.path.as_posix())
    return frame


def _required_file_status(flight_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_FLIGHT_FILES if not (flight_dir / name).exists()]
    return len(missing) == 0, missing


def _dedup_key(icao24: Any, start: Any, end: Any) -> str:
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end)
    start_text = None if pd.isna(start_ts) else start_ts.isoformat()
    end_text = None if pd.isna(end_ts) else end_ts.isoformat()
    return f"{_normalize_scalar(icao24)}|{start_text}|{end_text}"


def _summary_lookup(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if summary_df.empty:
        return {}
    records = summary_df.to_dict(orient="records")
    return {str(record["segment_id"]): record for record in records if record.get("segment_id") is not None}


def _audit_run(run: RawRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = _load_summary(run)
    summary_by_segment = _summary_lookup(summary_df)
    flight_root = run.path / "flights"
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    if flight_root.exists():
        flight_dirs = sorted([path for path in flight_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    else:
        flight_dirs = []
        issues.append(
            {
                "source_run": run.name,
                "segment_id": None,
                "issue_type": "missing_flights_directory",
                "details": str(flight_root.as_posix()),
            }
        )

    seen_segments: set[str] = set()
    for flight_dir in flight_dirs:
        segment_id = flight_dir.name
        seen_segments.add(segment_id)
        summary_row = summary_by_segment.get(segment_id)
        all_required_present, missing_files = _required_file_status(flight_dir)

        metadata = None
        metadata_error = None
        if (flight_dir / "metadata.json").exists():
            try:
                metadata = _safe_read_metadata(flight_dir / "metadata.json")
            except Exception as exc:  # pragma: no cover - audit path
                metadata_error = f"{type(exc).__name__}: {exc}"
        else:
            metadata_error = "metadata.json missing"

        validation = (metadata or {}).get("validation", {}) if metadata else {}
        flight_match = (metadata or {}).get("flight_match", {}) if metadata else {}

        adsc_rows, adsc_error = _safe_row_count(flight_dir / "adsc.parquet") if (flight_dir / "adsc.parquet").exists() else (None, "adsc.parquet missing")
        before_rows, before_error = _safe_row_count(flight_dir / "adsb_before.parquet") if (flight_dir / "adsb_before.parquet").exists() else (None, "adsb_before.parquet missing")
        after_rows, after_error = _safe_row_count(flight_dir / "adsb_after.parquet") if (flight_dir / "adsb_after.parquet").exists() else (None, "adsb_after.parquet missing")
        stitched_rows, stitched_error = _safe_row_count(flight_dir / "stitched_minimal.parquet") if (flight_dir / "stitched_minimal.parquet").exists() else (None, "stitched_minimal.parquet missing")

        metadata_segment_id = None if metadata is None else _normalize_scalar(metadata.get("segment_id"))
        icao24 = None if metadata is None else _normalize_scalar(metadata.get("icao24"))
        start_time = None if metadata is None else _to_timestamp(metadata.get("segment_start_time"))
        end_time = None if metadata is None else _to_timestamp(metadata.get("segment_end_time"))
        gap_duration_minutes = None if metadata is None else _normalize_scalar(metadata.get("gap_duration_minutes"))
        adsc_point_count = None if metadata is None else _normalize_scalar(metadata.get("adsc_point_count"))
        metadata_valid = bool(validation.get("is_valid")) if validation else False
        reasons = _normalize_reason_list(validation.get("reasons"))

        summary_present = summary_row is not None
        summary_matches_folder = summary_present and str(summary_row.get("segment_id")) == segment_id
        summary_matches_metadata = True
        if summary_present and metadata is not None:
            summary_matches_metadata = (
                str(summary_row.get("segment_id")) == str(metadata_segment_id)
                and _normalize_scalar(summary_row.get("icao24")) == icao24
                and _to_timestamp(summary_row.get("segment_start_time")) == start_time
                and _to_timestamp(summary_row.get("segment_end_time")) == end_time
            )

        required_non_empty = all(
            value is not None and int(value) > 0
            for value in (adsc_rows, before_rows, after_rows, stitched_rows)
        )
        valid_saved_folder = all_required_present and metadata is not None and metadata_valid and required_non_empty

        row = {
            "source_run": run.name,
            "source_root": str(run.path.as_posix()),
            "segment_id": segment_id,
            "metadata_segment_id": metadata_segment_id,
            "flight_dir": str(flight_dir.as_posix()),
            "summary_present": summary_present,
            "summary_matches_folder": summary_matches_folder,
            "summary_matches_metadata": summary_matches_metadata,
            "metadata_present": metadata is not None,
            "metadata_error": metadata_error,
            "required_files_present": all_required_present,
            "missing_required_files": json.dumps(missing_files),
            "metadata_is_valid": metadata_valid,
            "valid_saved_folder": valid_saved_folder,
            "icao24": icao24,
            "segment_start_time": start_time,
            "segment_end_time": end_time,
            "gap_duration_minutes": gap_duration_minutes,
            "adsc_point_count": adsc_point_count,
            "flight_callsign": _normalize_scalar(flight_match.get("callsign")),
            "flight_start_time": _to_timestamp(flight_match.get("flight_start_time")),
            "flight_end_time": _to_timestamp(flight_match.get("flight_end_time")),
            "estdepartureairport": _normalize_scalar(flight_match.get("estdepartureairport")),
            "estarrivalairport": _normalize_scalar(flight_match.get("estarrivalairport")),
            "before_count_metadata": _normalize_scalar(validation.get("before_count")),
            "during_count_metadata": _normalize_scalar(validation.get("during_count")),
            "after_count_metadata": _normalize_scalar(validation.get("after_count")),
            "before_boundary_speed_kts": _normalize_scalar(validation.get("before_boundary_speed_kts")),
            "after_boundary_speed_kts": _normalize_scalar(validation.get("after_boundary_speed_kts")),
            "max_internal_speed_kts": _normalize_scalar(validation.get("max_internal_speed_kts")),
            "median_internal_speed_kts": _normalize_scalar(validation.get("median_internal_speed_kts")),
            "validation_reasons": json.dumps(reasons),
            "adsc_rows": adsc_rows,
            "adsb_before_rows": before_rows,
            "adsb_after_rows": after_rows,
            "stitched_rows": stitched_rows,
            "adsc_read_error": adsc_error,
            "adsb_before_read_error": before_error,
            "adsb_after_read_error": after_error,
            "stitched_read_error": stitched_error,
            "dedup_key": _dedup_key(icao24, start_time, end_time),
            "summary_gap_duration_minutes": None if summary_row is None else summary_row.get("gap_duration_minutes"),
            "summary_adsc_point_count": None if summary_row is None else summary_row.get("adsc_point_count"),
            "summary_reasons": None if summary_row is None else json.dumps(_normalize_reason_list(summary_row.get("reasons"))),
            "summary_artifact_dir": None if summary_row is None else summary_row.get("artifact_dir"),
        }
        rows.append(row)

        if not summary_present:
            issues.append(
                {
                    "source_run": run.name,
                    "segment_id": segment_id,
                    "issue_type": "saved_folder_missing_from_summary",
                    "details": str(flight_dir.as_posix()),
                }
            )
        if not all_required_present:
            issues.append(
                {
                    "source_run": run.name,
                    "segment_id": segment_id,
                    "issue_type": "missing_required_files",
                    "details": ", ".join(missing_files),
                }
            )
        if metadata_error is not None:
            issues.append(
                {
                    "source_run": run.name,
                    "segment_id": segment_id,
                    "issue_type": "metadata_read_error",
                    "details": metadata_error,
                }
            )
        if metadata is not None and str(metadata_segment_id) != segment_id:
            issues.append(
                {
                    "source_run": run.name,
                    "segment_id": segment_id,
                    "issue_type": "metadata_segment_id_mismatch",
                    "details": f"metadata_segment_id={metadata_segment_id}",
                }
            )
        for label, count in (("adsc", adsc_rows), ("adsb_before", before_rows), ("adsb_after", after_rows), ("stitched", stitched_rows)):
            if count is not None and int(count) <= 0:
                issues.append(
                    {
                        "source_run": run.name,
                        "segment_id": segment_id,
                        "issue_type": f"empty_{label}_file",
                        "details": f"{label}_rows={count}",
                    }
                )

    if not summary_df.empty:
        for segment_id in sorted(set(summary_df["segment_id"].astype(str)) - seen_segments):
            issues.append(
                {
                    "source_run": run.name,
                    "segment_id": segment_id,
                    "issue_type": "summary_row_missing_saved_folder",
                    "details": str(run.path.as_posix()),
                }
            )

    audited_df = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)
    return audited_df, issues_df


def _choose_master_records(valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid_df.empty:
        return valid_df.copy(), pd.DataFrame()

    ranked = valid_df.copy()
    discovered_runs = sorted(ranked["source_run"].dropna().astype(str).unique().tolist())
    source_priority = {name: index for index, name in enumerate(discovered_runs)}
    ranked["source_priority"] = ranked["source_run"].map(source_priority).fillna(len(source_priority)).astype(int)
    ranked["summary_priority"] = ranked["summary_present"].astype(int) * -1
    ranked["metadata_priority"] = ranked["metadata_present"].astype(int) * -1
    ranked["size_priority"] = -(
        ranked[["adsc_rows", "adsb_before_rows", "adsb_after_rows", "stitched_rows"]]
        .fillna(0)
        .sum(axis=1)
    )
    ranked = ranked.sort_values(
        [
            "dedup_key",
            "summary_priority",
            "metadata_priority",
            "source_priority",
            "size_priority",
            "segment_id",
        ],
        ascending=[True, True, True, True, True, True],
        ignore_index=True,
    )
    ranked["duplicate_rank"] = ranked.groupby("dedup_key").cumcount() + 1
    ranked["is_master_record"] = ranked["duplicate_rank"] == 1

    duplicates_df = ranked[ranked["dedup_key"].duplicated(keep=False)].copy()
    master_df = ranked[ranked["is_master_record"]].copy()
    return master_df, duplicates_df


def _copy_master_flights(master_df: pd.DataFrame, master_root: Path) -> pd.DataFrame:
    flights_root = master_root / "flights"
    flights_root.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, Any]] = []
    for record in master_df.to_dict(orient="records"):
        source_dir = Path(record["flight_dir"])
        target_dir = flights_root / str(record["segment_id"])
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

        copied = dict(record)
        copied["master_flight_dir"] = str(target_dir.as_posix())
        copied["master_adsc_path"] = str((target_dir / "adsc.parquet").as_posix())
        copied["master_adsb_before_path"] = str((target_dir / "adsb_before.parquet").as_posix())
        copied["master_adsb_after_path"] = str((target_dir / "adsb_after.parquet").as_posix())
        copied["master_stitched_minimal_path"] = str((target_dir / "stitched_minimal.parquet").as_posix())
        copied["master_metadata_path"] = str((target_dir / "metadata.json").as_posix())
        copied_rows.append(copied)
    return pd.DataFrame(copied_rows)


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path_without_suffix.with_suffix(".parquet"), index=False)
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)


def _build_summary_payload(
    all_audited_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    master_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    runs: list[RawRun],
) -> dict[str, Any]:
    duplicate_keys = 0 if duplicates_df.empty else int(duplicates_df["dedup_key"].nunique())
    duplicate_extra_records = max(int(len(valid_df) - len(master_df)), 0)
    return {
        "raw_runs_considered": [run.name for run in runs],
        "raw_saved_flight_folders_found": int(len(all_audited_df)),
        "valid_saved_flights_found": int(len(valid_df)),
        "duplicate_records_found": duplicate_extra_records,
        "duplicate_groups_found": duplicate_keys,
        "final_unique_master_flights": int(len(master_df)),
        "integrity_issue_records": int(len(issues_df)),
        "source_runs": (
            all_audited_df.groupby("source_run")
            .agg(
                raw_saved_flight_folders=("segment_id", "size"),
                valid_saved_flights=("valid_saved_folder", "sum"),
            )
            .reset_index()
            .to_dict(orient="records")
            if not all_audited_df.empty
            else []
        ),
    }


def _write_master_readme(master_root: Path, summary_payload: dict[str, Any]) -> None:
    raw_runs = summary_payload.get("raw_runs_considered", [])
    if raw_runs:
        raw_runs_text = "\n".join(f"- `artifacts/{run_name}`" for run_name in raw_runs)
    else:
        raw_runs_text = "- No `artifacts/step1_raw_*` folders were found"

    readme = f"""# Step 1 Master Dataset

This folder is a derived Step 1 deliverable built from the preserved raw Step 1 runs:

{raw_runs_text}

The raw runs were not overwritten. This master folder exists only to consolidate, audit, and deduplicate Step 1.

## Contents

- `catalog/all_saved_flight_folders.*`: one audit row per saved flight folder found in the raw runs
- `catalog/valid_saved_flights.*`: saved flight folders that passed the master validity checks
- `catalog/duplicates.*`: duplicate groups detected by the stable dedup key
- `catalog/master_flights_catalog.*`: final deduplicated Step 1 catalog
- `catalog/integrity_issues.*`: integrity findings discovered while auditing the raw runs
- `catalog/raw_summary_merged.*`: merged view of the raw per-run summary tables
- `flights/<segment_id>/...`: copied final unique master flight folders
- `audit_summary.json`: compact machine-readable Step 1 totals

## Stable dedup key

Duplicates are detected with:

`icao24 | segment_start_time | segment_end_time`

This key is derived from the saved metadata and matches the semantics already used by the Step 1 `segment_id`.

## Current totals

- Raw saved flight folders found: {summary_payload["raw_saved_flight_folders_found"]}
- Valid saved flights found: {summary_payload["valid_saved_flights_found"]}
- Duplicate extra records found: {summary_payload["duplicate_records_found"]}
- Final unique master flights: {summary_payload["final_unique_master_flights"]}
- Integrity issue records: {summary_payload["integrity_issue_records"]}

Use `catalog/master_flights_catalog.parquet` or `.csv` as the Step 1 source of truth for later work.
"""
    (master_root / "README.md").write_text(readme, encoding="utf-8")


def build_step1_master(
    raw_runs: list[RawRun] | None = None,
    master_root: Path | None = None,
) -> dict[str, Any]:
    runs = raw_runs or default_raw_runs()
    out_root = master_root or default_master_root()
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "catalog").mkdir(parents=True, exist_ok=True)

    audited_frames: list[pd.DataFrame] = []
    issue_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for run in runs:
        audited_df, issues_df = _audit_run(run)
        audited_frames.append(audited_df)
        issue_frames.append(issues_df)

        summary_df = _load_summary(run)
        if not summary_df.empty:
            summary_frames.append(summary_df)

    all_audited_df = pd.concat(audited_frames, ignore_index=True) if audited_frames else pd.DataFrame()
    issues_df = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
    raw_summary_merged_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    valid_df = (
        all_audited_df[all_audited_df["valid_saved_folder"]].copy().reset_index(drop=True)
        if not all_audited_df.empty
        else pd.DataFrame()
    )
    master_selection_df, duplicates_df = _choose_master_records(valid_df)
    master_catalog_df = _copy_master_flights(master_selection_df, out_root) if not master_selection_df.empty else master_selection_df.copy()

    _write_frame(all_audited_df, out_root / "catalog" / "all_saved_flight_folders")
    _write_frame(valid_df, out_root / "catalog" / "valid_saved_flights")
    _write_frame(duplicates_df, out_root / "catalog" / "duplicates")
    _write_frame(master_catalog_df, out_root / "catalog" / "master_flights_catalog")
    _write_frame(issues_df, out_root / "catalog" / "integrity_issues")
    _write_frame(raw_summary_merged_df, out_root / "catalog" / "raw_summary_merged")

    summary_payload = _build_summary_payload(
        all_audited_df=all_audited_df,
        valid_df=valid_df,
        duplicates_df=duplicates_df,
        master_df=master_catalog_df,
        issues_df=issues_df,
        runs=runs,
    )
    (out_root / "audit_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_master_readme(out_root, summary_payload)
    return summary_payload


def main() -> None:
    summary = build_step1_master()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
