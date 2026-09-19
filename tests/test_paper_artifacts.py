import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
V3 = ROOT / "runs" / "v3_main"
AUDIT = ROOT / "runs" / "convergence_audit"
EXPOSURE = ROOT / "reviewer_response" / "exposure_parity"
EXPOSURE_ZIP = ROOT / "packages" / "exposure_parity_validation_artifacts_2026-09-15.zip"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".sha256", ".txt", ".yaml", ".yml"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def portable_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.casefold() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def test_reviewer_package_has_complete_aggregate_evidence():
    required = [
        ROOT / "SUPPLEMENTARY_MATERIAL.md",
        ROOT / "RUNS_AND_PROVENANCE.md",
        ROOT / "CACIC_2026_Supplementary_Material.docx",
        ROOT / "CACIC_2026_Supplementary_Material.pdf",
        ROOT / "MANIFEST.sha256",
        TABLES / "final_test_metrics.csv",
        TABLES / "model_nomenclature.csv",
        TABLES / "family_test_comparison.csv",
        TABLES / "validation_leaders.csv",
        TABLES / "paired_bootstrap_updated.csv",
        TABLES / "per_label_test.csv",
        TABLES / "classifier_convergence.csv",
        V3 / "results_validation_grid.csv",
        V3 / "dataset_statistics.csv",
        V3 / "fulltext_coverage.csv",
        V3 / "text_unit_statistics.csv",
        AUDIT / "results_validation.csv",
        AUDIT / "results_test.csv",
        AUDIT / "bootstrap_ci.csv",
        AUDIT / "thresholds.json",
        ROOT / "figures" / "dataset_construction_workflow.png",
        ROOT / "figures" / "experimental_protocol_workflow.png",
        EXPOSURE / "README.md",
        EXPOSURE / "REPORT.md",
        EXPOSURE / "MANIFEST.sha256",
        EXPOSURE / "tables" / "metrics_by_seed.csv",
        EXPOSURE / "tables" / "metrics_aggregate.csv",
        EXPOSURE / "tables" / "convergence_summary.csv",
        EXPOSURE / "tables" / "exposure_totals.csv",
        EXPOSURE / "tables" / "per_label_validation_aggregate.csv",
        EXPOSURE / "tables" / "thresholds_by_seed.csv",
        EXPOSURE_ZIP,
        EXPOSURE_ZIP.with_suffix(".zip.sha256"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing


def test_dense_model_nomenclature_is_explicit_and_traceable():
    mapping = rows(TABLES / "model_nomenclature.csv")
    by_identifier = {row["internal_identifier"]: row for row in mapping}
    assert "DistilUSE" in by_identifier["sbert; sbert_frozen"]["scientific_display_name"]
    assert "sequence classifier" in by_identifier["sbert_finetuned"]["scientific_display_name"]
    assert "sequence classifier" in by_identifier["labse_finetuned"]["scientific_display_name"]


def test_paper_counts_and_model_grid_match_frozen_artifacts():
    dataset = rows(V3 / "dataset_statistics.csv")[0]
    units = rows(V3 / "text_unit_statistics.csv")[0]
    coverage = rows(V3 / "fulltext_coverage.csv")[0]
    grid = rows(V3 / "results_validation_grid.csv")
    fine_tuned = rows(TABLES / "transformer_validation.csv")
    audit_grid = rows(AUDIT / "results_validation.csv")

    assert dataset["N_documents"] == "19710"
    assert dataset["N_labels"] == "37"
    assert dataset["N_assignments"] == "24326"
    assert units["N_abstract_segments"] == "24411"
    assert units["N_fulltext_files"] == "20750"
    assert units["items_with_multiple_abstracts"] == "4365"
    assert units["items_with_multiple_fulltexts"] == "463"
    assert float(coverage["represented_character_percentage"]) == pytest.approx(83.4486886759438)
    assert coverage["truncated_documents"] == "510"
    assert len(grid) == 233
    families = {}
    for row in grid:
        families[row["evaluation_family"]] = families.get(row["evaluation_family"], 0) + 1
    assert sum(families[name] for name in ("bow", "tfidf", "bm25")) == 189
    assert families["sbert_frozen"] + families["labse_frozen"] == 42
    assert families["sbert_finetuned"] + families["labse_finetuned"] == 2
    assert len(fine_tuned) == 2
    assert len(audit_grid) == 8


def test_final_metrics_and_confidence_intervals_match_manuscript():
    final = rows(TABLES / "final_test_metrics.csv")[0]
    intervals = {row["metric"]: row for row in rows(AUDIT / "bootstrap_ci.csv")}

    assert float(final["f1_macro"]) == pytest.approx(0.7522276914464162)
    assert float(final["f1_micro"]) == pytest.approx(0.768457672980286)
    assert float(final["recall_at_3"]) == pytest.approx(0.9319589951560211)
    assert float(intervals["f1_macro"]["ci_lower"]) == pytest.approx(0.7372236272436351)
    assert float(intervals["f1_macro"]["ci_upper"]) == pytest.approx(0.76431303946699)
    assert float(intervals["f1_micro"]["ci_lower"]) == pytest.approx(0.7566867283950617)
    assert float(intervals["f1_micro"]["ci_upper"]) == pytest.approx(0.7797311271975181)


def test_public_artifacts_exclude_item_level_and_model_payloads():
    forbidden_suffixes = {".parquet", ".npy", ".pt", ".safetensors", ".key", ".pem"}
    forbidden_columns = {"handle", "text", "abstract", "fulltext", "file_id", "uri", "path"}
    forbidden_text = ("/content/drive/mydrive/", "c:\\users\\", "h:\\mi unidad", "github_pat_", "ghp_")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in forbidden_suffixes
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            assert not forbidden_columns.intersection(name.casefold() for name in header)
        if path.suffix.lower() in {".md", ".csv", ".json"}:
            content = path.read_text(encoding="utf-8-sig").casefold()
            assert not any(token in content for token in forbidden_text)

    with (AUDIT / "thresholds.json").open(encoding="utf-8") as stream:
        assert isinstance(json.load(stream), dict)


def test_public_artifact_manifest_matches_every_file():
    manifest = ROOT / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest

    actual_paths = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and path != manifest
    }
    assert set(expected) == set(actual_paths)
    for relative, path in actual_paths.items():
        assert hashlib.sha256(portable_bytes(path)).hexdigest() == expected[relative]


def test_exposure_parity_metrics_and_scope_are_exact():
    aggregate = rows(EXPOSURE / "tables" / "metrics_aggregate.csv")
    keyed = {(row["split"], row["model"], row["metric"]): row for row in aggregate}
    assert float(keyed[("validation", "bm25_matched", "f1_macro")]["mean"]) == pytest.approx(
        0.7573298252205513
    )
    assert float(keyed[("validation", "bm25_matched", "f1_micro")]["mean"]) == pytest.approx(
        0.7700862895493767
    )
    assert float(keyed[("validation", "labse_matched", "f1_macro")]["mean"]) == pytest.approx(
        0.7397944287273
    )
    assert float(keyed[("validation", "labse_matched", "f1_macro")]["sample_sd"]) == pytest.approx(
        0.0019794606107488726
    )
    assert float(keyed[("validation", "labse_matched", "f1_micro")]["mean"]) == pytest.approx(
        0.7537219918974052
    )

    convergence = {int(row["seed"]): row for row in rows(EXPOSURE / "tables" / "convergence_summary.csv")}
    assert {seed: int(row["selected_epoch"]) for seed, row in convergence.items()} == {13: 14, 42: 9, 73: 9}
    assert int(convergence[73]["numerical_peak_epoch"]) == 10
    assert float(convergence[73]["peak_minus_selected_f1_macro"]) < 0.002

    totals = {row["split"]: row for row in rows(EXPOSURE / "tables" / "exposure_totals.csv")}
    non_test = totals["all_non_test"]
    assert non_test["n_documents"] == "16751"
    assert non_test["capped_source_characters"] == "866897087"
    assert non_test["selected_source_characters"] == "136758120"
    assert float(non_test["matched_character_coverage_percentage"]) == pytest.approx(15.775588827189127)

    assert len(rows(EXPOSURE / "tables" / "per_label_validation_by_seed.csv")) == 3 * 2 * 37
    assert len(rows(EXPOSURE / "tables" / "per_label_validation_aggregate.csv")) == 2 * 37
    assert len(rows(EXPOSURE / "tables" / "thresholds_by_seed.csv")) == 3 * 2 * 37
    provenance = json.loads((EXPOSURE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["followup_test_inference_performed"] is False
    assert provenance["followup_test_evaluation_performed"] is False
    assert provenance["historical_test_results_previously_inspected"] is True
    assert "final_test_opened" not in provenance
    report = (EXPOSURE / "REPORT.md").read_text(encoding="utf-8")
    compact_report = " ".join(line.removeprefix("> ").strip() for line in report.splitlines())
    assert "per original full-text unit" in compact_report
    assert "No test inference or evaluation was performed" in compact_report
    assert "validation-only" in compact_report
    assert "no probability or score calibration is performed" in compact_report
    assert "Historical execution source revision" in report
    assert "Thresholds are calibrated" not in report
    assert "calibration-only threshold fitting" not in report


def test_exposure_parity_archive_is_complete_private_payload_free_and_checksummed():
    package_manifest = {}
    for line in (EXPOSURE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        package_manifest[relative] = digest
    package_files = {
        path.relative_to(EXPOSURE).as_posix(): path
        for path in EXPOSURE.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    assert set(package_manifest) == set(package_files)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == package_manifest[relative]
               for relative, path in package_files.items())

    for config in (EXPOSURE / "configs").glob("*.yaml"):
        text = config.read_text(encoding="utf-8").casefold()
        assert "run_final_test: false" in text
        assert "${prepared_parquet}" in text
        assert "/content/" not in text
        assert "h:\\" not in text

    prefix = EXPOSURE_ZIP.stem + "/"
    with ZipFile(EXPOSURE_ZIP) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        archived = {name.removeprefix(prefix) for name in names}
        expected = {path.relative_to(EXPOSURE).as_posix() for path in EXPOSURE.rglob("*") if path.is_file()}
        assert archived == expected
        forbidden_suffixes = {".npy", ".pt", ".pth", ".ckpt", ".parquet", ".log"}
        assert not any(Path(name).suffix.casefold() in forbidden_suffixes for name in names)
        combined_text = "\n".join(
            archive.read(name).decode("utf-8-sig")
            for name in names
            if Path(name).suffix.casefold() in {".csv", ".json", ".md", ".yaml", ".sha256"}
        ).casefold()
        assert "/content/drive/mydrive/" not in combined_text
        assert "h:\\mi unidad" not in combined_text
        assert "c:\\users\\" not in combined_text

    expected_digest, expected_name = EXPOSURE_ZIP.with_suffix(".zip.sha256").read_text(
        encoding="utf-8"
    ).split()
    assert expected_name == EXPOSURE_ZIP.name
    assert hashlib.sha256(EXPOSURE_ZIP.read_bytes()).hexdigest() == expected_digest
