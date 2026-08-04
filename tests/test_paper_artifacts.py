import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
V3 = ROOT / "runs" / "v3_main"
AUDIT = ROOT / "runs" / "convergence_audit"
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
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing


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
