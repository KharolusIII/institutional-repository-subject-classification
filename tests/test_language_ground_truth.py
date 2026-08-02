import pandas as pd

from ir_subject_classification.language_ground_truth import load_abstract_ground_truth, text_fingerprint


def test_ground_truth_records_agreements_and_manual_corrections(tmp_path):
    frame = pd.DataFrame(
        {
            "dc.identifier.uri": ["http://sedici.unlp.edu.ar/handle/10915/1", "10915/2"],
            "resumen": ["Texto español", "English text"],
            "Auditoria_Ultima_Entrenamiento": ["es", "en"],
            "Idiomacatalogado": ["es", "es"],
            "LangID": ["es", "en"],
        }
    )
    path = tmp_path / "ground_truth.csv"
    frame.to_csv(path, index=False)
    lookup, conflicts = load_abstract_ground_truth(path)
    assert conflicts.empty
    assert lookup[("10915/1", text_fingerprint("Texto español"))]["source"] == "human_langid_agreement"
    assert lookup[("10915/2", text_fingerprint("English text"))]["source"] == "manual_audit"
