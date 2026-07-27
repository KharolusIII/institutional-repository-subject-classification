from ir_subject_classification.language import HeuristicLanguageDetector, LanguagePrediction


def test_language_detector_interface_and_detected_over_declared():
    detector = HeuristicLanguageDetector()
    prediction = detector.detect("The research study and the results are presented in the paper.")
    declared_language = "es"
    assert isinstance(prediction, LanguagePrediction)
    assert prediction.language == "en"
    assert prediction.language != declared_language


def test_short_text_is_und():
    assert HeuristicLanguageDetector().detect("AI").language == "und"

