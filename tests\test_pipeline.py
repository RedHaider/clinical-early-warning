from clinical_warning.data import features, make_sequences
from clinical_warning.experiment import evaluate


def test_feature_shapes():
    sequence, _, _ = make_sequences(25, seed=2)
    assert features(sequence, False).shape == (25, 4)
    assert features(sequence, True).shape == (25, 12)


def test_report_metrics_are_valid():
    report = evaluate(6)
    assert all(0 <= value <= 1 for model in report.values() for group in model.values() for value in group.values())
