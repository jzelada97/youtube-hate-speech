from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.splitter import stratified_split
from hatedet.models.baseline import build_baseline_pipeline


def test_pipeline_fits_and_predicts_on_real_data():
    df = load_raw_comments()
    train, _, test = stratified_split(df)

    pipeline = build_baseline_pipeline()
    pipeline.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])

    preds = pipeline.predict(test[TEXT_COLUMN])
    probs = pipeline.predict_proba(test[TEXT_COLUMN])

    assert len(preds) == len(test)
    assert probs.shape == (len(test), 2)
    assert set(preds.tolist()) <= {True, False}


def test_pipeline_is_a_single_serializable_object():
    """Regresion de la decision 'sin skew train/serve': todo debe vivir en un unico Pipeline."""
    pipeline = build_baseline_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["cleaner", "normalizer", "tfidf", "clf"]


def test_generalization_gap_meets_client_requirement():
    """NFR-1 del cliente: gap train/test en F1-macro < 5 puntos porcentuales."""
    from hatedet.models.evaluate import evaluate, generalization_gap_pp

    df = load_raw_comments()
    train, _, test = stratified_split(df)

    pipeline = build_baseline_pipeline()
    pipeline.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])

    train_report = evaluate(
        train[TARGET_COLUMN],
        pipeline.predict(train[TEXT_COLUMN]),
        pipeline.predict_proba(train[TEXT_COLUMN])[:, 1],
    )
    test_report = evaluate(
        test[TARGET_COLUMN],
        pipeline.predict(test[TEXT_COLUMN]),
        pipeline.predict_proba(test[TEXT_COLUMN])[:, 1],
    )

    gap = generalization_gap_pp(train_report, test_report)
    assert gap < 5.0
