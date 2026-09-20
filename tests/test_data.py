import pandas as pd
import pytest

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import REQUIRED_COLUMNS, TARGET_COLUMN, VIDEO_ID_COLUMN, validate_schema
from hatedet.data.splitter import stratified_split, video_holdout_split


pytestmark = pytest.mark.requires_data


def test_load_raw_comments_has_required_columns_and_no_duplicate_text():
    df = load_raw_comments()
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df["Text"].duplicated().sum() == 0
    assert len(df) > 0


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"Text": ["hi"]})
    with pytest.raises(ValueError):
        validate_schema(df)


def test_stratified_split_preserves_class_ratio_and_no_overlap():
    df = load_raw_comments()
    train, val, test = stratified_split(df, test_size=0.2, val_size=0.1)

    assert len(train) + len(val) + len(test) == len(df)

    train_ids = set(train["CommentId"])
    val_ids = set(val["CommentId"])
    test_ids = set(test["CommentId"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    full_rate = df[TARGET_COLUMN].mean()
    test_rate = test[TARGET_COLUMN].mean()
    assert abs(full_rate - test_rate) < 0.10


def test_video_holdout_split_keeps_videos_disjoint():
    df = load_raw_comments()
    train, test = video_holdout_split(df, n_test_videos=3)

    train_videos = set(train[VIDEO_ID_COLUMN])
    test_videos = set(test[VIDEO_ID_COLUMN])
    assert train_videos.isdisjoint(test_videos)
    assert len(test_videos) == 3
    assert len(train) + len(test) == len(df)
