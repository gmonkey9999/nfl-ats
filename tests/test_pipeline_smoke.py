import pandas as pd
from pathlib import Path


def test_core_and_enriched_exist_and_have_rows():
    core = Path('data/nfl_ats_model_dataset.csv')
    enriched = Path('data/nfl_ats_model_dataset_with_players.csv')
    assert core.exists(), f"Core ATS CSV not found: {core}"
    assert enriched.exists(), f"Enriched ATS CSV not found: {enriched}"

    df = pd.read_csv(core)
    df2 = pd.read_csv(enriched)

    assert len(df) > 0, "Core ATS CSV is empty"
    assert len(df2) > 0, "Enriched ATS CSV is empty"


def test_core_has_required_columns():
    df = pd.read_csv('data/nfl_ats_model_dataset.csv')
    for c in ('game_id', 'season'):
        assert c in df.columns, f"Required column missing in core ATS CSV: {c}"


def test_enriched_has_qb_like_columns():
    df2 = pd.read_csv('data/nfl_ats_model_dataset_with_players.csv')
    cols = set(df2.columns)
    qb_like = [c for c in cols if ('qb' in c.lower()) or ('passer' in c.lower()) or ('dropback' in c.lower())]
    assert qb_like, f"No QB-like columns found in enriched CSV. Columns: {list(cols)[:20]}"
