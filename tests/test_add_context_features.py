import logging
import textwrap
import pandas as pd
import numpy as np
import pytest

import add_context_features as acf


def test_load_stadiums_dedupe_warn(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    data = textwrap.dedent("""
    team,lat,lon,roof,surface,stadium
    A,34.0,-118.25,DOME,TURF,StadiumA
    A,34.1,-118.20,DOME,TURF,StadiumA_alt
    H,40.0,-75.0,OUTDOOR,GRASS,StadiumH
    """)
    p = tmp_path / "stadiums.csv"
    p.write_text(data)

    df = acf.load_stadiums(str(p))
    # warning about duplicate rows
    assert any('duplicate stadium rows' in r.message.lower() for r in caplog.records)
    # duplicates removed (team A only one row remains)
    assert df['team'].nunique() == 2


def test_compute_venue_coords_missing_home_warn(caplog):
    caplog.set_level(logging.WARNING)
    ats = pd.DataFrame({
        'game_id': ['G1'],
        'home_team': ['X'],
        'away_team': ['A'],
    })
    stadiums = pd.DataFrame({
        'team': ['A'],
        'lat': [34.0],
        'lon': [-118.25],
        'roof': ['DOME'],
        'surface': ['TURF'],
    })

    out = acf.compute_venue_coords(ats, stadiums, None)
    # compute_venue_coords logs a warning when home stadium mapping missing
    assert any('compute_venue_coords' in r.message.lower() for r in caplog.records)
    # venue_lat should be NaN because home mapping missing
    assert pd.isna(out.loc[0, 'venue_lat'])


def test_attach_travel_distances_missing_origins_warn(caplog):
    caplog.set_level(logging.WARNING)
    ats = pd.DataFrame({
        'game_id': ['G1'],
        'home_team': ['H'],
        'away_team': ['Z'],
        'venue_lat': [40.0],
        'venue_lon': [-75.0],
    })
    # stadiums only contains H (home) but missing away Z
    stadiums = pd.DataFrame({
        'team': ['H'],
        'lat': [40.0],
        'lon': [-75.0],
    })

    out = acf.attach_travel_distances(ats, stadiums)
    assert any('attach_travel_distances' in r.message.lower() for r in caplog.records)
    # away_travel_miles should be NaN because away origin missing
    assert pd.isna(out.loc[0, 'away_travel_miles'])


def test_spread_flags_tie_and_na():
    ats = pd.DataFrame({
        'game_id': ['G1', 'G2', 'G3'],
        'home_spread_close': [-3, 0, np.nan]
    })

    out = acf.attach_spread_flags(ats)

    # G1: home favorite
    assert out.loc[0, 'home_favorite'] == 1
    assert out.loc[0, 'away_favorite'] == 0
    assert out.loc[0, 'spread_tie'] == 0

    # G2: tie
    assert out.loc[1, 'home_favorite'] == 0
    assert out.loc[1, 'away_favorite'] == 0
    assert out.loc[1, 'spread_tie'] == 1

    # G3: NaN preserved as NA (nullable Int64 dtype)
    assert pd.isna(out.loc[2, 'home_favorite'])
    assert pd.isna(out.loc[2, 'away_favorite'])
    assert pd.isna(out.loc[2, 'spread_tie'])


if __name__ == '__main__':
    pytest.main([__file__])
