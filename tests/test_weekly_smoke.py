import sys
from pathlib import Path
import pandas as pd


def test_weekly_predictions_smoke(tmp_path, monkeypatch):
    """Smoke test: run weekly_predictions.main with mocked schedule + subprocess calls.

    This test avoids network calls by patching `import_schedules` to return a
    tiny schedule DataFrame, and patches `subprocess.check_call` to create the
    expected filled and prediction CSVs instead of invoking external scripts.
    The test runs the main function in an isolated tmp_path so it doesn't
    modify the repo working tree.
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import weekly_predictions

    # Build a minimal schedule: one game tomorrow
    today = pd.Timestamp.today()
    sched = pd.DataFrame([
        {
            "season": int(today.year),
            "week": 1,
            "game_id": f"{today.year}_01_HOME_AWAY",
            "home_team": "HOME",
            "away_team": "AWAY",
            "game_date": today + pd.Timedelta(days=1),
            "home_spread_close": -3.5,
            "over_under_line": 45.0,
        }
    ])

    monkeypatch.setattr(weekly_predictions, "import_schedules", lambda season: sched)

    # Fake subprocess.check_call to materialize expected files
    def fake_check_call(cmd):
        # cmd is like [python, script, --in_csv, in, --out_csv, out]
        cmd_str = " ".join(map(str, cmd))
        if "auto_fill_predict_input.py" in cmd_str:
            in_csv = cmd[cmd.index("--in_csv") + 1]
            out_csv = cmd[cmd.index("--out_csv") + 1]
            df = pd.read_csv(in_csv)
            # produce a filled CSV with minimal extra columns
            df["home_qb_epa_per_play"] = pd.NA
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            return 0
        if "predict_ats_outcomes.py" in cmd_str:
            in_csv = cmd[cmd.index("--in_csv") + 1]
            out_csv = cmd[cmd.index("--out_csv") + 1]
            df = pd.read_csv(in_csv)
            preds = df[["game_id", "season", "week", "home_team", "away_team"]].copy()
            preds["p_away_cover"] = 0.1
            preds["p_push"] = 0.1
            preds["p_home_cover"] = 0.8
            preds["pred_label"] = "home_cover"
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out_csv, index=False)
            return 0
        return 0

    monkeypatch.setattr(weekly_predictions.subprocess, "check_call", fake_check_call)

    # Run in temporary cwd so files are created under tmp_path/data/autogen
    monkeypatch.chdir(tmp_path)

    # Point model to an existing fast_run model if available in repo; otherwise use default
    model_path = repo_root / "models" / "fast_run" / "xgb_model.json"
    argv = ["weekly_predictions.py", "--py", sys.executable]
    if model_path.exists():
        argv += ["--model", str(model_path)]

    monkeypatch.setattr(sys, "argv", argv)

    # Execute main (should not raise)
    weekly_predictions.main()

    # Verify predictions file exists
    # The script writes to data/autogen/predictions_WEEK{N}.csv; find any file
    out_dir = tmp_path / "data" / "autogen"
    files = list(out_dir.glob("predictions_*.csv"))
    assert files, "No predictions CSV produced"
    # Basic sanity checks
    dfp = pd.read_csv(files[0])
    assert "p_home_cover" in dfp.columns
    assert len(dfp) >= 1
