"""
Backtesting & Self-Learning Engine
Predicts, compares with actuals, computes error, and adjusts feature weights.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class BacktestMatch:
    match_id: str
    players: List[Dict]           # [{name, team, role, batting_order, ...}]
    context: Dict                 # pitch_type, venue, weather, etc.
    actual_points: Dict[str, float]   # {player_name: actual_d11_points}


@dataclass
class BacktestResult:
    match_id: str
    player_errors: Dict[str, float]     # {player: abs_error}
    mae: float                          # Mean Absolute Error
    rmse: float
    top_pick_accuracy: float            # Were captain/VC right?
    player_predictions: Dict[str, float]


class PlayerWeightStore:
    """Stores per-player multiplier adjustments learned from backtesting."""

    def __init__(self):
        self.weights: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "batting_mult": 1.0,
            "bowling_mult": 1.0,
            "fielding_mult": 1.0,
            "n_matches": 0,
        })

    def update(self, player_name: str, predicted: float, actual: float):
        """
        Incremental update: nudge multipliers toward reducing error.
        Uses exponential moving average with alpha=0.2.
        """
        alpha = 0.2
        ratio = actual / (predicted + 1e-6)
        ratio = np.clip(ratio, 0.3, 3.0)   # Prevent wild swings

        w = self.weights[player_name]
        # Blend current multiplier toward the correction ratio
        w["batting_mult"] = (1 - alpha) * w["batting_mult"] + alpha * ratio
        w["bowling_mult"] = (1 - alpha) * w["bowling_mult"] + alpha * ratio
        w["n_matches"] += 1

    def get(self, player_name: str) -> Dict:
        return self.weights[player_name]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"player": k, **v} for k, v in self.weights.items()]
        return pd.DataFrame(rows)


class BacktestEngine:
    """
    Historical backtesting system.
    1. For each past match, generate predictions.
    2. Compare vs actual Dream11 points.
    3. Update PlayerWeightStore.
    4. Report accuracy metrics.
    """

    def __init__(self, ml_model, simulator):
        self.ml_model = ml_model
        self.simulator = simulator
        self.weight_store = PlayerWeightStore()
        self.results: List[BacktestResult] = []

    def run(self, matches: List[BacktestMatch]) -> pd.DataFrame:
        """Run backtest on a list of historical matches. Returns summary DataFrame."""
        print(f"[Backtest] Running {len(matches)} matches...")

        for match in matches:
            result = self._backtest_single(match)
            self.results.append(result)
            # Update weights based on errors
            for player_name, pred in result.player_predictions.items():
                actual = match.actual_points.get(player_name, 0)
                self.weight_store.update(player_name, pred, actual)

        return self._summary_dataframe()

    def _backtest_single(self, match: BacktestMatch) -> BacktestResult:
        from core.ml_model import MatchContext
        ctx = MatchContext(**{k: match.context.get(k, "") for k in [
            "venue", "pitch_type", "weather", "batting_first_team",
            "toss_winner", "toss_decision", "team_a", "team_b"
        ]})

        baselines = self.ml_model.predict_baseline(match.players, ctx)
        sim_results = self.simulator.run_all_scenarios(baselines, context=match.context)
        summary = self.simulator.summarise(sim_results)

        preds = {row["player"]: row["predicted_points"] for row in summary}
        errors = {}
        for player_name, actual in match.actual_points.items():
            pred = preds.get(player_name, 0)
            errors[player_name] = abs(pred - actual)

        error_vals = list(errors.values())
        mae = float(np.mean(error_vals)) if error_vals else 0
        rmse = float(np.sqrt(np.mean(np.array(error_vals) ** 2))) if error_vals else 0

        # Top-5 accuracy: are the top 5 predicted in the top 5 actual?
        top_pred = sorted(preds, key=preds.get, reverse=True)[:5]
        top_actual = sorted(match.actual_points, key=match.actual_points.get, reverse=True)[:5]
        overlap = len(set(top_pred) & set(top_actual))
        top_5_acc = overlap / 5.0

        return BacktestResult(
            match_id=match.match_id,
            player_errors=errors,
            mae=mae,
            rmse=rmse,
            top_pick_accuracy=top_5_acc,
            player_predictions=preds,
        )

    def _summary_dataframe(self) -> pd.DataFrame:
        rows = [{
            "match_id": r.match_id,
            "mae": round(r.mae, 2),
            "rmse": round(r.rmse, 2),
            "top_5_accuracy": round(r.top_pick_accuracy * 100, 1),
        } for r in self.results]
        df = pd.DataFrame(rows)
        if not df.empty:
            print(f"[Backtest] Overall MAE: {df['mae'].mean():.2f} | RMSE: {df['rmse'].mean():.2f} | Top-5 Acc: {df['top_5_accuracy'].mean():.1f}%")
        return df

    def get_adjusted_baselines(self, baselines, player_names: List[str]):
        """Apply learned weight corrections to baseline predictions."""
        for b in baselines:
            w = self.weight_store.get(b.name)
            b.predicted_runs *= w["batting_mult"]
            b.predicted_wickets *= w["bowling_mult"]
        return baselines

    def generate_sample_backtest_data(self) -> List[BacktestMatch]:
        """Generate synthetic historical data for demo backtesting."""
        matches = []
        np.random.seed(42)
        teams = [
            {"name": "India", "players": list({"Rohit Sharma": "bat", "Virat Kohli": "bat",
                "Suryakumar Yadav": "bat", "Hardik Pandya": "allround",
                "Jasprit Bumrah": "bowl", "Ravindra Jadeja": "allround",
                "Rishabh Pant": "wk", "Axar Patel": "bowl",
                "Mohammed Siraj": "bowl", "Kuldeep Yadav": "bowl", "Shubman Gill": "bat"}.items())},
            {"name": "Australia", "players": list({"David Warner": "bat", "Travis Head": "bat",
                "Steve Smith": "bat", "Glenn Maxwell": "allround",
                "Tim David": "bat", "Mitchell Marsh": "allround",
                "Matthew Wade": "wk", "Pat Cummins": "bowl",
                "Mitchell Starc": "bowl", "Josh Hazlewood": "bowl", "Adam Zampa": "bowl"}.items())},
        ]

        venues = ["Wankhede Stadium", "Eden Gardens", "Narendra Modi Stadium"]
        pitches = ["batting_friendly", "balanced", "spin_friendly", "pace_friendly"]

        for i in range(5):  # 5 sample backtest matches
            team_a, team_b = teams[0], teams[1]
            players = [
                {"name": n, "team": team_a["name"], "role": r, "batting_order": j + 1}
                for j, (n, r) in enumerate(team_a["players"])
            ] + [
                {"name": n, "team": team_b["name"], "role": r, "batting_order": j + 1}
                for j, (n, r) in enumerate(team_b["players"])
            ]

            # Simulated actual points
            actual = {}
            for p in players:
                base = {"bat": 45, "bowl": 40, "allround": 55, "wk": 42}.get(p["role"], 40)
                actual[p["name"]] = max(4, float(np.random.normal(base, 20)))

            matches.append(BacktestMatch(
                match_id=f"MATCH_{i+1:03d}",
                players=players,
                context={
                    "venue": venues[i % len(venues)],
                    "pitch_type": pitches[i % len(pitches)],
                    "weather": "normal",
                    "batting_first_team": team_a["name"],
                    "toss_winner": team_a["name"],
                    "toss_decision": "bat",
                    "team_a": team_a["name"],
                    "team_b": team_b["name"],
                },
                actual_points=actual,
            ))

        return matches
