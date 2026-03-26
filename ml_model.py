"""
ML Prediction Model
Generates baseline player performance predictions used as inputs to the simulation engine.
Uses RandomForest + feature engineering with fallback historical averages.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

from core.simulation_engine import BaselinePlayer
from data.historical_data import PLAYER_HISTORICAL_STATS, VENUE_STATS


@dataclass
class MatchContext:
    venue: str
    pitch_type: str          # batting_friendly / spin_friendly / pace_friendly / balanced
    weather: str             # normal / dew / rain_interrupted
    batting_first_team: str
    toss_winner: str
    toss_decision: str       # bat / field
    team_a: str
    team_b: str


class FeatureEngineer:
    """Builds feature vectors for ML prediction."""

    def build_batting_features(
        self,
        player_name: str,
        batting_order: int,
        context: MatchContext,
        stats: Dict,
    ) -> np.ndarray:
        avg = stats.get("batting_avg", 25)
        sr = stats.get("strike_rate", 125)
        recent_form = stats.get("recent_form_runs", avg)
        venue_avg = VENUE_STATS.get(context.venue, {}).get("avg_batting_score", 145)

        pitch_bat = {"batting_friendly": 1.2, "balanced": 1.0, "spin_friendly": 0.85, "pace_friendly": 0.90}.get(context.pitch_type, 1.0)
        dew_bonus = 1.1 if context.weather == "dew" and context.batting_first_team != player_stats_team(player_name) else 1.0
        order_factor = max(0.3, 1.0 - (batting_order - 1) * 0.07)

        return np.array([
            avg, sr, recent_form, venue_avg,
            pitch_bat, dew_bonus, order_factor, batting_order,
            stats.get("t20_matches", 50), stats.get("consistency", 0.6),
        ])

    def build_bowling_features(
        self,
        player_name: str,
        context: MatchContext,
        stats: Dict,
    ) -> np.ndarray:
        wickets_per_match = stats.get("wickets_per_match", 1.2)
        economy = stats.get("economy", 7.5)
        recent_wpm = stats.get("recent_form_wickets", wickets_per_match)

        pitch_bowl = {"batting_friendly": 0.8, "balanced": 1.0, "spin_friendly": 1.2, "pace_friendly": 1.15}.get(context.pitch_type, 1.0)

        is_spinner = 1 if stats.get("bowling_type", "pace") == "spin" else 0
        spin_pitch_bonus = 1.3 if is_spinner and context.pitch_type == "spin_friendly" else 1.0

        return np.array([
            wickets_per_match, economy, recent_wpm,
            pitch_bowl, spin_pitch_bonus, is_spinner,
            stats.get("t20_bowling_matches", 50),
            stats.get("avg_overs_per_match", 3.5),
        ])


def player_stats_team(player_name: str) -> str:
    """Quick lookup helper."""
    for team, players in PLAYER_HISTORICAL_STATS.items():
        if player_name in players:
            return team
    return "unknown"


class CricketMLModel:
    """
    Trains simple regression models for runs/wickets prediction.
    Falls back gracefully to historical averages when data is sparse.
    """

    MODEL_PATH = "models/"

    def __init__(self):
        self.feat_eng = FeatureEngineer()
        self.batting_model = self._build_pipeline()
        self.bowling_model = self._build_pipeline()
        self._trained = False
        os.makedirs(self.MODEL_PATH, exist_ok=True)

    def _build_pipeline(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ])

    def train(self, training_df: pd.DataFrame):
        """
        training_df columns: player, batting_avg, strike_rate, recent_form_runs,
        wickets_per_match, economy, actual_runs, actual_wickets, ...
        """
        if training_df is None or len(training_df) < 20:
            print("[ML] Not enough training data — using fallback averages only.")
            return

        bat_df = training_df[training_df["predicted_runs"].notna()]
        bowl_df = training_df[training_df["predicted_wickets"].notna()]

        bat_features = bat_df[["batting_avg", "strike_rate", "recent_form_runs",
                                "venue_avg", "pitch_bat", "order_factor"]].values
        bowl_features = bowl_df[["wickets_per_match", "economy", "recent_wpm",
                                  "pitch_bowl", "is_spinner", "avg_overs"]].values

        self.batting_model.fit(bat_features, bat_df["actual_runs"].values)
        self.bowling_model.fit(bowl_features, bowl_df["actual_wickets"].values)
        self._trained = True
        print("[ML] Models trained successfully.")

    def save(self):
        joblib.dump(self.batting_model, f"{self.MODEL_PATH}batting_model.pkl")
        joblib.dump(self.bowling_model, f"{self.MODEL_PATH}bowling_model.pkl")

    def load(self):
        try:
            self.batting_model = joblib.load(f"{self.MODEL_PATH}batting_model.pkl")
            self.bowling_model = joblib.load(f"{self.MODEL_PATH}bowling_model.pkl")
            self._trained = True
        except FileNotFoundError:
            print("[ML] No saved model found — using fallback.")

    def predict_baseline(
        self,
        players: List[Dict],
        context: MatchContext,
    ) -> List[BaselinePlayer]:
        """
        Main method: takes raw player dicts + match context,
        returns BaselinePlayer list ready for simulation.
        """
        baselines = []

        for p in players:
            name = p["name"]
            team = p["team"]
            role = p.get("role", "bat")
            order = p.get("batting_order", 8)

            # Look up historical stats (with safe defaults)
            stats = PLAYER_HISTORICAL_STATS.get(team, {}).get(name, {})
            stats = self._apply_defaults(stats, role)

            # ── Batting baseline ───────────────────────────────
            pred_runs = self._predict_runs(name, order, stats, context)
            pred_balls = max(1, pred_runs / (stats["strike_rate"] / 100 + 0.01))
            pred_fours = pred_runs * 0.28 / 4
            pred_sixes = pred_runs * 0.12 / 6

            # ── Bowling baseline ───────────────────────────────
            pred_wickets = 0.0
            pred_overs = 0.0
            pred_runs_c = 0.0
            pred_maidens = 0.0

            if role in ("bowl", "allround"):
                pred_wickets = self._predict_wickets(name, stats, context)
                pred_overs = stats.get("avg_overs_per_match", 3.5)
                pred_runs_c = pred_overs * stats.get("economy", 7.5)
                pred_maidens = max(0, (pred_overs - pred_runs_c / 6) * 0.05)

            baselines.append(BaselinePlayer(
                name=name,
                team=team,
                role=role,
                batting_order=order,
                predicted_runs=pred_runs,
                predicted_balls=pred_balls,
                predicted_fours=pred_fours,
                predicted_sixes=pred_sixes,
                predicted_wickets=pred_wickets,
                predicted_overs=pred_overs,
                predicted_runs_conceded=pred_runs_c,
                predicted_maidens=pred_maidens,
                predicted_catches=0.15 if role != "wk" else 0.0,
                predicted_stumpings=0.25 if role == "wk" else 0.0,
                predicted_run_outs=0.05,
                out_probability=stats.get("out_probability", 0.88),
                duck_probability=stats.get("duck_probability", 0.04),
                catch_probability=0.25 if role == "wk" else 0.12,
            ))

        return baselines

    # ── private helpers ──────────────────────────────────────────

    def _predict_runs(self, name: str, order: int, stats: Dict, ctx: MatchContext) -> float:
        base = stats["batting_avg"]
        pitch_mod = {"batting_friendly": 1.2, "balanced": 1.0, "spin_friendly": 0.88, "pace_friendly": 0.92}.get(ctx.pitch_type, 1.0)
        order_mod = max(0.25, 1.0 - (order - 1) * 0.06)
        venue_mod = VENUE_STATS.get(ctx.venue, {}).get("batting_modifier", 1.0)
        form_weight = 0.4
        form_pred = stats.get("recent_form_runs", base) * form_weight + base * (1 - form_weight)
        return max(0, form_pred * pitch_mod * order_mod * venue_mod)

    def _predict_wickets(self, name: str, stats: Dict, ctx: MatchContext) -> float:
        base = stats.get("wickets_per_match", 1.0)
        pitch_mod = {"batting_friendly": 0.75, "balanced": 1.0, "spin_friendly": 1.25, "pace_friendly": 1.15}.get(ctx.pitch_type, 1.0)
        is_spinner = stats.get("bowling_type", "pace") == "spin"
        if is_spinner and ctx.pitch_type == "spin_friendly":
            pitch_mod *= 1.2
        return max(0, base * pitch_mod)

    @staticmethod
    def _apply_defaults(stats: Dict, role: str) -> Dict:
        defaults = {
            "bat": {"batting_avg": 28, "strike_rate": 130, "recent_form_runs": 25,
                    "out_probability": 0.88, "duck_probability": 0.04},
            "bowl": {"batting_avg": 10, "strike_rate": 100, "wickets_per_match": 1.2,
                     "economy": 7.8, "avg_overs_per_match": 3.5, "bowling_type": "pace",
                     "recent_form_wickets": 1.0, "out_probability": 0.92},
            "allround": {"batting_avg": 22, "strike_rate": 125, "wickets_per_match": 1.0,
                         "economy": 7.5, "avg_overs_per_match": 3.0, "bowling_type": "pace",
                         "out_probability": 0.88},
            "wk": {"batting_avg": 25, "strike_rate": 130, "out_probability": 0.88,
                   "duck_probability": 0.05},
        }
        merged = {**defaults.get(role, defaults["bat"]), **stats}
        return merged
