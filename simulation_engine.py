"""
Monte Carlo Simulation Engine
Runs N simulations per match, varying player performance per scenario.
Completely decoupled from the ML model — plugs in baseline predictions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random

from core.scoring_engine import (
    Dream11ScoringEngine,
    PlayerPerformance,
    BattingPerformance,
    BowlingPerformance,
    FieldingPerformance,
)


class Scenario(str, Enum):
    BATTING_COLLAPSE = "batting_collapse"
    STRONG_BATTING = "strong_batting"
    BALANCED = "balanced"
    BOWLING_DOMINANCE = "bowling_dominance"
    IDEAL_FANTASY = "ideal_fantasy"


# ── Scenario modifier config ────────────────────────────────────────────────
SCENARIO_MODIFIERS: Dict[Scenario, Dict] = {
    Scenario.BATTING_COLLAPSE: {
        "label": "Batting Collapse",
        "description": "Top order fails early; lower order gets more balls; bowlers dominate",
        "batting_run_multiplier": (0.4, 0.75),   # (min, max) random range
        "batting_balls_multiplier": (0.5, 0.85),
        "bowling_wicket_multiplier": (1.5, 2.5),
        "bowling_econ_multiplier": (0.7, 0.9),   # lower economy = better
        "fielding_catch_prob_boost": 0.15,
        "top_order_penalty": 0.4,               # Top 4 batsmen hit hard
    },
    Scenario.STRONG_BATTING: {
        "label": "Strong Batting Performance",
        "description": "Top order flourishes; high SR bonuses; bowlers get fewer wickets",
        "batting_run_multiplier": (1.3, 1.8),
        "batting_balls_multiplier": (1.2, 1.5),
        "bowling_wicket_multiplier": (0.3, 0.6),
        "bowling_econ_multiplier": (1.3, 1.7),
        "fielding_catch_prob_boost": -0.05,
        "top_order_penalty": 0.0,
    },
    Scenario.BALANCED: {
        "label": "Balanced Match",
        "description": "Normal distribution of runs, wickets, and fielding",
        "batting_run_multiplier": (0.85, 1.15),
        "batting_balls_multiplier": (0.9, 1.1),
        "bowling_wicket_multiplier": (0.8, 1.2),
        "bowling_econ_multiplier": (0.9, 1.1),
        "fielding_catch_prob_boost": 0.0,
        "top_order_penalty": 0.0,
    },
    Scenario.BOWLING_DOMINANCE: {
        "label": "Bowling Dominance",
        "description": "Low scoring match; economy bonuses for bowlers; many wickets",
        "batting_run_multiplier": (0.3, 0.6),
        "batting_balls_multiplier": (0.5, 0.8),
        "bowling_wicket_multiplier": (1.8, 3.0),
        "bowling_econ_multiplier": (0.5, 0.75),
        "fielding_catch_prob_boost": 0.20,
        "top_order_penalty": 0.0,
    },
    Scenario.IDEAL_FANTASY: {
        "label": "Ideal Fantasy Scenario",
        "description": "Optimal mix: runs + wickets + fielding contributions",
        "batting_run_multiplier": (1.1, 1.5),
        "batting_balls_multiplier": (1.0, 1.3),
        "bowling_wicket_multiplier": (1.2, 1.8),
        "bowling_econ_multiplier": (0.75, 0.95),
        "fielding_catch_prob_boost": 0.10,
        "top_order_penalty": 0.0,
    },
}


@dataclass
class BaselinePlayer:
    """ML-predicted baseline for a single player."""
    name: str
    team: str
    role: str                         # bat / bowl / allround / wk
    batting_order: int = 11           # 1–11
    predicted_runs: float = 0.0
    predicted_balls: float = 0.0
    predicted_fours: float = 0.0
    predicted_sixes: float = 0.0
    predicted_wickets: float = 0.0
    predicted_overs: float = 0.0
    predicted_runs_conceded: float = 0.0
    predicted_maidens: float = 0.0
    predicted_catches: float = 0.0
    predicted_stumpings: float = 0.0
    predicted_run_outs: float = 0.0
    out_probability: float = 0.9      # P(dismissed)
    duck_probability: float = 0.05
    catch_probability: float = 0.10   # P(taking a catch per match)


@dataclass
class SimulationResult:
    player_name: str
    scenario: str
    simulations: int
    avg_points: float
    max_points: float
    min_points: float
    std_dev: float
    risk_factor: str        # Low / Medium / High
    consistency_score: float  # 0–100
    all_points: List[float] = field(default_factory=list)


class MonteCarloSimulator:
    """
    Runs N independent match simulations for each scenario.
    Each run perturbs player performance around ML baselines.
    """

    def __init__(self, n_simulations: int = 500, seed: Optional[int] = None):
        self.n = n_simulations
        self.scorer = Dream11ScoringEngine()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    # ────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────

    def run_all_scenarios(
        self,
        players: List[BaselinePlayer],
        active_scenarios: Optional[List[Scenario]] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, Dict[Scenario, SimulationResult]]:
        """
        Returns {player_name: {scenario: SimulationResult}} for all players.
        """
        scenarios = active_scenarios or list(Scenario)
        context = context or {}

        results: Dict[str, Dict[Scenario, SimulationResult]] = {}

        for scenario in scenarios:
            mods = SCENARIO_MODIFIERS[scenario]
            sim_matrix = self._run_scenario(players, mods, context)

            for player, points_list in zip(players, sim_matrix):
                if player.name not in results:
                    results[player.name] = {}
                results[player.name][scenario] = self._aggregate(
                    player.name, scenario, points_list
                )

        return results

    def summarise(
        self,
        results: Dict[str, Dict[Scenario, SimulationResult]],
    ) -> List[Dict]:
        """
        Produces a flat player summary across all scenarios for the dashboard.
        """
        summary = []
        for player_name, scenario_map in results.items():
            all_avg = [r.avg_points for r in scenario_map.values()]
            all_max = [r.max_points for r in scenario_map.values()]
            all_min = [r.min_points for r in scenario_map.values()]
            all_std = [r.std_dev for r in scenario_map.values()]

            scenario_breakdown = {
                s.value: {
                    "avg": round(r.avg_points, 2),
                    "max": round(r.max_points, 2),
                    "min": round(r.min_points, 2),
                    "risk": r.risk_factor,
                    "consistency": round(r.consistency_score, 1),
                }
                for s, r in scenario_map.items()
            }

            # Use balanced scenario as the headline
            balanced = scenario_map.get(Scenario.BALANCED)
            predicted = balanced.avg_points if balanced else np.mean(all_avg)
            std = balanced.std_dev if balanced else np.mean(all_std)

            summary.append({
                "player": player_name,
                "predicted_points": round(predicted, 2),
                "max_possible": round(max(all_max), 2),
                "min_possible": round(min(all_min), 2),
                "risk_factor": self._classify_risk(std),
                "consistency_score": round(100 - min(std / (predicted + 1e-6) * 100, 100), 1),
                "confidence_pct": round(max(0, 100 - (std / (predicted + 1e-6) * 50)), 1),
                "scenario_breakdown": scenario_breakdown,
            })

        # Sort by predicted points descending
        summary.sort(key=lambda x: x["predicted_points"], reverse=True)
        return summary

    def recommend_captain(self, summary: List[Dict]) -> Tuple[str, str]:
        """
        Returns (captain_name, vice_captain_name) based on highest avg + consistency.
        """
        scored = sorted(
            summary,
            key=lambda x: x["predicted_points"] * 0.7 + x["consistency_score"] * 0.3,
            reverse=True,
        )
        captain = scored[0]["player"] if len(scored) > 0 else "N/A"
        vc = scored[1]["player"] if len(scored) > 1 else "N/A"
        return captain, vc

    # ────────────────────────────────────────────────────────────
    # PRIVATE SIMULATION CORE
    # ────────────────────────────────────────────────────────────

    def _run_scenario(
        self,
        players: List[BaselinePlayer],
        mods: Dict,
        context: Dict,
    ) -> List[List[float]]:
        """
        Returns list-of-lists: sim_matrix[player_idx][sim_idx] = fantasy_points
        """
        matrix = [[] for _ in players]

        run_lo, run_hi = mods["batting_run_multiplier"]
        balls_lo, balls_hi = mods["batting_balls_multiplier"]
        wkt_lo, wkt_hi = mods["bowling_wicket_multiplier"]
        econ_lo, econ_hi = mods["bowling_econ_multiplier"]
        catch_boost = mods["fielding_catch_prob_boost"]
        top_penalty = mods["top_order_penalty"]

        # Context multipliers
        pitch_bat_mod = self._pitch_bat_modifier(context.get("pitch_type", "balanced"))
        pitch_bowl_mod = self._pitch_bowl_modifier(context.get("pitch_type", "balanced"))
        dew_mod = 1.1 if context.get("weather") == "dew" and context.get("batting_first") is False else 1.0

        for sim_idx in range(self.n):
            # Per-simulation global variance factor (simulates match-level randomness)
            global_noise = np.random.normal(1.0, 0.08)

            for p_idx, player in enumerate(players):
                # ── BATTING ──────────────────────────────────────
                run_mult = np.random.uniform(run_lo, run_hi) * pitch_bat_mod * dew_mod
                if player.batting_order <= 4 and top_penalty > 0:
                    run_mult *= (1.0 - top_penalty * np.random.uniform(0, 1))

                raw_runs = max(0, np.random.poisson(max(1, player.predicted_runs * run_mult * global_noise)))
                balls_mult = np.random.uniform(balls_lo, balls_hi)
                raw_balls = max(1, int(player.predicted_balls * balls_mult))

                out = np.random.random() < player.out_probability
                duck = (raw_runs == 0) and out and np.random.random() < player.duck_probability

                fours = max(0, int(np.random.poisson(player.predicted_fours * run_mult)))
                sixes = max(0, int(np.random.poisson(player.predicted_sixes * run_mult)))
                fours = min(fours, raw_runs // 4)
                sixes = min(sixes, raw_runs // 6)

                batting = BattingPerformance(
                    runs=raw_runs,
                    balls_faced=raw_balls,
                    fours=fours,
                    sixes=sixes,
                    is_out=out,
                    duck=duck,
                )

                # ── BOWLING ──────────────────────────────────────
                wkt_mult = np.random.uniform(wkt_lo, wkt_hi) * pitch_bowl_mod
                raw_wickets = 0
                raw_overs = 0.0
                raw_runs_conceded = 0
                raw_maidens = 0

                if player.predicted_overs > 0:
                    econ_mult = np.random.uniform(econ_lo, econ_hi)
                    raw_overs = max(0.1, float(np.random.normal(player.predicted_overs, 0.5)))
                    raw_wickets = max(0, int(np.random.poisson(max(0.1, player.predicted_wickets * wkt_mult))))
                    raw_wickets = min(raw_wickets, 10)
                    raw_runs_conceded = max(0, int(player.predicted_runs_conceded * econ_mult * (raw_overs / max(player.predicted_overs, 0.1))))
                    raw_maidens = max(0, int(np.random.poisson(player.predicted_maidens)))

                bowling = BowlingPerformance(
                    wickets=raw_wickets,
                    overs_bowled=round(raw_overs, 1),
                    runs_conceded=raw_runs_conceded,
                    maidens=raw_maidens,
                )

                # ── FIELDING ─────────────────────────────────────
                eff_catch_prob = min(0.99, max(0, player.catch_probability + catch_boost))
                catches = int(np.random.poisson(eff_catch_prob))
                stumpings = int(np.random.poisson(player.predicted_stumpings)) if player.role == "wk" else 0
                run_outs_direct = int(np.random.poisson(player.predicted_run_outs * 0.5))
                run_outs_indirect = int(np.random.poisson(player.predicted_run_outs * 0.5))

                fielding = FieldingPerformance(
                    catches=catches,
                    stumpings=stumpings,
                    run_outs_direct=run_outs_direct,
                    run_outs_indirect=run_outs_indirect,
                )

                perf = PlayerPerformance(batting=batting, bowling=bowling, fielding=fielding)
                score = self.scorer.compute(perf)
                matrix[p_idx].append(score["total"])

        return matrix

    def _aggregate(
        self, player_name: str, scenario: Scenario, points_list: List[float]
    ) -> SimulationResult:
        arr = np.array(points_list)
        avg = float(np.mean(arr))
        std = float(np.std(arr))
        risk = self._classify_risk(std)
        consistency = round(100 - min(std / (avg + 1e-6) * 100, 100), 1)
        return SimulationResult(
            player_name=player_name,
            scenario=scenario.value,
            simulations=len(points_list),
            avg_points=round(avg, 2),
            max_points=round(float(np.max(arr)), 2),
            min_points=round(float(np.min(arr)), 2),
            std_dev=round(std, 2),
            risk_factor=risk,
            consistency_score=max(0, consistency),
            all_points=points_list,
        )

    @staticmethod
    def _classify_risk(std: float) -> str:
        if std < 8:
            return "Low"
        elif std < 18:
            return "Medium"
        return "High"

    @staticmethod
    def _pitch_bat_modifier(pitch_type: str) -> float:
        return {"batting_friendly": 1.25, "balanced": 1.0, "spin_friendly": 0.85, "pace_friendly": 0.90}.get(pitch_type, 1.0)

    @staticmethod
    def _pitch_bowl_modifier(pitch_type: str) -> float:
        return {"batting_friendly": 0.80, "balanced": 1.0, "spin_friendly": 1.20, "pace_friendly": 1.15}.get(pitch_type, 1.0)
