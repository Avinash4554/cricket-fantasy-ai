"""
Cricket Fantasy Prediction API
FastAPI backend — all prediction, simulation, and backtest endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import traceback

from core.ml_model import CricketMLModel, MatchContext
from core.simulation_engine import MonteCarloSimulator, Scenario
from core.backtest_engine import BacktestEngine
from core.scoring_engine import (
    Dream11ScoringEngine, PlayerPerformance,
    BattingPerformance, BowlingPerformance, FieldingPerformance
)
from data.historical_data import PLAYER_HISTORICAL_STATS, VENUE_STATS

app = FastAPI(
    title="Cricket Fantasy Prediction System",
    description="ML + Monte Carlo simulation engine for Dream11 fantasy points prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise engines ───────────────────────────────────────────────────────
ml_model = CricketMLModel()
simulator = MonteCarloSimulator(n_simulations=500)
backtest_engine = BacktestEngine(ml_model, simulator)
scorer = Dream11ScoringEngine()


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class PlayerInput(BaseModel):
    name: str
    team: str
    role: str = Field(..., description="bat / bowl / allround / wk")
    batting_order: int = Field(default=8, ge=1, le=11)

class MatchInput(BaseModel):
    team_a: str
    team_b: str
    batting_first_team: str
    players: List[PlayerInput]
    venue: str = "default"
    pitch_type: str = Field(default="balanced", description="batting_friendly / spin_friendly / pace_friendly / balanced")
    weather: str = Field(default="normal", description="normal / dew / rain_interrupted")
    toss_winner: str = ""
    toss_decision: str = Field(default="bat", description="bat / field")
    n_simulations: int = Field(default=500, ge=100, le=2000)
    scenarios: List[str] = []

class ManualPerformanceInput(BaseModel):
    runs: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = True
    duck: bool = False
    wickets: int = 0
    overs_bowled: float = 0.0
    runs_conceded: int = 0
    maidens: int = 0
    catches: int = 0
    stumpings: int = 0
    run_outs_direct: int = 0
    run_outs_indirect: int = 0


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Cricket Fantasy Prediction API is running 🏏"}

@app.get("/venues")
def get_venues():
    return {"venues": list(VENUE_STATS.keys())}

@app.get("/players/{team}")
def get_players(team: str):
    team_data = PLAYER_HISTORICAL_STATS.get(team)
    if not team_data:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
    return {"team": team, "players": list(team_data.keys())}

@app.get("/teams")
def get_teams():
    return {"teams": list(PLAYER_HISTORICAL_STATS.keys())}

@app.post("/predict")
def predict(match: MatchInput):
    """
    Main prediction endpoint.
    Runs ML baseline + Monte Carlo simulation for all scenarios.
    Returns full dashboard data.
    """
    try:
        ctx = MatchContext(
            venue=match.venue,
            pitch_type=match.pitch_type,
            weather=match.weather,
            batting_first_team=match.batting_first_team,
            toss_winner=match.toss_winner or match.batting_first_team,
            toss_decision=match.toss_decision,
            team_a=match.team_a,
            team_b=match.team_b,
        )

        players_dict = [p.model_dump() for p in match.players]

        # Step 1: ML baseline prediction
        baselines = ml_model.predict_baseline(players_dict, ctx)

        # Apply backtest-learned corrections
        baselines = backtest_engine.get_adjusted_baselines(baselines, [b.name for b in baselines])

        # Step 2: Choose scenarios
        active_scenarios = None
        if match.scenarios:
            active_scenarios = [Scenario(s) for s in match.scenarios if s in Scenario._value2member_map_]

        sim = MonteCarloSimulator(n_simulations=match.n_simulations)

        # Step 3: Monte Carlo simulation
        context_dict = {
            "pitch_type": match.pitch_type,
            "weather": match.weather,
            "batting_first": match.batting_first_team == match.team_a,
        }
        results = sim.run_all_scenarios(baselines, active_scenarios, context=context_dict)

        # Step 4: Summarise
        summary = sim.summarise(results)

        # Step 5: Captain / VC recommendation
        captain, vc = sim.recommend_captain(summary)

        # Step 6: Baseline breakdown for transparency
        baseline_info = [
            {
                "name": b.name,
                "team": b.team,
                "role": b.role,
                "batting_order": b.batting_order,
                "predicted_runs": round(b.predicted_runs, 1),
                "predicted_wickets": round(b.predicted_wickets, 2),
                "predicted_overs": round(b.predicted_overs, 1),
            }
            for b in baselines
        ]

        return {
            "status": "success",
            "match": {
                "team_a": match.team_a,
                "team_b": match.team_b,
                "venue": match.venue,
                "pitch_type": match.pitch_type,
                "weather": match.weather,
                "batting_first": match.batting_first_team,
            },
            "captain": captain,
            "vice_captain": vc,
            "players": summary,
            "baselines": baseline_info,
            "simulations_run": match.n_simulations,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/score/manual")
def score_manual(perf: ManualPerformanceInput):
    """Compute Dream11 points for a manually entered performance."""
    player_perf = PlayerPerformance(
        batting=BattingPerformance(
            runs=perf.runs, balls_faced=perf.balls_faced,
            fours=perf.fours, sixes=perf.sixes,
            is_out=perf.is_out, duck=perf.duck,
        ),
        bowling=BowlingPerformance(
            wickets=perf.wickets, overs_bowled=perf.overs_bowled,
            runs_conceded=perf.runs_conceded, maidens=perf.maidens,
        ),
        fielding=FieldingPerformance(
            catches=perf.catches, stumpings=perf.stumpings,
            run_outs_direct=perf.run_outs_direct,
            run_outs_indirect=perf.run_outs_indirect,
        ),
    )
    return scorer.compute(player_perf)


@app.get("/backtest/run")
def run_backtest():
    """Run backtest on sample historical data and return metrics."""
    try:
        sample_matches = backtest_engine.generate_sample_backtest_data()
        df = backtest_engine.run(sample_matches)
        results = df.to_dict(orient="records")
        return {
            "matches_tested": len(results),
            "overall_mae": round(df["mae"].mean(), 2),
            "overall_rmse": round(df["rmse"].mean(), 2),
            "avg_top5_accuracy_pct": round(df["top_5_accuracy"].mean(), 1),
            "per_match": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/weights")
def get_weights():
    """Return learned per-player prediction weights."""
    df = backtest_engine.weight_store.to_dataframe()
    if df.empty:
        return {"message": "No backtest run yet. Call /backtest/run first.", "weights": []}
    return {"weights": df.to_dict(orient="records")}

@app.get("/scenarios")
def list_scenarios():
    """List all available scenarios with descriptions."""
    from core.simulation_engine import SCENARIO_MODIFIERS
    return {
        "scenarios": [
            {"id": s.value, "label": m["label"], "description": m["description"]}
            for s, m in SCENARIO_MODIFIERS.items()
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_trained": ml_model._trained}
