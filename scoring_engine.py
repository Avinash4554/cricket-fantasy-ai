"""
Dream11 Fantasy Cricket Scoring Engine
Strictly follows Dream11 T20 scoring rules.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BattingPerformance:
    runs: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = True
    duck: bool = False


@dataclass
class BowlingPerformance:
    wickets: int = 0
    overs_bowled: float = 0.0
    runs_conceded: int = 0
    maidens: int = 0
    dots: int = 0


@dataclass
class FieldingPerformance:
    catches: int = 0
    stumpings: int = 0
    run_outs_direct: int = 0
    run_outs_indirect: int = 0


@dataclass
class PlayerPerformance:
    batting: BattingPerformance = field(default_factory=BattingPerformance)
    bowling: BowlingPerformance = field(default_factory=BowlingPerformance)
    fielding: FieldingPerformance = field(default_factory=FieldingPerformance)
    is_playing: bool = True


class Dream11ScoringEngine:
    """
    Computes Dream11 fantasy points for T20 matches.
    Rules updated as per Dream11 T20 scoring system.
    """

    # ── BATTING POINTS ──────────────────────────────────────────
    BATTING_RUN = 1              # Per run
    BATTING_BOUNDARY_4 = 1      # Per four
    BATTING_BOUNDARY_6 = 2      # Per six
    BATTING_HALF_CENTURY = 8    # 50 runs bonus
    BATTING_CENTURY = 16        # 100 runs bonus (rare in T20)
    BATTING_DUCK = -2           # Duck (batsmen only)

    # Strike rate bonuses (min 10 balls faced)
    SR_BONUS_170_PLUS = 6
    SR_BONUS_150_TO_170 = 4
    SR_BONUS_130_TO_150 = 2
    SR_PENALTY_60_TO_70 = -2
    SR_PENALTY_50_TO_60 = -4
    SR_PENALTY_BELOW_50 = -6

    # ── BOWLING POINTS ──────────────────────────────────────────
    BOWLING_WICKET = 25         # Per wicket
    BOWLING_LBW_OR_BOWLED = 8  # Extra for LBW/Bowled
    BOWLING_3_WICKETS = 4       # Bonus for 3 wickets
    BOWLING_4_WICKETS = 8       # Bonus for 4 wickets
    BOWLING_5_PLUS_WICKETS = 16 # Bonus for 5+ wickets
    BOWLING_MAIDEN = 12         # Per maiden over

    # Economy bonuses (min 2 overs bowled)
    ECON_BONUS_BELOW_5 = 6
    ECON_BONUS_5_TO_6 = 4
    ECON_BONUS_6_TO_7 = 2
    ECON_PENALTY_10_TO_11 = -2
    ECON_PENALTY_11_TO_12 = -4
    ECON_PENALTY_ABOVE_12 = -6

    # ── FIELDING POINTS ─────────────────────────────────────────
    FIELDING_CATCH = 8
    FIELDING_3_CATCHES_BONUS = 4   # Bonus for 3+ catches
    FIELDING_STUMPING = 12
    FIELDING_RUN_OUT_DIRECT = 12
    FIELDING_RUN_OUT_INDIRECT = 6

    def compute(self, perf: PlayerPerformance) -> dict:
        """
        Returns full breakdown of fantasy points for a player.
        """
        if not perf.is_playing:
            return {"total": 0, "breakdown": {}, "playing_xi_points": 0}

        playing_xi_bonus = 4  # Points for being in playing XI

        batting_pts = self._batting_points(perf.batting)
        bowling_pts = self._bowling_points(perf.bowling)
        fielding_pts = self._fielding_points(perf.fielding)

        total = playing_xi_bonus + batting_pts["total"] + bowling_pts["total"] + fielding_pts["total"]

        return {
            "total": round(total, 2),
            "playing_xi_points": playing_xi_bonus,
            "batting": batting_pts,
            "bowling": bowling_pts,
            "fielding": fielding_pts,
        }

    def _batting_points(self, b: BattingPerformance) -> dict:
        pts = {}

        pts["runs"] = b.runs * self.BATTING_RUN
        pts["fours"] = b.fours * self.BATTING_BOUNDARY_4
        pts["sixes"] = b.sixes * self.BATTING_BOUNDARY_6

        if b.runs >= 100:
            pts["milestone"] = self.BATTING_CENTURY
        elif b.runs >= 50:
            pts["milestone"] = self.BATTING_HALF_CENTURY
        else:
            pts["milestone"] = 0

        if b.duck and b.is_out:
            pts["duck"] = self.BATTING_DUCK
        else:
            pts["duck"] = 0

        pts["strike_rate_bonus"] = self._sr_bonus(b.runs, b.balls_faced)

        pts["total"] = sum(pts.values())
        return pts

    def _sr_bonus(self, runs: int, balls: int) -> float:
        if balls < 10:
            return 0
        sr = (runs / balls) * 100
        if sr >= 170:
            return self.SR_BONUS_170_PLUS
        elif sr >= 150:
            return self.SR_BONUS_150_TO_170
        elif sr >= 130:
            return self.SR_BONUS_130_TO_150
        elif sr < 50:
            return self.SR_PENALTY_BELOW_50
        elif sr < 60:
            return self.SR_PENALTY_50_TO_60
        elif sr < 70:
            return self.SR_PENALTY_60_TO_70
        return 0

    def _bowling_points(self, b: BowlingPerformance) -> dict:
        pts = {}

        pts["wickets"] = b.wickets * self.BOWLING_WICKET

        if b.wickets >= 5:
            pts["wicket_milestone"] = self.BOWLING_5_PLUS_WICKETS
        elif b.wickets >= 4:
            pts["wicket_milestone"] = self.BOWLING_4_WICKETS
        elif b.wickets >= 3:
            pts["wicket_milestone"] = self.BOWLING_3_WICKETS
        else:
            pts["wicket_milestone"] = 0

        pts["maidens"] = b.maidens * self.BOWLING_MAIDEN
        pts["economy_bonus"] = self._economy_bonus(b.runs_conceded, b.overs_bowled)

        pts["total"] = sum(pts.values())
        return pts

    def _economy_bonus(self, runs: int, overs: float) -> float:
        if overs < 2:
            return 0
        econ = runs / overs if overs > 0 else 0
        if econ < 5:
            return self.ECON_BONUS_BELOW_5
        elif econ < 6:
            return self.ECON_BONUS_5_TO_6
        elif econ < 7:
            return self.ECON_BONUS_6_TO_7
        elif econ < 11:
            return 0
        elif econ < 12:
            return self.ECON_PENALTY_10_TO_11
        elif econ < 13:
            return self.ECON_PENALTY_11_TO_12
        else:
            return self.ECON_PENALTY_ABOVE_12

    def _fielding_points(self, f: FieldingPerformance) -> dict:
        pts = {}

        pts["catches"] = f.catches * self.FIELDING_CATCH
        pts["catches_bonus"] = self.FIELDING_3_CATCHES_BONUS if f.catches >= 3 else 0
        pts["stumpings"] = f.stumpings * self.FIELDING_STUMPING
        pts["run_outs_direct"] = f.run_outs_direct * self.FIELDING_RUN_OUT_DIRECT
        pts["run_outs_indirect"] = f.run_outs_indirect * self.FIELDING_RUN_OUT_INDIRECT

        pts["total"] = sum(pts.values())
        return pts
