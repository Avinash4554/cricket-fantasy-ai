import { useState, useCallback } from "react";

const API = "http://localhost:8000";

// ─── Colour palette & constants ────────────────────────────────────────────
const ROLES = ["bat", "bowl", "allround", "wk"];
const PITCHES = ["balanced", "batting_friendly", "spin_friendly", "pace_friendly"];
const WEATHERS = ["normal", "dew", "rain_interrupted"];
const VENUES = [
  "Wankhede Stadium","Eden Gardens","Chinnaswamy Stadium",
  "MA Chidambaram Stadium","Narendra Modi Stadium","MCG","Lords","default"
];
const TEAMS = ["India","Australia","England","Pakistan"];

const SCENARIO_COLORS = {
  batting_collapse:"#ef4444", strong_batting:"#22c55e",
  balanced:"#3b82f6", bowling_dominance:"#a855f7", ideal_fantasy:"#f59e0b"
};
const RISK_COLORS = { Low:"#22c55e", Medium:"#f59e0b", High:"#ef4444" };
const ROLE_ICONS = { bat:"🏏", bowl:"⚡", allround:"🔄", wk:"🧤" };

// ─── Utility ────────────────────────────────────────────────────────────────
function cn(...classes) { return classes.filter(Boolean).join(" "); }

function Badge({ children, color = "#3b82f6" }) {
  return (
    <span style={{ background: color + "22", color, border: `1px solid ${color}44` }}
      className="px-2 py-0.5 rounded-full text-xs font-bold tracking-wide">{children}</span>
  );
}

function Card({ children, className = "", glowing = false }) {
  return (
    <div className={cn(
      "rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm p-5",
      glowing && "ring-2 ring-amber-400/30 shadow-amber-400/10 shadow-xl",
      className
    )}>{children}</div>
  );
}

function MiniBar({ value, max, color }) {
  const pct = Math.min(100, (value / (max || 1)) * 100);
  return (
    <div className="h-1.5 rounded-full bg-white/10 overflow-hidden w-24 inline-block align-middle ml-2">
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

// ─── Player Row ──────────────────────────────────────────────────────────────
function PlayerRow({ player, isCaptain, isVC, maxPts }) {
  const [open, setOpen] = useState(false);
  const scenarios = player.scenario_breakdown || {};

  return (
    <>
      <tr
        onClick={() => setOpen(o => !o)}
        className={cn(
          "cursor-pointer transition-colors border-b border-white/5",
          isCaptain ? "bg-amber-400/10" : isVC ? "bg-violet-400/10" : "hover:bg-white/5"
        )}
      >
        <td className="py-3 px-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">{ROLE_ICONS[player.role] || "🏏"}</span>
            <div>
              <div className="font-semibold text-white text-sm flex items-center gap-2">
                {player.player}
                {isCaptain && <Badge color="#f59e0b">C</Badge>}
                {isVC && <Badge color="#a855f7">VC</Badge>}
              </div>
            </div>
          </div>
        </td>
        <td className="py-3 px-4 text-center">
          <span className="text-emerald-400 font-bold text-base">{player.predicted_points}</span>
          <MiniBar value={player.predicted_points} max={maxPts} color="#10b981" />
        </td>
        <td className="py-3 px-4 text-center text-white/60 text-sm">{player.max_possible}</td>
        <td className="py-3 px-4 text-center text-white/60 text-sm">{player.min_possible}</td>
        <td className="py-3 px-4 text-center">
          <Badge color={RISK_COLORS[player.risk_factor] || "#3b82f6"}>{player.risk_factor}</Badge>
        </td>
        <td className="py-3 px-4 text-center">
          <div className="flex items-center justify-center gap-1">
            <span className="text-white/80 text-sm">{player.confidence_pct}%</span>
            <div className="h-1.5 w-16 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full rounded-full bg-blue-400" style={{ width: `${player.confidence_pct}%` }} />
            </div>
          </div>
        </td>
        <td className="py-3 px-4 text-center">
          <div className="h-1.5 w-16 rounded-full bg-white/10 overflow-hidden mx-auto">
            <div className="h-full rounded-full bg-violet-400" style={{ width: `${player.consistency_score}%` }} />
          </div>
          <span className="text-white/50 text-xs">{player.consistency_score}</span>
        </td>
        <td className="py-3 px-4 text-center text-white/40 text-xs">{open ? "▲" : "▼"}</td>
      </tr>
      {open && (
        <tr className="bg-white/[0.02] border-b border-white/5">
          <td colSpan={8} className="px-6 py-3">
            <div className="flex flex-wrap gap-3">
              {Object.entries(scenarios).map(([sc, data]) => (
                <div key={sc} className="rounded-xl p-3 text-xs min-w-36"
                  style={{ background: (SCENARIO_COLORS[sc] || "#666") + "18", border: `1px solid ${SCENARIO_COLORS[sc] || "#666"}33` }}>
                  <div className="font-bold mb-1" style={{ color: SCENARIO_COLORS[sc] || "#aaa" }}>
                    {sc.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                  <div className="text-white/70">Avg: <span className="font-semibold text-white">{data.avg}</span></div>
                  <div className="text-white/50">Max: {data.max} | Min: {data.min}</div>
                  <div className="mt-1"><Badge color={RISK_COLORS[data.risk] || "#aaa"}>{data.risk}</Badge></div>
                </div>
              ))}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ─── Point Distribution Chart ─────────────────────────────────────────────
function PointsChart({ players }) {
  if (!players || players.length === 0) return null;
  const maxPts = Math.max(...players.map(p => p.predicted_points));
  const top10 = [...players].slice(0, 10);

  return (
    <Card className="mt-6">
      <h3 className="text-white font-bold mb-4 text-sm tracking-widest uppercase opacity-70">
        Points Distribution — Top 10
      </h3>
      <div className="space-y-2">
        {top10.map((p, i) => {
          const pct = (p.predicted_points / maxPts) * 100;
          return (
            <div key={p.player} className="flex items-center gap-3 text-sm">
              <span className="text-white/40 w-4 text-right text-xs">{i + 1}</span>
              <span className="text-white/80 w-36 truncate">{p.player}</span>
              <div className="flex-1 h-5 rounded bg-white/5 overflow-hidden relative">
                <div
                  className="h-full rounded transition-all flex items-center"
                  style={{
                    width: `${pct}%`,
                    background: `linear-gradient(90deg, ${
                      i === 0 ? "#f59e0b" : i === 1 ? "#a855f7" : "#3b82f6"
                    }cc, ${i === 0 ? "#f59e0b" : i === 1 ? "#a855f7" : "#3b82f6"}44)`,
                  }}
                />
                <span className="absolute right-2 top-0 h-full flex items-center text-white/60 text-xs">
                  {p.predicted_points}
                </span>
              </div>
              <Badge color={RISK_COLORS[p.risk_factor]}>{p.risk_factor}</Badge>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ─── Player Builder ───────────────────────────────────────────────────────
function PlayerBuilder({ players, setPlayers }) {
  const [name, setName] = useState("");
  const [team, setTeam] = useState("India");
  const [role, setRole] = useState("bat");
  const [order, setOrder] = useState(1);

  const add = () => {
    if (!name.trim()) return;
    setPlayers(p => [...p, { name: name.trim(), team, role, batting_order: Number(order) }]);
    setName(""); setOrder(o => Math.min(11, o + 1));
  };

  const quick = (teamName, playerName, playerRole, playerOrder) => {
    setPlayers(p => {
      if (p.find(x => x.name === playerName)) return p;
      return [...p, { name: playerName, team: teamName, role: playerRole, batting_order: playerOrder }];
    });
  };

  const QUICK_TEAMS = {
    India: [
      ["Rohit Sharma","bat",1],["Shubman Gill","bat",2],["Virat Kohli","bat",3],
      ["Suryakumar Yadav","bat",4],["Hardik Pandya","allround",5],
      ["Rishabh Pant","wk",6],["Ravindra Jadeja","allround",7],
      ["Axar Patel","bowl",8],["Jasprit Bumrah","bowl",9],
      ["Mohammed Siraj","bowl",10],["Kuldeep Yadav","bowl",11],
    ],
    Australia: [
      ["David Warner","bat",1],["Travis Head","bat",2],["Steve Smith","bat",3],
      ["Glenn Maxwell","allround",4],["Tim David","bat",5],
      ["Mitchell Marsh","allround",6],["Matthew Wade","wk",7],
      ["Pat Cummins","bowl",8],["Mitchell Starc","bowl",9],
      ["Josh Hazlewood","bowl",10],["Adam Zampa","bowl",11],
    ],
  };

  return (
    <div>
      {/* Quick-fill buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(QUICK_TEAMS).map(([t, playerList]) => (
          <button key={t} onClick={() => playerList.forEach(([n, r, o]) => quick(t, n, r, o))}
            className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 text-xs font-semibold transition-colors">
            ⚡ {t} XI
          </button>
        ))}
        <button onClick={() => setPlayers([])}
          className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs font-semibold transition-colors">
          Clear All
        </button>
      </div>

      {/* Manual add */}
      <div className="flex flex-wrap gap-2 items-end">
        <div>
          <label className="block text-white/50 text-xs mb-1">Player Name</label>
          <input value={name} onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === "Enter" && add()}
            className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm w-44 focus:outline-none focus:border-blue-400"
            placeholder="e.g. Virat Kohli" />
        </div>
        <div>
          <label className="block text-white/50 text-xs mb-1">Team</label>
          <select value={team} onChange={e => setTeam(e.target.value)}
            className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm">
            {TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-white/50 text-xs mb-1">Role</label>
          <select value={role} onChange={e => setRole(e.target.value)}
            className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm">
            {ROLES.map(r => <option key={r} value={r}>{ROLE_ICONS[r]} {r}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-white/50 text-xs mb-1">Order</label>
          <input type="number" min={1} max={11} value={order} onChange={e => setOrder(e.target.value)}
            className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm w-16 focus:outline-none" />
        </div>
        <button onClick={add}
          className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-400 text-white text-sm font-bold transition-colors">
          + Add
        </button>
      </div>

      {/* Player list */}
      {players.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {players.map((p, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/10 text-xs">
              <span>{ROLE_ICONS[p.role]}</span>
              <span className="text-white/90 font-medium">{p.name}</span>
              <span className="text-white/40">({p.team})</span>
              <button onClick={() => setPlayers(prev => prev.filter((_, j) => j !== i))}
                className="text-red-400/70 hover:text-red-400 ml-1">×</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  // Form state
  const [players, setPlayers] = useState([]);
  const [teamA, setTeamA] = useState("India");
  const [teamB, setTeamB] = useState("Australia");
  const [battingFirst, setBattingFirst] = useState("India");
  const [venue, setVenue] = useState("Wankhede Stadium");
  const [pitchType, setPitchType] = useState("balanced");
  const [weather, setWeather] = useState("normal");
  const [nSim, setNSim] = useState(500);
  const [selectedScenarios, setSelectedScenarios] = useState([]);

  // Results state
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backtest, setBacktest] = useState(null);
  const [btLoading, setBtLoading] = useState(false);
  const [tab, setTab] = useState("predict");

  const SCENARIOS = [
    { id: "batting_collapse", label: "Batting Collapse", icon: "📉", color: "#ef4444" },
    { id: "strong_batting", label: "Strong Batting", icon: "📈", color: "#22c55e" },
    { id: "balanced", label: "Balanced", icon: "⚖️", color: "#3b82f6" },
    { id: "bowling_dominance", label: "Bowling Dom.", icon: "🎯", color: "#a855f7" },
    { id: "ideal_fantasy", label: "Ideal Fantasy", icon: "⭐", color: "#f59e0b" },
  ];

  const toggleScenario = (id) => {
    setSelectedScenarios(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);
  };

  const predict = useCallback(async () => {
    if (players.length < 2) { setError("Add at least 2 players"); return; }
    setError(""); setLoading(true); setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          team_a: teamA, team_b: teamB, batting_first_team: battingFirst,
          players, venue, pitch_type: pitchType, weather,
          n_simulations: nSim,
          scenarios: selectedScenarios,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [players, teamA, teamB, battingFirst, venue, pitchType, weather, nSim, selectedScenarios]);

  const runBacktest = async () => {
    setBtLoading(true);
    try {
      const res = await fetch(`${API}/backtest/run`);
      setBacktest(await res.json());
    } catch (e) { setError(e.message); }
    finally { setBtLoading(false); }
  };

  const maxPts = result?.players ? Math.max(...result.players.map(p => p.predicted_points)) : 100;

  return (
    <div className="min-h-screen text-white" style={{
      background: "radial-gradient(ellipse at 20% 0%, #0f172a 0%, #020617 60%, #0f1629 100%)",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    }}>
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-3xl">🏏</div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-white">
              Cricket Fantasy <span className="text-emerald-400">AI</span>
            </h1>
            <p className="text-white/40 text-xs">ML + Monte Carlo Simulation Engine</p>
          </div>
        </div>
        <div className="flex gap-1 bg-white/5 rounded-xl p-1">
          {["predict","backtest"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-semibold capitalize transition-all",
                tab === t ? "bg-blue-500 text-white" : "text-white/50 hover:text-white"
              )}>
              {t === "predict" ? "🎯 Predict" : "📊 Backtest"}
            </button>
          ))}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {tab === "predict" && (
          <>
            {/* Match Config */}
            <Card>
              <h2 className="text-white font-bold text-sm tracking-widest uppercase opacity-60 mb-4">
                Match Configuration
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {[
                  { label: "Team A", value: teamA, set: setTeamA, options: TEAMS },
                  { label: "Team B", value: teamB, set: setTeamB, options: TEAMS },
                  { label: "Batting First", value: battingFirst, set: setBattingFirst, options: [teamA, teamB] },
                  { label: "Venue", value: venue, set: setVenue, options: VENUES },
                ].map(({ label, value, set, options }) => (
                  <div key={label}>
                    <label className="block text-white/50 text-xs mb-1">{label}</label>
                    <select value={value} onChange={e => set(e.target.value)}
                      className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-400">
                      {options.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-white/50 text-xs mb-1">Pitch Type</label>
                  <select value={pitchType} onChange={e => setPitchType(e.target.value)}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none">
                    {PITCHES.map(p => <option key={p} value={p}>{p.replace("_", " ")}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-white/50 text-xs mb-1">Weather</label>
                  <select value={weather} onChange={e => setWeather(e.target.value)}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none">
                    {WEATHERS.map(w => <option key={w} value={w}>{w}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-white/50 text-xs mb-1">Simulations</label>
                  <select value={nSim} onChange={e => setNSim(Number(e.target.value))}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none">
                    {[100, 250, 500, 1000, 2000].map(n => <option key={n} value={n}>{n} runs</option>)}
                  </select>
                </div>
              </div>

              {/* Scenario toggles */}
              <div className="mt-4">
                <label className="block text-white/50 text-xs mb-2">Active Scenarios (empty = all)</label>
                <div className="flex flex-wrap gap-2">
                  {SCENARIOS.map(s => {
                    const active = selectedScenarios.includes(s.id);
                    return (
                      <button key={s.id} onClick={() => toggleScenario(s.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all border"
                        style={{
                          background: active ? s.color + "33" : "transparent",
                          borderColor: active ? s.color : "#ffffff22",
                          color: active ? s.color : "#ffffff66",
                        }}>
                        {s.icon} {s.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </Card>

            {/* Players */}
            <Card>
              <h2 className="text-white font-bold text-sm tracking-widest uppercase opacity-60 mb-4">
                Playing XI Builder <span className="text-white/30 font-normal">({players.length} added)</span>
              </h2>
              <PlayerBuilder players={players} setPlayers={setPlayers} />
            </Card>

            {/* Predict button */}
            <div className="flex gap-3 items-center">
              <button onClick={predict} disabled={loading}
                className={cn(
                  "px-8 py-3 rounded-xl font-black text-base tracking-wide transition-all",
                  loading
                    ? "bg-blue-500/40 text-white/50 cursor-not-allowed"
                    : "bg-gradient-to-r from-blue-500 to-emerald-500 hover:from-blue-400 hover:to-emerald-400 text-white shadow-lg shadow-blue-500/25"
                )}>
                {loading ? "⚙️ Running Simulations…" : "🚀 Run Prediction Engine"}
              </button>
              {loading && (
                <div className="text-white/50 text-sm animate-pulse">
                  Running {nSim} Monte Carlo simulations across 5 scenarios…
                </div>
              )}
            </div>

            {error && (
              <div className="rounded-xl bg-red-500/15 border border-red-500/30 px-4 py-3 text-red-400 text-sm">
                ⚠️ {error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div>
                {/* Summary cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {[
                    { label: "👑 Captain", value: result.captain, color: "#f59e0b" },
                    { label: "🥈 Vice Captain", value: result.vice_captain, color: "#a855f7" },
                    { label: "🔬 Simulations", value: result.simulations_run?.toLocaleString(), color: "#3b82f6" },
                    { label: "👥 Players Analysed", value: result.players?.length, color: "#22c55e" },
                  ].map(({ label, value, color }) => (
                    <Card key={label} glowing={label.includes("Captain")}>
                      <div className="text-white/50 text-xs mb-1">{label}</div>
                      <div className="font-black text-base" style={{ color }}>{value}</div>
                    </Card>
                  ))}
                </div>

                {/* Main table */}
                <Card className="overflow-hidden p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-white/40 text-xs uppercase tracking-wider">
                          {["Player","Predicted Pts","Max","Min","Risk","Confidence","Consistency",""].map(h => (
                            <th key={h} className="py-3 px-4 text-left font-semibold">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.players.map(p => (
                          <PlayerRow key={p.player} player={p}
                            isCaptain={p.player === result.captain}
                            isVC={p.player === result.vice_captain}
                            maxPts={maxPts} />
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                <PointsChart players={result.players} />

                <p className="text-white/30 text-xs text-center mt-4">
                  Click any row to expand scenario breakdown · Predictions based on Monte Carlo simulation + ML baselines
                </p>
              </div>
            )}
          </>
        )}

        {tab === "backtest" && (
          <div className="space-y-6">
            <Card>
              <h2 className="font-bold text-white mb-2">📊 Self-Learning Backtesting Engine</h2>
              <p className="text-white/50 text-sm mb-4">
                Runs prediction on historical matches, compares with actual Dream11 points, and adjusts
                player weight multipliers to reduce future prediction error.
              </p>
              <button onClick={runBacktest} disabled={btLoading}
                className="px-6 py-2.5 rounded-xl font-bold bg-violet-500 hover:bg-violet-400 text-white transition-colors disabled:opacity-50">
                {btLoading ? "⚙️ Running Backtest…" : "▶ Run Backtest on Sample Data"}
              </button>
            </Card>

            {backtest && (
              <>
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: "Overall MAE", value: backtest.overall_mae, color: "#f59e0b", unit: "pts" },
                    { label: "Overall RMSE", value: backtest.overall_rmse, color: "#ef4444", unit: "pts" },
                    { label: "Top-5 Accuracy", value: backtest.avg_top5_accuracy_pct, color: "#22c55e", unit: "%" },
                  ].map(({ label, value, color, unit }) => (
                    <Card key={label}>
                      <div className="text-white/50 text-xs mb-1">{label}</div>
                      <div className="text-2xl font-black" style={{ color }}>{value}{unit}</div>
                    </Card>
                  ))}
                </div>

                <Card className="overflow-hidden p-0">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-white/40 text-xs uppercase tracking-wider">
                        {["Match","MAE","RMSE","Top-5 Accuracy"].map(h => (
                          <th key={h} className="py-3 px-5 text-left font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {backtest.per_match?.map((m, i) => (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                          <td className="py-3 px-5 font-semibold text-white">{m.match_id}</td>
                          <td className="py-3 px-5 text-amber-400">{m.mae}</td>
                          <td className="py-3 px-5 text-red-400">{m.rmse}</td>
                          <td className="py-3 px-5">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 w-24 rounded-full bg-white/10 overflow-hidden">
                                <div className="h-full bg-emerald-400 rounded-full"
                                  style={{ width: `${m.top_5_accuracy}%` }} />
                              </div>
                              <span className="text-emerald-400">{m.top_5_accuracy}%</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
