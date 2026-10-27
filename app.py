"""
app.py — CricketIQ 2.0
Player-Level IPL Match Winner Prediction
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_data, CURRENT_TEAMS
from src.player_features import (
    build_player_stats, build_global_player_stats, get_bvb_stats
)
from src.squad_features import (
    compute_xi_features, squad_batting_features, squad_bowling_features
)
from src.venue_features import build_venue_stats
from src.feature_pipeline import ALL_FEATURES
from src.squad_data import get_squad_data

BASE_DIR  = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cricketiq_model.pkl"

st.set_page_config(
    page_title="CricketIQ 2.0 | IPL Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);min-height:100vh;}

.hero-title{font-size:3rem;font-weight:800;
  background:linear-gradient(90deg,#f093fb,#f5576c,#fda085);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;text-align:center;margin:0;letter-spacing:-1px;}
.hero-sub{text-align:center;color:rgba(255,255,255,.5);font-size:1rem;margin-top:.4rem;margin-bottom:1.2rem;}

.sec-title{font-size:.82rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#f093fb;margin-bottom:.7rem;margin-top:.2rem;}

.glass{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
  border-radius:18px;padding:1.3rem 1.5rem;margin-bottom:.9rem;}
.glass:hover{border-color:rgba(240,147,251,.35);}

.team-badge{display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem;}
.dot-blue{width:10px;height:10px;border-radius:50%;background:#667eea;flex-shrink:0;}
.dot-red {width:10px;height:10px;border-radius:50%;background:#f5576c;flex-shrink:0;}
.team-lbl{font-size:1rem;font-weight:700;color:white;}

/* Perf badge */
.pb{display:inline-block;border-radius:12px;padding:.12rem .5rem;
    font-size:.7rem;font-weight:700;margin-left:.3rem;}
.pb-hi {background:rgba(72,199,142,.2);color:#48c78e;border:1px solid rgba(72,199,142,.4);}
.pb-md {background:rgba(250,200,70,.2);color:#fac846;border:1px solid rgba(250,200,70,.4);}
.pb-lo {background:rgba(245,87,92,.2);color:#f5576c;border:1px solid rgba(245,87,92,.4);}

/* Winner */
.winner-banner{background:linear-gradient(135deg,#667eea,#764ba2);border-radius:14px;
  padding:1rem 1.5rem;text-align:center;font-size:1.5rem;font-weight:800;color:white;
  margin:1rem 0;box-shadow:0 6px 30px rgba(102,126,234,.45);animation:glow 2s infinite;}
@keyframes glow{0%,100%{box-shadow:0 6px 30px rgba(102,126,234,.45);}
                50%{box-shadow:0 6px 50px rgba(102,126,234,.75);}}

/* Prob cards */
.prob-card{border-radius:14px;padding:1.3rem 1rem;text-align:center;}
.prob-team-lbl{font-size:.85rem;font-weight:600;color:rgba(255,255,255,.75);margin-bottom:.3rem;}
.prob-val{font-size:2.8rem;font-weight:800;line-height:1;}
.prob-desc{font-size:.7rem;color:rgba(255,255,255,.4);margin-top:.3rem;}

/* Score gauge bar */
.gauge-wrap{height:8px;background:rgba(255,255,255,.1);border-radius:4px;overflow:hidden;margin-top:.3rem;}
.gauge-fill{height:100%;border-radius:4px;}

/* Factor chips */
.chip-g{display:inline-block;background:rgba(72,199,142,.14);border:1px solid rgba(72,199,142,.4);
  color:#48c78e;border-radius:20px;padding:.25rem .7rem;font-size:.78rem;font-weight:500;margin:.12rem;}
.chip-r{display:inline-block;background:rgba(245,87,92,.14);border:1px solid rgba(245,87,92,.4);
  color:#f5576c;border-radius:20px;padding:.25rem .7rem;font-size:.78rem;font-weight:500;margin:.12rem;}

/* Matchup rows */
.mu-row{display:flex;justify-content:space-between;align-items:center;
  padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.06);}
.mu-p{font-weight:600;color:white;font-size:.86rem;}
.mu-s{color:rgba(255,255,255,.45);font-size:.78rem;}

/* Player row */
.prow{display:flex;align-items:center;gap:.4rem;padding:.28rem 0;
  border-bottom:1px solid rgba(255,255,255,.05);}
.pnum{font-size:.72rem;color:rgba(255,255,255,.3);width:1.2rem;text-align:right;}
.pname{color:white;font-weight:600;font-size:.85rem;}
.pstat{color:rgba(255,255,255,.38);font-size:.72rem;margin-left:auto;}

/* Analysis panels */
.an-head{font-size:.78rem;font-weight:700;color:#a5b4fc;letter-spacing:.8px;
  text-transform:uppercase;margin-bottom:.5rem;}
.an-row{display:flex;justify-content:space-between;
  padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,.05);}
.an-lbl{color:rgba(255,255,255,.6);font-size:.82rem;}
.an-val{color:white;font-weight:600;font-size:.82rem;}

/* Pitch meter */
.pitch-bar-wrap{background:rgba(255,255,255,.08);border-radius:8px;overflow:hidden;height:12px;margin-top:.2rem;}
.pitch-bar{height:100%;border-radius:8px;}

/* Info box */
.ibox{background:rgba(250,200,70,.1);border:1px solid rgba(250,200,70,.3);border-radius:10px;
  color:#fac846;padding:.45rem .85rem;font-size:.8rem;margin-bottom:.8rem;}

/* Verdict box */
.verdict{border-radius:14px;padding:1.2rem 1.4rem;margin-top:.8rem;
  border:2px solid rgba(255,255,255,.15);}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label{color:rgba(255,255,255,.65)!important;font-size:.8rem!important;}
div[data-testid="stRadio"] label{color:rgba(255,255,255,.65)!important;}
div[data-testid="stMetricValue"]{color:white!important;font-size:1.6rem!important;}
div[data-testid="stMetricLabel"]{color:rgba(255,255,255,.5)!important;}
.stButton>button{background:linear-gradient(135deg,#f093fb,#f5576c)!important;color:white!important;
  border:none!important;border-radius:12px!important;font-weight:700!important;font-size:1.05rem!important;
  box-shadow:0 4px 20px rgba(240,147,251,.3)!important;transition:all .3s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(240,147,251,.5)!important;}
label{color:rgba(255,255,255,.65)!important;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def perf_badge(score: int) -> str:
    if score >= 70:
        return f'<span class="pb pb-hi">⚡{score}%</span>'
    elif score >= 50:
        return f'<span class="pb pb-md">📈{score}%</span>'
    else:
        return f'<span class="pb pb-lo">📉{score}%</span>'


def role_of(player: str, gs: dict) -> str:
    s = gs.get(player, {})
    ib = s.get("is_batter", False)
    iw = s.get("is_bowler", False)
    if ib and iw:  return "all"
    if iw:         return "bowl"
    if ib:         return "bat"
    return "bat"


ROLE_EMO = {"bat": "🏏", "bowl": "🎳", "all": "⭐"}
ROLE_LBL = {"bat": "BAT", "bowl": "BOWL", "all": "ALL"}
ROLE_CLR = {"bat": "#a5b4fc", "bowl": "#fca5a5", "all": "#fde68a"}


def stat_line(player: str, gs: dict) -> str:
    s = gs.get(player, {})
    if not s:
        return "Debut / No data"
    parts = []
    if s.get("bat_innings", 0) >= 1 or s.get("bat_runs", 0) > 0:
        parts.append(f"Avg {s.get('bat_avg', 0):.1f}")
        parts.append(f"SR {s.get('bat_sr', 0):.0f}")
        if s.get("bat_runs", 0):
            parts.append(f"{s['bat_runs']}r")
    if s.get("bowl_balls", 0) >= 6 or s.get("bowl_wickets", 0) > 0:
        parts.append(f"Econ {s.get('bowl_econ', 0):.1f}")
        parts.append(f"{s.get('bowl_wickets', 0)}w")
    return " · ".join(parts) if parts else "Debut / No data"


def top_n_by(xi, gs, key, n=3, hi=True):
    scored = [(p, gs.get(p, {}).get(key, 0)) for p in xi]
    return [p for p, _ in sorted(scored, key=lambda x: x[1], reverse=hi)[:n]]


def select_bvb_bowlers(xi: list[str], gs: dict, n: int = 5) -> list[str]:
    """
    Selects legitimate frontline bowlers and primary bowling all-rounders from the Playing XI.
    Disqualifies pure specialist batters and wicketkeepers (e.g. Phil Salt, Buttler, Kohli).
    Sorts candidate bowlers by bowling authority, career balls bowled, and wickets.
    """
    candidates = []
    for p in xi:
        s = gs.get(p, {})
        role = role_of(p, gs)
        balls = s.get("bowl_balls", 0)
        wkts = s.get("bowl_wickets", 0)

        # Strictly disqualify pure batters who do not regularly bowl in IPL
        if role == "bat" and balls < 60:
            continue
        if balls == 0 and role != "bowl":
            continue

        # Priority score: specialist bowlers get top priority + proven workload
        score = (3000 if role == "bowl" else (1500 if role == "all" else 0)) + balls + (wkts * 10)
        candidates.append((p, score, balls, wkts))

    candidates.sort(key=lambda x: x[1], reverse=True)
    bowlers = [p for p, _, _, _ in candidates[:n]]

    # Fallback to reverse XI (tail-end bowlers) if fewer than n qualify
    if len(bowlers) < n:
        for p in reversed(xi):
            if p not in bowlers:
                bowlers.append(p)
            if len(bowlers) == n:
                break
    return bowlers[:n]


def select_bvb_batters(xi: list[str], gs: dict, n: int = 5) -> list[str]:
    """
    Selects top 5 primary batsmen from Playing XI according to batting order and role.
    Excludes tailenders and pure bowlers.
    """
    batters = []
    for p in xi:
        s = gs.get(p, {})
        role = role_of(p, gs)
        if role in ("bat", "all") or s.get("bat_runs", 0) >= 100:
            batters.append(p)
        if len(batters) == n:
            break
    if len(batters) < n:
        for p in xi:
            if p not in batters:
                batters.append(p)
            if len(batters) == n:
                break
    return batters[:n]


def bvb_eval(s: dict) -> tuple[str, str, str]:
    """
    Returns (label, text_color, border_color).
    Uses professional T20 cricket analytics balancing run rate and dismissal threat:
    - Net BvB Impact Index = (Runs - 18 * Dismissals) / Balls * 100.
    - If balls < 6: Neutral / Limited H2H data.
    - Bowler dominance: Multiple wickets, low avg, or heavy dot-ball pressure.
    - Batter dominance: High strike rate without getting out.
    - Balanced: Fair fight, standard T20 parity.
    """
    b = s.get("bvb_balls", 0)
    r = s.get("bvb_runs", 0)
    w = s.get("bvb_dismissals", 0)
    sr = s.get("bvb_sr", 125.0)
    avg = s.get("bvb_avg", 22.0)
    net = s.get("bvb_net_impact", 40.0)

    if b < 6:
        return "⚪ Limited H2H (<6b)", "rgba(255,255,255,0.45)", "rgba(255,255,255,0.12)"

    # Deep Bowler Dominance (Wickets threat or heavily suppressed runs)
    if (w >= 2 and avg <= 22.0) or (w >= 1 and b <= 12 and r <= 12) or (net <= 10.0 and b >= 8):
        return "🔴 Bowler Dominates", "#f87171", "rgba(248,113,113,0.4)"

    # Moderate Bowler Advantage
    if (w >= 1 and sr <= 120.0) or (sr < 108.0) or (net < 32.0):
        return "🟠 Bowler Advantage", "#fb923c", "rgba(251,146,60,0.4)"

    # Deep Batter Dominance (Heavy boundary scoring, zero/minimal dismissals)
    if (sr >= 165.0 and w == 0 and b >= 10) or (sr >= 180.0 and w <= 1 and b >= 15) or (net >= 95.0 and b >= 10):
        return "🟢 Batter Dominates", "#4ade80", "rgba(74,222,128,0.4)"

    # Moderate Batter Advantage
    if (sr >= 142.0 and w == 0) or (sr >= 152.0 and avg >= 32.0) or (net > 65.0):
        return "🟢 Batter Advantage", "#86efac", "rgba(134,239,172,0.4)"

    # Balanced / Even contest
    return "⚖️ Balanced Matchup", "#fde047", "rgba(253,224,71,0.4)"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Building player stats & IPL 2026 squad intelligence — first run ~45 s…")
def load_all():
    matches, deliveries = load_data(BASE_DIR)
    player_stats_all, bvb = build_player_stats(matches, deliveries)
    global_stats          = build_global_player_stats(matches, deliveries)
    venue_stats           = build_venue_stats(matches, deliveries)
    last_mid              = matches["id"].iloc[-1]
    current_ps            = player_stats_all.get(last_mid, {})
    venues                = sorted(matches["venue"].dropna().unique().tolist())
    squad_data            = get_squad_data(BASE_DIR)
    return matches, deliveries, player_stats_all, bvb, venue_stats, current_ps, global_stats, venues, squad_data


@st.cache_resource(show_spinner="🤖 Loading model…")
def load_model():
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict):
            return loaded
        # Robust fallback: wrap bare scikit-learn Pipeline/estimator
        FEATURE_COLS = [
            'diff_bat_avg', 'diff_bat_sr', 'diff_bat_death_sr', 'diff_bat_pp_sr',
            'diff_boundary_rate', 'diff_bowl_econ', 'diff_wicket_rate',
            'diff_pp_econ', 'diff_death_econ', 'diff_bowl_dot_rate', 'toss_venue_edge'
        ]
        return {
            "model": loaded,
            "model_name": "Symmetric Calibrated Logistic Regression",
            "features": FEATURE_COLS,
            "test_acc": 0.616,
            "roc_auc": 0.671
        }
    return None


try:
    (matches, deliveries, player_stats_all, bvb,
     venue_stats, current_ps, global_stats, venues,
     squad_data) = load_all()
except Exception as e:
    st.error(f"❌ Data load failed: {e}")
    st.stop()

model_bundle = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏏 CricketIQ 2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Player-Level IPL Match Prediction · Ball-by-Ball Intelligence · IPL 2026 Roster Engine</div>',
            unsafe_allow_html=True)

if model_bundle:
    acc = model_bundle.get("test_acc", 0)
    st.markdown(
        f'<div style="text-align:center;margin-bottom:.8rem">'
        f'<span style="background:rgba(72,199,142,.15);border:1px solid rgba(72,199,142,.4);'
        f'color:#48c78e;border-radius:20px;padding:.28rem 1rem;font-size:.82rem;font-weight:600">'
        f'✅ {model_bundle.get("model_name","Model")} · Chronological Test Accuracy: {acc:.1%} · '
        f'{len(matches):,} matches · {len(global_stats):,} players tracked'
        f'</span></div>', unsafe_allow_html=True)

st.divider()

# ── Match Setup ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">⚙️ Match Setup</div>', unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    team1_name = st.selectbox("🔵 Team 1", CURRENT_TEAMS, index=5)
with mc2:
    t2_opts = [t for t in CURRENT_TEAMS if t != team1_name]
    team2_name = st.selectbox("🔴 Team 2", t2_opts, index=0)
with mc3:
    venue = st.selectbox("🏟️ Venue", venues)

tc1, tc2 = st.columns(2)
with tc1:
    toss_w_sel = st.selectbox("🪙 Toss Winner", [team1_name, team2_name])
    toss_winner = "team1" if toss_w_sel == team1_name else "team2"
with tc2:
    toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

st.divider()

# ── Playing XI Selector ───────────────────────────────────────────────────────
st.markdown('<div class="sec-title">👥 Playing XI — Confirmed IPL 2026 Starting Rosters</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="ibox">🏏 Each team is pre-filled with their official <b>IPL 2026 Playing XI</b>. '
    'You can swap any player using the dropdowns — <b>only official 2026 squad members are listed</b>. '
    'Real-time career batting averages, strike rates, wickets, economy, and performance scores (⚡%) update dynamically for every selection.</div>',
    unsafe_allow_html=True)


def xi_selector(team: str, dot_cls: str, col_key: str, sdata: dict) -> tuple[list[str], str | None]:
    sd        = sdata.get(team, {})
    last_xi   = sd.get("last_xi", [])
    squad     = sd.get("squad", [])

    # Complete squad set
    all_squad = sorted(set(squad) | set(last_xi))

    st.markdown(
        f'<div class="team-badge"><div class="{dot_cls}"></div>'
        f'<span class="team-lbl">{team}</span>'
        f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.45);margin-left:auto;">{len(all_squad)} squad players</span>'
        f'</div>',
        unsafe_allow_html=True)

    team_slug = team.replace(" ", "_").replace("&", "_")
    chosen: list[str] = []

    def format_player(p: str) -> str:
        gs_p = global_stats.get(p, {})
        score = gs_p.get("perf_score", 40)
        sl = stat_line(p, global_stats)
        role_e = ROLE_EMO.get(role_of(p, global_stats), "🏏")
        return f"{role_e} {p}  [{score}%]  —  {sl}"

    for i in range(11):
        fallback = last_xi[i] if i < len(last_xi) else (all_squad[i] if i < len(all_squad) else all_squad[0])
        slot_key = f"{col_key}_{team_slug}_slot_{i}"

        # If slot_key is in session_state, use that value, else fallback
        current_val = st.session_state.get(slot_key, fallback)
        if current_val not in all_squad:
            current_val = fallback

        # Option list: all squad members
        sel_idx = all_squad.index(current_val) if current_val in all_squad else 0

        sel = st.selectbox(
            f"Player {i+1}",
            options=all_squad,
            index=sel_idx,
            format_func=format_player,
            key=slot_key,
        )
        chosen.append(sel)

        # Show live updated stats of the CURRENTLY SELECTED player right underneath
        s_sel = global_stats.get(sel, {})
        perf_sel = s_sel.get("perf_score", 40)
        role_sel = role_of(sel, global_stats)
        r_em = ROLE_EMO.get(role_sel, "🏏")
        r_lb = ROLE_LBL.get(role_sel, "BAT")
        r_cl = ROLE_CLR.get(role_sel, "#a5b4fc")

        stat_p = []
        if s_sel.get("bat_innings", 0) >= 1 or s_sel.get("bat_runs", 0) > 0:
            stat_p.append(f"🏏 Bat: <b>Avg {s_sel.get('bat_avg', 0):.1f}</b> · <b>SR {s_sel.get('bat_sr', 0):.0f}</b> · {s_sel.get('bat_runs', 0)}r (HS {s_sel.get('bat_hs', 0)})")
        if s_sel.get("bowl_balls", 0) >= 6 or s_sel.get("bowl_wickets", 0) > 0:
            stat_p.append(f"🎳 Bowl: <b>Econ {s_sel.get('bowl_econ', 0):.1f}</b> · <b>{s_sel.get('bowl_wickets', 0)}w</b> · SR {s_sel.get('bowl_sr', 0):.0f}")

        stat_text = " &nbsp;|&nbsp; ".join(stat_p) if stat_p else "Debut / Minimal IPL ball data"

        st.markdown(
            f"""<div style="margin-top:-6px;margin-bottom:10px;padding:5px 9px;
                 background:rgba(255,255,255,0.05);border-radius:8px;border:1px solid rgba(255,255,255,0.08);
                 font-size:0.75rem;color:rgba(255,255,255,0.85);display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                <span style="color:{r_cl};font-weight:700;font-size:0.7rem;background:{r_cl}22;padding:1px 6px;border-radius:4px;">{r_em} {r_lb}</span>
                {perf_badge(perf_sel)}
                <span>{stat_text}</span>
            </div>""",
            unsafe_allow_html=True
        )

    # ── Impact Player Option ──────────────────────────────────────────────────
    bench_options = ["None (Standard 11)"] + [p for p in all_squad if p not in chosen]
    impact_key = f"{col_key}_{team_slug}_impact"
    impact_sel = st.selectbox(
        f"⚡ {team} Designated Impact Sub",
        options=bench_options,
        index=1 if len(bench_options) > 1 else 0,
        key=impact_key,
        format_func=lambda p: p if p.startswith("None") else format_player(p),
        help="The Impact Player can be substituted tactically as an extra batter or bowler."
    )
    chosen_impact = None if impact_sel.startswith("None") else impact_sel

    # Check for duplicates
    dups = [p for p in set(chosen) if chosen.count(p) > 1]
    if dups:
        st.warning(f"⚠️ Duplicate player selected in {team}: **{', '.join(dups)}**. Please ensure 11 unique players.")

    return chosen, chosen_impact


xi_col1, xi_col2 = st.columns(2)
with xi_col1:
    team1_xi, team1_impact = xi_selector(team1_name, "dot-blue", "t1", squad_data)
with xi_col2:
    team2_xi, team2_impact = xi_selector(team2_name, "dot-red", "t2", squad_data)

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
go = st.button("🔮 Analyse & Predict Match Winner", type="primary", use_container_width=True)

if go:

    # ── Compute venue stats ────────────────────────────────────────────────────
    vm    = matches[matches["venue"] == venue]
    v_cnt = len(vm)
    if v_cnt > 0:
        vst = venue_stats.get(vm["id"].iloc[-1], {})
    else:
        vst = {}
    v_avg   = vst.get("venue_avg_score",     160.0)
    v_pp    = vst.get("venue_avg_pp_score",   45.0)
    v_death = vst.get("venue_avg_death_score",45.0)
    v_chase = vst.get("venue_chase_win_rate",  0.5)
    v_n     = vst.get("venue_matches",          0)

    # ── Squad stats for analysis ───────────────────────────────────────────────
    effective_ps = {**global_stats, **current_ps}
    t1b  = squad_batting_features(effective_ps, team1_xi)
    t2b  = squad_batting_features(effective_ps, team2_xi)
    t1bw = squad_bowling_features(effective_ps, team1_xi)
    t2bw = squad_bowling_features(effective_ps, team2_xi)

    # ── Batters vs Bowlers H2H Selection ──────────────────────────────────────
    t1_bats = select_bvb_batters(team1_xi, global_stats, 5)
    t2_bowls = select_bvb_bowlers(team2_xi, global_stats, 5)

    t2_bats = select_bvb_batters(team2_xi, global_stats, 5)
    t1_bowls = select_bvb_bowlers(team1_xi, global_stats, 5)

    # Pro BvB Matchup Integration
    bvb_net_list_t1 = []
    for bat in t1_bats:
        for bowl in t2_bowls:
            s_pair = get_bvb_stats(bvb, bat, bowl)
            if s_pair["bvb_balls"] >= 6:
                bvb_net_list_t1.append(s_pair["bvb_net_impact"])

    bvb_net_list_t2 = []
    for bat in t2_bats:
        for bowl in t1_bowls:
            s_pair = get_bvb_stats(bvb, bat, bowl)
            if s_pair["bvb_balls"] >= 6:
                bvb_net_list_t2.append(s_pair["bvb_net_impact"])

    bvb_net1 = float(np.mean(bvb_net_list_t1)) if bvb_net_list_t1 else 40.0
    bvb_net2 = float(np.mean(bvb_net_list_t2)) if bvb_net_list_t2 else 40.0
    bvb_delta = float(np.clip((bvb_net1 - bvb_net2) * 0.0012, -0.07, 0.07))

    # ── Strategic Toss & Venue Advantage ──────────────────────────────────────
    t1_won_toss = (toss_winner == "team1")
    chose_bat   = (toss_decision == "bat")
    venue_favour_chase = (v_chase >= 0.50)
    decision_optimal   = (venue_favour_chase and not chose_bat) or (not venue_favour_chase and chose_bat)
    toss_magnitude     = abs(v_chase - 0.50)
    if t1_won_toss:
        toss_venue_edge = toss_magnitude if decision_optimal else -toss_magnitude
    else:
        toss_venue_edge = -toss_magnitude if decision_optimal else toss_magnitude

    # ── Calculate Decision Factors ────────────────────────────────────────────
    factors_t1 = []
    factors_t2 = []

    checks = [
        (t1b['squad_bat_avg'],          t2b['squad_bat_avg'],          "Batting Avg",        True,  0.5, "🏏"),
        (t1b['squad_bat_sr'],           t2b['squad_bat_sr'],           "Strike Rate",        True,  2.0, "⚡"),
        (t1b['squad_top3_avg'],         t2b['squad_top3_avg'],         "Top-3 Average",      True,  1.0, "👑"),
        (t1b['squad_mid3_avg'],         t2b['squad_mid3_avg'],         "Mid-Order Avg",      True,  1.0, "💪"),
        (t1b['squad_bat_depth'],        t2b['squad_bat_depth'],        "Batting Depth",      True,  3.0, "🪜"),
        (t1b['squad_bat_death_sr'],     t2b['squad_bat_death_sr'],     "Death Bat SR",       True,  3.0, "💀"),
        (t1b['squad_bat_pp_sr'],        t2b['squad_bat_pp_sr'],        "PP Bat SR",          True,  2.0, "🔵"),
        (t1bw['squad_bowl_wicket_rate'],t2bw['squad_bowl_wicket_rate'],"Wicket Rate",        True,  0.003,"🎯"),
        (t1bw['squad_bowl_econ'],       t2bw['squad_bowl_econ'],       "Economy Rate",       False, 0.2, "🎳"),
        (t1bw['squad_pp_econ'],         t2bw['squad_pp_econ'],         "PP Economy",         False, 0.2, "🔵"),
        (t1bw['squad_death_econ'],      t2bw['squad_death_econ'],      "Death Economy",      False, 0.2, "🔴"),
        (t1bw['squad_bowl_dot_rate'],   t2bw['squad_bowl_dot_rate'],   "Dot Ball Rate",      True,  0.02,"📍"),
    ]
    for v1, v2, lbl, hi, thr, em in checks:
        diff = v1 - v2
        if hi:
            if diff >  thr: factors_t1.append(f"{em} {lbl}: {v1:.2f} vs {v2:.2f}")
            if diff < -thr: factors_t2.append(f"{em} {lbl}: {v2:.2f} vs {v1:.2f}")
        else:
            if diff < -thr: factors_t1.append(f"{em} {lbl}: {v1:.2f} vs {v2:.2f}")
            if diff >  thr: factors_t2.append(f"{em} {lbl}: {v2:.2f} vs {v1:.2f}")

    if bvb_net1 > bvb_net2 + 8.0:
        factors_t1.append(f"⚔️ Pro H2H Matchup Edge (Net BvB Index: {bvb_net1:.1f} vs {bvb_net2:.1f})")
    elif bvb_net2 > bvb_net1 + 8.0:
        factors_t2.append(f"⚔️ Pro H2H Matchup Edge (Net BvB Index: {bvb_net2:.1f} vs {bvb_net1:.1f})")

    toss_team = team1_name if toss_winner == "team1" else team2_name
    if toss_decision == "field" and v_chase > 0.53:
        toss_fact = f"🪙 {toss_team} chose to field — chasing suits {venue} ({v_chase:.0%} chase rate)"
        (factors_t1 if toss_winner == "team1" else factors_t2).append(toss_fact)
    elif toss_decision == "bat" and v_chase < 0.47:
        toss_fact = f"🪙 {toss_team} chose to bat — defending suits {venue} ({1-v_chase:.0%} defend rate)"
        (factors_t1 if toss_winner == "team1" else factors_t2).append(toss_fact)

    edge_t1 = len(factors_t1)
    edge_t2 = len(factors_t2)
    adv_diff = (edge_t1 - edge_t2) / max(edge_t1 + edge_t2, 1)

    # ── Model prediction ───────────────────────────────────────────────────────
    match_row = {
        'diff_bat_avg':       t1b.get('squad_bat_avg', 0.0) - t2b.get('squad_bat_avg', 0.0),
        'diff_bat_sr':        t1b.get('squad_bat_sr', 0.0) - t2b.get('squad_bat_sr', 0.0),
        'diff_bat_death_sr':  t1b.get('squad_bat_death_sr', 0.0) - t2b.get('squad_bat_death_sr', 0.0),
        'diff_bat_pp_sr':     t1b.get('squad_bat_pp_sr', 0.0) - t2b.get('squad_bat_pp_sr', 0.0),
        'diff_boundary_rate': t1b.get('squad_boundary_rate', 0.0) - t2b.get('squad_boundary_rate', 0.0),
        'diff_bowl_econ':     t2bw.get('squad_bowl_econ', 0.0) - t1bw.get('squad_bowl_econ', 0.0),
        'diff_wicket_rate':   t1bw.get('squad_bowl_wicket_rate', 0.0) - t2bw.get('squad_bowl_wicket_rate', 0.0),
        'diff_pp_econ':       t2bw.get('squad_pp_econ', 0.0) - t1bw.get('squad_pp_econ', 0.0),
        'diff_death_econ':    t2bw.get('squad_death_econ', 0.0) - t1bw.get('squad_death_econ', 0.0),
        'diff_bowl_dot_rate': t1bw.get('squad_bowl_dot_rate', 0.0) - t2bw.get('squad_bowl_dot_rate', 0.0),
        'toss_venue_edge':    toss_venue_edge,
    }

    if model_bundle and "features" in model_bundle:
        X_df = pd.DataFrame([{f: match_row.get(f, 0.0) for f in model_bundle["features"]}])
        proba = model_bundle["model"].predict_proba(X_df)[0]
        cls   = list(model_bundle["model"].classes_)
        raw_t1p = float(proba[cls.index(1)] if 1 in cls else proba[1])
    else:
        raw_t1p = t1b["squad_bat_avg"] / max(t1b["squad_bat_avg"] + t2b["squad_bat_avg"], 1)

    # Fuse symmetric model prediction with domain evidence differential
    base_logit = float(np.log(np.clip(raw_t1p, 0.05, 0.95) / np.clip(1.0 - raw_t1p, 0.05, 0.95)))
    fused_logit = base_logit + 0.65 * adv_diff + 1.2 * bvb_delta
    t1p = float(np.clip(1.0 / (1.0 + np.exp(-fused_logit)), 0.08, 0.92))
    t2p = 1.0 - t1p

    winner   = team1_name if t1p >= t2p else team2_name
    win_prob = max(t1p, t2p)
    conf_lbl = "High" if win_prob > .65 else ("Moderate" if win_prob > .55 else "Low")

    # ── WINNER BANNER ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="winner-banner">
        🏆 Predicted Winner: {winner}
        <span style="font-size:.88rem;font-weight:400;opacity:.75;margin-left:1rem">
        {conf_lbl} confidence · {win_prob:.0%}
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Prob Cards ─────────────────────────────────────────────────────────────
    pc1, pc2 = st.columns(2)
    for (col, team, prob, clr) in [
        (pc1, team1_name, t1p, "#667eea" if t1p >= t2p else "rgba(255,255,255,.18)"),
        (pc2, team2_name, t2p, "#f5576c" if t2p > t1p  else "rgba(255,255,255,.18)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="prob-card"
                 style="background:linear-gradient(135deg,{clr}22,{clr}44);border:2px solid {clr}66">
              <div class="prob-team-lbl">{team}</div>
              <div class="prob-val" style="color:{clr}">{prob:.1%}</div>
              <div class="prob-desc">Estimated win probability</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(prob))

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # DETAILED ANALYSIS TABS
    # ════════════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "🏏 Player Scores",
        "⚔️ BvB Matchups",
        "🏟️ Pitch & Venue",
        "🔍 Key Factors & AI Verdict",
    ])

    # ── TAB 1: Player Scores ───────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🏏 Individual Player Scorecards & Lineup")
        st.markdown(
            '<div class="ibox">Career IPL stats from 2008–2024 ball-by-ball deliveries. '
            'Performance% = normalized player score benchmarked against IPL career standards.</div>',
            unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        for (col, xi, impact_p, team, clr) in [
            (sc1, team1_xi, team1_impact, team1_name, "#667eea"),
            (sc2, team2_xi, team2_impact, team2_name, "#f5576c"),
        ]:
            with col:
                st.markdown(
                    f'<div style="font-weight:700;color:{clr};font-size:.95rem;'
                    f'margin-bottom:.6rem">🏟 {team} (11 Starters + Sub)</div>',
                    unsafe_allow_html=True)

                for i, player in enumerate(xi, 1):
                    s    = global_stats.get(player, {})
                    perf = s.get("perf_score", 40)
                    role = role_of(player, global_stats)
                    r_em = ROLE_EMO.get(role, "🏏")
                    r_lb = ROLE_LBL.get(role, "BAT")
                    r_cl = ROLE_CLR.get(role, "#a5b4fc")

                    # Build detail lines
                    bat_line  = "—"
                    bowl_line = "—"
                    form_line = "—"

                    if s.get("bat_innings", 0) >= 1 or s.get("bat_runs", 0) > 0:
                        bat_line = (
                            f"{s.get('bat_innings',0)} inn · "
                            f"{s.get('bat_runs',0)} runs · "
                            f"Avg {s.get('bat_avg',0):.1f} · "
                            f"SR {s.get('bat_sr',0):.0f} · "
                            f"HS {s.get('bat_hs',0)} · "
                            f"4s:{s.get('bat_fours',0)} 6s:{s.get('bat_sixes',0)}"
                        )
                        form_line = (
                            f"PP SR {s.get('bat_pp_sr',0):.0f} · "
                            f"Death SR {s.get('bat_death_sr',0):.0f} · "
                            f"Recent avg {s.get('bat_recent_avg',0):.1f}"
                        )

                    if s.get("bowl_balls", 0) >= 6 or s.get("bowl_wickets", 0) > 0:
                        bowl_line = (
                            f"{s.get('bowl_wickets',0)} wkts · "
                            f"Econ {s.get('bowl_econ',0):.1f} · "
                            f"SR {s.get('bowl_sr',0):.0f} · "
                            f"Dot {s.get('bowl_dot_rate',0)*100:.0f}% · "
                            f"PP Econ {s.get('bowl_pp_econ',0):.1f} · "
                            f"Death Econ {s.get('bowl_death_econ',0):.1f}"
                        )

                    # Gauge bar fill colour
                    g_clr = "#48c78e" if perf >= 70 else ("#fac846" if perf >= 50 else "#f5576c")

                    st.markdown(f"""
<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
     border-radius:12px;padding:.7rem 1rem;margin-bottom:.45rem;">
  <div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.35rem;">
    <span style="color:rgba(255,255,255,.3);font-size:.72rem">{i}</span>
    <span style="font-size:.72rem;font-weight:700;color:{r_cl};background:{r_cl}22;
                 border:1px solid {r_cl}55;border-radius:8px;padding:.1rem .4rem">{r_em}{r_lb}</span>
    <span style="color:white;font-weight:700;font-size:.88rem">{player}</span>
    {perf_badge(perf)}
  </div>
  <div class="gauge-wrap"><div class="gauge-fill"
       style="width:{perf}%;background:linear-gradient(90deg,{g_clr},{clr})"></div></div>
  <div style="margin-top:.4rem;font-size:.74rem;color:rgba(255,255,255,.55)">
    🏏 <b>Bat:</b> {bat_line}
  </div>
  <div style="font-size:.74rem;color:rgba(255,255,255,.55)">
    🎳 <b>Bowl:</b> {bowl_line}
  </div>
  <div style="font-size:.74rem;color:rgba(255,255,255,.4)">
    📈 <b>Phase:</b> {form_line}
  </div>
</div>""", unsafe_allow_html=True)

                # Render Impact Player card if specified
                if impact_p:
                    s_imp = global_stats.get(impact_p, {})
                    perf_imp = s_imp.get("perf_score", 40)
                    role_imp = role_of(impact_p, global_stats)
                    r_em_imp = ROLE_EMO.get(role_imp, "🏏")
                    r_lb_imp = ROLE_LBL.get(role_imp, "BAT")
                    r_cl_imp = ROLE_CLR.get(role_imp, "#a5b4fc")
                    g_clr_imp = "#48c78e" if perf_imp >= 70 else ("#fac846" if perf_imp >= 50 else "#f5576c")

                    bat_p_imp = f"{s_imp.get('bat_runs',0)}r (Avg {s_imp.get('bat_avg',0):.1f}, SR {s_imp.get('bat_sr',0):.0f})" if s_imp.get('bat_runs',0) > 0 else "—"
                    bowl_p_imp = f"{s_imp.get('bowl_wickets',0)}w (Econ {s_imp.get('bowl_econ',0):.1f})" if s_imp.get('bowl_balls',0) >= 6 else "—"

                    st.markdown(f"""
<div style="background:rgba(250,200,70,.06);border:1px dashed rgba(250,200,70,.5);
     border-radius:12px;padding:.7rem 1rem;margin-top:.6rem;margin-bottom:.45rem;">
  <div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.35rem;">
    <span style="font-size:.7rem;font-weight:800;color:#fac846;background:rgba(250,200,70,.2);
                 border:1px solid rgba(250,200,70,.6);border-radius:6px;padding:.1rem .4rem">⚡ IMPACT SUB</span>
    <span style="font-size:.72rem;font-weight:700;color:{r_cl_imp};background:{r_cl_imp}22;
                 border:1px solid {r_cl_imp}55;border-radius:8px;padding:.1rem .4rem">{r_em_imp}{r_lb_imp}</span>
    <span style="color:white;font-weight:700;font-size:.88rem">{impact_p}</span>
    {perf_badge(perf_imp)}
  </div>
  <div class="gauge-wrap"><div class="gauge-fill"
       style="width:{perf_imp}%;background:linear-gradient(90deg,{g_clr_imp},{clr})"></div></div>
  <div style="margin-top:.4rem;font-size:.74rem;color:rgba(255,255,255,.65)">
    🏏 <b>Bat:</b> {bat_p_imp} &nbsp;|&nbsp; 🎳 <b>Bowl:</b> {bowl_p_imp}
  </div>
</div>""", unsafe_allow_html=True)

    # ── TAB 2: BvB Matchups ────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### ⚔️ Batter vs Bowler — Head-to-Head Analysis")
        st.markdown(
            '<div class="ibox">Confidence: ✅ ≥30 balls · 🟡 10–29 balls · ⚠️ &lt;10 balls (shrinkage applied)</div>',
            unsafe_allow_html=True)

        # Top 5 primary batters vs top 5 frontline bowlers each side
        t1_bats  = select_bvb_batters(team1_xi, global_stats, 5)
        t2_bowls = select_bvb_bowlers(team2_xi, global_stats, 5)
        t2_bats  = select_bvb_batters(team2_xi, global_stats, 5)
        t1_bowls = select_bvb_bowlers(team1_xi, global_stats, 5)

        bv1, bv2 = st.columns(2)

        with bv1:
            st.markdown(
                f'<div style="font-weight:700;color:#667eea;margin-bottom:.5rem">'
                f'🏏 {team1_name} BAT  vs  🎳 {team2_name} BOWL</div>',
                unsafe_allow_html=True)
            for bat in t1_bats:
                bat_s = global_stats.get(bat, {})
                bat_avg = bat_s.get("bat_avg", 0)
                bat_sr  = bat_s.get("bat_sr", 0)
                st.markdown(
                    f'<div style="color:#a5b4fc;font-size:.8rem;font-weight:600;'
                    f'margin-top:.5rem">🏏 {bat} — Avg {bat_avg:.1f} · SR {bat_sr:.0f}</div>',
                    unsafe_allow_html=True)
                for bowl in t2_bowls:
                    s   = get_bvb_stats(bvb, bat, bowl)
                    b   = s["bvb_balls"]
                    ci  = "✅ High" if b >= 30 else ("🟡 Med" if b >= 10 else "⚠️ Low")
                    dom_lbl, dom_clr, dom_bdr = bvb_eval(s)
                    bowl_s = global_stats.get(bowl, {})
                    b_econ = bowl_s.get("bowl_econ", 8.5)
                    b_wkts = bowl_s.get("bowl_wickets", 0)
                    sr_color = "#4ade80" if s['bvb_sr'] >= 145 else ("#fde047" if s['bvb_sr'] >= 115 else "#fb923c")
                    w_color = "#f87171" if s.get('bvb_dismissals', 0) >= 1 else "rgba(255,255,255,0.5)"
                    st.markdown(f"""
<div style="background:rgba(255,255,255,.04);border-radius:10px;
     padding:.55rem .85rem;margin-bottom:.35rem;border:1px solid rgba(255,255,255,.08)">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:white;font-size:.82rem">🎳 <b>{bowl}</b>
      <span style="color:rgba(255,255,255,.45);font-size:.72rem">
        (Econ {b_econ:.1f} · {b_wkts} wkts)</span></span>
    <span style="font-size:.72rem;padding:2px 8px;border-radius:12px;font-weight:700;color:{dom_clr};border:1px solid {dom_bdr};background:{dom_clr}18">{dom_lbl}</span>
  </div>
  <div style="display:flex;gap:1.1rem;margin-top:.35rem;font-size:.76rem;align-items:center">
    <span style="color:{sr_color}">SR <b>{s['bvb_sr']:.0f}</b></span>
    <span style="color:#fca5a5">Avg <b>{s['bvb_avg']:.1f}</b></span>
    <span style="color:{w_color}">Wickets <b>{s.get('bvb_dismissals',0)}</b></span>
    <span style="color:rgba(255,255,255,.5)">{b} balls · {s.get('bvb_runs',0)}r</span>
    <span style="color:rgba(255,255,255,.3);margin-left:auto;font-size:0.7rem">{ci} sample</span>
  </div>
</div>""", unsafe_allow_html=True)

        with bv2:
            st.markdown(
                f'<div style="font-weight:700;color:#f5576c;margin-bottom:.5rem">'
                f'🏏 {team2_name} BAT  vs  🎳 {team1_name} BOWL</div>',
                unsafe_allow_html=True)
            for bat in t2_bats:
                bat_s = global_stats.get(bat, {})
                bat_avg = bat_s.get("bat_avg", 0)
                bat_sr  = bat_s.get("bat_sr", 0)
                st.markdown(
                    f'<div style="color:#fca5a5;font-size:.8rem;font-weight:600;'
                    f'margin-top:.5rem">🏏 {bat} — Avg {bat_avg:.1f} · SR {bat_sr:.0f}</div>',
                    unsafe_allow_html=True)
                for bowl in t1_bowls:
                    s   = get_bvb_stats(bvb, bat, bowl)
                    b   = s["bvb_balls"]
                    ci  = "✅ High" if b >= 30 else ("🟡 Med" if b >= 10 else "⚠️ Low")
                    dom_lbl, dom_clr, dom_bdr = bvb_eval(s)
                    bowl_s = global_stats.get(bowl, {})
                    b_econ = bowl_s.get("bowl_econ", 8.5)
                    b_wkts = bowl_s.get("bowl_wickets", 0)
                    sr_color = "#4ade80" if s['bvb_sr'] >= 145 else ("#fde047" if s['bvb_sr'] >= 115 else "#fb923c")
                    w_color = "#f87171" if s.get('bvb_dismissals', 0) >= 1 else "rgba(255,255,255,0.5)"
                    st.markdown(f"""
<div style="background:rgba(255,255,255,.04);border-radius:10px;
     padding:.55rem .85rem;margin-bottom:.35rem;border:1px solid rgba(255,255,255,.08)">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:white;font-size:.82rem">🎳 <b>{bowl}</b>
      <span style="color:rgba(255,255,255,.45);font-size:.72rem">
        (Econ {b_econ:.1f} · {b_wkts} wkts)</span></span>
    <span style="font-size:.72rem;padding:2px 8px;border-radius:12px;font-weight:700;color:{dom_clr};border:1px solid {dom_bdr};background:{dom_clr}18">{dom_lbl}</span>
  </div>
  <div style="display:flex;gap:1.1rem;margin-top:.35rem;font-size:.76rem;align-items:center">
    <span style="color:{sr_color}">SR <b>{s['bvb_sr']:.0f}</b></span>
    <span style="color:#fca5a5">Avg <b>{s['bvb_avg']:.1f}</b></span>
    <span style="color:{w_color}">Wickets <b>{s.get('bvb_dismissals',0)}</b></span>
    <span style="color:rgba(255,255,255,.5)">{b} balls · {s.get('bvb_runs',0)}r</span>
    <span style="color:rgba(255,255,255,.3);margin-left:auto;font-size:0.7rem">{ci} sample</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── TAB 3: Pitch & Venue ───────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"### 🏟️ Pitch & Venue Analysis — {venue}")
        st.markdown(
            '<div class="ibox">Venue scoring patterns from historical IPL data. '
            'Helps determine whether the pitch favours batters, pacers or spinners, '
            'and whether chasing or defending is the better strategy.</div>',
            unsafe_allow_html=True)

        if v_n > 0:
            vm_all = matches[matches["venue"] == venue].copy()
            all_ids = vm_all["id"].tolist()
            v_del = deliveries[deliveries["match_id"].isin(all_ids)]

            # First innings score distribution
            inn1 = v_del[v_del["inning"] == 1].groupby("match_id")["total_runs"].sum()
            inn2 = v_del[v_del["inning"] == 2].groupby("match_id")["total_runs"].sum()

            avg1 = inn1.mean() if len(inn1) else 160.0
            avg2 = inn2.mean() if len(inn2) else 155.0

            # PP / middle / death avg
            pp_avg   = v_del[v_del["over"] < 6]["total_runs"].sum() / max(v_n, 1)
            mid_avg  = v_del[(v_del["over"] >= 6) & (v_del["over"] < 15)]["total_runs"].sum() / max(v_n, 1)
            death_avg= v_del[v_del["over"] >= 15]["total_runs"].sum() / max(v_n, 1)

            # Chase success
            chased = sum(1 for _, r in vm_all.iterrows()
                         if r.get("toss_decision") == "field" and r["winner"] == r["toss_winner"])
            bat_first_wins = sum(1 for _, r in vm_all.iterrows()
                                 if r.get("toss_decision") == "bat" and r["winner"] == r["toss_winner"])

            # Wickets in pp vs death
            wkts_pp    = v_del[(v_del["over"] < 6) & (v_del["is_wicket"] == 1)].shape[0]
            wkts_death = v_del[(v_del["over"] >= 15) & (v_del["is_wicket"] == 1)].shape[0]

            # Highest / lowest scores
            hi_score = int(inn1.max()) if len(inn1) else 0
            lo_score = int(inn1.min()) if len(inn1) else 0

            # Classify pitch
            if avg1 > 175:
                pitch_type = "🏏 High-scoring batting paradise"
                pitch_col  = "#48c78e"
            elif avg1 > 160:
                pitch_type = "⚖️ Balanced pitch — bat & bowl"
                pitch_col  = "#fac846"
            else:
                pitch_type = "🎳 Low-scoring bowlers' track"
                pitch_col  = "#f5576c"

            # Chase vs defend
            if v_chase > 0.55:
                chase_verdict = f"✅ Chaser-friendly ({v_chase:.0%} chase win rate)"
                chase_col = "#48c78e"
            elif v_chase < 0.45:
                chase_verdict = f"🛡️ Defender-friendly ({v_chase:.0%} chase win rate)"
                chase_col = "#f5576c"
            else:
                chase_verdict = f"⚖️ Balanced ({v_chase:.0%} chase win rate)"
                chase_col = "#fac846"

            # Toss advantage
            total_toss = chased + bat_first_wins
            if total_toss > 0:
                toss_pct = (max(chased, bat_first_wins) / total_toss) * 100
                toss_advice = (
                    f"{'Field' if chased > bat_first_wins else 'Bat'} after winning toss "
                    f"({toss_pct:.0f}% win rate)"
                )
            else:
                toss_advice = "Insufficient toss data"

            # Pitch verdict for this match's toss decision
            if toss_decision == "field" and v_chase > 0.5:
                pitch_toss_match = f"✅ Toss decision matches venue pattern — chasing suits {venue}"
            elif toss_decision == "bat" and v_chase < 0.5:
                pitch_toss_match = f"✅ Toss decision matches venue pattern — defending suits {venue}"
            else:
                pitch_toss_match = f"⚠️ Toss decision goes against historical venue trend"

            # Layout
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Avg 1st Inn Score",  f"{avg1:.0f}")
            pm2.metric("Avg 2nd Inn Score",  f"{avg2:.0f}")
            pm3.metric("Matches at Venue",   str(v_n))
            pm4.metric("Highest 1st Inn",    str(hi_score))

            pm5, pm6, pm7, pm8 = st.columns(4)
            pm5.metric("PP Avg (6 ov)",       f"{pp_avg:.0f}")
            pm6.metric("Middle Avg (9 ov)",   f"{mid_avg:.0f}")
            pm7.metric("Death Avg (5 ov)",    f"{death_avg:.0f}")
            pm8.metric("Lowest 1st Inn",      str(lo_score))

            st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
     border-radius:14px;padding:1.1rem 1.4rem;margin-top:.6rem">

  <div style="margin-bottom:.8rem">
    <span style="font-size:1rem;font-weight:700;color:{pitch_col}">{pitch_type}</span>
  </div>

  <div class="an-row">
    <span class="an-lbl">🌿 Chase vs Defend</span>
    <span style="color:{chase_col};font-weight:600">{chase_verdict}</span>
  </div>
  <div class="an-row">
    <span class="an-lbl">🪙 Best Toss Decision</span>
    <span class="an-val">{toss_advice}</span>
  </div>
  <div class="an-row">
    <span class="an-lbl">🎯 Your Toss Decision</span>
    <span style="color:{'#48c78e' if '✅' in pitch_toss_match else '#fac846'};font-weight:600">{pitch_toss_match}</span>
  </div>
  <div class="an-row">
    <span class="an-lbl">🎳 PP Wickets (avg/match)</span>
    <span class="an-val">{wkts_pp / max(v_n,1):.1f} wkts in powerplay</span>
  </div>
  <div class="an-row">
    <span class="an-lbl">💥 Death Wickets (avg/match)</span>
    <span class="an-val">{wkts_death / max(v_n,1):.1f} wkts in death overs</span>
  </div>

  <div style="margin-top:.8rem">
    <div class="an-head">Phase Scoring Distribution</div>
    <div style="font-size:.78rem;color:rgba(255,255,255,.5);margin-bottom:.3rem">PP (0–5) · Mid (6–14) · Death (15–19)</div>
    <div style="display:flex;gap:.4rem;height:40px;align-items:flex-end">
      <div style="flex:1;text-align:center">
        <div style="background:#667eea;border-radius:4px 4px 0 0;
             height:{int(pp_avg / max(death_avg,1) * 32)}px"></div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.5);margin-top:.2rem">PP {pp_avg:.0f}</div>
      </div>
      <div style="flex:1;text-align:center">
        <div style="background:#f093fb;border-radius:4px 4px 0 0;
             height:{int(mid_avg / max(death_avg,1) * 32)}px"></div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.5);margin-top:.2rem">Mid {mid_avg:.0f}</div>
      </div>
      <div style="flex:1;text-align:center">
        <div style="background:#f5576c;border-radius:4px 4px 0 0;height:32px"></div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.5);margin-top:.2rem">Death {death_avg:.0f}</div>
      </div>
    </div>
  </div>

  <div style="margin-top:.9rem">
    <div class="an-head">Player Performance at {venue}</div>
    <div style="display:flex;gap:1rem">
""", unsafe_allow_html=True)

            # Top scorers and wicket-takers at venue — from all XI players
            all_xi = team1_xi + team2_xi
            venue_bat = []
            venue_bowl = []
            for p in all_xi:
                ps = global_stats.get(p, {})
                # Approximate venue average from current_ps venue stats if available
                cp = current_ps.get(p, {})
                vba = cp.get("bat_venue_avg", ps.get("bat_avg", 22.0))
                vbsr = cp.get("bat_venue_sr", ps.get("bat_sr", 125.0))
                vbe = cp.get("bowl_venue_econ", ps.get("bowl_econ", 8.5))
                inns = ps.get("bat_innings", 0)
                wkts = ps.get("bowl_wickets", 0)
                if inns >= 2:
                    venue_bat.append((p, vba, vbsr, inns))
                if wkts >= 1:
                    venue_bowl.append((p, vbe, wkts))

            venue_bat.sort(key=lambda x: x[1], reverse=True)
            venue_bowl.sort(key=lambda x: x[1])

            vb1, vb2 = st.columns(2)
            with vb1:
                st.markdown("**🏏 Best Batters at this Venue**")
                for p, va, vsr, inn in venue_bat[:5]:
                    t = team1_name if p in team1_xi else team2_name
                    dot_clr = "#667eea" if p in team1_xi else "#f5576c"
                    st.markdown(
                        f'<div class="an-row">'
                        f'<span class="an-lbl"><span style="color:{dot_clr}">●</span> {p} <span style="color:rgba(255,255,255,.3);font-size:.72rem">({t})</span></span>'
                        f'<span class="an-val">Avg {va:.1f} · SR {vsr:.0f}</span>'
                        f'</div>', unsafe_allow_html=True)

            with vb2:
                st.markdown("**🎳 Best Bowlers at this Venue**")
                for p, ve, vw in venue_bowl[:5]:
                    t = team1_name if p in team1_xi else team2_name
                    dot_clr = "#667eea" if p in team1_xi else "#f5576c"
                    st.markdown(
                        f'<div class="an-row">'
                        f'<span class="an-lbl"><span style="color:{dot_clr}">●</span> {p} <span style="color:rgba(255,255,255,.3);font-size:.72rem">({t})</span></span>'
                        f'<span class="an-val">Econ {ve:.1f} · {vw} wkts</span>'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            st.info(f"No historical IPL data found for **{venue}**.")

    # ── TAB 4: Key Factors ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 🔍 Key Decision Factors & Matchup Dynamics")

        kf1, kf2 = st.columns(2)
        with kf1:
            st.markdown(f"**✅ {team1_name} Verified Advantages ({edge_t1})**")
            if factors_t1:
                for f in factors_t1:
                    st.markdown(f'<span class="chip-g">{f}</span>', unsafe_allow_html=True)
            else:
                st.caption("No clear statistical edge found.")

        with kf2:
            st.markdown(f"**✅ {team2_name} Verified Advantages ({edge_t2})**")
            if factors_t2:
                for f in factors_t2:
                    st.markdown(f'<span class="chip-r">{f}</span>', unsafe_allow_html=True)
            else:
                st.caption("No clear statistical edge found.")

        # ── Final AI Verdict Paragraph ─────────────────────────────────────────
        winner_edges = edge_t1 if winner == team1_name else edge_t2
        loser_edges  = edge_t2 if winner == team1_name else edge_t1
        loser_team   = team2_name if winner == team1_name else team1_name
        verdict_color = "#667eea" if winner == team1_name else "#f5576c"

        win_bat = t1b if winner == team1_name else t2b
        lose_bat = t2b if winner == team1_name else t1b
        win_bowl = t1bw if winner == team1_name else t2bw
        lose_bowl = t2bw if winner == team1_name else t1bw

        # Dynamically determine the winner's actual key differentiators
        strengths = []
        if win_bat.get('squad_bat_avg', 0) > lose_bat.get('squad_bat_avg', 0) + 0.5:
            strengths.append("superior batting average")
        if win_bat.get('squad_bat_death_sr', 0) > lose_bat.get('squad_bat_death_sr', 0) + 2.0:
            strengths.append("death-overs scoring acceleration")
        if win_bat.get('squad_bat_depth', 0) > lose_bat.get('squad_bat_depth', 0) + 3.0:
            strengths.append("lineup batting depth")
        if win_bowl.get('squad_bowl_econ', 99) < lose_bowl.get('squad_bowl_econ', 99) - 0.2:
            strengths.append("disciplined bowling economy")
        if win_bowl.get('squad_bowl_wicket_rate', 0) > lose_bowl.get('squad_bowl_wicket_rate', 0) + 0.003:
            strengths.append("wicket-taking strike rate")
        if (winner == team1_name and bvb_net1 > bvb_net2 + 8) or (winner == team2_name and bvb_net2 > bvb_net1 + 8):
            strengths.append("favorable player head-to-head matchups")

        if strengths:
            strength_desc = f"Their {', '.join(strengths[:2])} are decisive differentiators at {venue}."
        else:
            strength_desc = f"Their balanced squad composition and venue compatibility provide the tactical edge."

        # Strategic toss/venue summary
        winner_won_toss = toss_winner == ("team1" if winner == team1_name else "team2")
        if (toss_decision == "field" and v_chase > 0.52 and winner_won_toss):
            chase_desc = f"Additionally, winning the toss to bowl first leverages the {v_chase:.0%} chasing bias at {venue}."
        elif (toss_decision == "bat" and v_chase < 0.48 and winner_won_toss):
            chase_desc = f"Additionally, batting first aligns with the defensive pitch profile at {venue}."
        else:
            chase_desc = f"{'Chasing' if v_chase > 0.52 else 'Defending'} has historically succeeded here ({max(v_chase, 1-v_chase):.0%})."

        verdict_text = (
            f"<b>{winner}</b> hold the statistical upper hand with <b>{winner_edges} verified advantages</b> "
            f"(vs {loser_edges} for {loser_team}). {strength_desc} {chase_desc} "
            f"The model predicts <b>{winner} with {win_prob:.0%} confidence ({conf_lbl})</b>."
        )
        st.markdown(f"""
<div class="verdict" style="background:linear-gradient(135deg,{verdict_color}18,{verdict_color}08)">
  <div style="font-size:.88rem;font-weight:700;color:{verdict_color};margin-bottom:.4rem">
    🧠 AI Verdict
  </div>
  <div style="color:rgba(255,255,255,.8);font-size:.86rem;line-height:1.6">{verdict_text}</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "⚠️ CricketIQ 2.0 is a BCA educational ML project. "
        "Data: Cricsheet.org (IPL 2008–2024). Predictions are statistical estimates only. "
        "Player performance % = normalised batting + bowling score vs IPL career averages.")

st.markdown(
    '<div style="text-align:center;padding:2rem 0 1rem;color:rgba(255,255,255,.2);font-size:.76rem">'
    'CricketIQ 2.0 · BCA Machine Learning Project · Player-Level Prediction Engine'
    '</div>', unsafe_allow_html=True)
