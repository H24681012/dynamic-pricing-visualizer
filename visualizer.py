"""
Interactive Dynamic Pricing Visualizer — Streamlit App
Run: streamlit run visualizer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict

# ── Engine (copied from notebook cells 2-4) ──────────────────────────

@dataclass
class Segment:
    name: str
    N: int
    v_mean: float
    v_std: float
    beta: float
    delta: float
    gamma: float

COLORS = {
    'hardcore': '#e63946', 'casual': '#457b9d', 'bargain': '#2a9d8f',
    'uniform': '#6c757d', 'segment': '#e63946', 'fairness': '#f4a261',
}

def mnl_purchase_prob(v, beta, price, delta, expected_cs_next,
                      gamma=0.0, price_bar=0.0, fairness=False):
    u_buy = beta * (v - price)
    if fairness and price > price_bar:
        u_buy -= gamma * (price - price_bar)
    u_wait = delta * expected_cs_next
    u_exit = 0.0
    u_max = max(u_buy, u_wait, u_exit)
    exp_buy = np.exp(u_buy - u_max)
    exp_wait = np.exp(u_wait - u_max)
    exp_exit = np.exp(u_exit - u_max)
    return exp_buy / (exp_buy + exp_wait + exp_exit)

def expected_consumer_surplus(v, beta, price, delta, expected_cs_next,
                               gamma=0.0, price_bar=0.0, fairness=False):
    u_buy = beta * (v - price)
    if fairness and price > price_bar:
        u_buy -= gamma * (price - price_bar)
    u_wait = delta * expected_cs_next
    u_exit = 0.0
    u_max = max(u_buy, u_wait, u_exit)
    return u_max + np.log(np.exp(u_buy - u_max) + np.exp(u_wait - u_max) + np.exp(u_exit - u_max))

@dataclass
class DPResult:
    prices: Dict[str, List[float]]
    revenues: Dict[str, List[float]]
    quantities: Dict[str, List[float]]
    remaining: Dict[str, List[float]]
    total_revenue: float
    surplus: Dict[str, float]
    scenario: str

def solve_dp(segments, T, valuation_decay, scenario='uniform',
             sequel_hazard=0.0, hazard_start=0, salvage_value=0.0,
             price_grid=None):
    if price_grid is None:
        price_grid = np.arange(5, 71, 1.0)

    use_fairness = (scenario == 'fairness')
    use_segment = (scenario in ('segment', 'fairness'))

    def v_at(seg, t):
        return seg.v_mean * max(0.0, 1 - valuation_decay * t)

    def q_at(t):
        if sequel_hazard <= 0 or t <= hazard_start:
            return 0.0
        return sequel_hazard

    ecs = {s.name: [0.0] * (T + 1) for s in segments}
    vf = {s.name: [0.0] * (T + 1) for s in segments}
    optimal_prices = {s.name: [0.0] * T for s in segments}

    v_salvage = {}
    for s in segments:
        if salvage_value > 0:
            v_end = v_at(s, T)
            prob_s = mnl_purchase_prob(v_end, s.beta, salvage_value, 0, 0)
            v_salvage[s.name] = prob_s * salvage_value
        else:
            v_salvage[s.name] = 0.0

    for t in range(T - 1, -1, -1):
        q_next = q_at(t + 1) if t + 1 <= T else 0.0

        if use_segment:
            candidate_prices, candidate_vals = [], []
            for s in segments:
                v = v_at(s, t)
                ecs_next = (1 - q_next) * ecs[s.name][t + 1]
                vf_cont = (1 - q_next) * vf[s.name][t + 1] + q_next * v_salvage[s.name]
                best_p, best_val = price_grid[0], -np.inf
                for p in price_grid:
                    prob = mnl_purchase_prob(v, s.beta, p, s.delta, ecs_next)
                    val = prob * p + (1 - prob) * vf_cont
                    if val > best_val:
                        best_val = val; best_p = p
                candidate_prices.append(best_p); candidate_vals.append(best_val)

            if use_fairness:
                for _ in range(3):
                    p_bar = np.mean(candidate_prices)
                    new_prices, new_vals = [], []
                    for i, s in enumerate(segments):
                        v = v_at(s, t)
                        ecs_next = (1 - q_next) * ecs[s.name][t + 1]
                        vf_cont = (1 - q_next) * vf[s.name][t + 1] + q_next * v_salvage[s.name]
                        best_p, best_val = price_grid[0], -np.inf
                        for p in price_grid:
                            prob = mnl_purchase_prob(v, s.beta, p, s.delta, ecs_next,
                                                     gamma=s.gamma, price_bar=p_bar, fairness=True)
                            val = prob * p + (1 - prob) * vf_cont
                            if val > best_val:
                                best_val = val; best_p = p
                        new_prices.append(best_p); new_vals.append(best_val)
                    candidate_prices = new_prices; candidate_vals = new_vals

            p_bar = np.mean(candidate_prices) if use_fairness else 0.0
            for i, s in enumerate(segments):
                optimal_prices[s.name][t] = candidate_prices[i]
                v = v_at(s, t); p = candidate_prices[i]
                ecs_next = (1 - q_next) * ecs[s.name][t + 1]
                vf_cont = (1 - q_next) * vf[s.name][t + 1] + q_next * v_salvage[s.name]
                prob = mnl_purchase_prob(v, s.beta, p, s.delta, ecs_next,
                                         gamma=s.gamma, price_bar=p_bar, fairness=use_fairness)
                vf[s.name][t] = prob * p + (1 - prob) * vf_cont
                ecs[s.name][t] = expected_consumer_surplus(v, s.beta, p, s.delta, ecs_next,
                                         gamma=s.gamma, price_bar=p_bar, fairness=use_fairness)
        else:
            best_p, best_total = price_grid[0], -np.inf
            for p in price_grid:
                total = 0.0
                for s in segments:
                    v = v_at(s, t)
                    ecs_next = (1 - q_next) * ecs[s.name][t + 1]
                    vf_cont = (1 - q_next) * vf[s.name][t + 1] + q_next * v_salvage[s.name]
                    prob = mnl_purchase_prob(v, s.beta, p, s.delta, ecs_next)
                    total += s.N * (prob * p + (1 - prob) * vf_cont)
                if total > best_total:
                    best_total = total; best_p = p
            for s in segments:
                optimal_prices[s.name][t] = best_p
                v = v_at(s, t)
                ecs_next = (1 - q_next) * ecs[s.name][t + 1]
                vf_cont = (1 - q_next) * vf[s.name][t + 1] + q_next * v_salvage[s.name]
                prob = mnl_purchase_prob(v, s.beta, best_p, s.delta, ecs_next)
                vf[s.name][t] = prob * best_p + (1 - prob) * vf_cont
                ecs[s.name][t] = expected_consumer_surplus(v, s.beta, best_p, s.delta, ecs_next)

    # Forward simulation
    result_prices = {s.name: [] for s in segments}
    result_revenues = {s.name: [] for s in segments}
    result_quantities = {s.name: [] for s in segments}
    result_remaining = {s.name: [] for s in segments}
    result_surplus = {s.name: 0.0 for s in segments}
    N_rem = {s.name: float(s.N) for s in segments}
    survival_prob = 1.0

    for t in range(T):
        prices_at_t = [optimal_prices[s.name][t] for s in segments]
        p_bar = np.mean(prices_at_t) if use_fairness else 0.0
        q_next = q_at(t + 1) if t + 1 <= T else 0.0
        for s in segments:
            v = v_at(s, t); p = optimal_prices[s.name][t]
            ecs_next = (1 - q_next) * ecs[s.name][t + 1]
            prob = mnl_purchase_prob(v, s.beta, p, s.delta, ecs_next,
                                     gamma=s.gamma, price_bar=p_bar, fairness=use_fairness)
            n_buy = N_rem[s.name] * prob
            rev = n_buy * p * survival_prob
            result_surplus[s.name] += n_buy * max(0, v - p) * survival_prob
            result_prices[s.name].append(p)
            result_revenues[s.name].append(rev)
            result_quantities[s.name].append(n_buy * survival_prob)
            result_remaining[s.name].append(N_rem[s.name])
            N_rem[s.name] -= n_buy
        survival_prob *= (1 - q_next)

    salvage_rev = 0.0
    if salvage_value > 0 and sequel_hazard > 0:
        for s in segments:
            salvage_rev += N_rem[s.name] * v_salvage[s.name]

    total_rev = sum(sum(result_revenues[s.name]) for s in segments) + salvage_rev
    return DPResult(result_prices, result_revenues, result_quantities,
                    result_remaining, total_rev, result_surplus, scenario)


# ── Streamlit App ─────────────────────────────────────────────────────

st.set_page_config(page_title="Dynamic Pricing Visualizer", layout="wide")
st.title("Dynamic Pricing Simulation")
st.caption("Personalized vs. Uniform Pricing on PlayStation Store  |  HBA 4520")

# ── Sidebar: Experiment picker + parameter tweaks ─────────────────────
st.sidebar.header("Experiment")
experiment = st.sidebar.radio("Select experiment:", ["FIFA (12 months)", "Spider-Man (24 quarters)"])

is_fifa = experiment.startswith("FIFA")

st.sidebar.markdown("---")
st.sidebar.header("Tweak Parameters")

if is_fifa:
    T = st.sidebar.slider("Horizon (months)", 4, 24, 12)
    valuation_decay = st.sidebar.slider("Valuation decay (%/period)", 0.0, 10.0, 3.0, 0.5) / 100
    sequel_hazard = 0.0
    hazard_start = 0
    salvage_value = 0.0
    period_label = "Month"
else:
    T = st.sidebar.slider("Horizon (quarters)", 8, 36, 24)
    valuation_decay = st.sidebar.slider("Valuation decay (%/period)", 0.0, 5.0, 0.8, 0.1) / 100
    sequel_hazard = st.sidebar.slider("Sequel hazard (q)", 0.0, 0.50, 0.125, 0.025)
    hazard_start = st.sidebar.slider("Hazard starts at period", 0, 24, 12)
    salvage_value = st.sidebar.slider("Salvage value ($)", 0.0, 30.0, 10.0, 1.0)
    period_label = "Quarter"

st.sidebar.markdown("---")
st.sidebar.header("Consumer Segments")

with st.sidebar.expander("Hardcore Gamers", expanded=False):
    h_N = st.slider("Population", 100, 5000, 1000, 100, key="h_N")
    h_v = st.slider("Valuation ($)", 10, 100, 60, 1, key="h_v")
    h_beta = st.slider("Price sensitivity (beta)", 0.01, 0.50, 0.08, 0.01, key="h_beta")
    h_delta = st.slider("Patience (delta)", 0.0, 1.0, 0.3, 0.05, key="h_delta")
    h_gamma = st.slider("Fairness sensitivity (gamma)", 0.0, 0.30, 0.04, 0.01, key="h_gamma")

with st.sidebar.expander("Casual Players", expanded=False):
    c_N = st.slider("Population", 100, 5000, 2000, 100, key="c_N")
    c_v = st.slider("Valuation ($)", 10, 100, 40, 1, key="c_v")
    c_beta = st.slider("Price sensitivity (beta)", 0.01, 0.50, 0.12, 0.01, key="c_beta")
    c_delta = st.slider("Patience (delta)", 0.0, 1.0, 0.6, 0.05, key="c_delta")
    c_gamma = st.slider("Fairness sensitivity (gamma)", 0.0, 0.30, 0.08, 0.01, key="c_gamma")

with st.sidebar.expander("Bargain Hunters", expanded=False):
    b_N = st.slider("Population", 100, 5000, 3000, 100, key="b_N")
    b_v = st.slider("Valuation ($)", 5, 100, 20, 1, key="b_v")
    b_beta = st.slider("Price sensitivity (beta)", 0.01, 0.50, 0.20, 0.01, key="b_beta")
    b_delta = st.slider("Patience (delta)", 0.0, 1.0, 0.9, 0.05, key="b_delta")
    b_gamma = st.slider("Fairness sensitivity (gamma)", 0.0, 0.30, 0.15, 0.01, key="b_gamma")

# Build segments from sliders
segments = [
    Segment("Hardcore", h_N, h_v, 8, h_beta, h_delta, h_gamma),
    Segment("Casual", c_N, c_v, 10, c_beta, c_delta, c_gamma),
    Segment("Bargain", b_N, b_v, 8, b_beta, b_delta, b_gamma),
]

# ── Run solver ────────────────────────────────────────────────────────
with st.spinner("Running DP solver..."):
    r_uni = solve_dp(segments, T, valuation_decay, 'uniform',
                     sequel_hazard, hazard_start, salvage_value)
    r_seg = solve_dp(segments, T, valuation_decay, 'segment',
                     sequel_hazard, hazard_start, salvage_value)
    r_fair = solve_dp(segments, T, valuation_decay, 'fairness',
                      sequel_hazard, hazard_start, salvage_value)

# ── Metrics ───────────────────────────────────────────────────────────
lift = (r_seg.total_revenue - r_uni.total_revenue) / r_uni.total_revenue * 100
erosion = (r_seg.total_revenue - r_fair.total_revenue) / r_seg.total_revenue * 100
net = (r_fair.total_revenue - r_uni.total_revenue) / r_uni.total_revenue * 100

col1, col2, col3 = st.columns(3)
col1.metric("Uniform Revenue", f"${r_uni.total_revenue:,.0f}")
col2.metric("Segment Revenue", f"${r_seg.total_revenue:,.0f}", f"+{lift:.1f}%")
col3.metric("Segment + Fairness", f"${r_fair.total_revenue:,.0f}", f"+{net:.1f}% net")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Price Paths", "Revenue Breakdown", "Consumer Depletion"])

periods = np.arange(1, T + 1)

with tab1:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (r, title) in zip(axes, [
        (r_uni, "Uniform"), (r_seg, "Segment"), (r_fair, "Segment + Fairness")
    ]):
        for s in segments:
            ax.plot(periods, r.prices[s.name], color=COLORS[s.name.lower()],
                    linewidth=2, label=s.name, marker='o', markersize=3)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(period_label)
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Optimal Price Paths", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_labels = ['Uniform', 'Segment', 'Seg + Fairness']
    x = np.arange(3)
    width = 0.5
    bottom = np.zeros(3)
    for s in segments:
        vals = [sum(r.revenues[s.name]) for r in [r_uni, r_seg, r_fair]]
        ax.bar(x, vals, width, bottom=bottom, label=s.name,
               color=COLORS[s.name.lower()], alpha=0.85)
        bottom += vals
    for i, r in enumerate([r_uni, r_seg, r_fair]):
        ax.text(x[i], bottom[i] + 500, f'${r.total_revenue:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.legend(); ax.set_title("Total Revenue by Scenario", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (r, title) in zip(axes, [
        (r_uni, "Uniform"), (r_seg, "Segment"), (r_fair, "Segment + Fairness")
    ]):
        for s in segments:
            ax.plot(periods, r.remaining[s.name], color=COLORS[s.name.lower()],
                    linewidth=2, label=s.name)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(period_label)
        ax.set_ylabel("Remaining Consumers")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Consumer Depletion Over Time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Summary")
summary = pd.DataFrame({
    'Metric': ['Uniform Revenue', 'Segment Revenue', 'Segment + Fairness Revenue',
               'Personalization Lift', 'Fairness Erosion', 'Net Benefit'],
    'Value': [f'${r_uni.total_revenue:,.0f}', f'${r_seg.total_revenue:,.0f}',
              f'${r_fair.total_revenue:,.0f}', f'+{lift:.1f}%', f'-{erosion:.1f}%',
              f'+{net:.1f}%' if net >= 0 else f'{net:.1f}%']
})
st.dataframe(summary, use_container_width=True, hide_index=True)
