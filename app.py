import io
import json
import math
import textwrap
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="LuxePricing.ai | Integrated Capstone Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Dark premium CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #070A12;
        --panel: #0E1422;
        --panel2: #111A2B;
        --line: rgba(255,255,255,.09);
        --muted: #9AA7BC;
        --text: #F8FAFC;
        --gold: #C59B5A;
        --blue: #5B8CFF;
        --green: #34D399;
        --red: #F87171;
        --amber: #FBBF24;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, rgba(91,140,255,.13), transparent 32%),
                    radial-gradient(circle at top right, rgba(197,155,90,.12), transparent 30%),
                    #070A12;
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080B13 0%, #0B1220 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * { color: #D8DEE9; }

    h1, h2, h3 { letter-spacing: -0.035em; }

    .hero {
        padding: 26px 28px;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(17,26,43,.96), rgba(8,12,22,.96));
        box-shadow: 0 20px 60px rgba(0,0,0,.26);
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }

    .hero p {
        margin: 9px 0 0;
        color: var(--muted);
        max-width: 920px;
        line-height: 1.55;
        font-size: 14px;
    }

    .kpi-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 17px 18px;
        background: linear-gradient(180deg, rgba(17,26,43,.94), rgba(14,20,34,.94));
        box-shadow: 0 12px 32px rgba(0,0,0,.18);
        height: 100%;
    }

    .kpi-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: 11px;
        font-weight: 800;
    }

    .kpi-value {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 800;
        margin-top: 7px;
    }

    .kpi-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .module-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        background: rgba(14,20,34,.82);
        box-shadow: 0 12px 36px rgba(0,0,0,.16);
        margin-bottom: 16px;
    }

    .module-card h3 { margin-top: 0; }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 9px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.05);
        color: #DDE7F5;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .badge.gold { color: #F7D9A6; border-color: rgba(197,155,90,.45); background: rgba(197,155,90,.10); }
    .badge.green { color: #A7F3D0; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.10); }
    .badge.blue { color: #C7D2FE; border-color: rgba(91,140,255,.35); background: rgba(91,140,255,.10); }
    .badge.red { color: #FECACA; border-color: rgba(248,113,113,.35); background: rgba(248,113,113,.10); }

    .note-box {
        border-left: 4px solid var(--gold);
        background: rgba(197,155,90,.08);
        border-radius: 14px;
        padding: 14px 16px;
        color: #E8EEF8;
        line-height: 1.55;
        margin: 10px 0 16px;
        font-size: 14px;
    }

    .small-muted { color: var(--muted); font-size: 13px; line-height: 1.5; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 8px 14px;
        background: rgba(255,255,255,.04);
        border: 1px solid var(--line);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(197,155,90,.16);
        border: 1px solid rgba(197,155,90,.45);
    }

    div[data-testid="stMetric"] {
        background: rgba(17,26,43,.72);
        border: 1px solid var(--line);
        padding: 14px 16px;
        border-radius: 16px;
    }

    .dataframe { border-radius: 14px; overflow: hidden; }

    code, pre {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def money(value, currency="€", compact=False):
    try:
        value = float(value)
    except Exception:
        return f"{currency}0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if compact:
        if value >= 1_000_000:
            return f"{sign}{currency}{value/1_000_000:.1f}M"
        if value >= 1_000:
            return f"{sign}{currency}{value/1_000:.0f}k"
    return f"{sign}{currency}{value:,.0f}"


def pct(value, decimals=1):
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "0.0%"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def kpi(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle, badges=None):
    badges_html = ""
    if badges:
        badges_html = "<div style='margin-top:14px'>" + "".join([f"<span class='badge {b.get('style','')}'>{b['text']}</span>" for b in badges]) + "</div>"
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badges_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text):
    st.markdown(f"<div class='note-box'>{text}</div>", unsafe_allow_html=True)


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8")

# ------------------------------------------------------------
# Sample data
# ------------------------------------------------------------
def sample_competitor_feed():
    return pd.DataFrame([
        {"Hotel": "Mandarin Oriental Paris", "Tier": "Luxury", "Room Type": "Deluxe King", "Refundable Rate": 1040, "Non-refundable Rate": 970, "Available": True, "Source": "approved_rate_shopper", "Captured At": "2026-06-16 08:15"},
        {"Hotel": "Ritz Paris", "Tier": "Ultra luxury", "Room Type": "Deluxe King", "Refundable Rate": 1320, "Non-refundable Rate": 1250, "Available": True, "Source": "approved_rate_shopper", "Captured At": "2026-06-16 08:16"},
        {"Hotel": "Le Bristol Paris", "Tier": "Ultra luxury", "Room Type": "Deluxe King", "Refundable Rate": 1180, "Non-refundable Rate": 1100, "Available": True, "Source": "approved_rate_shopper", "Captured At": "2026-06-16 08:16"},
        {"Hotel": "The Peninsula Paris", "Tier": "Luxury", "Room Type": "Deluxe King", "Refundable Rate": 980, "Non-refundable Rate": 920, "Available": True, "Source": "approved_rate_shopper", "Captured At": "2026-06-16 08:17"},
        {"Hotel": "Cheval Blanc Paris", "Tier": "Ultra luxury", "Room Type": "Deluxe King", "Refundable Rate": 1450, "Non-refundable Rate": 1380, "Available": True, "Source": "approved_rate_shopper", "Captured At": "2026-06-16 08:17"},
    ])


def sample_demand_calendar():
    base = date(2026, 7, 1)
    rows = []
    events = ["Paris Fashion Week", "Summer leisure", "Corporate congress", "High season", "Weekend leisure", "No major event", "Art fair"]
    for i in range(14):
        rows.append({
            "Date": base + timedelta(days=i),
            "Season": "Peak" if i in [0, 1, 2, 3] else "High",
            "Event": events[i % len(events)],
            "Event Impact Score": [92, 88, 74, 80, 68, 35, 70][i % 7],
            "Search Index": [82, 78, 72, 76, 65, 52, 69][i % 7],
            "Expected Compression": ["High", "High", "Medium", "Medium", "Medium", "Low", "Medium"][i % 7],
        })
    return pd.DataFrame(rows)


def sample_ops_df():
    return pd.DataFrame([
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-01", 150, 118, 30680, 260, 78.7, "Leisure", "Direct", "EUR", "PMS", "2026-06-14", 245, "No", 64],
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-01", 150, 118, 30680, 260, 78.7, "Leisure", "Direct", "EUR", "PMS", "2026-06-14", 245, "No", 64],
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-02", 150, 129, 36120, 280, 86.0, "Corporate", "OTA", "EUR", "RMS", "2026-06-14", 270, "Yes", 72],
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-03", 150, 151, 40770, 270, 100.7, "Leisure", "OTA", "EUR", "PMS", "2026-06-03", 295, "Yes", 85],
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-04", 150, 96, 24000, 260, 64.0, "Group", "GDS", "EURO", "PMS", "2026-06-14", 255, "No", 59],
        ["LP-LIS-001", "Luxe Lisbon Riverside", "2026-07-05", 150, np.nan, 21000, 245, 58.0, "Leisure", "Direct", "EUR", "PMS", "2026-06-14", 250, "No", 51],
        ["LP-MAD-002", "Luxe Madrid Collection", "2026-07-06", 120, 92, 23000, 250, 76.7, "Leisure", "Direct", "EUR", "RMS", "2026-06-13", 240, "No", 62],
        ["LP-PAR-003", "Luxe Paris Saint-Germain", "2026-07-07", 98, 88, 43120, 490, 89.8, "Corporate", "Direct", "EUR", "PMS", "2026-06-14", 505, "Yes", 79],
    ], columns=["hotel_id", "property_name", "record_date", "rooms_available", "rooms_sold", "rooms_revenue", "adr", "occupancy_pct", "market_segment", "channel", "currency", "source_system", "updated_at", "competitor_adr", "event_flag", "booking_pace_pct"])


def sample_marketing_calendar():
    return pd.DataFrame([
        {"Date": "2026-07-03", "Channel": "Instagram", "Theme": "Opening Teaser", "Objective": "Awareness", "Audience": "Affluent leisure travelers", "Offer": "No offer", "Asset Type": "Carousel", "CTA": "Join the private opening list", "Notes": "Focus on sense of place"},
        {"Date": "2026-07-09", "Channel": "LinkedIn", "Theme": "Development Story", "Objective": "Brand credibility", "Audience": "Owners and partners", "Offer": "Opening announcement", "Asset Type": "Text + image", "CTA": "Request the opening brief", "Notes": "Corporate tone"},
        {"Date": "2026-07-15", "Channel": "Meta Ad", "Theme": "Summer City Escape", "Objective": "Lead generation", "Audience": "Couples and weekend travelers", "Offer": "Opening package", "Asset Type": "Paid social ad", "CTA": "Discover availability", "Notes": "Avoid discount language"},
        {"Date": "2026-07-22", "Channel": "Email", "Theme": "Members Preview", "Objective": "Conversion", "Audience": "CRM luxury prospects", "Offer": "Private preview stay", "Asset Type": "Email campaign", "CTA": "Reserve a preview stay", "Notes": "Soft luxury tone"},
    ])

# ------------------------------------------------------------
# Agent calculations
# ------------------------------------------------------------
def calculate_rate_recommendation(current_bar, rooms_available, on_books, pickup_7d, search_index, event_impact, floor, ceiling, comp_df, max_change_pct):
    clean = comp_df[(comp_df["Available"] == True) & (comp_df["Refundable Rate"] > 0)].copy()
    comp_median = float(clean["Refundable Rate"].median()) if len(clean) else current_bar
    occ_on_books = on_books / rooms_available if rooms_available else 0
    pickup_pressure = min(100, (pickup_7d / max(1, rooms_available)) * 500)
    demand_score = 0.35 * occ_on_books * 100 + 0.20 * pickup_pressure + 0.25 * search_index + 0.20 * event_impact
    comp_gap = (comp_median - current_bar) / max(current_bar, 1)
    demand_factor = (demand_score - 60) / 100
    raw = current_bar * (1 + 0.30 * comp_gap + 0.22 * demand_factor)
    raw = 0.65 * raw + 0.35 * comp_median
    max_up = current_bar * (1 + max_change_pct / 100)
    max_down = current_bar * (1 - max_change_pct / 100)
    rec = clamp(raw, max_down, max_up)
    rec = clamp(rec, floor, ceiling)
    rec = round(rec / 5) * 5
    confidence = 55 + min(22, len(clean) * 4) + min(13, demand_score / 10) - min(12, abs(comp_gap) * 20)
    confidence = int(clamp(confidence, 45, 96))
    return {
        "clean_comp_count": len(clean),
        "comp_median": comp_median,
        "occupancy_on_books": occ_on_books,
        "pickup_pressure": pickup_pressure,
        "demand_score": demand_score,
        "raw_recommendation": raw,
        "recommended_bar": rec,
        "delta": rec - current_bar,
        "delta_pct": (rec / current_bar - 1) * 100 if current_bar else 0,
        "confidence": confidence,
    }

# ------------------------------------------------------------
# Ops scan
# ------------------------------------------------------------
def scan_data_quality(df):
    required = ["hotel_id", "property_name", "record_date", "rooms_available", "rooms_sold", "rooms_revenue", "adr", "occupancy_pct", "market_segment", "channel", "currency", "source_system", "updated_at"]
    exceptions = []
    df = df.copy()
    df["quality_status"] = "Clean"
    df["quality_issues"] = ""

    def add_issue(idx, severity, rule, issue, owner):
        exceptions.append({"Row": idx + 1, "Severity": severity, "Rule": rule, "Issue": issue, "Owner": owner})
        df.loc[idx, "quality_status"] = "Blocked" if severity == "High" else "Review"
        current = df.loc[idx, "quality_issues"]
        df.loc[idx, "quality_issues"] = (current + "; " if current else "") + issue

    for col in required:
        if col not in df.columns:
            continue
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for idx in df[missing].index:
            add_issue(idx, "High", "Completeness", f"Missing required field: {col}", "Data Ops")

    for idx, r in df.iterrows():
        rooms_available = pd.to_numeric(r.get("rooms_available"), errors="coerce")
        rooms_sold = pd.to_numeric(r.get("rooms_sold"), errors="coerce")
        rooms_revenue = pd.to_numeric(r.get("rooms_revenue"), errors="coerce")
        adr = pd.to_numeric(r.get("adr"), errors="coerce")
        occ = pd.to_numeric(r.get("occupancy_pct"), errors="coerce")
        if pd.notna(occ) and (occ < 0 or occ > 100):
            add_issue(idx, "High", "Validity", "Occupancy is outside 0–100%", "Revenue Manager")
        if pd.notna(rooms_sold) and pd.notna(rooms_available) and rooms_sold > rooms_available:
            add_issue(idx, "High", "Validity", "Rooms sold exceeds rooms available", "Revenue Manager")
        if pd.notna(rooms_revenue) and pd.notna(rooms_sold) and rooms_sold > 0 and pd.notna(adr):
            calc_adr = rooms_revenue / rooms_sold
            if abs(calc_adr - adr) / max(adr, 1) > 0.05:
                add_issue(idx, "Medium", "Consistency", "ADR does not match rooms revenue / rooms sold", "Ops Analyst")
        if r.get("currency") not in ["EUR", "USD", "GBP", "AED", "HKD"]:
            add_issue(idx, "Medium", "Validity", "Currency is not standardized", "CRM Admin")
        try:
            updated = pd.to_datetime(r.get("updated_at"))
            if (pd.Timestamp("2026-06-16") - updated).days > 7:
                add_issue(idx, "Medium", "Timeliness", "Record is older than 7 days", "Data Ops")
        except Exception:
            add_issue(idx, "Medium", "Timeliness", "Updated_at date is invalid", "Data Ops")

    key_cols = ["hotel_id", "record_date", "market_segment", "channel"]
    if all(c in df.columns for c in key_cols):
        dupes = df.duplicated(subset=key_cols, keep=False)
        for idx in df[dupes].index:
            add_issue(idx, "Medium", "Uniqueness", "Duplicate hotel-date-segment-channel key", "CRM Admin")

    total_checks = max(1, len(df) * 5)
    penalty = sum(2 if e["Severity"] == "High" else 1 for e in exceptions)
    score = int(clamp(100 - penalty / total_checks * 100, 0, 100))
    clean_df = df[df["quality_status"] == "Clean"].copy()
    return df, pd.DataFrame(exceptions), clean_df, score

# ------------------------------------------------------------
# Finance calculations
# ------------------------------------------------------------
def finance_model(keys, days, current_adr, current_occ_pct, other_rev, rooms_exp_pct, other_margin_pct, undistributed, fixed_charges, adr_uplift_pct, occ_change_pp, ancillary_uplift_pct, flow_through_pct, implementation_cost, annual_subscription, mgmt_fee_pct, ffe_pct, cap_rate_pct, discount_rate_pct, adr_floor, adr_ceiling):
    occ = current_occ_pct / 100
    rooms_available = keys * days
    rooms_sold = rooms_available * occ
    current_rooms_rev = rooms_sold * current_adr
    current_total_rev = current_rooms_rev + other_rev
    current_rooms_profit = current_rooms_rev * (1 - rooms_exp_pct / 100)
    current_other_profit = other_rev * other_margin_pct / 100
    current_gop = current_rooms_profit + current_other_profit - undistributed
    current_mgmt = current_total_rev * mgmt_fee_pct / 100
    current_ffe = current_total_rev * ffe_pct / 100
    current_noi = current_gop - current_mgmt - current_ffe - fixed_charges
    current_revpar = current_rooms_rev / max(rooms_available, 1)

    scenario_adr = clamp(current_adr * (1 + adr_uplift_pct / 100), adr_floor, adr_ceiling)
    scenario_occ = clamp(occ + occ_change_pp / 100, 0, 0.98)
    scenario_rooms_sold = rooms_available * scenario_occ
    scenario_rooms_rev = scenario_rooms_sold * scenario_adr
    rooms_rev_delta = scenario_rooms_rev - current_rooms_rev
    occupancy_ratio = scenario_rooms_sold / max(rooms_sold, 1)
    scenario_other_rev = other_rev * occupancy_ratio * (1 + ancillary_uplift_pct / 100)
    other_delta = scenario_other_rev - other_rev
    scenario_total_rev = scenario_rooms_rev + scenario_other_rev
    incremental_profit = rooms_rev_delta * flow_through_pct / 100 + other_delta * other_margin_pct / 100
    scenario_gop = current_gop + incremental_profit
    scenario_mgmt = scenario_total_rev * mgmt_fee_pct / 100
    scenario_ffe = scenario_total_rev * ffe_pct / 100
    scenario_noi = scenario_gop - scenario_mgmt - scenario_ffe - fixed_charges
    noi_uplift = scenario_noi - current_noi
    recurring_noi_after_subscription = noi_uplift - annual_subscription
    net_benefit_y1 = noi_uplift - annual_subscription - implementation_cost
    investment = implementation_cost + annual_subscription
    roi_y1 = net_benefit_y1 / max(investment, 1)
    monthly_recurring = recurring_noi_after_subscription / 12
    payback_months = implementation_cost / monthly_recurring if monthly_recurring > 0 else np.inf
    discount_rate = discount_rate_pct / 100
    npv_3yr = -implementation_cost + sum(recurring_noi_after_subscription / ((1 + discount_rate) ** y) for y in [1, 2, 3])
    asset_uplift = recurring_noi_after_subscription / max(cap_rate_pct / 100, 0.0001)
    scenario_revpar = scenario_rooms_rev / max(rooms_available, 1)
    return locals()

# ------------------------------------------------------------
# Synthetic guest data / elasticity
# ------------------------------------------------------------
SEGMENTS = {
    "Luxury Leisure": {"elasticity": -0.45, "base": 0.76},
    "Corporate Negotiated": {"elasticity": -0.75, "base": 0.67},
    "Transient Leisure": {"elasticity": -1.25, "base": 0.61},
    "Group / MICE": {"elasticity": -0.95, "base": 0.63},
    "OTA Price Sensitive": {"elasticity": -1.85, "base": 0.53},
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_guest_data(n=800, base_adr=420, seed=42, market_type="Urban Luxury"):
    rng = np.random.default_rng(seed)
    seasons = {"Low": .82, "Shoulder": 1.00, "High": 1.18, "Peak": 1.38}
    events = {"None": 1.00, "City Event": 1.10, "Trade Fair": 1.18, "School Vacation": 1.12, "Major Compression Event": 1.38}
    channels = ["Direct Web", "OTA", "GDS", "Corporate Portal", "Travel Advisor"]
    room_types = {"Classic": 1.00, "Deluxe": 1.22, "Junior Suite": 1.55, "Signature Suite": 2.05}
    rows = []
    for i in range(n):
        segment = rng.choice(list(SEGMENTS.keys()))
        season = rng.choice(list(seasons.keys()), p=[.18, .30, .34, .18])
        event = rng.choice(list(events.keys()), p=[.38, .24, .12, .14, .12])
        channel = rng.choice(channels)
        room_type = rng.choice(list(room_types.keys()))
        comp_adr = base_adr * seasons[season] * events[event] * room_types[room_type] * rng.normal(1, .065)
        offered_adr = comp_adr * rng.normal(1.04, .10)
        price_ratio = offered_adr / comp_adr
        occupancy_pace = np.clip(({"Low": .45, "Shoulder": .62, "High": .74, "Peak": .86}[season]) + (0 if event == "None" else .07) + rng.normal(0, .07), .20, .98)
        demand_index = np.clip(38 + occupancy_pace * 52 + (0 if event == "None" else 8) + rng.normal(0, 6), 15, 100)
        e = SEGMENTS[segment]["elasticity"]
        logit = -.35 + (demand_index - 60) / 32 + e * math.log(price_ratio) + (.04 if channel == "Direct Web" else -.05 if channel == "OTA" else .01)
        prob = sigmoid(logit)
        booked = int(rng.random() < prob)
        los = max(1, int(round(rng.normal(2.2, 1.0))))
        rows.append({
            "Record_ID": i + 1,
            "Guest_Segment": segment,
            "Channel": channel,
            "Room_Type": room_type,
            "Market_Type": market_type,
            "Season": season,
            "Event": event,
            "Competitor_ADR": round(comp_adr, 2),
            "Offered_ADR": round(offered_adr, 2),
            "Price_Ratio": round(price_ratio, 3),
            "Demand_Index": round(demand_index, 1),
            "Occupancy_Pace": round(occupancy_pace, 3),
            "Length_of_Stay": los,
            "Booking_Probability": round(prob, 3),
            "Booked": booked,
            "Expected_Revenue": round(booked * offered_adr * los, 2),
        })
    return pd.DataFrame(rows)


def estimate_segment_elasticity(df):
    rows = []
    for seg, group in df.groupby("Guest_Segment"):
        if len(group) < 20:
            continue
        high = group[group["Price_Ratio"] >= group["Price_Ratio"].median()]
        low = group[group["Price_Ratio"] < group["Price_Ratio"].median()]
        high_conv = high["Booked"].mean() if len(high) else 0
        low_conv = low["Booked"].mean() if len(low) else 0
        high_price = high["Price_Ratio"].mean() if len(high) else 1
        low_price = low["Price_Ratio"].mean() if len(low) else 1
        if high_conv <= 0 or low_conv <= 0 or high_price <= 0 or low_price <= 0 or high_price == low_price:
            est = SEGMENTS.get(seg, {"elasticity": -1.0})["elasticity"]
        else:
            est = math.log(high_conv / low_conv) / math.log(high_price / low_price)
        rows.append({
            "Segment": seg,
            "Records": len(group),
            "Observed Conversion": group["Booked"].mean(),
            "Estimated Elasticity": float(np.clip(est, -2.6, -0.15)),
            "Interpretation": "Highly price sensitive" if est < -1.3 else "Moderate sensitivity" if est < -.8 else "Low price sensitivity",
        })
    return pd.DataFrame(rows).sort_values("Estimated Elasticity")


def recommendation_from_elasticity(segment, comp_adr, current_adr, demand_index, occupancy_pace, guardrail_pct, elasticity_lookup):
    e = float(elasticity_lookup.get(segment, SEGMENTS.get(segment, {"elasticity": -1.0})["elasticity"]))
    ratios = np.linspace(.80, 1.25, 60)
    rows = []
    for ratio in ratios:
        offered = comp_adr * ratio
        price_ratio = offered / comp_adr
        prob = sigmoid(-.35 + (demand_index - 60) / 32 + e * math.log(price_ratio) + (occupancy_pace - .65) * .9)
        revpar_proxy = offered * prob
        rows.append({"Price Ratio": ratio, "Offered ADR": offered, "Booking Probability": prob, "RevPAR Proxy": revpar_proxy})
    curve = pd.DataFrame(rows)
    best = curve.iloc[curve["RevPAR Proxy"].idxmax()]
    lower = current_adr * (1 - guardrail_pct / 100)
    upper = current_adr * (1 + guardrail_pct / 100)
    recommended = round(clamp(best["Offered ADR"], lower, upper) / 5) * 5
    return recommended, curve, e

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏨 LuxePricing.ai")
    st.caption("Integrated Streamlit capstone app")
    st.divider()
    workspace = st.selectbox(
        "Main navigation",
        [
            "Live Demo & Analytics",
            "Operations & Finance",
            "Data Science Sandbox",
            "GTM Strategy & Corporate",
        ],
        index=0,
    )
    st.divider()
    st.markdown("**Deliverables included**")
    st.checkbox("1. AI Marketing", True, disabled=True)
    st.checkbox("2. AI Sales", True, disabled=True)
    st.checkbox("3. AI HR / CRO", True, disabled=True)
    st.checkbox("4. AI Finance ROI", True, disabled=True)
    st.checkbox("5. AI Ops Data Quality", True, disabled=True)
    st.checkbox("6. AI Tech Elasticity", True, disabled=True)
    st.checkbox("7. AI Agent Demo", True, disabled=True)
    st.divider()
    st.caption("Prototype mode: no live PMS, CRS, CRM, STR or OTA connection. All API writebacks are simulated and require human approval.")

# ------------------------------------------------------------
# Workspace 1: Live Demo & Analytics
# ------------------------------------------------------------
if workspace == "Live Demo & Analytics":
    hero(
        "Live Demo & Analytics",
        "Combines Deliverable 7: AI Agent Demo with competitor rate-shopper simulation, demand signals, recommendation logic, approval queue and audit trail. This is the part to demonstrate live during the pitch.",
        badges=[{"text": "Deliverable 7", "style": "gold"}, {"text": "Daily BAR recommendation", "style": "blue"}, {"text": "Human approval required", "style": "green"}],
    )

    if "comp_df" not in st.session_state:
        st.session_state.comp_df = sample_competitor_feed()
    if "agent_audit" not in st.session_state:
        st.session_state.agent_audit = []
    if "approved_rates" not in st.session_state:
        st.session_state.approved_rates = []

    left, right = st.columns([.95, 1.05], gap="large")
    with left:
        st.markdown("### Property + demand inputs")
        c1, c2 = st.columns(2)
        hotel_name = c1.text_input("Hotel / asset", "Maison Lumière Paris")
        target_date = c2.date_input("Target rate date", date(2026, 7, 1))
        c3, c4, c5 = st.columns(3)
        current_bar = c3.number_input("Current BAR", min_value=0, value=980, step=10)
        floor = c4.number_input("Rate floor", min_value=0, value=780, step=10)
        ceiling = c5.number_input("Rate ceiling", min_value=0, value=1450, step=10)
        c6, c7, c8 = st.columns(3)
        rooms_available = c6.number_input("Rooms available", min_value=1, value=120)
        on_books = c7.number_input("Rooms on books", min_value=0, value=86)
        pickup_7d = c8.number_input("7-day pickup", min_value=0, value=18)
        c9, c10, c11 = st.columns(3)
        search_index = c9.slider("Search / demand index", 0, 100, 74)
        event_impact = c10.slider("Event impact score", 0, 100, 90)
        max_change_pct = c11.slider("Max daily change guardrail", 5, 35, 18)
        event_name = st.text_input("Demand driver / event", "Paris Fashion Week")

        uploaded_comp = st.file_uploader("Upload competitor feed CSV", type=["csv"], help="Columns should include Hotel, Refundable Rate, Available. Sample data is loaded by default.")
        if uploaded_comp is not None:
            try:
                comp_uploaded = pd.read_csv(uploaded_comp)
                for col in ["Refundable Rate", "Non-refundable Rate"]:
                    if col in comp_uploaded.columns:
                        comp_uploaded[col] = pd.to_numeric(comp_uploaded[col], errors="coerce")
                if "Available" in comp_uploaded.columns:
                    comp_uploaded["Available"] = comp_uploaded["Available"].astype(str).str.lower().isin(["true", "yes", "1", "available"])
                st.session_state.comp_df = comp_uploaded
                st.success("Competitor feed uploaded.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        with st.expander("Competitor rate-shopper feed", expanded=True):
            st.dataframe(st.session_state.comp_df, use_container_width=True, hide_index=True)
            st.download_button("Download competitor feed", csv_download(st.session_state.comp_df), "competitor_feed.csv", "text/csv")

    rec = calculate_rate_recommendation(current_bar, rooms_available, on_books, pickup_7d, search_index, event_impact, floor, ceiling, st.session_state.comp_df, max_change_pct)

    with right:
        st.markdown("### Agent output")
        a, b, c, d = st.columns(4)
        with a: kpi("Current BAR", money(current_bar), "from PMS/RMS")
        with b: kpi("Comp median", money(rec["comp_median"]), "clean available rates")
        with c: kpi("Demand score", f"{rec['demand_score']:.0f}/100", f"event: {event_name}")
        with d: kpi("Recommended BAR", money(rec["recommended_bar"]), f"{rec['delta_pct']:+.1f}% vs current")

        st.markdown("#### Reasoning terminal")
        terminal = f"""
        [INIT] Loaded property: {hotel_name} | target date: {target_date}
        [DATA] Clean competitor rates found: {rec['clean_comp_count']} | median comp rate: {money(rec['comp_median'])}
        [DEMAND] On-books occupancy: {rec['occupancy_on_books']*100:.1f}% | 7-day pickup: {pickup_7d} rooms
        [CALENDAR] Event signal: {event_name} | event impact score: {event_impact}/100
        [GUARDRAIL] Floor: {money(floor)} | Ceiling: {money(ceiling)} | Max move: ±{max_change_pct}%
        [OUTPUT] Recommended BAR: {money(rec['recommended_bar'])} | confidence: {rec['confidence']}%
        [CONTROL] Recommendation requires revenue-manager approval before PMS/CRS push.
        """
        st.code(textwrap.dedent(terminal).strip(), language="text")

        explanation = (
            f"The agent recommends **{money(rec['recommended_bar'])}** because demand pressure is "
            f"{rec['demand_score']:.0f}/100, current BAR is {'below' if current_bar < rec['comp_median'] else 'above'} the competitor median, "
            f"and the final rate stays inside the approved floor/ceiling and daily-change guardrails."
        )
        note(explanation)
        col_a, col_b, col_c = st.columns([1, 1, 1])
        if col_a.button("Approve rate", type="primary"):
            row = {"Hotel": hotel_name, "Date": str(target_date), "Room Type": "Deluxe King", "Approved BAR": rec["recommended_bar"], "Current BAR": current_bar, "Confidence": rec["confidence"], "Status": "Approved"}
            st.session_state.approved_rates.append(row)
            st.session_state.agent_audit.append({"Time": datetime.now().strftime("%H:%M:%S"), "Action": "Approved rate", "Output": money(rec["recommended_bar"]), "User": "Revenue Manager"})
            st.success("Rate moved to approval queue.")
        if col_b.button("Reject recommendation"):
            st.session_state.agent_audit.append({"Time": datetime.now().strftime("%H:%M:%S"), "Action": "Rejected rate", "Output": money(rec["recommended_bar"]), "User": "Revenue Manager"})
            st.warning("Recommendation rejected and logged.")
        if col_c.button("Simulate rate-shopper refresh"):
            df = sample_competitor_feed()
            noise = np.random.default_rng().normal(0, 35, len(df))
            df["Refundable Rate"] = (df["Refundable Rate"] + noise).round(-1).astype(int)
            st.session_state.comp_df = df
            st.session_state.agent_audit.append({"Time": datetime.now().strftime("%H:%M:%S"), "Action": "Rate-shopper refresh", "Output": "Competitor feed updated", "User": "Agent"})
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["Approval queue", "Demand calendar", "LLM prompt pack"])
    with tab1:
        approved_df = pd.DataFrame(st.session_state.approved_rates)
        audit_df = pd.DataFrame(st.session_state.agent_audit)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Approved recommendations")
            if len(approved_df):
                st.dataframe(approved_df, use_container_width=True, hide_index=True)
                st.download_button("Download approved rates", csv_download(approved_df), "approved_rates.csv", "text/csv")
            else:
                st.info("No rates approved yet.")
        with c2:
            st.markdown("#### Audit trail")
            if len(audit_df):
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("No audit events yet.")
    with tab2:
        demand_df = sample_demand_calendar()
        st.dataframe(demand_df, use_container_width=True, hide_index=True)
        fig = px.line(demand_df, x="Date", y=["Event Impact Score", "Search Index"], markers=True, title="Demand calendar signals")
        fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        provider = st.selectbox("LLM provider logic", ["ChatGPT / OpenAI", "Gemini", "Claude"], key="agent_provider")
        prompt = f"""
        You are a hotel revenue-management agent inside LuxePricing.ai.
        Provider selected: {provider}

        Task: Recommend the daily BAR for {hotel_name} on {target_date}.
        Inputs:
        - Current BAR: {money(current_bar)}
        - Rate floor/ceiling: {money(floor)} / {money(ceiling)}
        - Rooms available: {rooms_available}
        - Rooms on books: {on_books}
        - 7-day pickup: {pickup_7d}
        - Demand search index: {search_index}/100
        - Event: {event_name} with impact {event_impact}/100
        - Competitor median: {money(rec['comp_median'])}
        - Recommended BAR from deterministic model: {money(rec['recommended_bar'])}

        Output required:
        1. Explain the pricing recommendation in plain English.
        2. Mention risks and guardrails.
        3. Produce a revenue-manager approval note.
        4. Do not claim live OTA scraping; state that competitor data comes from approved rate-shopping feeds or uploaded CSV.
        """
        st.code(textwrap.dedent(prompt).strip(), language="text")

# ------------------------------------------------------------
# Workspace 2: Operations & Finance
# ------------------------------------------------------------
elif workspace == "Operations & Finance":
    hero(
        "Operations & Finance",
        "Combines Deliverable 5: CRM ingestion + data-quality monitor with Deliverable 4: AI ROI calculator. It shows how clean hotel data flows into financial decision-making.",
        badges=[{"text": "Deliverable 5", "style": "blue"}, {"text": "Deliverable 4", "style": "gold"}, {"text": "USALI-style bridge", "style": "green"}],
    )

    ops_tab, finance_tab, integration_tab = st.tabs(["AI Ops: CRM ingestion + quality", "AI Finance: ROI calculator", "How Ops feeds Finance"])

    with ops_tab:
        st.markdown("### CRM ingestion source map")
        source_map = pd.DataFrame([
            ["PMS / CRS", "Rooms sold, rooms available, arrivals, cancellations", "Revenue Snapshot", "Ops / IT"],
            ["Revenue reports / RMS", "ADR, RevPAR, pickup, future pace", "Market Signal", "Revenue Manager"],
            ["USALI finance reports", "Rooms revenue, GOP, NOI, management fees, FF&E", "Finance Snapshot", "Finance"],
            ["CRM", "Owner, operator, contacts, property record", "Account / Property", "CRM Admin"],
            ["Rate-shopper feed", "Competitor ADR, availability, capture timestamp", "Market Signal", "Revenue Ops"],
            ["Campaign calendar", "Launches, events, packages, demand pushes", "Campaign Object", "Marketing"],
        ], columns=["Source", "Data received", "CRM object", "Owner"])
        st.dataframe(source_map, use_container_width=True, hide_index=True)

        st.markdown("### Upload or use sample data")
        ops_upload = st.file_uploader("Upload hotel revenue / CRM ingestion CSV", type=["csv"], key="ops_upload")
        if ops_upload:
            ops_df = pd.read_csv(ops_upload)
        else:
            ops_df = sample_ops_df()
        scanned_df, exceptions_df, clean_df, score = scan_data_quality(ops_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Records loaded", f"{len(ops_df)}", "source rows")
        with c2: kpi("Quality score", f"{score}%", "push threshold: 90%")
        with c3: kpi("Open exceptions", f"{len(exceptions_df)}", "blocked/review rows")
        with c4: kpi("Clean records", f"{len(clean_df)}", "ready for export")

        st.markdown("#### Quality monitor output")
        st.dataframe(scanned_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Exception queue")
            if len(exceptions_df):
                st.dataframe(exceptions_df, use_container_width=True, hide_index=True)
                st.download_button("Download exception queue", csv_download(exceptions_df), "luxepricing_exception_queue.csv", "text/csv")
            else:
                st.success("No exceptions detected.")
        with c2:
            st.markdown("#### Clean CRM export")
            st.dataframe(clean_df, use_container_width=True, hide_index=True)
            st.download_button("Download clean export", csv_download(clean_df), "luxepricing_clean_crm_export.csv", "text/csv")

        st.markdown("#### Data-quality prompt pack")
        st.code(textwrap.dedent(f"""
        You are the LuxePricing.ai data-quality analyst.
        Review this CRM/PMS ingestion scan.
        Records loaded: {len(ops_df)}
        Quality score: {score}%
        Open exceptions: {len(exceptions_df)}

        Required output:
        - Summarize key data risks.
        - Prioritize fixes by business impact.
        - Explain which rows should be blocked before entering the pricing and finance models.
        - Do not overwrite data automatically; propose fixes for human approval.
        """).strip(), language="text")

    with finance_tab:
        st.markdown("### AI ROI calculator for pricing uplift scenarios")
        left, right = st.columns([1.05, .95], gap="large")
        with left:
            st.markdown("#### Property and USALI-style assumptions")
            f1, f2, f3 = st.columns(3)
            keys = f1.number_input("Keys / rooms", value=150, min_value=1)
            days = f2.number_input("Operating days", value=365, min_value=1)
            currency = f3.selectbox("Currency", ["€", "$", "£", "AED", "HK$"], index=0)
            f4, f5, f6 = st.columns(3)
            current_adr = f4.number_input("Current ADR", value=215.0, step=5.0)
            current_occ = f5.number_input("Current occupancy %", value=75.0, step=.5)
            other_rev = f6.number_input("Other operated revenue / year", value=4_200_000.0, step=50_000.0)
            f7, f8, f9 = st.columns(3)
            rooms_exp = f7.number_input("Rooms expense ratio %", value=28.0, step=.5)
            other_margin = f8.number_input("Other dept profit margin %", value=35.0, step=.5)
            undistributed = f9.number_input("Undistributed expenses", value=4_800_000.0, step=50_000.0)
            f10, f11, f12 = st.columns(3)
            fixed_charges = f10.number_input("Fixed charges", value=950_000.0, step=25_000.0)
            mgmt_fee = f11.number_input("Management fee %", value=3.0, step=.25)
            ffe = f12.number_input("FF&E reserve %", value=4.0, step=.25)

            st.markdown("#### Scenario controls")
            s1, s2, s3 = st.columns(3)
            adr_uplift = s1.slider("ADR uplift %", -5.0, 20.0, 5.0, .5)
            occ_change = s2.slider("Occupancy change, p.p.", -5.0, 8.0, 1.5, .5)
            ancillary_uplift = s3.slider("Ancillary revenue uplift %", -5.0, 10.0, 1.0, .5)
            s4, s5, s6 = st.columns(3)
            flow_through = s4.slider("Incremental rooms flow-through %", 40.0, 95.0, 78.0, 1.0)
            adr_floor = s5.number_input("ADR floor", value=160.0, step=5.0)
            adr_ceiling = s6.number_input("ADR ceiling", value=340.0, step=5.0)
            s7, s8, s9, s10 = st.columns(4)
            implementation_cost = s7.number_input("Implementation cost", value=85_000.0, step=5_000.0)
            annual_subscription = s8.number_input("Annual subscription", value=120_000.0, step=5_000.0)
            cap_rate = s9.number_input("Cap rate %", value=6.25, step=.05)
            discount_rate = s10.number_input("Discount rate %", value=10.0, step=.25)

        fm = finance_model(keys, days, current_adr, current_occ, other_rev, rooms_exp, other_margin, undistributed, fixed_charges, adr_uplift, occ_change, ancillary_uplift, flow_through, implementation_cost, annual_subscription, mgmt_fee, ffe, cap_rate, discount_rate, adr_floor, adr_ceiling)
        with right:
            st.markdown("#### Finance outputs")
            k1, k2 = st.columns(2)
            with k1: kpi("Current RevPAR", money(fm["current_revpar"], currency), "ADR × occupancy")
            with k2: kpi("Scenario RevPAR", money(fm["scenario_revpar"], currency), f"{money(fm['scenario_revpar']-fm['current_revpar'], currency)} change")
            k3, k4 = st.columns(2)
            with k3: kpi("Year 1 net benefit", money(fm["net_benefit_y1"], currency, True), "NOI uplift - tool cost")
            with k4: kpi("Asset value uplift", money(fm["asset_uplift"], currency, True), "recurring NOI / cap rate")
            k5, k6 = st.columns(2)
            with k5: kpi("Payback", "∞" if not np.isfinite(fm["payback_months"]) else f"{fm['payback_months']:.1f} months", "implementation cost / monthly benefit")
            with k6: kpi("Year 1 ROI", pct(fm["roi_y1"] * 100, 0), "net benefit / investment")

            memo = f"""
            AI finance memo

            Scenario: {adr_uplift:.1f}% ADR uplift and {occ_change:+.1f} p.p. occupancy change.
            The model increases RevPAR from {money(fm['current_revpar'], currency)} to {money(fm['scenario_revpar'], currency)}.
            Estimated incremental NOI is {money(fm['noi_uplift'], currency, True)} before tool subscription and implementation cost.
            After implementation and annual subscription, Year 1 net benefit equals {money(fm['net_benefit_y1'], currency, True)}.
            The recurring net NOI uplift capitalizes into approximately {money(fm['asset_uplift'], currency, True)} of asset value at a {cap_rate:.2f}% cap rate.

            Recommendation: {'Proceed to pilot if the hotel has reliable PMS/RMS data and revenue-manager approval rules.' if fm['net_benefit_y1'] > 0 else 'Do not proceed under this scenario; revise assumptions or reduce implementation cost.'}
            """
            st.text_area("AI finance memo", textwrap.dedent(memo).strip(), height=220)

        bridge = pd.DataFrame([
            ["Rooms revenue", fm["current_rooms_rev"], fm["scenario_rooms_rev"], fm["scenario_rooms_rev"] - fm["current_rooms_rev"]],
            ["Other operated revenue", fm["other_rev"], fm["scenario_other_rev"], fm["scenario_other_rev"] - fm["other_rev"]],
            ["Total revenue", fm["current_total_rev"], fm["scenario_total_rev"], fm["scenario_total_rev"] - fm["current_total_rev"]],
            ["GOP", fm["current_gop"], fm["scenario_gop"], fm["scenario_gop"] - fm["current_gop"]],
            ["Management fee", fm["current_mgmt"], fm["scenario_mgmt"], fm["scenario_mgmt"] - fm["current_mgmt"]],
            ["FF&E reserve", fm["current_ffe"], fm["scenario_ffe"], fm["scenario_ffe"] - fm["current_ffe"]],
            ["NOI", fm["current_noi"], fm["scenario_noi"], fm["noi_uplift"]],
        ], columns=["USALI-style metric", "Current", "Scenario", "Change"])
        bridge_display = bridge.copy()
        for col in ["Current", "Scenario", "Change"]:
            bridge_display[col] = bridge_display[col].apply(lambda x: money(x, currency))
        st.markdown("#### USALI-style current vs scenario bridge")
        st.dataframe(bridge_display, use_container_width=True, hide_index=True)

        sensitivity_rows = []
        for name, adr_u, occ_u in [("Stress", 1.5, -1.0), ("Conservative", 3.0, 0.5), ("Base", adr_uplift, occ_change), ("Upside", 8.0, 2.5), ("Aggressive", 12.0, 4.0)]:
            m = finance_model(keys, days, current_adr, current_occ, other_rev, rooms_exp, other_margin, undistributed, fixed_charges, adr_u, occ_u, ancillary_uplift, flow_through, implementation_cost, annual_subscription, mgmt_fee, ffe, cap_rate, discount_rate, adr_floor, adr_ceiling)
            sensitivity_rows.append({"Scenario": name, "ADR uplift": f"{adr_u:.1f}%", "Occ change": f"{occ_u:+.1f} p.p.", "RevPAR": money(m["scenario_revpar"], currency), "Incremental NOI": money(m["noi_uplift"], currency, True), "Year 1 ROI": pct(m["roi_y1"] * 100, 0), "Asset uplift": money(m["asset_uplift"], currency, True)})
        st.markdown("#### Scenario sensitivity table")
        st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True, hide_index=True)

    with integration_tab:
        note("This tab explains the integrated logic: the Ops module protects the Finance model from bad data. If rooms sold, ADR, occupancy or revenue fields are missing or inconsistent, the ROI calculator should not be trusted until those records are corrected.")
        st.markdown("### End-to-end operating model")
        flow = pd.DataFrame([
            ["1", "Upload PMS/RMS/finance report", "Ops monitor maps fields and flags issues", "CSV upload or API connector"],
            ["2", "Clean revenue snapshot", "Only clean data enters Finance and Tech models", "Completeness, validity, consistency"],
            ["3", "Run pricing uplift scenario", "Finance converts ADR/occupancy uplift into NOI and asset value", "USALI-style bridge"],
            ["4", "Approve scenario", "Human owner accepts pilot or rejects assumptions", "Audit log required"],
        ], columns=["Step", "Process", "What AI supports", "Control"])
        st.dataframe(flow, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Workspace 3: Data Science Sandbox
# ------------------------------------------------------------
elif workspace == "Data Science Sandbox":
    hero(
        "Data Science Sandbox",
        "Deliverable 6: Pricing elasticity predictive model with synthetic guest data. This is the technical proof behind the daily rate agent.",
        badges=[{"text": "Deliverable 6", "style": "gold"}, {"text": "Synthetic guest data", "style": "blue"}, {"text": "Elasticity model", "style": "green"}],
    )

    if "guest_df" not in st.session_state:
        st.session_state.guest_df = generate_synthetic_guest_data()

    data_tab, model_tab, scenario_tab, governance_tab = st.tabs(["Data Studio", "Train Model", "Rate Scenario", "Prompt + Governance"])

    with data_tab:
        c1, c2, c3, c4 = st.columns(4)
        n_records = c1.number_input("Synthetic records", min_value=100, max_value=5000, value=800, step=100)
        base_adr = c2.number_input("Base ADR", min_value=50, max_value=2000, value=420, step=10)
        market_type = c3.selectbox("Market type", ["Urban Luxury", "Resort Luxury", "Business Hotel", "Lifestyle Boutique"])
        seed = c4.number_input("Random seed", value=42, step=1)
        b1, b2 = st.columns([1, 5])
        if b1.button("Generate data", type="primary"):
            st.session_state.guest_df = generate_synthetic_guest_data(n_records, base_adr, int(seed), market_type)
        uploaded_guest = b2.file_uploader("Or upload guest quote dataset CSV", type=["csv"], key="guest_upload")
        if uploaded_guest:
            st.session_state.guest_df = pd.read_csv(uploaded_guest)
        df = st.session_state.guest_df
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Records", f"{len(df):,}", "guest quote observations")
        with k2: kpi("Avg offered ADR", money(df["Offered_ADR"].mean(), "$"), "synthetic quote rate")
        with k3: kpi("Conversion", pct(df["Booked"].mean() * 100), "booked / quoted")
        with k4: kpi("Data quality", "98%", "synthetic complete dataset")
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)
        st.download_button("Download synthetic guest data", csv_download(df), "luxepricing_synthetic_guest_data.csv", "text/csv")

        fig = px.scatter(df.sample(min(400, len(df))), x="Price_Ratio", y="Booking_Probability", color="Guest_Segment", size="Demand_Index", hover_data=["Season", "Event", "Offered_ADR"], title="Price ratio vs booking probability")
        fig.update_layout(template="plotly_dark", height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with model_tab:
        df = st.session_state.guest_df
        elasticity_df = estimate_segment_elasticity(df)
        k1, k2, k3 = st.columns(3)
        with k1: kpi("Model type", "Interpretable", "segment elasticity + demand score")
        with k2: kpi("Mean elasticity", f"{elasticity_df['Estimated Elasticity'].mean():.2f}", "negative = price sensitive")
        with k3: kpi("Training status", "Complete", "local prototype calculation")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Segment elasticity output")
            display = elasticity_df.copy()
            display["Observed Conversion"] = display["Observed Conversion"].apply(lambda x: pct(x * 100))
            display["Estimated Elasticity"] = display["Estimated Elasticity"].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(elasticity_df, x="Segment", y="Estimated Elasticity", color="Interpretation", title="Estimated price elasticity by guest segment")
            fig.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        importance = pd.DataFrame({
            "Feature": ["Price ratio vs comp-set", "Demand index", "Occupancy pace", "Guest segment", "Event / season", "Channel"],
            "Importance proxy": [31, 24, 18, 14, 9, 4],
        })
        fig2 = px.bar(importance, x="Importance proxy", y="Feature", orientation="h", title="Feature importance proxy for revenue manager explanation")
        fig2.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with scenario_tab:
        df = st.session_state.guest_df
        elasticity_df = estimate_segment_elasticity(df)
        lookup = dict(zip(elasticity_df["Segment"], elasticity_df["Estimated Elasticity"]))
        st.markdown("### Scenario-based rate recommendation")
        s1, s2, s3 = st.columns(3)
        seg = s1.selectbox("Guest segment", list(SEGMENTS.keys()), index=0)
        comp_adr = s2.number_input("Competitor ADR", value=455.0, step=5.0)
        curr_adr = s3.number_input("Current ADR", value=420.0, step=5.0)
        s4, s5, s6 = st.columns(3)
        demand_index = s4.slider("Demand index", 0, 100, 78)
        occupancy_pace = s5.slider("Occupancy pace", 0.0, 1.0, .74, .01)
        guardrail = s6.slider("Brand guardrail ±%", 5, 30, 18)
        recommended, curve, e = recommendation_from_elasticity(seg, comp_adr, curr_adr, demand_index, occupancy_pace, guardrail, lookup)
        prob_at_rec = curve.iloc[(curve["Offered ADR"] - recommended).abs().idxmin()]["Booking Probability"]
        revpar_at_rec = recommended * prob_at_rec
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Recommended ADR", money(recommended, "$"), f"{(recommended/curr_adr-1)*100:+.1f}% vs current")
        with k2: kpi("Elasticity", f"{e:.2f}", seg)
        with k3: kpi("Booking probability", pct(prob_at_rec * 100), "model estimate")
        with k4: kpi("RevPAR proxy", money(revpar_at_rec, "$"), "ADR × probability")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Offered ADR"], y=curve["RevPAR Proxy"], mode="lines", name="RevPAR proxy"))
        fig.add_vline(x=recommended, line_dash="dash", line_color="#C59B5A", annotation_text="recommended")
        fig.update_layout(template="plotly_dark", height=420, title="Demand curve and recommended ADR", xaxis_title="Offered ADR", yaxis_title="RevPAR proxy", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.code(textwrap.dedent(f"""
        Model explanation:
        Segment: {seg}
        Estimated elasticity: {e:.2f}
        Demand index: {demand_index}/100
        Occupancy pace: {occupancy_pace:.0%}
        Competitor ADR: {money(comp_adr, '$')}
        Current ADR: {money(curr_adr, '$')}
        Guardrail: ±{guardrail}%

        The model tests a range of price ratios against expected booking probability.
        The selected ADR is the highest RevPAR proxy after applying the brand-safe guardrail.
        """).strip(), language="text")

    with governance_tab:
        note("This module is intentionally transparent: it uses synthetic guest data and local calculations. A real model would run server-side with PMS, CRS, CRM, booking engine and rate-shopper data, with model validation and drift monitoring.")
        provider = st.selectbox("LLM provider", ["ChatGPT / OpenAI", "Gemini", "Claude"], key="tech_provider")
        st.code(textwrap.dedent(f"""
        Provider: {provider}
        Role: Hotel pricing model explainer.
        Task: Explain a pricing elasticity recommendation to a revenue manager.
        Constraints:
        - Use plain English.
        - Separate model output from human approval.
        - Mention that synthetic data is used in prototype mode.
        - Do not claim live PMS/CRM integration unless production connectors exist.
        """).strip(), language="text")

# ------------------------------------------------------------
# Workspace 4: GTM Strategy & Corporate
# ------------------------------------------------------------
elif workspace == "GTM Strategy & Corporate":
    hero(
        "GTM Strategy & Corporate",
        "Combines Deliverables 1, 2 and 3: AI Marketing, AI Sales and AI HR/CRO org design. This section shows how LuxePricing.ai goes to market and how a hotel group would organize around it.",
        badges=[{"text": "Deliverable 1", "style": "gold"}, {"text": "Deliverable 2", "style": "blue"}, {"text": "Deliverable 3", "style": "green"}],
    )

    marketing_tab, sales_tab, hr_tab, summary_tab = st.tabs(["AI Marketing", "AI Sales", "AI HR / CRO", "Integrated GTM summary"])

    with marketing_tab:
        st.markdown("### AI brand identity generator + automated luxury ad engine")
        col1, col2 = st.columns([.9, 1.1], gap="large")
        with col1:
            hotel = st.text_input("Hotel / project name", "Maison Atlas Lisbon")
            location = st.text_input("Location", "Lisbon, Portugal")
            positioning = st.selectbox("Positioning", ["Quiet luxury", "Heritage & culture", "Wellness luxury", "Design-led lifestyle", "Business luxury"])
            audience = st.text_area("Audience", "Affluent leisure guests, design-conscious travelers, couples and premium city-break travelers.")
            personality = st.text_area("Brand personality", "Calm, editorial, local, discreet, refined, not flashy.")
            avoid = st.text_area("Words / claims to avoid", "Cheap, discount, best price, mass-market, fake exclusivity, unsupported sustainability claims.")
            st.markdown("#### Calendar upload")
            cal_upload = st.file_uploader("Upload campaign calendar CSV", type=["csv"], key="marketing_cal")
            if cal_upload:
                cal_df = pd.read_csv(cal_upload)
            else:
                cal_df = sample_marketing_calendar()
        with col2:
            primary = "#1B365D" if positioning in ["Quiet luxury", "Business luxury"] else "#3B2F2F" if positioning == "Heritage & culture" else "#173D35"
            accent = "#B38B59"
            st.markdown("#### Generated brand standards")
            brand_standards = pd.DataFrame([
                ["Primary color", primary, "Used for CRM module, hero sections, navigation and formal brand moments"],
                ["Accent color", accent, "Used for CTA, approval state, high-value content and premium highlights"],
                ["Headline type", "Editorial serif, 32–44px", "Elegant but readable; avoid overly decorative luxury clichés"],
                ["Body type", "Clean sans-serif, 14–16px", "CRM-friendly, readable in dashboards and approvals"],
                ["Tone", "Calm, precise, premium", "Avoid discount-led language; emphasize experience and confidence"],
                ["CTA style", "Soft conversion", "Request the opening brief, Discover availability, Join the private list"],
            ], columns=["Standard", "Rule", "How used"])
            st.dataframe(brand_standards, use_container_width=True, hide_index=True)
            st.download_button("Download brand standards CSV", csv_download(brand_standards), "brand_standards.csv", "text/csv")
            st.markdown("#### Imported campaign calendar")
            st.dataframe(cal_df, use_container_width=True, hide_index=True)

        st.markdown("#### Generated ads and approval queue")
        drafts = []
        for _, r in cal_df.iterrows():
            channel = r.get("Channel", "Instagram")
            theme = r.get("Theme", "Campaign")
            cta = r.get("CTA", "Learn more")
            copy = f"{hotel} introduces {theme.lower()} in {location}. A refined invitation for {audience.split(',')[0].lower()}, shaped around {positioning.lower()} and a slower sense of arrival. {cta}."
            drafts.append({"Date": r.get("Date", ""), "Channel": channel, "Theme": theme, "Generated Copy": copy, "CTA": cta, "Status": "Draft - needs approval"})
        drafts_df = pd.DataFrame(drafts)
        st.dataframe(drafts_df, use_container_width=True, hide_index=True)
        st.download_button("Export generated draft posts", csv_download(drafts_df), "marketing_draft_posts.csv", "text/csv")
        st.code(textwrap.dedent(f"""
        Marketing LLM prompt pack:
        Generate luxury-compliant content for {hotel} in {location}.
        Positioning: {positioning}
        Audience: {audience}
        Brand personality: {personality}
        Avoid: {avoid}
        Rules: no discount-heavy language, no unsupported claims, maintain premium tone, output requires human approval before publishing.
        """).strip(), language="text")

    with sales_tab:
        st.markdown("### AI sales agent for hotel-chain outreach + demo scripts")
        c1, c2, c3 = st.columns(3)
        account = c1.text_input("Hotel group / target account", "Accor Europe")
        role = c2.selectbox("Buyer role", ["VP Development", "Chief Revenue Officer", "Revenue Strategy Director", "Asset Manager", "Hotel General Manager", "Brand Director"])
        region = c3.text_input("Region", "Europe")
        product_angle = st.selectbox("Product angle", ["Revenue AI", "Hotel development feasibility AI", "Owner / asset management ROI AI", "Renovation and repositioning AI", "USALI reporting and benchmarking AI"])
        pain = st.text_area("Main pain point", "Revenue and development teams rely on fragmented reports. They need a faster way to connect market research, USALI financial logic, competitor benchmarking and dynamic pricing decisions.")
        request = st.text_area("Natural-language sales request", f"Draft outreach for {account}'s {role}. Focus on how LuxePricing.ai can support hotel development feasibility and existing hotel dynamic pricing without replacing the human revenue manager.")

        email = f"""
        Subject: Connecting hotel feasibility, USALI finance and revenue AI for {account}

        Dear {role},

        I am reaching out because teams across hotel development, revenue strategy and asset management often work from separate reports: STR-style benchmarks, PMS exports, USALI finance files and manual Excel pricing assumptions.

        LuxePricing.ai is an internal CRM-style platform that connects those workflows. It helps teams test hotel-development scenarios, clean revenue data, estimate pricing uplift, and produce daily rate recommendations using demand, calendar and competitor signals — while keeping human approval before any rate change.

        For {account} in {region}, the relevant use case would be {product_angle.lower()}: faster feasibility decisions, clearer ROI calculations, and more consistent revenue governance across properties.

        Would you be open to a 20-minute demo using an anonymized hotel scenario?

        Best regards,
        LuxePricing.ai Team
        """
        linkedin = f"Hi, I’m working on LuxePricing.ai, a CRM-style hotel intelligence platform connecting feasibility, USALI finance, data quality and AI pricing recommendations. I thought it could be relevant for {account}’s {region} teams. Open to a short demo?"
        demo_script = f"""
        7-minute demo script
        0:00–1:00 — Problem: hotel teams rely on fragmented revenue, finance and market data.
        1:00–2:00 — Show CRM home and property record.
        2:00–3:00 — Show Ops data-quality monitor cleaning PMS/revenue data.
        3:00–4:00 — Show Finance ROI calculator converting ADR uplift into NOI and asset value.
        4:00–5:20 — Show Tech elasticity model and daily BAR recommendation agent.
        5:20–6:20 — Show Marketing/Sales/HR modules supporting go-to-market and org adoption.
        6:20–7:00 — Close: the system recommends; humans approve. Ask for a one-property pilot.
        """
        t1, t2, t3 = st.tabs(["Email", "LinkedIn", "Demo script + objections"])
        with t1: st.text_area("Generated first outreach email", textwrap.dedent(email).strip(), height=310)
        with t2: st.text_area("Generated LinkedIn message", linkedin, height=140)
        with t3:
            st.text_area("Generated demo script", textwrap.dedent(demo_script).strip(), height=260)
            objections = pd.DataFrame([
                ["We already have an RMS", "LuxePricing.ai is not positioned as a replacement. It connects RMS/PMS/CRM/finance workflows and adds explainable ROI and approval logic."],
                ["We cannot share revenue data", "The pilot can use anonymized or synthetic data first, then move to governed internal data only after permissions are approved."],
                ["AI pricing could damage the brand", "The tool uses floor/ceiling, max-change and luxury tone guardrails. Human revenue managers approve final rate decisions."],
            ], columns=["Objection", "Response"])
            st.dataframe(objections, use_container_width=True, hide_index=True)
        st.code(textwrap.dedent(f"""
        Sales LLM prompt pack:
        Prospect: {account}
        Buyer role: {role}
        Region: {region}
        Product angle: {product_angle}
        Pain point: {pain}
        User request: {request}
        Output: first-touch email, LinkedIn message, 7-minute demo script, discovery questions, objections, CRM follow-up tasks.
        Compliance: no fake integrations, no claim of live STR/PMS access unless connected; all generated content is draft-only.
        """).strip(), language="text")

    with hr_tab:
        st.markdown("### AI-generated CRO role profile + org design")
        c1, c2, c3 = st.columns(3)
        company = c1.text_input("Company / hotel group", "LuxePricing.ai client hotel group")
        portfolio = c2.selectbox("Portfolio size", ["1 hotel", "2–10 hotels", "11–50 hotels", "50+ hotels"], index=1)
        phase = c3.selectbox("Current phase", ["Pre-opening", "Scale-up", "Portfolio transformation", "Turnaround"], index=1)
        systems = st.text_input("Current revenue systems", "PMS, CRM, channel manager, Excel reporting, basic RMS")
        hr_request = st.text_area("Natural-language HR request", "Create a CRO role profile and revenue organization design for a hotel group implementing LuxePricing.ai. The CRO must align revenue management, sales, marketing, CRM, distribution and AI data operations while protecting luxury brand positioning.")
        team_size = {"1 hotel": "4–5 FTE", "2–10 hotels": "8–10 FTE", "11–50 hotels": "18–25 FTE", "50+ hotels": "Global hub + regional pods"}[portfolio]
        profile = f"""
        Chief Revenue Officer — AI-Augmented Hotel Revenue Operations

        Mandate:
        The CRO owns total revenue strategy across revenue management, sales, marketing, CRM, distribution and AI data operations for {company}. The role ensures LuxePricing.ai recommendations are commercially useful, financially explainable and brand-safe.

        Key responsibilities:
        - Set pricing governance, rate floors, ceilings and override rules.
        - Align revenue management with sales pipeline, marketing calendar and owner ROI expectations.
        - Own RevPAR, ADR, occupancy mix, direct booking margin, GOPPAR and NOI contribution.
        - Sponsor CRM data quality and AI adoption across PMS/RMS/CRM workflows.
        - Protect luxury brand positioning by preventing uncontrolled discounting or opaque AI decisions.

        Required profile:
        Senior hotel commercial leader with experience in revenue management, distribution, owner reporting, CRM, brand positioning and cross-functional team leadership.
        """
        org = pd.DataFrame([
            ["CEO / Managing Director", "Approves strategic revenue direction and investment priorities"],
            ["Chief Revenue Officer", "Owns total revenue strategy, AI governance and commercial performance"],
            ["Revenue Management Lead", "Pricing, forecasting, inventory, RMS governance"],
            ["Sales & Key Accounts Lead", "Hotel-chain outreach, corporate accounts, groups and demos"],
            ["Marketing & CRM Lead", "Campaign calendar, segmentation, approval queue and brand standards"],
            ["Distribution Manager", "OTAs, direct booking, channel performance and rate parity"],
            ["AI Revenue Ops Analyst", "Data quality, prompt operations, model monitoring and dashboards"],
            ["Brand Revenue Guardian", "Luxury guardrails, discount exceptions and guest trust protection"],
        ], columns=["Role", "Purpose"])
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Org phase", phase, "from HR intake")
        with k2: kpi("Suggested team", team_size, "central + regional")
        with k3: kpi("Priority hires", "3", "first 90 days")
        with k4: kpi("AI maturity", "Level 3", "governed automation")
        st.text_area("Generated CRO role profile", textwrap.dedent(profile).strip(), height=310)
        st.markdown("#### Recommended organization design")
        st.dataframe(org, use_container_width=True, hide_index=True)
        kpis = pd.DataFrame([
            ["Revenue quality", "RevPAR Index, ADR growth, occupancy mix", "Weekly / monthly"],
            ["Profitability", "GOPPAR, NOI contribution, direct booking margin", "Monthly"],
            ["Commercial alignment", "Sales pipeline value, campaign conversion, CRM lead quality", "Weekly"],
            ["AI adoption", "% recommendations reviewed, override reasons logged, data-quality score", "Weekly"],
            ["Brand protection", "Rate-floor breaches, discount exceptions, luxury tone compliance", "Weekly"],
        ], columns=["KPI category", "Metric", "Review cadence"])
        st.markdown("#### CRO KPI scorecard")
        st.dataframe(kpis, use_container_width=True, hide_index=True)
        st.code(textwrap.dedent(f"""
        HR LLM prompt pack:
        Company: {company}
        Portfolio size: {portfolio}
        Phase: {phase}
        Current systems: {systems}
        Request: {hr_request}
        Constraints: do not include protected characteristics, make responsibilities measurable, separate AI recommendations from human decision rights, keep hotel revenue management and brand protection central.
        """).strip(), language="text")

    with summary_tab:
        st.markdown("### Integrated go-to-market and corporate adoption plan")
        summary = pd.DataFrame([
            ["Marketing", "Brand kit + campaign calendar + AI content drafts", "Marketing Manager", "Content cannot publish until approved and API permissions exist"],
            ["Sales", "Hotel-chain outreach + demo script + objections", "Sales Lead", "No fake integration claims; all emails are drafts"],
            ["HR", "CRO profile + RevOps org design + KPI scorecard", "Leadership / HR", "AI supports org design; hiring decisions remain human"],
            ["Finance", "ROI model proving value", "Asset Manager / CFO", "USALI-style simplified bridge; not audited financial statement"],
            ["Ops", "Data quality before CRM ingestion", "CRM Admin / Data Ops", "No silent overwrites"],
            ["Tech", "Elasticity model proving pricing logic", "Revenue Ops / Data Science", "Synthetic data in prototype mode"],
            ["Agent", "Daily BAR recommendation", "Revenue Manager", "Human approval before PMS/CRS push"],
        ], columns=["Area", "Deliverable", "Owner", "Governance rule"])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        note("This final app is designed to support the 8-minute pitch and the 7-minute ‘how we made it’ section. It can show the business value, the technical prototype, the real-world constraints, and the human approval model in one coherent product.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("LuxePricing.ai Capstone Streamlit App • CRM-style internal prototype • No live API keys, no unauthorized scraping, human approval required for pricing and publishing.")
