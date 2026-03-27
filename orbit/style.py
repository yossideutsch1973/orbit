"""Orbit CSS and theming constants."""

ACCENT = "#4F8EF7"
ACCENT_DARK = "#3A6FD8"
BG_DARK = "#0E1117"
BG_CARD = "#1A1D23"
TEXT_PRIMARY = "#FAFAFA"
TEXT_SECONDARY = "#A0A4AB"
SUCCESS = "#00C853"
WARNING = "#FFB300"
ERROR = "#FF5252"

CUSTOM_CSS = f"""
<style>
/* ---------- global ---------- */
[data-testid="stAppViewContainer"] {{
    background-color: {BG_DARK};
}}

/* ---------- accent buttons ---------- */
div.stButton > button {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    transition: background-color 0.2s;
}}
div.stButton > button:hover {{
    background-color: {ACCENT_DARK};
    color: white;
    border: none;
}}

/* ---------- cards ---------- */
div[data-testid="stExpander"] {{
    background-color: {BG_CARD};
    border-radius: 10px;
    border: 1px solid #2A2D35;
}}

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {{
    background-color: {BG_CARD};
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #2A2D35;
}}
div[data-testid="stMetric"] label {{
    color: {TEXT_SECONDARY};
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {ACCENT};
    font-weight: 700;
}}

/* ---------- headers ---------- */
h1 {{
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.5px;
}}
h2, h3 {{
    color: {TEXT_PRIMARY} !important;
}}

/* ---------- success/info banners ---------- */
div[data-testid="stAlert"] {{
    border-radius: 8px;
}}

/* ---------- hide default Streamlit branding ---------- */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* ---------- step indicators ---------- */
.step-badge {{
    display: inline-block;
    background-color: {ACCENT};
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    text-align: center;
    line-height: 32px;
    font-weight: 700;
    font-size: 1rem;
    margin-right: 8px;
}}
.step-title {{
    display: inline-block;
    font-size: 1.3rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    vertical-align: middle;
}}
.step-complete {{
    background-color: {SUCCESS};
}}

/* ---------- tagline ---------- */
.tagline {{
    color: {TEXT_SECONDARY};
    font-size: 1.1rem;
    margin-top: -12px;
    margin-bottom: 24px;
}}
</style>
"""

MPL_STYLE_PARAMS = {
    "figure.facecolor": BG_DARK,
    "axes.facecolor": "#1E2129",
    "axes.edgecolor": "#3A3D45",
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": TEXT_PRIMARY,
    "grid.color": "#2A2D35",
    "grid.alpha": 0.5,
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": "#3A3D45",
    "legend.fontsize": 10,
    "font.size": 11,
    "lines.linewidth": 2,
}
