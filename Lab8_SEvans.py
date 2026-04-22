# =============================================================================
# Lab 8 – Impact of Remote Work on Mental Health
# Author  : S. Evans
# Course  : INSC 489 – Spring 2026
# Dataset : Impact_of_Remote_Work_on_Mental_Health.csv (Kaggle)
# Tech    : Python · Pandas · Plotly · Dash
# =============================================================================
#
# STORY  ──────────────────────────────────────────────────────────────────────
#   Audience : HR managers, workplace-wellness officers, and executives who
#              set remote-work policy.
#   Questions:
#     1. Does work location (Remote / Hybrid / Onsite) relate to mental-health
#        conditions reported by employees?
#     2. Which industries and roles carry the highest stress burden?
#     3. How do hours worked, virtual meetings, and social isolation interact
#        with stress?
#     4. Does company support for remote work correlate with better sleep,
#        higher satisfaction, or improved productivity?
#     5. Are there demographic differences (age, gender) in mental-health
#        outcomes across work models?

import pathlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── 0. Paths ────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "Impact_of_Remote_Work_on_Mental_Health.csv"

# ── 1. Load & clean ─────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)

# Stress level → numeric order for sorting axes
STRESS_ORDER = {"Low": 1, "Medium": 2, "High": 3}
df_raw["Stress_Num"] = df_raw["Stress_Level"].map(STRESS_ORDER)

# Sleep quality → ordered category
SLEEP_ORDER = ["Poor", "Average", "Good"]

# Satisfaction → ordered category
SAT_ORDER = ["Unsatisfied", "Neutral", "Satisfied"]

# Drop rows where both key fields are missing
df = df_raw.dropna(subset=["Work_Location", "Mental_Health_Condition"]).copy()

# ── 2. Colour palette (accessible, limited set) ─────────────────────────────
LOCATION_COLORS = {
    "Remote": "#6C63FF",
    "Hybrid": "#43B89C",
    "Onsite": "#F5A623",
}
MENTAL_COLORS = {
    "Burnout":    "#E05C5C",
    "Anxiety":    "#F5A623",
    "Depression": "#6C63FF",
}
STRESS_COLORS = {
    "Low":    "#43B89C",
    "Medium": "#F5A623",
    "High":   "#E05C5C",
}

# Global Plotly theme overrides
CHART_THEME = "plotly_dark"
FONT_FAMILY = "Inter, Segoe UI, sans-serif"
BG_COLOR    = "#12131A"
PAPER_COLOR = "#1C1D2B"
GRID_COLOR  = "#2A2B3D"

def apply_layout(fig, title="", height=400):
    """Apply consistent dark-theme styling to every figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family=FONT_FAMILY, color="#EAEAEA"), x=0.02),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family=FONT_FAMILY, color="#C8C9D4"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    )
    return fig

# ── 3. Helper: filter df by sidebar controls ─────────────────────────────────
def filter_df(locations, genders, industries):
    mask = (
        df["Work_Location"].isin(locations) &
        df["Gender"].isin(genders) &
        df["Industry"].isin(industries)
    )
    return df[mask]

# ── 4. Build app ─────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP],
    title="Remote Work & Mental Health",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ── 4a. Sidebar filter panel ─────────────────────────────────────────────────
all_locations  = sorted(df["Work_Location"].unique())
all_genders    = sorted(df["Gender"].unique())
all_industries = sorted(df["Industry"].unique())

sidebar = dbc.Card(
    [
        html.H6("FILTERS", className="text-uppercase fw-bold mb-3",
                style={"letterSpacing": "2px", "color": "#6C63FF"}),
        html.Label("Work Location", className="small fw-bold",
                   style={"color": "#B8BDD4", "display": "block", "marginBottom": "4px"}),
        dcc.Checklist(
            id="filter-location",
            options=[{"label": f"  {v}", "value": v} for v in all_locations],
            value=all_locations,
            className="mb-3",
            inputStyle={"marginRight": "6px", "accentColor": "#6C63FF"},
            labelStyle={"color": "#E8E9F0", "fontSize": "0.875rem",
                        "display": "block", "marginBottom": "3px", "cursor": "pointer"},
        ),
        html.Label("Gender", className="small fw-bold",
                   style={"color": "#B8BDD4", "display": "block", "marginBottom": "4px"}),
        dcc.Checklist(
            id="filter-gender",
            options=[{"label": f"  {v}", "value": v} for v in all_genders],
            value=all_genders,
            className="mb-3",
            inputStyle={"marginRight": "6px", "accentColor": "#6C63FF"},
            labelStyle={"color": "#E8E9F0", "fontSize": "0.875rem",
                        "display": "block", "marginBottom": "3px", "cursor": "pointer"},
        ),
        html.Label("Industry", className="small fw-bold",
                   style={"color": "#B8BDD4", "display": "block", "marginBottom": "4px"}),
        dcc.Checklist(
            id="filter-industry",
            options=[{"label": f"  {v}", "value": v} for v in all_industries],
            value=all_industries,
            className="mb-3",
            inputStyle={"marginRight": "6px", "accentColor": "#6C63FF"},
            labelStyle={"color": "#E8E9F0", "fontSize": "0.875rem",
                        "display": "block", "marginBottom": "3px", "cursor": "pointer"},
        ),
        html.Hr(style={"borderColor": "#2A2B3D"}),
        html.Label("Age Range", className="small fw-bold",
                   style={"color": "#B8BDD4", "display": "block", "marginBottom": "4px"}),
        dcc.RangeSlider(
            id="filter-age",
            min=int(df["Age"].min()),
            max=int(df["Age"].max()),
            step=1,
            value=[int(df["Age"].min()), int(df["Age"].max())],
            marks={
                int(df["Age"].min()): str(int(df["Age"].min())),
                int(df["Age"].max()): str(int(df["Age"].max())),
            },
            tooltip={"placement": "bottom", "always_visible": False},
            className="mb-3",
        ),
        html.Div(id="record-count",
                 className="small mt-2 text-center",
                 style={"color": "#8E91A8"}),
    ],
    body=True,
    style={"backgroundColor": PAPER_COLOR, "border": "1px solid #2A2B3D",
           "position": "sticky", "top": "10px"},
)

# ── 4b. KPI row ──────────────────────────────────────────────────────────────
def kpi_card(icon, label, value_id):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.I(className=f"bi {icon} fs-3 mb-1",
                       style={"color": "#6C63FF"}),
                html.Div(id=value_id, className="fs-4 fw-bold"),
                html.Div(label, className="small text-muted"),
            ], className="text-center p-2"),
            style={"backgroundColor": PAPER_COLOR, "border": "1px solid #2A2B3D"},
        ),
        xs=6, md=3, className="mb-3",
    )

kpi_row = dbc.Row([
    kpi_card("bi-people-fill",        "Employees",               "kpi-employees"),
    kpi_card("bi-emoji-frown-fill",   "Burnout Rate",            "kpi-burnout"),
    kpi_card("bi-thermometer-half",   "High Stress Rate",        "kpi-stress"),
    kpi_card("bi-clock-history",      "Avg Hours / Week",        "kpi-hours"),
], className="mb-2")

# ── 4c. Tab group 1 – Overview ───────────────────────────────────────────────
tab_overview = dbc.Tab(label="🏠 Overview", tab_id="tab-overview", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mh-location"), md=6),
        dbc.Col(dcc.Graph(id="chart-stress-industry"), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-sleep-location"), md=6),
        dbc.Col(dcc.Graph(id="chart-productivity"), md=6),
    ]),
])

# ── 4d. Tab group 2 – Stress & Workload ─────────────────────────────────────
tab_stress = dbc.Tab(label="🔥 Stress & Workload", tab_id="tab-stress", children=[
    dbc.Row([
        dbc.Col([
            html.Label("Color by:", className="small fw-bold"),
            dcc.RadioItems(
                id="scatter-color-by",
                options=[
                    {"label": " Work Location", "value": "Work_Location"},
                    {"label": " Stress Level",  "value": "Stress_Level"},
                    {"label": " Mental Health", "value": "Mental_Health_Condition"},
                ],
                value="Stress_Level",
                inline=True,
                inputStyle={"marginRight": "5px", "marginLeft": "12px"},
                className="mb-2",
            ),
        ], md=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-scatter-hours-stress"), md=8),
        dbc.Col(dcc.Graph(id="chart-social-isolation"), md=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-virtual-meetings-stress"), md=6),
        dbc.Col(dcc.Graph(id="chart-wlb-stress"), md=6),
    ]),
])

# ── 4e. Tab group 3 – Support & Satisfaction ─────────────────────────────────
tab_support = dbc.Tab(label="🤝 Support & Satisfaction", tab_id="tab-support", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-support-mh"), md=6),
        dbc.Col(dcc.Graph(id="chart-satisfaction-location"), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-physical-activity"), md=5),
        dbc.Col(dcc.Graph(id="chart-heatmap"), md=7),
    ]),
])

# ── 4f. Tab group 4 – Demographics ───────────────────────────────────────────
tab_demographics = dbc.Tab(label="👥 Demographics", tab_id="tab-demographics", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-age-mh"), md=6),
        dbc.Col(dcc.Graph(id="chart-gender-stress"), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-role-mh"), md=12),
    ]),
])

# ── 4g. Full layout ──────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG_COLOR, "minHeight": "100vh", "paddingBottom": "40px"},
    children=[
        # Header
        html.Div(
            [
                html.Span("🧠", style={"fontSize": "2.2rem", "marginRight": "12px"}),
                html.Div([
                    html.H1("Remote Work & Mental Health",
                            style={"fontSize": "1.8rem", "margin": 0,
                                   "fontFamily": FONT_FAMILY, "color": "#EAEAEA"}),
                    html.P("An interactive analysis for HR leaders and workplace wellness teams",
                           className="mb-0 text-muted small"),
                ]),
            ],
            className="d-flex align-items-center py-3 px-3 mb-3",
            style={"borderBottom": "1px solid #2A2B3D",
                   "background": "linear-gradient(90deg,#12131A 70%,#1a1b2e)"},
        ),

        dbc.Row([
            # Sidebar
            dbc.Col(sidebar, xs=12, md=2, className="pe-0"),

            # Main content
            dbc.Col([
                kpi_row,
                dbc.Tabs(
                    [tab_overview, tab_stress, tab_support, tab_demographics],
                    id="main-tabs",
                    active_tab="tab-overview",
                    className="mb-3",
                ),
            ], xs=12, md=10),
        ]),
    ],
)

# =============================================================================
# 5. Callbacks
# =============================================================================

# ── shared filter helper ─────────────────────────────────────────────────────
def get_filtered(locations, genders, industries, age_range):
    mask = (
        df["Work_Location"].isin(locations) &
        df["Gender"].isin(genders) &
        df["Industry"].isin(industries) &
        df["Age"].between(age_range[0], age_range[1])
    )
    return df[mask]


# ── KPIs ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-employees", "children"),
    Output("kpi-burnout",   "children"),
    Output("kpi-stress",    "children"),
    Output("kpi-hours",     "children"),
    Output("record-count",  "children"),
    Input("filter-location",  "value"),
    Input("filter-gender",    "value"),
    Input("filter-industry",  "value"),
    Input("filter-age",       "value"),
)
def update_kpis(locations, genders, industries, age_range):
    d = get_filtered(locations, genders, industries, age_range)
    if d.empty:
        return "0", "—", "—", "—", "No data"
    n = len(d)
    burnout_pct   = f"{(d['Mental_Health_Condition']=='Burnout').mean()*100:.1f}%"
    high_stress   = f"{(d['Stress_Level']=='High').mean()*100:.1f}%"  # % reporting High stress
    avg_hours     = f"{d['Hours_Worked_Per_Week'].mean():.1f}"
    count_txt     = f"{n:,} records selected"
    return f"{n:,}", burnout_pct, high_stress, avg_hours, count_txt


# ── Overview Tab ─────────────────────────────────────────────────────────────
@app.callback(
    Output("chart-mh-location",  "figure"),
    Output("chart-stress-industry", "figure"),
    Output("chart-sleep-location",  "figure"),
    Output("chart-productivity",    "figure"),
    Input("filter-location",  "value"),
    Input("filter-gender",    "value"),
    Input("filter-industry",  "value"),
    Input("filter-age",       "value"),
)
def update_overview(locations, genders, industries, age_range):
    d = get_filtered(locations, genders, industries, age_range)

    # Chart 1 – Mental health condition split by work location (100% stacked bar)
    if d.empty:
        fig1 = go.Figure()
    else:
        grp = (d.groupby(["Work_Location", "Mental_Health_Condition"])
                .size().reset_index(name="Count"))
        totals = grp.groupby("Work_Location")["Count"].transform("sum")
        grp["Pct"] = grp["Count"] / totals * 100
        fig1 = px.bar(
            grp, x="Work_Location", y="Pct",
            color="Mental_Health_Condition",
            color_discrete_map=MENTAL_COLORS,
            barmode="stack",
            labels={"Pct": "% of Employees", "Work_Location": "Work Model",
                    "Mental_Health_Condition": "Condition"},
            category_orders={"Work_Location": ["Remote", "Hybrid", "Onsite"]},
            template=CHART_THEME,
        )
    apply_layout(fig1, "Mental Health Condition by Work Model")

    # Chart 2 – Average stress by industry (horizontal bar)
    if d.empty:
        fig2 = go.Figure()
    else:
        stress_by_ind = (d.groupby("Industry")["Stress_Num"]
                          .mean().reset_index(name="Avg_Stress")
                          .sort_values("Avg_Stress", ascending=True))
        fig2 = px.bar(
            stress_by_ind, x="Avg_Stress", y="Industry",
            orientation="h",
            color="Avg_Stress",
            color_continuous_scale=["#43B89C", "#F5A623", "#E05C5C"],
            range_color=[1, 3],
            labels={"Avg_Stress": "Avg Stress (1=Low, 3=High)"},
            template=CHART_THEME,
        )
        fig2.update(layout_coloraxis_showscale=False)
    apply_layout(fig2, "Average Stress Level by Industry")

    # Chart 3 – Sleep quality distribution by work location
    sleep_order_val = SLEEP_ORDER
    if d.empty:
        fig3 = go.Figure()
    else:
        sleep_grp = (d[d["Sleep_Quality"].isin(sleep_order_val)]
                      .groupby(["Work_Location", "Sleep_Quality"])
                      .size().reset_index(name="Count"))
        fig3 = px.bar(
            sleep_grp, x="Work_Location", y="Count",
            color="Sleep_Quality",
            barmode="group",
            color_discrete_sequence=["#E05C5C", "#F5A623", "#43B89C"],
            category_orders={
                "Sleep_Quality": sleep_order_val,
                "Work_Location": ["Remote", "Hybrid", "Onsite"],
            },
            labels={"Work_Location": "Work Model", "Count": "# Employees",
                    "Sleep_Quality": "Sleep Quality"},
            template=CHART_THEME,
        )
    apply_layout(fig3, "Sleep Quality by Work Model")

    # Chart 4 – Productivity change distribution (donut)
    if d.empty:
        fig4 = go.Figure()
    else:
        prod = d["Productivity_Change"].value_counts().reset_index()
        prod.columns = ["Productivity_Change", "Count"]
        prod_colors = {"Increase": "#43B89C", "No Change": "#6C63FF", "Decrease": "#E05C5C"}
        fig4 = px.pie(
            prod, names="Productivity_Change", values="Count",
            hole=0.52,
            color="Productivity_Change",
            color_discrete_map=prod_colors,
            template=CHART_THEME,
        )
        fig4.update_traces(textinfo="percent+label", pull=[0.03, 0.03, 0.03])
    apply_layout(fig4, "Reported Productivity Change")

    return fig1, fig2, fig3, fig4


# ── Stress & Workload Tab ────────────────────────────────────────────────────
@app.callback(
    Output("chart-scatter-hours-stress",  "figure"),
    Output("chart-social-isolation",       "figure"),
    Output("chart-virtual-meetings-stress","figure"),
    Output("chart-wlb-stress",             "figure"),
    Input("filter-location",  "value"),
    Input("filter-gender",    "value"),
    Input("filter-industry",  "value"),
    Input("filter-age",       "value"),
    Input("scatter-color-by", "value"),
)
def update_stress(locations, genders, industries, age_range, color_by):
    d = get_filtered(locations, genders, industries, age_range)

    color_map = None
    if color_by == "Work_Location":
        color_map = LOCATION_COLORS
    elif color_by == "Stress_Level":
        color_map = STRESS_COLORS
    elif color_by == "Mental_Health_Condition":
        color_map = MENTAL_COLORS

    # Chart 1 – Scatter: hours worked vs social isolation, coloured by chosen field
    if d.empty:
        fig1 = go.Figure()
    else:
        sample = d.sample(min(1000, len(d)), random_state=42)
        fig1 = px.scatter(
            sample,
            x="Hours_Worked_Per_Week",
            y="Social_Isolation_Rating",
            color=color_by,
            color_discrete_map=color_map,
            opacity=0.65,
            labels={"Hours_Worked_Per_Week": "Hrs / Week",
                    "Social_Isolation_Rating": "Social Isolation (1–5)"},
            template=CHART_THEME,
            hover_data=["Job_Role", "Industry", "Stress_Level"],
        )
    apply_layout(fig1, "Hours Worked vs Social Isolation", height=420)

    # Chart 2 – Box: social isolation by stress level
    if d.empty:
        fig2 = go.Figure()
    else:
        fig2 = px.box(
            d, x="Stress_Level", y="Social_Isolation_Rating",
            color="Stress_Level",
            color_discrete_map=STRESS_COLORS,
            category_orders={"Stress_Level": ["Low", "Medium", "High"]},
            labels={"Stress_Level": "Stress Level",
                    "Social_Isolation_Rating": "Isolation (1–5)"},
            template=CHART_THEME,
        )
        fig2.update_traces(showlegend=False)
    apply_layout(fig2, "Isolation by Stress", height=420)

    # Chart 3 – Bar: avg virtual meetings by stress level × work location
    if d.empty:
        fig3 = go.Figure()
    else:
        vm = (d.groupby(["Stress_Level", "Work_Location"])
               ["Number_of_Virtual_Meetings"]
               .mean().reset_index(name="Avg_Meetings"))
        fig3 = px.bar(
            vm, x="Stress_Level", y="Avg_Meetings",
            color="Work_Location",
            barmode="group",
            color_discrete_map=LOCATION_COLORS,
            category_orders={
                "Stress_Level": ["Low", "Medium", "High"],
                "Work_Location": ["Remote", "Hybrid", "Onsite"],
            },
            labels={"Avg_Meetings": "Avg Meetings / Day",
                    "Stress_Level": "Stress Level"},
            template=CHART_THEME,
        )
    apply_layout(fig3, "Virtual Meetings / Day by Stress & Work Model")

    # Chart 4 – Violin: work-life balance rating by stress level
    if d.empty:
        fig4 = go.Figure()
    else:
        fig4 = px.violin(
            d, x="Stress_Level", y="Work_Life_Balance_Rating",
            color="Stress_Level",
            color_discrete_map=STRESS_COLORS,
            box=True, points=False,
            category_orders={"Stress_Level": ["Low", "Medium", "High"]},
            labels={"Work_Life_Balance_Rating": "WLB Rating (1–5)",
                    "Stress_Level": "Stress Level"},
            template=CHART_THEME,
        )
        fig4.update_traces(showlegend=False)
    apply_layout(fig4, "Work-Life Balance Rating by Stress Level")

    return fig1, fig2, fig3, fig4


# ── Support & Satisfaction Tab ───────────────────────────────────────────────
@app.callback(
    Output("chart-support-mh",          "figure"),
    Output("chart-satisfaction-location","figure"),
    Output("chart-physical-activity",   "figure"),
    Output("chart-heatmap",             "figure"),
    Input("filter-location",  "value"),
    Input("filter-gender",    "value"),
    Input("filter-industry",  "value"),
    Input("filter-age",       "value"),
)
def update_support(locations, genders, industries, age_range):
    d = get_filtered(locations, genders, industries, age_range)

    # Chart 1 – Grouped bar: mental health condition by access to MH resources
    if d.empty:
        fig1 = go.Figure()
    else:
        grp = (d.groupby(["Access_to_Mental_Health_Resources", "Mental_Health_Condition"])
                .size().reset_index(name="Count"))
        fig1 = px.bar(
            grp, x="Mental_Health_Condition", y="Count",
            color="Access_to_Mental_Health_Resources",
            barmode="group",
            color_discrete_sequence=["#E05C5C", "#43B89C"],
            labels={"Mental_Health_Condition": "Condition",
                    "Access_to_Mental_Health_Resources": "MH Resources"},
            template=CHART_THEME,
        )
    apply_layout(fig1, "Mental Health Conditions vs Access to Resources")

    # Chart 2 – Stacked bar: satisfaction with remote work by location
    if d.empty:
        fig2 = go.Figure()
    else:
        sat_grp = (d[d["Satisfaction_with_Remote_Work"].isin(SAT_ORDER)]
                    .groupby(["Work_Location", "Satisfaction_with_Remote_Work"])
                    .size().reset_index(name="Count"))
        totals = sat_grp.groupby("Work_Location")["Count"].transform("sum")
        sat_grp["Pct"] = sat_grp["Count"] / totals * 100
        fig2 = px.bar(
            sat_grp, x="Work_Location", y="Pct",
            color="Satisfaction_with_Remote_Work",
            barmode="stack",
            color_discrete_sequence=["#E05C5C", "#F5A623", "#43B89C"],
            category_orders={
                "Satisfaction_with_Remote_Work": SAT_ORDER,
                "Work_Location": ["Remote", "Hybrid", "Onsite"],
            },
            labels={"Pct": "% of Employees", "Work_Location": "Work Model",
                    "Satisfaction_with_Remote_Work": "Satisfaction"},
            template=CHART_THEME,
        )
    apply_layout(fig2, "Remote Work Satisfaction by Work Model")

    # Chart 3 – Bar: physical activity frequency distribution
    if d.empty:
        fig3 = go.Figure()
    else:
        pa = d["Physical_Activity"].value_counts().reset_index()
        pa.columns = ["Activity", "Count"]
        fig3 = px.bar(
            pa, x="Activity", y="Count",
            color="Activity",
            color_discrete_sequence=["#6C63FF", "#43B89C", "#F5A623", "#E05C5C"],
            template=CHART_THEME,
            labels={"Activity": "Physical Activity", "Count": "# Employees"},
        )
        fig3.update_traces(showlegend=False)
    apply_layout(fig3, "Physical Activity Frequency")

    # Chart 4 – Heatmap: stress × satisfaction (count)
    if d.empty:
        fig4 = go.Figure()
    else:
        pivot = (d[d["Satisfaction_with_Remote_Work"].isin(SAT_ORDER)]
                  .groupby(["Stress_Level", "Satisfaction_with_Remote_Work"])
                  .size().unstack(fill_value=0))
        pivot = pivot.reindex(index=["Low","Medium","High"],
                              columns=SAT_ORDER,
                              fill_value=0)
        fig4 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#12131A"],[0.5,"#6C63FF"],[1,"#E05C5C"]],
            text=pivot.values,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#C8C9D4")),
        ))
        fig4.update_layout(
            xaxis_title="Satisfaction with Remote Work",
            yaxis_title="Stress Level",
            template=CHART_THEME,
        )
    apply_layout(fig4, "Stress vs Remote Work Satisfaction (Heatmap)")

    return fig1, fig2, fig3, fig4


# ── Demographics Tab ─────────────────────────────────────────────────────────
@app.callback(
    Output("chart-age-mh",     "figure"),
    Output("chart-gender-stress","figure"),
    Output("chart-role-mh",    "figure"),
    Input("filter-location",  "value"),
    Input("filter-gender",    "value"),
    Input("filter-industry",  "value"),
    Input("filter-age",       "value"),
)
def update_demographics(locations, genders, industries, age_range):
    d = get_filtered(locations, genders, industries, age_range)

    # Chart 1 – Histogram: age distribution coloured by mental health condition
    if d.empty:
        fig1 = go.Figure()
    else:
        fig1 = px.histogram(
            d, x="Age", color="Mental_Health_Condition",
            color_discrete_map=MENTAL_COLORS,
            barmode="overlay", opacity=0.75,
            nbins=20,
            labels={"Mental_Health_Condition": "Condition"},
            template=CHART_THEME,
        )
    apply_layout(fig1, "Age Distribution by Mental Health Condition")

    # Chart 2 – Grouped bar: stress level by gender
    if d.empty:
        fig2 = go.Figure()
    else:
        gen_stress = (d.groupby(["Gender", "Stress_Level"])
                       .size().reset_index(name="Count"))
        fig2 = px.bar(
            gen_stress, x="Gender", y="Count",
            color="Stress_Level",
            barmode="group",
            color_discrete_map=STRESS_COLORS,
            category_orders={"Stress_Level": ["Low", "Medium", "High"]},
            labels={"Count": "# Employees"},
            template=CHART_THEME,
        )
    apply_layout(fig2, "Stress Level by Gender")

    # Chart 3 – Grouped bar: job role vs mental health condition (horizontal)
    if d.empty:
        fig3 = go.Figure()
    else:
        role_mh = (d.groupby(["Job_Role", "Mental_Health_Condition"])
                    .size().reset_index(name="Count"))
        fig3 = px.bar(
            role_mh, x="Count", y="Job_Role",
            color="Mental_Health_Condition",
            barmode="group",
            orientation="h",
            color_discrete_map=MENTAL_COLORS,
            labels={"Count": "# Employees", "Job_Role": "Job Role",
                    "Mental_Health_Condition": "Condition"},
            template=CHART_THEME,
        )
    apply_layout(fig3, "Mental Health Conditions by Job Role", height=360)

    return fig1, fig2, fig3


# =============================================================================
# 6. Run
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
