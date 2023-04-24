import dash_gif_component as gif
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import gc
from scipy.stats import pearsonr
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import numpy as np
from plotly_calplot import calplot
from datetime import date
import dash_bootstrap_components as dbc
from typing import List, Dict, Tuple, Union

load_figure_template("slate")


country_name_to_iso2 = {
    "Austria": "AT",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Czech_Republic": "CZ",
    "Denmark": "DK",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Montenegro": "ME",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "United_Kingdom": "GB",
}


all_country: List = list(country_name_to_iso2.values())

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])


# df = pd.read_csv('dataset_compressed.csv.gz', compression='gzip', usecols=[
#                  "country", "capacity_factor", "year", "month", "hour", "day"])
df = pd.read_csv("out.csv.gz", compression="gzip")

df_indices = pd.read_csv(
    "daily_indices_82_to_19.csv",
    usecols=[
        "timestamp",
        "nao",
        "ao",
        "mjo80e",
        "mjo40w",
        "mjo20e",
        "mjo160e",
        "mjo10w",
        "nino34",
    ],
)

df_prices = pd.read_csv("prices.csv")


fcols = df.select_dtypes("float").columns
icols = df.select_dtypes("integer").columns

df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")


df_hourly_all_years = (
    df.drop(["month"], axis=1)
    .groupby(["year", "hour", "country"])
    .mean(numeric_only=True)
    .reset_index()
)
df_month_all_years = (
    df.drop(["hour", "day"], axis=1)
    .groupby(["year", "month", "country"])
    .mean(numeric_only=True)
    .reset_index()
)


df_dict = {
    "hour": df_hourly_all_years.groupby(["hour", "country"])
    .mean(numeric_only=True)
    .reset_index(),
    "month": df_month_all_years.groupby(["month", "country"])
    .mean(numeric_only=True)
    .reset_index(),
    "year": df.groupby(["year", "country"]).mean(numeric_only=True).reset_index(),
}


df_daily_cp = (
    df.drop(["month"], axis=1)
    .groupby(["country", "day"])
    .mean(numeric_only=True)
    .reset_index()
)


del df
gc.collect()


daily_cp_eu_entire_period = (
    df_daily_cp.groupby("day").mean(numeric_only=True).reset_index()
)


monthly_cp_eu = (
    df_month_all_years.groupby(["month", "year"]).mean(numeric_only=True).reset_index()
)


df_prices_dict = {
    country_iso: pd.merge(
        left=df_daily_cp[df_daily_cp.country == country_iso].loc[
            :, ["day", "capacity_factor_w"]
        ],
        right=df_prices[df_prices["Country"] == country_iso],
        left_on="day",
        right_on="timestamp",
    )
    for country_iso in country_name_to_iso2.values()
}


def compute_combined_cp(df: pd.DataFrame, wind_share_percent: int) -> pd.DataFrame:
    wind_share = wind_share_percent / 100
    df.loc[:, "capacity_factor"] = (
        wind_share * df["capacity_factor_w"]
        + (1 - wind_share) * df["capacity_factor_s"]
    )
    return df.drop(["capacity_factor_w", "capacity_factor_s"], axis=1)


def compute_df_daily_cp_corr(wind_share_percent: int) -> pd.DataFrame:
    df = compute_combined_cp(df_daily_cp, wind_share_percent)
    return pd.DataFrame(
        np.array(
            [df[df.country == country].capacity_factor for country in all_country]
        ).T,
        columns=all_country,
    )


def compute_df_daily_lp_corr(wind_share_percent: int) -> pd.DataFrame:
    df = compute_combined_cp(df_daily_cp, wind_share_percent)
    return pd.DataFrame(
        np.array(
            [df[df.country == country].capacity_factor < 0.1 for country in all_country]
        ).T,
        columns=all_country,
    )


def num_events(daily_cp: pd.DataFrame, mini_cons_day: int) -> int:
    events = (daily_cp.capacity_factor < 0.1).values
    counter = 0
    seq = 0

    for i, x in enumerate(events):
        if x == 1:
            seq += 1
        if x == 0 or i == len(events):
            if seq >= mini_cons_day:
                counter += 1
            seq = 0

    return counter


# ---------------------------------------------------------------
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(id="slider-output-container", style={"width": "20vw"}),
                html.Div(
                    [
                        dcc.Slider(
                            0,
                            100,
                            10,
                            value=100,
                            marks={
                                value: {"label": "{}/{}".format(value, 100 - value)}
                                for value in range(0, 110, 10)
                            },
                            id="my-slider",
                        )
                    ],
                    style={"width": "40vw"},
                ),
            ],
            style={
                "height": "10vh",
                "display": "flex",
                "justifyContent": "center",
                "flexDirection": "row",
                "alignItems": "center",
            },
        ),
        html.Div(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Tooltip(
                                            "Select the aggregation period and the statistics to display on map",
                                            target="period_stats",
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        {
                                                            "label": "Average",
                                                            "value": "avg",
                                                        },
                                                        {
                                                            "label": "Std of the",
                                                            "value": "std",
                                                        },
                                                    ],
                                                    id="avg_std",
                                                    searchable=False,
                                                    value="avg",
                                                    clearable=False,
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        {
                                                            "label": "yearly capacity factor",
                                                            "value": "year",
                                                        },
                                                        {
                                                            "label": "monthly capacity factor (1979-2019)",
                                                            "value": "month",
                                                        },
                                                        {
                                                            "label": "hourly capacity factor (1979-2019)",
                                                            "value": "hour",
                                                        },
                                                    ],
                                                    id="period_radio",
                                                    searchable=False,
                                                    value="year",
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                    ],
                                    id="period_stats",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Tooltip(
                                            html.Span(
                                                "Hold Shift to select multiple countries",
                                                style={
                                                    "color": "rgb(187,51,59)",
                                                    "font-weight": "bold",
                                                },
                                            ),
                                            target="map_row",
                                        ),
                                        dcc.Graph(id="map", style={"height": "48vh"}),
                                    ],
                                    id="map_row",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Tooltip(
                                            "Select the period to display on the map",
                                            target="range_slider_row",
                                        ),
                                        dcc.RangeSlider(
                                            1979,
                                            2019,
                                            1,
                                            value=[1979, 2019],
                                            id="range_slider",
                                            marks={
                                                i: {"label": str(i)}
                                                for i in range(1979, 2020, 5)
                                            },
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                    id="range_slider_row",
                                ),
                                dbc.Row(
                                    children=[],
                                    id="evolution",
                                    style={"height": "20vh"},
                                ),
                            ]
                        ),
                        style={"height": "90vh", "width": "45vw"},
                    ),
                    style={"display": "flex", "justifyContent": "center"},
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Tooltip(
                                                    "Select a plot to display",
                                                    target="plot_selector_col",
                                                ),
                                                dcc.Dropdown(
                                                    [
                                                        "Min, Max, Avg monthly capacity factor",
                                                        "Min, Max, Avg intra-day hourly capacity factor",
                                                        "Intra-year variation range of the monthly capacity factor",
                                                        "Intra-day variation range of the hourly capacity factor",
                                                        "Cumulative days above thresholds",
                                                        "LP events",
                                                        "Day-ahead prices (Daily average)",
                                                        "Climate indices and LP events",
                                                        "Inter-annual variability of the number of LP days",
                                                        "Spatial correlation between LP day distribution",
                                                        "Spatial correlation between daily capacity factor",
                                                        "Daily capacity factor distribution",
                                                        "YoY (year-over-year) monthly capacity factor comparison",
                                                    ],
                                                    "Intra-year variation range of the monthly capacity factor",
                                                    searchable=False,
                                                    optionHeight=60,
                                                    id="dropdown",
                                                    clearable=False,
                                                ),
                                            ],
                                            id="plot_selector_col",
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in range(1979, 2020)
                                                    ],
                                                    value=2019,
                                                    clearable=False,
                                                    searchable=False,
                                                    id="year_picker",
                                                )
                                            ],
                                            id="year_picker_div",
                                            style={"display": "none"},
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.DatePickerRange(
                                                    end_date=date(2019, 12, 1),
                                                    display_format="D/M/Y",
                                                    start_date_placeholder_text="D/M/Y",
                                                    end_date_placeholder_text="D/M/Y",
                                                    start_date=date(1979, 1, 1),
                                                    id="date_picker",
                                                    style={"height": "36px"},
                                                ),
                                            ],
                                            id="date_picker_div",
                                            style={"display": "none"},
                                            width=7,
                                        ),
                                    ]
                                ),
                                dcc.Loading(
                                    fullscreen=False,
                                    children=[
                                        dbc.Row(children=[], id="side_graph", style={})
                                    ],
                                ),
                            ]
                        ),
                        style={"height": "90vh", "width": "45vw"},
                    ),
                    style={"display": "flex", "justifyContent": "center"},
                ),
            ],
            style={
                "height": "100vh",
                "display": "flex",
                "justifyContent": "center",
                "flexDirection": "row",
            },
        ),  # align items center
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Help and notations")),
                dbc.ModalBody(
                    [
                        html.P(
                            "28C: Denotes the data aggregated over the 28 countries for which data is available, considered as a single region.",
                            style={"textAlign": "left"},
                        ),
                        html.Br(),
                        html.P(
                            "LP events: Short for Low Power events. Consecutive days during which the daily capacity factor is constantly below a threshold of 10 per cent.",
                            style={"textAlign": "left"},
                        ),
                    ]
                ),
            ],
            id="modal-lg",
            size="lg",
            is_open=False,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Get started")),
                dbc.ModalBody(
                    [
                        html.H4("Select your plot on the right card:"),
                        html.P("- Use the dropdown to switch between plots"),
                        html.Div(
                            gif.GifPlayer(
                                gif="assets/rec2.gif",
                                still="assets/rec2.png",
                                autoplay=True,
                            ),
                            style={"width": "50vw", "margin": "auto"},
                        ),
                        html.Br(),
                        html.H4("Filter data:"),
                        html.P(
                            "- Select countries on the left map to filter the data to show on the selected plot"
                        ),
                        html.P("- Use Shift ⇧ to select multiple countries"),
                        html.Div(
                            gif.GifPlayer(
                                gif="assets/rec.gif",
                                still="assets/rec.png",
                                autoplay=True,
                            ),
                            style={"width": "50vw", "margin": "auto"},
                        ),
                    ],
                    style={"textAlign": "left"},
                ),
            ],
            id="modal-lg-intro",
            size="lg",
            is_open=True,
        ),
        dbc.Alert(
            [
                html.I(className="bi bi-info-circle-fill me-2"),
                html.Span(
                    "Help and Notations",
                    id="open-lg",
                    n_clicks=0,
                    style={
                        "textDecoration": "underline",
                        "marginRight": "15px",
                        "cursor": "pointer",
                    },
                ),
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Span(
                    html.A("Data source", href="https://doi.org/10.17864/1947.272"),
                    style={
                        "textDecoration": "underline",
                        "marginRight": "15px",
                        "cursor": "pointer",
                    },
                ),
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Span(
                    html.A(
                        "Data generation (Bloomfield, H.C., Brayshaw, D. and Charlton-Perez, A. 2020)",
                        href="https://doi.org/10.1002/met.1858",
                    ),
                    style={
                        "textDecoration": "underline",
                        "marginRight": "15px",
                        "cursor": "pointer",
                    },
                ),
            ],
            color="dark",
        ),
    ]
)


# ---------------------------------------------------------------


@app.callback(
    Output("slider-output-container", "children"), Input("my-slider", "value")
)
def update_output(wind_share_percent: int) -> str:
    return f"The displayed data is for a mix of {wind_share_percent}% wind and {100 - wind_share_percent}% solar"


@app.callback(
    Output("modal-lg", "is_open"),
    Input("open-lg", "n_clicks"),
    State("modal-lg", "is_open"),
)
def toggle_modal(n1: bool, is_open: bool) -> bool:
    return not is_open if n1 else is_open


@app.callback(
    [
        Output(component_id="range_slider", component_property="min"),
        Output(component_id="range_slider", component_property="max"),
        Output(component_id="range_slider", component_property="marks"),
        Output(component_id="range_slider", component_property="value"),
    ],
    [Input(component_id="period_radio", component_property="value")],
)
def update_slider(
    period_radio: str,
) -> Tuple[int, int, Union[None, Dict[int, str]], List[int]]:
    marks = {}
    min = df_dict[period_radio][period_radio].min()
    max = df_dict[period_radio][period_radio].max()

    if period_radio == "hour":
        marks = {i: f"{str(i)}:00" for i in range(min, max + 1, 2)}

    elif period_radio == "month":
        months = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sept",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        marks = {i: months[i] for i in range(min, max + 1)}
    elif period_radio == "year":
        marks = {i: str(i) for i in range(min, max + 1, 5)}
    return min, max, marks, [min, max]


@app.callback(
    [
        Output(component_id="map", component_property="figure"),
        Output(component_id="map", component_property="selectedData"),
    ],
    [
        Input(component_id="period_radio", component_property="value"),
        Input(component_id="avg_std", component_property="value"),
        Input(component_id="range_slider", component_property="value"),
        Input(component_id="slider-output-container", component_property="children"),
    ],
    [State("my-slider", "value")],
)
def update_map(period_radio, avg_std, range, _, wind_share_percent):
    if period_radio is None:
        raise PreventUpdate

    df = df_dict[period_radio]
    mask = (df[period_radio] >= range[0]) & (df[period_radio] <= range[1])
    df = compute_combined_cp(df[mask], wind_share_percent)
    df = (
        df.groupby(["country"])["capacity_factor"].mean()
        if avg_std == "avg"
        else df.groupby(["country"])["capacity_factor"].std()
    )
    fig = px.choropleth_mapbox(
        df,
        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
        featureidkey="properties.ISO2",
        locations=df.index,
        color=df.values,
        zoom=1,
        center={"lat": 56.4, "lon": 15.0},
        mapbox_style="carto-positron",
        color_continuous_scale="Viridis",
        opacity=0.5,
        title="",
    )

    fig.update_layout(
        font=dict(
            size=10,
        ),
        xaxis=go.layout.XAxis(tickangle=45),
        coloraxis_colorbar=dict(
            title="Capacity <br>factor",
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_colorbar_x=-0.15,
        clickmode="event+select",
    )

    return fig, None


@app.callback(
    Output(component_id="evolution", component_property="children"),
    [
        Input(component_id="period_radio", component_property="value"),
        Input(component_id="range_slider", component_property="value"),
        Input("map", "selectedData"),
        Input(component_id="slider-output-container", component_property="children"),
    ],
    [State("my-slider", "value")],
)
def update_plot(
    period_radio: str, range: Tuple[int, int], click: Dict, _, wind_share_percent: int
):
    if range[0] == range[1]:
        return

    df_select = df_dict[period_radio]
    mask = (df_select[period_radio] >= range[0]) & (df_select[period_radio] <= range[1])
    df_select = compute_combined_cp(df_select[mask], wind_share_percent)
    df_all = df_select.groupby([period_radio]).mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig.add_trace(
        go.Scatter(
            x=df_all[period_radio],
            y=df_all["capacity_factor"],
            name="28C",
            line_shape="linear",
        )
    )
    if click is not None:
        selected_countries = [point["location"] for point in click["points"]]
        df = df_select.groupby([period_radio, "country"]).mean().reset_index()
        for selected_country in selected_countries:
            df_country = df[df.country == selected_country]
            fig.add_trace(
                go.Scatter(
                    x=df_country[period_radio],
                    y=df_country["capacity_factor"],
                    name=selected_country,
                    line_shape="linear",
                )
            )

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        margin=dict(l=5, r=10, t=10, b=5),
        yaxis_title="Capacity factor",
        xaxis_title=period_radio,
        font=dict(size=10),
        title=dict(x=0.5, y=1, font=dict()),
    )

    return dcc.Graph(figure=fig)


@app.callback(
    [
        Output("side_graph", "children"),
        Output("date_picker_div", "style"),
        Output("year_picker_div", "style"),
    ],
    [
        Input("map", "selectedData"),
        Input("dropdown", "value"),
        Input("year_picker", "value"),
        Input(component_id="slider-output-container", component_property="children"),
    ],
    [State("my-slider", "value")],
)
def update_side_graph(
    click: Dict, fig_choice: str, year: int, _, wind_share_percent: int
):
    if fig_choice is None:
        raise PreventUpdate

    if fig_choice == "Inter-annual variability of the number of LP days":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        lp = compute_combined_cp(daily_cp_eu_entire_period, wind_share_percent)[
            ["year", "capacity_factor"]
        ]
        lp["count"] = lp.capacity_factor < 0.1
        lp = lp[["count", "year"]].groupby("year").sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=lp["year"],
                y=lp["count"],
                name="28C",
                line_shape="linear",
                showlegend=True,
                marker=dict(color="rgb(187,51,59)"),
            )
        )

        if click:
            selected_countries = [point["location"] for point in click["points"]]

            lp = compute_combined_cp(
                df_daily_cp,
                wind_share_percent,
            )[["year", "capacity_factor", "country"]]
            lp["count"] = lp.capacity_factor < 0.1
            lp = lp[["count", "year"]].groupby(["year", "country"]).sum().reset_index()

            for selected_country in selected_countries:
                lp_country = lp[lp.country == selected_country]
                fig.add_trace(
                    go.Scatter(
                        x=lp_country["year"],
                        y=lp_country["count"],
                        name=selected_country,
                        showlegend=True,
                    )
                )

            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Year",
                yaxis_title="Number of low power days",
            )
        fig.update_yaxes(rangemode="tozero")

        return (
            dcc.Graph(figure=fig, style={"height": "80vh"}),
            {"display": "none"},
            {"display": "none"},
        )

    if fig_choice == "Day-ahead prices (Daily average)":
        fig = go.Figure()

        if click is None:
            df_prices_europe = df_prices.groupby(["timestamp"]).mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=df_prices_europe.timestamp,
                    y=df_prices_europe.price,
                    name="28C",
                    line={"color": "rgb(187,51,59)", "width": 1},
                    showlegend=True,
                    visible=True,
                )
            )

        else:
            fig.add_trace(go.Scatter(x=[], y=[]))
            selected_countries = [point["location"] for point in click["points"]]

            for country_iso in selected_countries:
                if len(df_prices_dict[country_iso]) == 0:
                    return (
                        [
                            html.Div(
                                "No data for {}".format(country_iso),
                                style={
                                    "marginBottom": "10vh",
                                    "marginTop": "10vh",
                                    "textAlign": "center",
                                },
                            )
                        ],
                        {"display": "none"},
                        {"display": "none"},
                    )

            df_concat = pd.concat(
                [df_prices_dict[country_iso] for country_iso in selected_countries]
            )
            by_row_index = df_concat.groupby([df_concat.timestamp])
            df_means = by_row_index.mean().reset_index()
            name = (
                click["points"][0]["location"]
                if len(click["points"]) == 1
                else "selected region "
            )
            color = "rgb(6, 168, 67)" if len(click["points"]) == 1 else "grey"
            corr = df_means.price.corr(df_means.capacity_factor_w)
            fig.add_trace(
                go.Scatter(
                    x=df_means.timestamp,
                    y=df_means.price,
                    name=name,
                    visible=True,
                    line_shape="linear",
                    showlegend=True,
                    line={"color": color, "width": 0.5},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_means.timestamp,
                    y=df_means.price.where(df_means.capacity_factor_w < 0.1),
                    name="Prices during low <b>wind</b> power days.<br>Correlation between <b>wind</b> capacity factor and electricity price ({}-{}) is {:.2f}".format(
                        df_means.timestamp.iloc[0][:4],
                        df_means.timestamp.iloc[-1][:4],
                        corr,
                    ),
                    mode="markers",
                    marker=dict(color="red", size=3),
                    showlegend=True,
                    visible="legendonly",
                )
            )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Price (EUR/MWh)",
            xaxis_title="Date",
            legend=dict(orientation="h"),
        )

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
            )
        )

        return (
            dcc.Graph(figure=fig, style={"height": "80vh"}),
            {"display": "none"},
            {"display": "none"},
        )

    if fig_choice == "YoY (year-over-year) monthly capacity factor comparison":
        if click:
            df_select = (
                compute_combined_cp(df_month_all_years, wind_share_percent)[
                    df_month_all_years.country.isin(
                        [point["location"] for point in click["points"]]
                    )
                ]
                .groupby(["month", "year"])
                .mean()
                .reset_index()
            )
            name = (
                click["points"][0]["location"] + "_"
                if len(click["points"]) == 1
                else "selected region "
            )
            color = "green" if len(click["points"]) == 1 else "grey"
        else:
            df_select = compute_combined_cp(monthly_cp_eu, wind_share_percent)
            name = "28C-"
            color = "rgb(187,51,59)"

        fig = go.Figure()
        for year_ in monthly_cp_eu.year.unique():
            if year_ != year:
                fig.add_trace(
                    go.Scatter(
                        x=df_select[df_select.year == year_].month,
                        y=df_select[df_select.year == year_]["capacity_factor"],
                        name=name + str(year_),
                        line_shape="linear",
                        line=dict(color="gray"),
                        opacity=0.2,
                        showlegend=False,
                    )
                )
            if year_ == year:
                fig.add_trace(
                    go.Scatter(
                        x=df_select[df_select.year == year_].month,
                        y=df_select[df_select.year == year_]["capacity_factor"],
                        name=name + str(year_),
                        line_shape="linear",
                        showlegend=True,
                        line=dict(color=color),
                    )
                )

        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="month",
        )

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Min, Max, Avg intra-day hourly capacity factor":
        df_hourly = compute_combined_cp(
            df_hourly_all_years.drop(["year"], axis=1)[
                df_hourly_all_years.year == year
            ],
            wind_share_percent,
        )

        eu_min_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().min()
        eu_max_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().max()
        eu_mean_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().mean()
        variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp

        min_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].min()
        max_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].max()
        mean_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].mean()
        variation_range_hourly = (max_hourly_cp - min_hourly_cp).sort_values()

        sort_hourly_mean = np.argsort(mean_hourly_cp)
        min_hourly_cp = min_hourly_cp[sort_hourly_mean]
        max_hourly_cp = max_hourly_cp[sort_hourly_mean]
        mean_hourly_cp = mean_hourly_cp[sort_hourly_mean]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(
            go.Scatter(
                x=["28C"],
                y=[eu_mean_hourly_cp],
                mode="markers",
                showlegend=False,
                name="28C",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[eu_mean_hourly_cp - eu_min_hourly_cp],
                    arrayminus=[eu_max_hourly_cp - eu_mean_hourly_cp],
                ),
            )
        )

        if click is None:
            selected_countries = all_country
            mask = np.isin(min_hourly_cp.index, selected_countries)

            x = min_hourly_cp[mask].index
            y = mean_hourly_cp[mask].values
            error = max_hourly_cp[mask].values - y
            error_min = y - min_hourly_cp[mask].values

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    showlegend=False,
                    hoverlabel=dict(namelength=0),
                    line=dict(color="gray"),
                    error_y=dict(
                        type="data", symmetric=False, array=error, arrayminus=error_min
                    ),
                )
            )

        else:
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries:
                mask = np.isin(min_hourly_cp.index, selected_country)
                x = min_hourly_cp[mask].index
                y = mean_hourly_cp[mask].values
                error = max_hourly_cp[mask].values - y
                error_min = y - min_hourly_cp[mask].values

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        showlegend=False,
                        hoverlabel=dict(namelength=0),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=error,
                            arrayminus=error_min,
                        ),
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="Countries",
        )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Intra-year variation range of the monthly capacity factor":
        df_month = compute_combined_cp(
            df_month_all_years.drop(["year"], axis=1)[df_month_all_years.year == year],
            wind_share_percent,
        )

        eu_min_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).min()
        )
        eu_max_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).max()
        )
        eu_mean_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).mean()
        )
        variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp

        min_monthly_cp = df_month.groupby(["country"])["capacity_factor"].min()
        max_monthly_cp = df_month.groupby(["country"])["capacity_factor"].max()
        mean_monthly_cp = df_month.groupby(["country"])["capacity_factor"].mean(
            numeric_only=True
        )
        variation_range = (max_monthly_cp - min_monthly_cp).sort_values()

        sort_monthly_mean = np.argsort(mean_monthly_cp)
        min_monthly_cp = min_monthly_cp[sort_monthly_mean]
        max_monthly_cp = max_monthly_cp[sort_monthly_mean]
        mean_monthly_cp = mean_monthly_cp[sort_monthly_mean]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(
            go.Bar(
                x=["28C"],
                y=[variation_range_eu],
                name="28C",
                showlegend=False,
            )
        )
        if click is None:
            selected_countries = all_country
            filtered_df = variation_range[
                np.isin(variation_range.index, selected_countries)
            ]
            x = filtered_df.index
            y = filtered_df.values

            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    hoverlabel=dict(namelength=0),
                    showlegend=False,
                    marker_color="gray",
                )
            )

        else:
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries:
                filtered_df = variation_range[
                    np.isin(variation_range.index, selected_country)
                ]
                x = filtered_df.index
                y = filtered_df.values

                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        hoverlabel=dict(namelength=0),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Variation range",
            xaxis_title="Countries",
        )
        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Intra-day variation range of the hourly capacity factor":
        start_time = time.time()

        df_hourly = compute_combined_cp(
            df_hourly_all_years.drop(["year"], axis=1)[
                df_hourly_all_years.year == year
            ],
            wind_share_percent,
        )

        print("--- %s seconds get df  ---" % (time.time() - start_time))
        start_time = time.time()

        eu_min_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().min()
        eu_max_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().max()
        eu_mean_hourly_cp = df_hourly.groupby("hour")["capacity_factor"].mean().mean()
        variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp

        min_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].min()
        max_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].max()
        mean_hourly_cp = df_hourly.groupby(["country"])["capacity_factor"].mean()
        variation_range_hourly = (max_hourly_cp - min_hourly_cp).sort_values()

        sort_hourly_mean = np.argsort(mean_hourly_cp)
        min_hourly_cp = min_hourly_cp[sort_hourly_mean]
        max_hourly_cp = max_hourly_cp[sort_hourly_mean]
        mean_hourly_cp = mean_hourly_cp[sort_hourly_mean]

        print("--- %s seconds do operations ---" % (time.time() - start_time))
        start_time = time.time()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(
            go.Bar(
                x=["28C"],
                y=[variation_range_eu_hourly],
                name="28C",
                showlegend=False,
            )
        )
        if click is None:
            selected_countries = all_country
            filtered_df = variation_range_hourly[
                np.isin(variation_range_hourly.index, selected_countries)
            ]
            x = filtered_df.index
            y = filtered_df.values

            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    hoverlabel=dict(namelength=0),
                    showlegend=False,
                    marker_color="gray",
                )
            )

        else:
            start_time = time.time()
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries:
                filtered_df = variation_range_hourly[
                    np.isin(variation_range_hourly.index, selected_country)
                ]
                x = filtered_df.index
                y = filtered_df.values
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        hoverlabel=dict(namelength=0),
                        showlegend=False,
                    )
                )

            print("--- %s seconds plot stuff ---" % (time.time() - start_time))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Variation range",
            xaxis_title="Countries",
        )

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Min, Max, Avg monthly capacity factor":
        df_month = compute_combined_cp(
            df_month_all_years.drop(["year"], axis=1)[df_month_all_years.year == year],
            wind_share_percent,
        )

        eu_min_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).min()
        )
        eu_max_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).max()
        )
        eu_mean_monthly_cp = (
            df_month.groupby("month")["capacity_factor"].mean(numeric_only=True).mean()
        )
        variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp

        min_monthly_cp = df_month.groupby(["country"])["capacity_factor"].min()
        max_monthly_cp = df_month.groupby(["country"])["capacity_factor"].max()
        mean_monthly_cp = df_month.groupby(["country"])["capacity_factor"].mean(
            numeric_only=True
        )
        variation_range = (max_monthly_cp - min_monthly_cp).sort_values()

        sort_monthly_mean = np.argsort(mean_monthly_cp)
        min_monthly_cp = min_monthly_cp[sort_monthly_mean]
        max_monthly_cp = max_monthly_cp[sort_monthly_mean]
        mean_monthly_cp = mean_monthly_cp[sort_monthly_mean]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(
            go.Scatter(
                x=["28C"],
                y=[eu_mean_monthly_cp],
                mode="markers",
                showlegend=False,
                name="28C",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[eu_mean_monthly_cp - eu_min_monthly_cp],
                    arrayminus=[eu_max_monthly_cp - eu_mean_monthly_cp],
                ),
            )
        )

        if click is None:
            selected_countries = all_country
            mask = np.isin(min_monthly_cp.index, selected_countries)

            x = min_monthly_cp[mask].index
            y = mean_monthly_cp[mask].values
            error = max_monthly_cp[mask].values - y
            error_min = y - min_monthly_cp[mask].values

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    showlegend=False,
                    hoverlabel=dict(namelength=0),
                    line=dict(color="gray"),
                    error_y=dict(
                        type="data", symmetric=False, array=error, arrayminus=error_min
                    ),
                )
            )

        else:
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries:
                mask = np.isin(min_monthly_cp.index, selected_country)
                x = min_monthly_cp[mask].index
                y = mean_monthly_cp[mask].values
                error = max_monthly_cp[mask].values - y
                error_min = y - min_monthly_cp[mask].values

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        showlegend=False,
                        hoverlabel=dict(namelength=0),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=error,
                            arrayminus=error_min,
                        ),
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="Countries",
        )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Cumulative days above thresholds":
        df = df_daily_cp[df_daily_cp.year == year]
        df = compute_combined_cp(df, wind_share_percent)
        daily_cp_eu = df.groupby("day").mean(numeric_only=True)
        cum_days_eu = [
            (daily_cp_eu.capacity_factor > i).mean() for i in np.linspace(0, 1, 100)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(
            go.Scatter(
                x=np.linspace(0, 1, 100),
                y=cum_days_eu,
                line_shape="linear",
                line=dict(color="rgb(187,51,59)"),
                name="28C",
            )
        )

        if click is not None:
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries:
                df = df[np.isin(df.country, selected_countries)]
                df_country = df[df.country == selected_country]["capacity_factor"]
                fig.add_trace(
                    go.Scatter(
                        x=np.linspace(0, 1, 100),
                        y=[(df_country > i).mean() for i in np.linspace(0, 1, 100)],
                        line_shape="linear",
                        name=selected_country,
                    )
                )

            if len(selected_countries) > 1:
                daily_cp_region = df.groupby(["day"])["capacity_factor"].mean(
                    numeric_only=True
                )
                cum_days_region = [
                    (daily_cp_region > i).mean() for i in np.linspace(0, 1, 100)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=np.linspace(0, 1, 100),
                        y=cum_days_region,
                        line_shape="linear",
                        name="selected region",
                    )
                )

        else:
            for selected_country in all_country:
                daily_cp_country = df[df.country == selected_country]["capacity_factor"]
                fig.add_trace(
                    go.Scatter(
                        x=np.linspace(0, 1, 100),
                        y=[
                            (daily_cp_country > i).mean()
                            for i in np.linspace(0, 1, 100)
                        ],
                        line_shape="linear",
                        name=selected_country,
                        line=dict(color="gray"),
                        opacity=0.2,
                        showlegend=False,
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Capacity factor threshold",
            yaxis_title="Proportion of days above threshold",
        )

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "Spatial correlation between daily capacity factor":
        if not click or len(click["points"]) != 1:
            return (
                html.Div(
                    "Select only one country on the left map to display the correlation between the LP day distribution of this country and the LP day distribution of the other European countries",
                    style={"marginTop": "10vh", "textAlign": "center"},
                ),
                {"display": "none"},
                {"display": "none"},
            )

        selected_countries = click["points"][0]["location"]
        daily_cp_df = compute_df_daily_cp_corr(wind_share_percent)
        rho = abs(daily_cp_df.corr()[selected_countries])
        pval = (
            daily_cp_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)  # type: ignore
        )[selected_countries]
        rho[pval > 0.05] = np.NaN
        fig = px.choropleth_mapbox(
            rho,
            geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
            featureidkey="properties.ISO2",
            locations=rho.index,
            color=rho,
            zoom=1,
            center={"lat": 56.4, "lon": 15.0},
            range_color=[0, 1.0],
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            opacity=0.5,
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Absolute<br>correlation<br>(1979-2019)",
            )
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )

        fig.add_annotation(
            text="only statistically significant correlations (p-value > 0.05) are displayed",
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=10, color="black"),
        )

        return (
            dcc.Graph(figure=fig, style={"height": "80vh"}),
            {"display": "none"},
            {"display": "none"},
        )

    if fig_choice == "Spatial correlation between LP day distribution":
        if not click or len(click["points"]) != 1:
            return (
                html.Div(
                    "Select one country on the left map to display the correlation between the LP day distribution of this country and the LP day distribution of the other European countries",
                    style={"marginTop": "10vh", "textAlign": "center"},
                ),
                {"display": "none"},
                {"display": "none"},
            )
        selected_countries = click["points"][0]["location"]
        daily_LP_df = compute_df_daily_lp_corr(wind_share_percent)
        rho = abs(daily_LP_df.corr()[selected_countries])
        pval = (
            daily_LP_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)  # type: ignore
        )[selected_countries]
        rho[pval > 0.05] = np.NaN
        fig = px.choropleth_mapbox(
            rho,
            geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
            featureidkey="properties.ISO2",
            locations=rho.index,
            color=rho,
            zoom=1,
            center={"lat": 56.4, "lon": 15.0},
            range_color=[0, 0.6],
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            opacity=0.5,
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Absolute<br>correlation<br>(1979-2019)",
            )
        )

        fig.add_annotation(
            text="only statistically significant correlations (p-value > 0.05) are displayed",
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=10, color="black"),
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )

        return (
            dcc.Graph(figure=fig, style={"height": "80vh"}),
            {"display": "none"},
            {"display": "none"},
        )

    if fig_choice == "Climate indices and LP events":
        if not click or len(click["points"]) < 1:
            return (
                [
                    html.Div(
                        "Select a region on the left map to display the number of low power (LP) events during the selected year",
                        style={
                            "marginBottom": "10vh",
                            "marginTop": "10vh",
                            "textAlign": "center",
                        },
                    ),
                    html.Br(),
                    gif.GifPlayer(
                        gif="assets/rec.gif",
                        still="assets/rec.png",
                        autoplay=True,
                    ),
                ],
                {"display": "none"},
                {"display": "none"},
            )

        if year < 1982:
            return (
                [
                    html.Div(
                        "No climate data for this year. Please choose a year between 1982 and 2019",
                        style={
                            "marginBottom": "10vh",
                            "marginTop": "10vh",
                            "textAlign": "center",
                        },
                    ),
                ],
                {"display": "none"},
                {},
            )

        selected_countries = [point["location"] for point in click["points"]]

        indices = [
            "nao",
            "ao",
            "mjo80e",
            "mjo40w",
            "mjo20e",
            "mjo160e",
            "mjo10w",
            "nino34",
        ]
        mask = dict()
        for selected_country in selected_countries:
            mask[selected_country] = (
                df_daily_cp[df_daily_cp.country == selected_country].year >= year
            ) & (df_daily_cp[df_daily_cp.country == selected_country].year <= year)

        dfs = [
            compute_combined_cp(
                df_daily_cp[df_daily_cp.country == selected_country][
                    mask[selected_country]
                ].reset_index(),
                wind_share_percent,
            )
            for selected_country in mask.keys()
        ]
        df_concat = pd.concat(dfs)
        by_row_index = df_concat.groupby(df_concat.day)
        df_means = by_row_index.mean().reset_index()

        merge = pd.merge(
            left=df_means, right=df_indices, right_on="timestamp", left_on="day"
        )

        fig = go.Figure()

        for index in indices:
            fig.add_trace(
                go.Scatter(
                    x=merge.timestamp,
                    y=merge[index],
                    line={"color": "grey", "width": 1},
                    name=index,
                    legendgroup=index,
                    showlegend=False,
                    visible=(False if index != "nao" else True),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=merge.timestamp,
                    y=merge[index].where(merge.capacity_factor < 0.1),
                    mode="markers",
                    name=index + " during LP days",
                    marker=dict(color="red", size=5),
                    showlegend=False,
                    legendgroup=index,
                    visible=(False if index != "nao" else True),
                )
            )

        fig.update_layout(
            updatemenus=[
                {
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.2,
                    "yanchor": "top",
                    "type": "buttons",
                    "direction": "left",
                    "buttons": [
                        {
                            "label": c,
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        c == c2 for c2 in indices for _ in range(2)
                                    ]
                                }
                            ],
                        }
                        for c in indices
                    ],
                }
            ]
        )

        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(
            margin=dict(t=100),
            yaxis_title="Selected climate index",
            xaxis_title="date",
        )
        fig.add_annotation(
            text="climate index values during lp days are indicated in red",
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=10, color="red"),
        )

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}

    if fig_choice == "LP events":
        if not click or len(click["points"]) < 1:
            return (
                [
                    html.Div(
                        "Select a region on the left map to display the number of low power (LP) events during the selected year",
                        style={
                            "marginBottom": "10vh",
                            "marginTop": "10vh",
                            "textAlign": "center",
                        },
                    ),
                    html.Br(),
                    gif.GifPlayer(
                        gif="assets/rec.gif",
                        still="assets/rec.png",
                        autoplay=True,
                    ),
                ],
                {"display": "none"},
                {"display": "none"},
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(go.Bar(x=[], y=[]))

        # fig.add_trace(
        #     go.Bar(x=x, y=y, marker_color="rgb(187,51,59)", name="28C", showlegend=True)
        # )

        if click is not None:
            selected_countries = [point["location"] for point in click["points"]]
            mask = (
                (df_daily_cp.year >= year)
                & (df_daily_cp.year <= year)
                & (np.isin(df_daily_cp.country, selected_countries))
            )
            df = compute_combined_cp(
                df_daily_cp[mask],
                wind_share_percent,
            )

            for selected_country in selected_countries:
                fig.add_trace(
                    go.Bar(
                        name=selected_country,
                        x=list(range(1, 9)),
                        y=[
                            num_events(
                                df[df.country == selected_country],
                                i,
                            )
                            for i in range(1, 9)
                        ],
                    )
                )

            by_row_index = df.groupby(df.day)
            df_means = by_row_index.mean().reset_index()
            dummy_df = pd.DataFrame(
                {
                    "ds": pd.date_range(
                        df_means.day.values[0], df_means.day.values[-1]
                    ),
                    "value": (df_means.capacity_factor < 0.1).values.astype(int),
                }
            )

            fig2 = calplot(
                dummy_df,
                x="ds",
                y="value",
                colorscale=[[0, "rgb(4,204,148)"], [1, "rgb(227,26,28)"]],  # type: ignore
            )
            fig2.update_xaxes(fixedrange=True)
            fig2.update_yaxes(fixedrange=True)

            fig2.update_layout(
                title={
                    "text": "Low power days in the selected area in {}".format(year),
                    "y": 0.09,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )

            if len(click["points"]) > 1:
                fig.add_trace(
                    go.Bar(
                        x=list(range(1, 9)),
                        y=[num_events(df_means, i) for i in range(1, 9)],
                        name="mean capacity<br>factor of selected<br>countries",
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Number of occurences",
            xaxis_title="Minimum duration of the low power event (days)",
        )

        return (
            [
                dcc.Graph(figure=fig, style={"height": "50vh"}),
                (dcc.Graph(figure=fig2, style={"marginTop": "1vh"}) if fig2 else None),
            ],
            {"display": "none"},
            {},
        )

    if fig_choice == "Daily capacity factor distribution":
        daily_cp_eu = compute_combined_cp(
            df_daily_cp[df_daily_cp.year == year].groupby("day").mean().reset_index(),
            wind_share_percent,
        )

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=[], y=[]))
        fig.add_trace(
            go.Histogram(
                xbins=dict(size=0.05),
                name="28C",
                showlegend=True,
                marker=dict(color="rgb(187,51,59)"),
                x=daily_cp_eu.capacity_factor,
                histnorm="percent",
            )
        )

        if click is not None:
            selected_countries = [point["location"] for point in click["points"]]
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = compute_combined_cp(
                df_daily_cp[df_daily_cp.year == year], wind_share_percent
            )
            for country in selected_countries:
                daily_cp_dict[country] = df_daily_cp_year[
                    df_daily_cp_year.country == country
                ]

            for selected_country in selected_countries:
                fig.add_trace(
                    go.Histogram(
                        x=daily_cp_dict[selected_country].capacity_factor,
                        histnorm="percent",
                        name=selected_country,
                        xbins=dict(size=0.05),
                    )
                )

            if len(selected_countries) > 1:
                fig.add_trace(
                    go.Histogram(
                        x=pd.Series(
                            np.mean(
                                np.stack(
                                    [
                                        daily_cp_dict[
                                            selected_country
                                        ].capacity_factor.tolist()
                                        for selected_country in selected_countries
                                    ]
                                ),
                                axis=0,
                            )
                        ),
                        histnorm="percent",
                        name="selected region",
                        xbins=dict(size=0.05),
                    )
                )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Capacity factor",
            yaxis_title="Percentage of days",
        )

        return dcc.Graph(figure=fig, style={"height": "80vh"}), {"display": "none"}, {}


if __name__ == "__main__":

    def sizeof_fmt(num, suffix="B"):
        """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, "Yi", suffix)

    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in locals().items()),
        key=lambda x: -x[1],
    )[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    app.run_server(port=8088, debug=True, use_reloader=True)
