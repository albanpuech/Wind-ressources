
import dash_gif_component as gif
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

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
load_figure_template("slate")


country_name_to_iso2 = {
    'Austria': 'AT',
    'Belgium': 'BE',
    'Bulgaria': 'BG',
    'Croatia': 'HR',
    'Czech_Republic': 'CZ',
    'Denmark': 'DK',
    'Finland': 'FI',
    'France': 'FR',
    'Germany': 'DE',
    'Greece': 'GR',
    'Hungary': 'HU',
    'Ireland': 'IE',
    'Italy': 'IT',
    'Latvia': 'LV',
    'Lithuania': 'LT',
    'Luxembourg': 'LU',
    'Montenegro': 'ME',
    'Netherlands': 'NL',
    'Norway': 'NO',
    'Poland': 'PL',
    'Portugal': 'PT',
    'Romania': 'RO',
    'Slovakia': 'SK',
    'Slovenia': 'SI',
    'Spain': 'ES',
    'Sweden': 'SE',
    'Switzerland': 'CH',
    'United_Kingdom': 'GB'
}


all_country = list(country_name_to_iso2.values())

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])


df = pd.read_csv('dataset_compressed.csv.gz', compression='gzip', usecols=[
                 "country", "capacity_factor", "year", "month", "hour", "day"])


df_hourly_all_years = df.drop(["month"], axis=1).groupby(
    ['year', 'hour', 'country']).mean(numeric_only=True).reset_index()
df_month_all_years = df.drop(["hour", "day"], axis=1).groupby(
    ['year', 'month', 'country']).mean(numeric_only=True).reset_index()

df_dict = {
    "hour": df_hourly_all_years.groupby(['hour', 'country']).mean(numeric_only=True).reset_index(),
    "month": df_month_all_years.groupby(['month', 'country']).mean(numeric_only=True).reset_index(),
    "year": df.groupby(['year', 'country']).mean(numeric_only=True).reset_index(),
}


fcols = df.select_dtypes('float').columns
icols = df.select_dtypes('integer').columns

df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')


print(df.memory_usage())


df_daily_cp = df.drop(["month"], axis=1).groupby(
    ["country", "day"]).mean(numeric_only=True).reset_index()
cum_days_dict_entire_period = dict()
daily_cp_dict_entire_period = dict()

for country in country_name_to_iso2.values():
    daily_cp_dict_entire_period[country] = df_daily_cp[df_daily_cp.country == country]
    cum_days_dict_entire_period[country] = [
        (daily_cp_dict_entire_period[country]['capacity_factor'] > i).mean() for i in np.linspace(0, 1, 100)]


daily_cp_eu_entire_period = df_daily_cp.groupby(
    "day").mean(numeric_only=True).reset_index()
cum_days_eu_entire_period = [
    (daily_cp_eu_entire_period.capacity_factor > i).mean() for i in np.linspace(0, 1, 100)]


monthly_cp_eu = df_month_all_years.groupby(
    ['month', 'year']).mean(numeric_only=True).reset_index()


daily_cp_df = pd.DataFrame(np.array(
    [daily_cp_dict_entire_period[country].capacity_factor for country in all_country]).T, columns=all_country)
daily_LWP_df = pd.DataFrame(np.array(
    [(daily_cp_dict_entire_period[country].capacity_factor) < 0.1 for country in all_country]).T, columns=all_country)


def num_events(daily_cp, mini_cons_day):
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
app.layout = html.Div([html.Div([
    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                        dbc.Tooltip("Select the aggregation period and the statistics to display on map",
                                target="period_stats",
                                    ),
                        dbc.Col([
                            dcc.Dropdown(options=[{'label': 'Average', 'value': 'avg'},
                                                  {'label': 'Std of the', 'value': 'std'}],
                                         id="avg_std", searchable=False, value="avg", clearable=False),
                        ], width=4),
                        dbc.Col([
                            dcc.Dropdown(options=[
                                {'label': 'yearly capacity factor', 'value': 'year'},
                                {'label': 'monthly capacity factor (1979-2019)',
                                    'value': 'month'},
                                {'label': 'hourly capacity factor (1979-2019)' , 'value': 'hour'}],
                                id="period_radio", searchable=False, value="year", clearable=False),
                        ]),

                        ], id="period_stats"),
                dbc.Row([
                    dbc.Tooltip(html.Span("Hold Shift to select multiple countries", style={
                                "color": "rgb(187,51,59)", "font-weight": "bold"}), target="map_row"),
                    dcc.Graph(id='map', style={'height': '48vh'})
                ], id="map_row"),



                dbc.Row([
                    dbc.Tooltip(
                        "Select the period to display on the map", target="range_slider_row"),
                    dcc.RangeSlider(1979, 2019, 1, value=[1979, 2019], id='range_slider', marks={
                        i: {"label": str(i)} for i in range(1979, 2020, 5)}, tooltip={"placement": "top", "always_visible": True}),
                ],
                    id="range_slider_row"
                ),

                dbc.Row(children=[], id='evolution', style={'height': '20vh'}
                        )
            ]), style={"height": "90vh", "width": "45vw"}
        ), style={"display": "flex", "justifyContent": "center"}),

    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Tooltip("Select a plot to display",
                                    target="plot_selector_col"),
                        dcc.Dropdown(['Min, Max, Avg monthly capacity factor',
                                      'Min, Max, Avg intra-day hourly capacity factor',
                                      'Intra-year variation range of the monthly capacity factor',
                                      'Intra-day variation range of the hourly capacity factor',
                                      "Cumulative days above thresholds",
                                      'LWP events',
                                      'Spatial correlation between LWP day distribution',
                                      'Spatial correlation between daily capacity factor',
                                      'Daily capacity factor distribution',
                                      'YoY (year-over-year) monthly capacity factor comparison'],
                                     'Intra-year variation range of the monthly capacity factor', searchable=False, optionHeight=60, id='dropdown', clearable=False),

                    ], id="plot_selector_col"),

                    dbc.Col([
                        dcc.Dropdown(options=[{'label': x, 'value': x} for x in range(
                            1979, 2019)], value=1979, clearable=False, searchable=False, id="year_picker")], id="year_picker_div", style={"display": "none"}),


                    dbc.Col([
                            dcc.DatePickerRange(
                                end_date=date(2019, 12, 1),
                                display_format='D/M/Y',
                                start_date_placeholder_text='D/M/Y',
                                end_date_placeholder_text='D/M/Y',
                                start_date=date(1979, 1, 1),
                                id="date_picker",
                                style={"height": "36px"}
                            ),
                            ], id="date_picker_div", style={"display": "none"}, width=7)

                ]),



                dcc.Loading(fullscreen=False, children=[
                            dbc.Row(children=[], id='side_graph', style={})]),






            ]), style={"height": "90vh", "width": "45vw"}), style={"display": "flex", "justifyContent": "center"})

], style={"height": "100vh", "display": "flex", "justifyContent": "center", "alignItems": "center", "flexDirection": "row"}), 

dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Help and notations")),
                dbc.ModalBody([
                    html.P("28C: Denotes the data aggregated over the 28 countries for which data is available, considered as a single region.",style={"textAlign": "left"}),
                    html.Br(),
                    html.P("LWP events: Short for Low Wind Power events. Consecutive days during which the daily capacity factor is constantly below a threshold of 10 per cent.",style={"textAlign": "left"})]),
            ],
            id="modal-lg",
            size="lg",
            is_open=False,
        ),


dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Get started")),
                dbc.ModalBody([
                html.H4("Select your plot on the right card:"),
                html.P("- Use the dropdown to switch between plots"),
                html.Div(
                gif.GifPlayer(
                                gif='assets/rec2.gif',
                                still='assets/rec2.png',
                                autoplay=True
                ) ,style={"width":"50vw","margin":"auto"}),
                html.Br(),
                html.H4("Filter data:"),
                html.P("- Select countries on the left map to filter the data to show on the selected plot"),
                html.P("- Use Shift ⇧ to select multiple countries"),
                html.Div(
                gif.GifPlayer(
                                gif='assets/rec.gif',
                                still='assets/rec.png',
                                autoplay=True
                ) ,style={"width":"50vw","margin":"auto"})
                            
            ], style={"textAlign": "left"})],
            id="modal-lg-intro",
            size="lg",
            is_open=True,
),

dbc.Alert(
    [   html.I(className="bi bi-info-circle-fill me-2"),
        html.Span("Help and Notations", id="open-lg", n_clicks=0, style={"textDecoration": "underline","marginRight": "15px","cursor":"pointer"}),
        html.I(className="bi bi-check-circle-fill me-2"),
        html.A('Data source: Bloomfield, Hannah, Brayshaw, David and Charlton-Perez, Andrew (2020)',
               href="https://doi.org/10.17864/1947.272"),
    ],
    color="dark",

),

])


# ---------------------------------------------------------------




@app.callback(
    Output("modal-lg", "is_open"),
    Input("open-lg", "n_clicks"),
    State("modal-lg", "is_open"))
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    [Output(component_id='range_slider', component_property='min'),
     Output(component_id='range_slider', component_property='max'),
     Output(component_id='range_slider', component_property='marks'),
     Output(component_id='range_slider', component_property='value'),],
    Input(component_id='period_radio', component_property='value')
)
def update_slider(period_radio):
    months = ['Jan', "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    min = df_dict[period_radio][period_radio].min()
    max = df_dict[period_radio][period_radio].max()
    if period_radio == 'month':
        marks = {i: months[i] for i in range(min, max, 1)}
    if period_radio == 'year':
        marks = {i: str(i) for i in range(min, max, 5)}
    if period_radio == 'hour':
        marks = {i: str(i)+':00' for i in range(min, max, 2)}

    return min, max, marks, [min, max]


@app.callback(
    [Output(component_id='map', component_property='figure'),
     Output(component_id='map', component_property='selectedData')],
    [Input(component_id='period_radio', component_property='value'),
     Input(component_id='avg_std', component_property='value'),
     Input(component_id='range_slider', component_property='value')],

)
def update_map(period_radio, avg_std, range):
    if period_radio is None:
        raise PreventUpdate
    else:
        df = df_dict[period_radio]
        mask = (df[period_radio] >= range[0]) & (df[period_radio] <= range[1])
        df = df[mask]
        if avg_std == "avg":
            df = df.groupby(['country'])['capacity_factor'].mean()
        else:
            df = df.groupby(['country'])['capacity_factor'].std()

        fig = px.choropleth_mapbox(df,
                                   geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                                   featureidkey='properties.ISO2',
                                   locations=df.index,
                                   color=df.values,
                                   zoom=1.5, center={"lat": 56.4, "lon": 15.0},
                                   mapbox_style="carto-positron",
                                   color_continuous_scale="Viridis",
                                   opacity=0.5,
                                   title='',
                                   )
        fig.update_layout(clickmode='event+select')
        fig.update_layout(coloraxis_colorbar_x=-0.15)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Capacity <br>factor",
            ))

        fig.update_layout(
            font=dict(
                size=10,
            )
        )

        fig.update_layout(
            xaxis=go.layout.XAxis(
                tickangle=45)
        )

    return fig, None


@app.callback(
    Output(component_id='evolution', component_property='children'),

    [Input(component_id='period_radio', component_property='value'),
     Input(component_id='range_slider', component_property='value'),
     Input("map", "selectedData")
     ]
)
def update_plot(period_radio, range, click):
    if range[0] == range[1]:
        return
    df_select = df_dict[period_radio]
    mask = (df_select[period_radio] >= range[0]) & (
        df_select[period_radio] <= range[1])
    df_select = df_select[mask]
    df_all = df_select.groupby([period_radio]).mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig.add_trace(go.Scatter(x=df_all[period_radio], y=df_all['capacity_factor'], name="28C",
                             line_shape='linear'))
    if click is not None:
        for point in click["points"]:
            selected_country = point["location"]
            df_country = df_select[df_select.country == selected_country].groupby(
                [period_radio]).mean().reset_index()
            fig.add_trace(go.Scatter(x=df_country[period_radio], y=df_country['capacity_factor'], name=selected_country,
                                     line_shape='linear'))
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(

        margin=dict(l=5, r=10, t=10, b=5),
        yaxis_title="Capacity factor",
        xaxis_title=period_radio,
        font=dict(
            size=10
        ),
        title=dict(
            x=0.5,
            y=1,
            font=dict(
            )
        ),
    )

    return dcc.Graph(figure=fig)


@app.callback(
    [Output('side_graph', 'children'),
     Output('date_picker_div', 'style'),
     Output('year_picker_div', 'style')],
    [Input("map", "selectedData"),
     Input("dropdown", "value"),
     Input("year_picker", "value"),
     ]
)
def update_side_graph(click, fig_choice, year):
    if fig_choice is None:
        raise PreventUpdate

    if fig_choice == "YoY (year-over-year) monthly capacity factor comparison":

        if click:
            df_select = df_month_all_years[df_month_all_years.country.isin(
                [point["location"] for point in click["points"]])].groupby(['month', 'year']).mean().reset_index()
            name = (click["points"][0]["location"] +
                    "_" if len(click["points"]) == 1 else "selected region ")
            color = ('green' if len(click["points"]) == 1 else 'white')
        else:
            df_select = monthly_cp_eu
            name = "28C-"
            color = "rgb(187,51,59)"

        fig = go.Figure()
        for year_ in monthly_cp_eu.year.unique():
            if year_ != year:
                fig.add_trace(go.Scatter(x=df_select[df_select.year == year_].month, y=df_select[df_select.year == year_]['capacity_factor'], name=name+str(year_),
                                         line_shape='linear', line=dict(color='gray'), opacity=0.2, showlegend=False))
            if year_ == year:
                fig.add_trace(go.Scatter(x=df_select[df_select.year == year_].month, y=df_select[df_select.year == year_]['capacity_factor'], name=name+str(year_),
                                         line_shape='linear', showlegend=True, line=dict(color=color)))

        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="month",
        )

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == 'Min, Max, Avg intra-day hourly capacity factor':

        df_hourly = df_hourly_all_years.drop(["year"], axis=1)[
            df_hourly_all_years.year == year]

        eu_min_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().min()
        eu_max_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().max()
        eu_mean_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().mean()
        variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp

        min_hourly_cp = df_hourly.groupby(['country'])['capacity_factor'].min()
        max_hourly_cp = df_hourly.groupby(['country'])['capacity_factor'].max()
        mean_hourly_cp = df_hourly.groupby(
            ['country'])['capacity_factor'].mean()
        variation_range_hourly = (max_hourly_cp - min_hourly_cp).sort_values()

        sort_hourly_mean = np.argsort(mean_hourly_cp)
        min_hourly_cp = min_hourly_cp[sort_hourly_mean]
        max_hourly_cp = max_hourly_cp[sort_hourly_mean]
        mean_hourly_cp = mean_hourly_cp[sort_hourly_mean]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(go.Scatter(
            x=["28C"], y=[eu_mean_hourly_cp],
            mode='markers',
            showlegend=False,
            name="28C",
            error_y=dict(
                type='data',
                symmetric=False,
                array=[eu_mean_hourly_cp-eu_min_hourly_cp],
                arrayminus=[eu_max_hourly_cp-eu_mean_hourly_cp])
        ))

        if click is None:
            selected_countries = all_country
            mask = np.isin(min_hourly_cp.index, selected_countries)

            x = min_hourly_cp[mask].index
            y = mean_hourly_cp[mask].values
            error = max_hourly_cp[mask].values-y
            error_min = y-min_hourly_cp[mask].values

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                showlegend=False,
                hoverlabel=dict(namelength=0),
                line=dict(color="gray"),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error,
                    arrayminus=error_min)
            ))

        else:
            selected_countries = [point["location"]
                                  for point in click["points"]]
            for selected_country in selected_countries:
                mask = np.isin(min_hourly_cp.index, selected_country)
                x = min_hourly_cp[mask].index
                y = mean_hourly_cp[mask].values
                error = max_hourly_cp[mask].values-y
                error_min = y-min_hourly_cp[mask].values

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    showlegend=False,
                    hoverlabel=dict(namelength=0),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error,
                        arrayminus=error_min)
                ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="Countries",

        )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == 'Intra-year variation range of the monthly capacity factor':

        df_month = df_month_all_years.drop(["year"], axis=1)[
            df_month_all_years.year == year]

        eu_min_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).min()
        eu_max_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).max()
        eu_mean_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).mean()
        variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp

        min_monthly_cp = df_month.groupby(['country'])['capacity_factor'].min()
        max_monthly_cp = df_month.groupby(['country'])['capacity_factor'].max()
        mean_monthly_cp = df_month.groupby(
            ['country'])['capacity_factor'].mean(numeric_only=True)
        variation_range = (max_monthly_cp - min_monthly_cp).sort_values()

        sort_monthly_mean = np.argsort(mean_monthly_cp)
        min_monthly_cp = min_monthly_cp[sort_monthly_mean]
        max_monthly_cp = max_monthly_cp[sort_monthly_mean]
        mean_monthly_cp = mean_monthly_cp[sort_monthly_mean]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(go.Bar(
            x=["28C"], y=[variation_range_eu],
            name="28C",
            showlegend=False,

        ))
        if click is None:
            selected_countries = all_country
            filtered_df = variation_range[np.isin(
                variation_range.index, selected_countries)]
            x = filtered_df.index
            y = filtered_df.values

            fig.add_trace(go.Bar(
                x=x, y=y,
                hoverlabel=dict(namelength=0),
                showlegend=False,
                marker_color='gray'
            ))

        else:
            selected_countries = [point["location"]
                                  for point in click["points"]]
            for selected_country in selected_countries:
                filtered_df = variation_range[np.isin(
                    variation_range.index, selected_country)]
                x = filtered_df.index
                y = filtered_df.values

                fig.add_trace(go.Bar(
                    x=x, y=y,
                    hoverlabel=dict(namelength=0),
                    showlegend=False,
                ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Variation range",
            xaxis_title="Countries",

        )
        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == 'Intra-day variation range of the hourly capacity factor':

        start_time = time.time()

        df_hourly = df_hourly_all_years.drop(["year"], axis=1)[
            df_hourly_all_years.year == year]

        print("--- %s seconds get df  ---" % (time.time() - start_time))
        start_time = time.time()

        eu_min_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().min()
        eu_max_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().max()
        eu_mean_hourly_cp = df_hourly.groupby(
            'hour')['capacity_factor'].mean().mean()
        variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp

        min_hourly_cp = df_hourly.groupby(['country'])['capacity_factor'].min()
        max_hourly_cp = df_hourly.groupby(['country'])['capacity_factor'].max()
        mean_hourly_cp = df_hourly.groupby(
            ['country'])['capacity_factor'].mean()
        variation_range_hourly = (max_hourly_cp - min_hourly_cp).sort_values()

        sort_hourly_mean = np.argsort(mean_hourly_cp)
        min_hourly_cp = min_hourly_cp[sort_hourly_mean]
        max_hourly_cp = max_hourly_cp[sort_hourly_mean]
        mean_hourly_cp = mean_hourly_cp[sort_hourly_mean]

        print("--- %s seconds do operations ---" % (time.time() - start_time))
        start_time = time.time()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(go.Bar(
            x=["28C"], y=[variation_range_eu_hourly],
            name="28C",
            showlegend=False,

        ))
        if click is None:
            selected_countries = all_country
            filtered_df = variation_range_hourly[np.isin(
                variation_range_hourly.index, selected_countries)]
            x = filtered_df.index
            y = filtered_df.values

            fig.add_trace(go.Bar(
                x=x, y=y,
                hoverlabel=dict(namelength=0),
                showlegend=False,
                marker_color='gray'
            ))

        else:

            start_time = time.time()
            selected_countries = [point["location"]
                                  for point in click["points"]]
            for selected_country in selected_countries:
                filtered_df = variation_range_hourly[np.isin(
                    variation_range_hourly.index, selected_country)]
                x = filtered_df.index
                y = filtered_df.values
                fig.add_trace(go.Bar(
                    x=x, y=y,
                    hoverlabel=dict(namelength=0),
                    showlegend=False,
                ))

            print("--- %s seconds plot stuff ---" % (time.time() - start_time))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Variation range",
            xaxis_title="Countries",
        )

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == 'Min, Max, Avg monthly capacity factor':

        df_month = df_month_all_years.drop(["year"], axis=1)[
            df_month_all_years.year == year]

        eu_min_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).min()
        eu_max_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).max()
        eu_mean_monthly_cp = df_month.groupby(
            'month')['capacity_factor'].mean(numeric_only=True).mean()
        variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp

        min_monthly_cp = df_month.groupby(['country'])['capacity_factor'].min()
        max_monthly_cp = df_month.groupby(['country'])['capacity_factor'].max()
        mean_monthly_cp = df_month.groupby(
            ['country'])['capacity_factor'].mean(numeric_only=True)
        variation_range = (max_monthly_cp - min_monthly_cp).sort_values()

        sort_monthly_mean = np.argsort(mean_monthly_cp)
        min_monthly_cp = min_monthly_cp[sort_monthly_mean]
        max_monthly_cp = max_monthly_cp[sort_monthly_mean]
        mean_monthly_cp = mean_monthly_cp[sort_monthly_mean]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(go.Scatter(
            x=["28C"], y=[eu_mean_monthly_cp],
            mode='markers',
            showlegend=False,
            name="28C",
            error_y=dict(
                type='data',
                symmetric=False,
                array=[eu_mean_monthly_cp-eu_min_monthly_cp],
                arrayminus=[eu_max_monthly_cp-eu_mean_monthly_cp])
        ))

        if click is None:
            selected_countries = all_country
            mask = np.isin(min_monthly_cp.index, selected_countries)

            x = min_monthly_cp[mask].index
            y = mean_monthly_cp[mask].values
            error = max_monthly_cp[mask].values-y
            error_min = y-min_monthly_cp[mask].values

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                showlegend=False,
                hoverlabel=dict(namelength=0),
                line=dict(color="gray"),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error,
                    arrayminus=error_min)
            ))

        else:
            selected_countries = [point["location"]
                                  for point in click["points"]]
            for selected_country in selected_countries:
                mask = np.isin(min_monthly_cp.index, selected_country)
                x = min_monthly_cp[mask].index
                y = mean_monthly_cp[mask].values
                error = max_monthly_cp[mask].values-y
                error_min = y-min_monthly_cp[mask].values

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    showlegend=False,
                    hoverlabel=dict(namelength=0),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error,
                        arrayminus=error_min)
                ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Capacity factor",
            xaxis_title="Countries",

        )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == "Cumulative days above thresholds":

        daily_cp_eu = df_daily_cp[df_daily_cp.year == year].groupby(
            "day").mean(numeric_only=True).reset_index()
        cum_days_eu = [(daily_cp_eu.capacity_factor > i).mean()
                       for i in np.linspace(0, 1, 100)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_days_eu,
                      line_shape='linear', line=dict(color="rgb(187,51,59)"), name="28C"))

        if click is not None:
            selected_countries = [point["location"]
                                  for point in click["points"]]
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year == year]
            for country in selected_countries:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]
                cum_days_dict[country] = [(daily_cp_dict[country]['capacity_factor'] > i).mean(
                ) for i in np.linspace(0, 1, 100)]

            for selected_country in selected_countries:
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100),
                              y=cum_days_dict[selected_country], line_shape='linear', name=selected_country))

            if len(selected_countries) > 1:
                cum_days_region = [(pd.Series(np.mean(np.stack([daily_cp_dict[selected_country].capacity_factor.tolist(
                ) for selected_country in selected_countries]), axis=0)) > i).mean() for i in np.linspace(0, 1, 100)]
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100),
                              y=cum_days_region, line_shape='linear', name="selected region"))

        else:
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year == year]
            for country in all_country:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]
                cum_days_dict[country] = [(daily_cp_dict[country]['capacity_factor'] > i).mean(
                ) for i in np.linspace(0, 1, 100)]

            for selected_country in all_country:
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_days_dict[selected_country], line_shape='linear', name=selected_country, line=dict(
                    color='gray'), opacity=0.2, showlegend=False))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Capacity factor threshold",
            yaxis_title="Proportion of days above threshold",
        )

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}

    if fig_choice == 'Spatial correlation between daily capacity factor':

        if not click or len(click["points"]) != 1:
            return html.Div("Select only one country on the left map to display the correlation between the LWP day distribution of this country and the LWP day distribution of the other European countries", style={"marginTop": "10vh", "textAlign": "center"}), {"display": "none"}, {"display": "none"}

        df_select = daily_cp_df.corr()[click["points"][0]['location']]
        fig = px.choropleth_mapbox(df_select,
                                   geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                                   featureidkey='properties.ISO2',
                                   locations=df_select.index,
                                   color=df_select,
                                   zoom=1, center={"lat": 56.4, "lon": 15.0},
                                   range_color=[0, 1.0],
                                   mapbox_style="carto-positron",
                                   color_continuous_scale="Viridis",
                                   opacity=0.5,
                                   )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Correlation<br>(1979-2019)',
            ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Spatial correlation between LWP day distribution':

        if not click or len(click["points"]) != 1:
            return html.Div("Select one country on the left map to display the correlation between the LWP day distribution of this country and the LWP day distribution of the other European countries", style={"marginTop": "10vh", "textAlign": "center"}), {"display": "none"}, {"display": "none"}

        df_select = daily_LWP_df.corr()[click["points"][0]['location']]
        fig = px.choropleth_mapbox(df_select,
                                   geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                                   featureidkey='properties.ISO2',
                                   locations=df_select.index,
                                   color=df_select,
                                   zoom=1, center={"lat": 56.4, "lon": 15.0},
                                   range_color=[0, 0.6],
                                   mapbox_style="carto-positron",
                                   color_continuous_scale="Viridis",
                                   opacity=0.5,
                                   )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Correlation<br>(1979-2019)',
            ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == 'LWP events':

        if not click or len(click["points"]) < 1:
            return [html.Div("Select a region on the left map to display the number of low wind power (LWP) events during the selected year", style={"marginBottom": "10vh", "marginTop": "10vh", "textAlign": "center"}),
                    html.Br(),
                    gif.GifPlayer(
                gif='assets/rec.gif',
                still='assets/rec.png',
                autoplay=True,
            )], {"display": "none"}, {"display": "none"}

        x = list(range(1, 9))
        mask = (daily_cp_eu_entire_period.year >= year) & (
            daily_cp_eu_entire_period.year <= year)
        y = [num_events(daily_cp_eu_entire_period[mask], i)
             for i in range(1, 9)]
        fig = go.Figure()
        fig2 = None
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(
            go.Bar(x=x, y=y, marker_color="rgb(187,51,59)", name="28C", showlegend=True))

        if click is not None:
            mask = dict()
            for point in click["points"]:
                selected_country = point["location"]
                mask[selected_country] = (daily_cp_dict_entire_period[selected_country].year >= year) & (
                    daily_cp_dict_entire_period[selected_country].year <= year)
                fig.add_trace(go.Bar(name=selected_country, x=list(range(1, 9)), y=[num_events(daily_cp_dict_entire_period[selected_country][mask[selected_country]], i) for i in range(1, 9)]
                                     ))

            dfs = [daily_cp_dict_entire_period[selected_country][mask[selected_country]
                                                                 ].reset_index() for selected_country in mask.keys()]
            df_concat = pd.concat(dfs)
            by_row_index = df_concat.groupby(df_concat.day)
            df_means = by_row_index.mean().reset_index()
            dummy_df = pd.DataFrame({
                "ds": pd.date_range(dfs[0].day.values[0], dfs[0].day.values[-1]),
                "value": (df_means.capacity_factor < 0.1).values.astype(int),

            })

            fig2 = calplot(dummy_df, x='ds', y='value', colorscale=[
                           [0, "rgb(4,204,148)"], [1, "rgb(227,26,28)"]])
            fig2.update_xaxes(fixedrange=True)
            fig2.update_yaxes(fixedrange=True)

            fig2.update_layout(
                title={
                    'text': "Low wind power days in the selected area in {}".format(year),
                    'y': 0.09,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }

            )

            if len(click["points"]) > 1:
                fig.add_trace(go.Bar(
                    x=list(range(1, 9)), y=[num_events(df_means, i) for i in range(1, 9)], name="mean capacity<br>factor of selected<br>countries"
                ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Number of occurences",
            xaxis_title="Minimum duration of the low wind power event (days)",

        )

        return [
            dcc.Graph(figure=fig, style={'height': '50vh'}),
            (dcc.Graph(figure=fig2, style={'marginTop': '1vh'}) if fig2 else None)],  {"display": "none"}, {}

    if fig_choice == 'Daily capacity factor distribution':

        daily_cp_eu = df_daily_cp[df_daily_cp.year ==
                                  year].groupby("day").mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=[], y=[]))
        fig.add_trace(go.Histogram(
            xbins=dict(
                size=0.05),
            name="28C",
            showlegend=True,
            marker=dict(color="rgb(187,51,59)"),
            x=daily_cp_eu.capacity_factor, histnorm='percent'))

        if click is not None:
            selected_countries = [point["location"]
                                  for point in click["points"]]
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year == year]
            for country in selected_countries:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]

            for selected_country in selected_countries:
                fig.add_trace(go.Histogram(
                    x=daily_cp_dict[selected_country].capacity_factor, histnorm='percent', name=selected_country, xbins=dict(size=0.05)))

            if len(selected_countries) > 1:
                fig.add_trace(go.Histogram(x=pd.Series(np.mean(np.stack([daily_cp_dict[selected_country].capacity_factor.tolist(
                ) for selected_country in selected_countries]), axis=0)), histnorm='percent', name="selected region", xbins=dict(size=0.05)))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Capacity factor",
            yaxis_title="Percentage of days",

        )

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}


if __name__ == '__main__':
    app.run_server(port=8088, debug=True, use_reloader=True)
