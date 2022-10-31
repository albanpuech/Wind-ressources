# Environment used: dash1_8_0_env
import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px
import plotly.graph_objects as go
import dash  # (version 1.8.0)
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import numpy as np
import matplotlib.pyplot as plt
from plotly_calplot import calplot
# import calplot
from datetime import date
import dash_bootstrap_components as dbc
load_figure_template("SLATE")

# print(px.data.gapminder()[:15])

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


app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])


# df = pd.read_csv('dataset_with_timestamp')
df = pd.read_csv('light')


df_dict = {
    "hour": df.drop(['timestamp'], axis=1).groupby(['hour', 'country']).mean(numeric_only=True).reset_index(),
    "month": df.drop(['timestamp'], axis=1).groupby(['month', 'country']).mean(numeric_only=True).reset_index(),
    "year": df.drop(['timestamp'], axis=1).groupby(['year', 'country']).mean(numeric_only=True).reset_index(),
    "day": df.drop(['timestamp'], axis=1).groupby(['day', 'country']).mean(numeric_only=True).reset_index()
}


df2 = df.copy()
df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='%Y-%m-%d %H:%M:%S')
df2["timestamp"] = df2["timestamp"].dt.floor('d')
df2 = df2[["country", "day", "capacity_factor", "timestamp"]
          ].groupby(["country", "timestamp"]).mean().reset_index()
cum_time = dict()
daily_cp = dict()

for country in country_name_to_iso2.values():
    daily_cp[country] = df2[df2.country == country]
    cum_time[country] = [(daily_cp[country]['capacity_factor'] > i).mean()
                         for i in np.linspace(0, 1, 100)]

daily_cp_eu = df2.groupby("timestamp").mean().reset_index()
cum_time_eu = [(daily_cp_eu.capacity_factor > i).mean()
               for i in np.linspace(0, 1, 100)]


eu_min_hourly_cp = df_dict["hour"].groupby(
    'hour')['capacity_factor'].mean().min()
eu_max_hourly_cp = df_dict["hour"].groupby('hour')['capacity_factor'].mean(
).max()  # mean over country then take min over hours
eu_mean_hourly_cp = df_dict["hour"].groupby(
    'hour')['capacity_factor'].mean().mean()
min_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].min()
max_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].max()
mean_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].mean()
variation_range_hourly = max_hourly_cp - min_hourly_cp
variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp
sort_hourly_range = np.argsort(variation_range_hourly)
sort_hourly = np.argsort(mean_hourly_cp)

eu_min_monthly_cp = df_dict["month"].groupby(
    'month')['capacity_factor'].mean().min()
eu_max_monthly_cp = df_dict["month"].groupby('month')['capacity_factor'].mean(
).max()  # mean over country then take min over months
eu_mean_monthly_cp = df_dict["month"].groupby(
    'month')['capacity_factor'].mean().mean()
min_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].min()
max_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].max()
mean_monthly_cp = df_dict["month"].groupby(
    ['country'])['capacity_factor'].mean()
std_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].std()
variation_range = max_monthly_cp - min_monthly_cp
variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp
sort_monthly_range = np.argsort(variation_range)
sort_monthly = np.argsort(mean_monthly_cp)


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
app.layout = html.Div([


    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                   
                        dbc.Col([
                        dcc.Dropdown(options=[
                            {'label': 'yearly capacity factor', 'value': 'year'},
                            {'label': 'monthly capacity factor', 'value': 'month'},
                            {'label': 'hourly capacity factor', 'value': 'hour'}],
                            id="period_radio", value="year", optionHeight=60),
                        ]),
                        dbc.Col([
                        dcc.Dropdown(options=[{'label': 'Average', 'value': 'avg'},
                                              {'label': 'Standard deviation of the', 'value': 'std'}],
                                     id="avg_std", value="avg", optionHeight=60,),
                        ]),
                ]),
                dbc.Row([
                 
                        dcc.Graph(id='map', style={'height': '50vh'})
                ]),


              
                dbc.Row(dcc.RangeSlider(1979, 2019, 1, value=[1979, 2019], id='range_slider', marks={
                    i: {"label":str(i)} for i in range(1979, 2020, 5)}, tooltip={"placement": "bottom", "always_visible": True})
                ),

                dbc.Row(
                    dcc.Graph(id='distribution', style={'height': '20vh'})
                )
            ]), style={"height": "90vh", "width": "45vw"}
        ), style={"display": "flex", "justifyContent":"center"}),

    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(['Min, Max, Avg monthly capacity factor',
                                                'Min, Max, Avg hourly capacity factor', 'Monthly variation range',
                                                'Hourly variation range',
                                                "Cumulative days above thresholds",
                                                'Low power events',
                                                'Daily capacity factor distribution'],
                                            'Hourly variation range', id='dropdown'),
                                
                        # 'height': '60vh'}),
                    ]),

                    dbc.Col([
                        dcc.Dropdown(options=[{'label': x, 'value': x} for x in range(
                            1979, 2019)], value=1979, id="year_picker")], id="year_picker_div", style={"display": "none"}),

                
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



                
                    dbc.Row(children=[], id='side_graph', style={}),





            ]), style={"height": "90vh", "width": "45vw"}), style={"display": "flex", "justifyContent":"center"})

], style={"height": "100vh","display": "flex","justifyContent":"center","alignItems":"center", "flexDirection": "row"})  # style={"display": "flex", })


# ---------------------------------------------------------------

@app.callback(
    [Output(component_id='range_slider', component_property='min'),
     Output(component_id='range_slider', component_property='max'),
     Output(component_id='range_slider', component_property='marks'),
     Output(component_id='range_slider', component_property='value'),],
    Input(component_id='period_radio', component_property='value')
)
def update_slider(period_radio):
    min = df_dict[period_radio][period_radio].min()
    max = df_dict[period_radio][period_radio].max()


    return min, max, {i: str(i) for i in range(min, max, 5)}, [min, max]


@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input(component_id='period_radio', component_property='value'),
     Input(component_id='avg_std', component_property='value'),
     Input(component_id='range_slider', component_property='value')]
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
                                   locations=df.index,  # column in dataframe
                                   # animation_group ='Area_Name',       #dataframe
                                   color=df.values,  # dataframe
                                   #    range_color=[0, 0.5],
                                   zoom=1.5, center={"lat": 56.4, "lon": 15.0},
                                   mapbox_style="carto-positron",
                                   color_continuous_scale="Viridis",
                                   opacity=0.5,

                                   title='',
                                   )
        # fig.update_layout(
        #     title={"text": 'Average capacity factor of European countries'})
        fig.update_layout(clickmode='event+select')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=40))
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Capacity <br>factor",
            ))

        fig.update_layout(
            font=dict(
                size=10,  # Set the font size here
            )
        )



        fig.update_layout(
            xaxis=go.layout.XAxis(
                tickangle=45)
        )

    return fig


@app.callback(
    Output(component_id='distribution', component_property='figure'),

    [Input(component_id='period_radio', component_property='value'),
     Input(component_id='range_slider', component_property='value'),
     Input("map", "selectedData")
     ]
)
def update_plot(period_radio, range, click):
    if period_radio is None:
        raise PreventUpdate
    else:
        df = df_dict[period_radio]
        mask = (df[period_radio] >= range[0]) & (df[period_radio] <= range[1])
        df = df[mask]

        df_all = df.groupby([period_radio]).mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_all[period_radio], y=df_all['capacity_factor'],
                                 line_shape='linear'))
        if click is not None:
            for point in click["points"]:
                selected_country = point["location"]
                df_country = df[df.country == selected_country].groupby(
                    [period_radio]).mean().reset_index()
                fig.add_trace(go.Scatter(x=df_country[period_radio], y=df_country['capacity_factor'],
                                         line_shape='linear'))

        fig.update_layout(
            margin=dict(l=5, r=10, t=10, b=5),
            yaxis={"mirror": "allticks", 'side': 'right'}
        )

    return fig


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

    if fig_choice == 'Min, Max, Avg hourly capacity factor':

        # x = np.concatenate([min_hourly_cp[sort][mask].index,["EU"]])
        # y = np.concatenate([mean_hourly_cp[sort][mask].values,[eu_mean_hourly_cp]])
        # error = y - np.concatenate([min_hourly_cp[sort][mask].values,[eu_min_hourly_cp]])
        # error_min =  np.concatenate([max_hourly_cp[sort][mask].values,[eu_max_hourly_cp]]) - y

        mask = [True for _ in min_hourly_cp.index]
        if click is not None:
            mask = [point["location"] for point in click["points"]]
            mask = np.isin(min_hourly_cp[sort_hourly].index, mask)

        x = min_hourly_cp[sort_hourly][mask].index
        y = mean_hourly_cp[sort_hourly][mask].values
        error = max_hourly_cp[sort_hourly][mask].values-y
        error_min = y-min_hourly_cp[sort_hourly][mask].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            showlegend=False,
            error_y=dict(
                type='data',
                symmetric=False,
                array=error,
                arrayminus=error_min)
        ))

        fig.add_trace(go.Scatter(
            x=["EU"], y=[eu_mean_hourly_cp],
            mode='markers',
            name="EU",
            error_y=dict(
                type='data',

                symmetric=False,
                array=[eu_mean_hourly_cp-eu_min_hourly_cp],
                arrayminus=[eu_max_hourly_cp-eu_mean_hourly_cp])
        ))

        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Monthly variation range':

        mask = [True for _ in min_hourly_cp.index]
        if click is not None:
            mask = [point["location"] for point in click["points"]]
            mask = np.isin(min_hourly_cp[sort_hourly].index, mask)

        x = variation_range[sort_monthly_range][mask].index
        y = variation_range[sort_monthly_range][mask].values

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x, y=y,
            showlegend=False,
        ))

        fig.add_trace(go.Bar(
            x=["EU"], y=[variation_range_eu],
            name="EU",
        ))

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}
    if fig_choice == 'Standard deviation of montly capacity factor':

        fig = px.choropleth_mapbox(std_monthly_cp,
                                   geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                                   featureidkey='properties.ISO2',
                                   locations=std_monthly_cp.index,  # column in dataframe
                                   # animation_group ='Area_Name',       #dataframe
                                   color='capacity_factor',  # dataframe
                                   #range_color=[0, 0.5],
                                   zoom=1, center={"lat": 56.4, "lon": 15.0},
                                   mapbox_style="carto-positron",
                                   color_continuous_scale="Viridis",
                                   opacity=0.5,
                                   title='',
                                   )
        # fig.update_layout(
        #         title={"text": 'Average capacity factor of European countries'})
        # fig.update_layout(clickmode='event+select')
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Std<br>Capacity<br>factor",
            ))

        fig.update_layout(
            font=dict(
                size=10,  # Set the font size here
            )
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

        fig.update_layout(
            xaxis=go.layout.XAxis(
                tickangle=45)
        )

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}
    if fig_choice == 'Hourly variation range':

        mask = [True for _ in min_hourly_cp.index]
        if click is not None:
            mask = [point["location"] for point in click["points"]]
            mask = np.isin(min_hourly_cp[sort_hourly].index, mask)

        x = variation_range_hourly[sort_hourly_range][mask].index
        y = variation_range_hourly[sort_hourly_range][mask].values

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x, y=y,
            showlegend=False,
        ))

        fig.add_trace(go.Bar(
            x=["EU"], y=[variation_range_eu_hourly],
            name="EU",
        ))

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Min, Max, Avg monthly capacity factor':

        mask = [True for _ in min_hourly_cp.index]
        if click is not None:
            mask = [point["location"] for point in click["points"]]
            mask = np.isin(min_hourly_cp[sort_hourly].index, mask)

        x = min_monthly_cp[sort_monthly][mask].index
        y = mean_monthly_cp[sort_monthly][mask].values
        error = max_monthly_cp[sort_monthly][mask].values-y
        error_min = y-min_monthly_cp[sort_monthly][mask].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            showlegend=False,
            error_y=dict(
                type='data',
                symmetric=False,
                array=error,
                arrayminus=error_min)
        ))

        fig.add_trace(go.Scatter(
            x=["EU"], y=[eu_mean_monthly_cp],
            mode='markers',
            name="EU",
            error_y=dict(
                type='data',

                symmetric=False,
                array=[eu_mean_monthly_cp-eu_min_monthly_cp],
                arrayminus=[eu_max_monthly_cp-eu_mean_monthly_cp])
        ))
        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}

    if fig_choice == "Cumulative days above thresholds":

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_time_eu,
                      line_shape='linear', line=dict(color='red'), name="EU"))

        if click is not None:
            for point in click["points"]:
                selected_country = point["location"]
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100),
                              y=cum_time[selected_country], line_shape='linear'))

        else:
            for selected_country in country_name_to_iso2.values():
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_time[selected_country], line_shape='linear', line=dict(
                    color='gray'), opacity=0.2, name=selected_country))

        # fig.update_layout(
        #     margin=dict(l=5, r=10, t=10, b=5),
        #     yaxis={"mirror": "allticks", 'side': 'right'}
        # )

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}



    if fig_choice == 'Low power events':
        print(year)
        x = list(range(1, 9))
        mask = (daily_cp_eu.timestamp.dt.year >= year) & (
            daily_cp_eu.timestamp.dt.year <= year)
        y = [num_events(daily_cp_eu[mask], i) for i in range(1, 9)]
        fig = go.Figure()
        fig2 = None
        fig.add_trace(go.Bar(x=x, y=y, marker_color="red", name="EU"))

        if click is not None:
            mask = dict()
            for point in click["points"]:
                selected_country = point["location"]
                mask[selected_country] = (daily_cp[selected_country].timestamp.dt.year >= year) & (
                    daily_cp[selected_country].timestamp.dt.year <= year)
                print(len(daily_cp[selected_country][mask[selected_country]]))
                fig.add_trace(go.Bar(
                    x=list(range(1, 9)), y=[num_events(daily_cp[selected_country][mask[selected_country]], i) for i in range(1, 9)],
                ))

            dfs = [daily_cp[selected_country][mask[selected_country]
                                              ].reset_index() for selected_country in mask.keys()]
            df_concat = pd.concat(dfs)
            by_row_index = df_concat.groupby(df_concat.timestamp)
            df_means = by_row_index.mean().reset_index()

            fig.add_trace(go.Bar(
                x=list(range(1, 9)), y=[num_events(df_means, i) for i in range(1, 9)],
            ))

        if click is not None:
            mask = dict()
            for point in click["points"]:
                selected_country = point["location"]
                mask[selected_country] = (daily_cp[selected_country].timestamp.dt.year >= year) & (
                    daily_cp[selected_country].timestamp.dt.year <= year)

            dfs = [daily_cp[selected_country][mask[selected_country]
                                              ].reset_index() for selected_country in mask.keys()]
            df_concat = pd.concat(dfs)
            by_row_index = df_concat.groupby(df_concat.timestamp)
            df_means = by_row_index.mean().reset_index()
            dummy_df = pd.DataFrame({
                "ds": pd.date_range(dfs[0].timestamp.values[0], dfs[0].timestamp.values[-1]),
                "value": (df_means.capacity_factor < 0.1).values.astype(int),

            })

            fig2 = calplot(dummy_df, x='ds', y='value', colorscale=[[0, "rgb(4,204,148)"], [1, "rgb(227,26,28)"]])

        # else :

        #     dummy_df = pd.DataFrame({
        #         "ds": pd.date_range(start, end),
        #         "value": (daily_cp["FR"][mask].capacity_factor<0.1).values.astype(int),

        #     })

        return [dcc.Graph(figure=fig), (dcc.Graph(figure=fig2) if fig2 else None)],  {"display":"none"}, ({} if True else {"display": "none"})

    if fig_choice == 'Daily capacity factor distribution':
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=daily_cp_eu.capacity_factor, histnorm='percent'))

        if click is not None:
            for point in click["points"]:
                selected_country = point["location"]
                fig.add_trace(go.Histogram(
                    x=daily_cp[selected_country].capacity_factor, histnorm='percent'))

        return dcc.Graph(figure=fig), {"display": "none"}, {"display": "none"}


if __name__ == '__main__':
    app.run_server(port=8088, debug=True, use_reloader=True)
