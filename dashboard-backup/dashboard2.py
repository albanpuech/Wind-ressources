import time
start = time.process_time()

# Environment used: dash1_8_0_env
import pandas as pd  # (version 1.0.0)
import plotly.express as px
import plotly.graph_objects as go
import dash  # (version 1.8.0)
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import numpy as np
from plotly_calplot import calplot
# import calplot
from datetime import date
import dash_bootstrap_components as dbc
load_figure_template("slate")
import dash_gif_component as gif
import dask.dataframe as pd



print("import",time.process_time() - start)

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

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])


start = time.process_time()
df = pd.read_csv('dataset_with_timestamp')
# df = pd.read_csv('light2')
# df = pd.read_csv('dataset_compressed.csv.gz', compression='gzip')

  
print("load df",time.process_time() - start)
start = time.process_time()

df_dict = {
    "hour": df.drop(['timestamp'], axis=1).groupby(['hour', 'country']).mean().reset_index(),
    "month": df.drop(['timestamp'], axis=1).groupby(['month', 'country']).mean().reset_index(),
    "year": df.drop(['timestamp'], axis=1).groupby(['year', 'country']).mean().reset_index(),
    "day": df.drop(['timestamp'], axis=1).groupby(['day', 'country']).mean().reset_index()
}


df_daily_cp = df.copy()
df_daily_cp['timestamp'] = pd.to_datetime(df_daily_cp['timestamp'], format='%Y-%m-%d %H:%M:%S')
df_daily_cp["timestamp"] = df_daily_cp["timestamp"].dt.floor('d')
df_daily_cp = df_daily_cp[["country", "day", "capacity_factor", "timestamp","year"]].groupby(["country", "timestamp"]).mean().reset_index()
cum_days_dict_entire_period = dict()
daily_cp_dict_entire_period = dict()

for country in country_name_to_iso2.values():
    daily_cp_dict_entire_period[country] = df_daily_cp[df_daily_cp.country == country]
    cum_days_dict_entire_period[country] = [(daily_cp_dict_entire_period[country]['capacity_factor'] > i).mean() for i in np.linspace(0, 1, 100)]

daily_cp_eu_entire_period = df_daily_cp.groupby("timestamp").mean().reset_index()
cum_days_eu_entire_period = [(daily_cp_eu_entire_period.capacity_factor > i).mean() for i in np.linspace(0, 1, 100)]



df_monthly_cp = df.copy()
df_montly_cp = df.drop(['timestamp','hour'], axis=1).groupby(['month','year','country']).mean().reset_index()
monthly_cp_dict = dict()

for country in country_name_to_iso2.values():
    monthly_cp_dict[country] = df_montly_cp[df_montly_cp.country == country].reset_index()

monthly_cp_eu = df_montly_cp.groupby(['month','year']).mean().reset_index()







eu_min_hourly_cp = df_dict["hour"].groupby('hour')['capacity_factor'].mean().min()
eu_max_hourly_cp = df_dict["hour"].groupby('hour')['capacity_factor'].mean().max()
eu_mean_hourly_cp = df_dict["hour"].groupby('hour')['capacity_factor'].mean().mean()
variation_range_eu_hourly = eu_max_hourly_cp - eu_min_hourly_cp



min_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].min()
max_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].max()
mean_hourly_cp = df_dict["hour"].groupby(['country'])['capacity_factor'].mean()
variation_range_hourly = (max_hourly_cp - min_hourly_cp)




eu_min_monthly_cp = df_dict["month"].groupby('month')['capacity_factor'].mean().min()
eu_max_monthly_cp = df_dict["month"].groupby('month')['capacity_factor'].mean().max()
eu_mean_monthly_cp = df_dict["month"].groupby('month')['capacity_factor'].mean().mean()
variation_range_eu = eu_max_monthly_cp - eu_min_monthly_cp

min_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].min()
max_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].max()
mean_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].mean()
std_monthly_cp = df_dict["month"].groupby(['country'])['capacity_factor'].std()
variation_range = (max_monthly_cp - min_monthly_cp)




## compute correlation
daily_cp_df = pd.DataFrame(np.array([daily_cp_dict_entire_period[country].capacity_factor for country in all_country]).T, columns = all_country)
daily_LWP_df= pd.DataFrame(np.array([(daily_cp_dict_entire_period[country].capacity_factor)<0.1 for country in all_country]).T, columns = all_country)
# daily_cp_df_melted = daily_cp_df.corr().melt(ignore_index=False, var_name="main_country")
# daily_LWP_df_melted = daily_LWP_df.corr().melt(ignore_index=False, var_name="main_country")


print("create df",time.process_time() - start)


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
                        dbc.Tooltip("Select the aggregation period and the statistics to display on map",
                                target="period_stats",
                            ),
                        dbc.Col([
                        dcc.Dropdown(options=[{'label': 'Average', 'value': 'avg'},
                                              {'label': 'Std of the', 'value': 'std'}],
                                     id="avg_std", searchable = False,value="avg"),
                        ],width=4),
                        dbc.Col([
                        dcc.Dropdown(options=[
                            {'label': 'yearly capacity factor', 'value': 'year'},
                            {'label': 'monthly capacity factor', 'value': 'month'},
                            {'label': 'hourly capacity factor', 'value': 'hour'}],
                            id="period_radio", searchable = False,value="year"),
                        ]),

                ], id="period_stats"),
                dbc.Row([
                    dbc.Tooltip(html.Span("Hold Shift to select multiple countries",style={"color":"rgb(187,51,59)", "font-weight": "bold"}),target="map_row"),
                    dcc.Graph(id='map', style={'height': '48vh'})
                ], id="map_row"),



                dbc.Row([
                    dbc.Tooltip("Select the period to display on the map",target="range_slider_row"),
                    dcc.RangeSlider(1979, 2019, 1, value=[1979, 2019], id='range_slider', marks={
                    i: {"label":str(i)} for i in range(1979, 2020, 5)}, tooltip={"placement": "top", "always_visible": True}),
                    ],
                    id="range_slider_row"
                ),

                dbc.Row(children=[], id='evolution', style={'height': '20vh'}
                )
            ]), style={"height": "90vh", "width": "45vw"}
        ), style={"display": "flex", "justifyContent":"center"}),

    dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Tooltip("Select a plot to display",target="plot_selector_col"),
                        dcc.Dropdown(['Min, Max, Avg monthly capacity factor',
                                                'Min, Max, Avg hourly capacity factor', 'Monthly variation range',
                                                'Hourly variation range',
                                                "Cumulative days above thresholds",
                                                'LWP events',
                                                'Spatial correlation between LWP day distribution',
                                                'Spatial correlation between daily capacity factor',
                                                'Daily capacity factor distribution',
                                                'YoY (year-over-year) monthly capacity factor comparison'],
                                            'Hourly variation range', searchable = False,optionHeight=60,id='dropdown'),

                    ], id="plot_selector_col"),

                    dbc.Col([
                        dcc.Dropdown(options=[{'label': x, 'value': x} for x in range(
                            1979, 2019)], value=1979, searchable = False,id="year_picker")], id="year_picker_div", style={"display": "none"}),


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



                    dcc.Loading(fullscreen=False,children=[dbc.Row(children=[], id='side_graph', style={})]),





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
    months = ['Jan', "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    min = df_dict[period_radio][period_radio].min()
    max = df_dict[period_radio][period_radio].max()
    if period_radio == 'month' :
        marks = {i: months[i] for i in range(min, max, 1)}
    if period_radio == 'year' :
        marks = {i: str(i) for i in range(min, max, 5)}
    if period_radio == 'hour' :
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
            # hover = "Country: "+df.index + "<br>Avg capacity factor: " + df.values.astype(str)
        else:
            df = df.groupby(['country'])['capacity_factor'].std()
            # hover = "Country: "+df.index + "<br>Std capacity factor: " + df.values.astype(str)


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
        fig.update_layout(coloraxis_colorbar_x=-0.15)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
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

    return fig, None


@app.callback(
    Output(component_id='evolution', component_property='children'),

    [Input(component_id='period_radio', component_property='value'),
     Input(component_id='range_slider', component_property='value'),
     Input("map", "selectedData")
     ]
)
def update_plot(period_radio, range, click):
    if range[0] == range[1] : return
    df = df_dict[period_radio]
    mask = (df[period_radio] >= range[0]) & (df[period_radio] <= range[1])
    df = df[mask]
    df_all = df.groupby([period_radio]).mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig.add_trace(go.Scatter(x=df_all[period_radio], y=df_all['capacity_factor'], name="EU",
                                line_shape='linear'))
    if click is not None:
        for point in click["points"]:
            selected_country = point["location"]
            df_country = df[df.country == selected_country].groupby(
                [period_radio]).mean().reset_index()
            fig.add_trace(go.Scatter(x=df_country[period_radio], y=df_country['capacity_factor'], name = selected_country,
                                        line_shape='linear'))
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(

        margin=dict(l=5, r=10, t=10, b=5),
        # yaxis={"mirror": "allticks", 'side': 'right'},
        yaxis_title="Capacity factor",
        xaxis_title=period_radio,
        font=dict(
            size=10
    #     family="system-ui, -apple-system, Segoe UI",
    ),
            title=dict(
            # text='Evolution of the capacity factor',
            x=0.5,
            y=1,
            # yanchor = 'middle',
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


    if fig_choice == "YoY (year-over-year) monthly capacity factor comparison" :

        # if not click or len(click["points"])!=1 :
        #     return html.Div("Select one country on the left map to display the correlation between the LWP day distribution of this country and the LWP day distribution of the other European countries", style={"marginTop":"10vh","textAlign":"center"}), {"display": "none"}, {"display": "none"}
        if click :
            df = df_montly_cp[df_montly_cp.country.isin([point["location"] for point in click["points"]])].groupby(['month','year']).mean().reset_index()
            name = ( click["points"][0]["location"] + "_" if len(click["points"])==1 else "selected regions " )
            color = ('green' if len(click["points"])==1 else 'white')
        else :
            df = monthly_cp_eu
            name = "EU "
            color = "rgb(187,51,59)"


        fig = go.Figure()
        for year_ in monthly_cp_eu.year.unique() :
            if year_ != year :
                fig.add_trace(go.Scatter(x=df[df.year == year_].month, y=df[df.year == year_]['capacity_factor'], name=name+str(year_),
                                    line_shape='linear',line=dict(color='gray'), opacity=0.2, showlegend = False))
            if year_ == year :
                fig.add_trace(go.Scatter(x=df[df.year == year_].month, y=df[df.year == year_]['capacity_factor'], name=name+str(year_),
                    line_shape='linear',showlegend = True, line=dict(color=color )))

        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Capacity factor",
        xaxis_title="month",
    )


        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {}






    if fig_choice == 'Min, Max, Avg hourly capacity factor':

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[],y=[]))
        fig.add_trace(go.Scatter(
        x=["EU"], y=[eu_mean_hourly_cp],
        mode='markers',
        showlegend=False,
        name="EU",
        error_y=dict(
            type='data',
            symmetric=False,
            array=[eu_mean_hourly_cp-eu_min_hourly_cp],
            arrayminus=[eu_max_hourly_cp-eu_mean_hourly_cp])
    ))

        if click is None :
            selected_countries = all_country
            mask = np.isin(min_hourly_cp.index,selected_countries)

            x = min_hourly_cp[mask].index
            y = mean_hourly_cp[mask].values
            error = max_hourly_cp[mask].values-y
            error_min = y-min_hourly_cp[mask].values

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                showlegend=False,
                hoverlabel = dict(namelength=0),
                line=dict(color="gray"),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error,
                    arrayminus=error_min)
            ))

        else :
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries :
                mask = np.isin(min_hourly_cp.index,selected_country)
                x = min_hourly_cp[mask].index
                y = mean_hourly_cp[mask].values
                error = max_hourly_cp[mask].values-y
                error_min = y-min_hourly_cp[mask].values

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    showlegend=False,
                    hoverlabel = dict(namelength=0),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error,
                        arrayminus=error_min)
                ))



        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        yaxis_title="Capacity factor",
        xaxis_title="Countries",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Monthly variation range':

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[],y=[]))
        fig.add_trace(go.Bar(
    x=["EU"], y=[variation_range_eu],
    name="EU",
    showlegend=False,

))
        if click is None :
            selected_countries = all_country
            filtered_df = variation_range[np.isin(variation_range.index,selected_countries)]
            x = filtered_df.index
            y = filtered_df.values


            fig.add_trace(go.Bar(
                x=x, y=y,
                hoverlabel = dict(namelength=0),
                showlegend=False,
                marker_color = 'gray'
            ))

        else :
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in  selected_countries :
                filtered_df = variation_range[np.isin(variation_range.index,selected_country)]
                x = filtered_df.index
                y = filtered_df.values


                fig.add_trace(go.Bar(
                    x=x, y=y,
                    hoverlabel = dict(namelength=0),
                    showlegend=False,
                ))




        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        yaxis_title="Capacity factor",
        xaxis_title="Countries",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )
        fig.update_yaxes(rangemode="tozero")


        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Hourly variation range':


        fig = go.Figure()
        fig.add_trace(go.Bar(x=[],y=[]))
        fig.add_trace(go.Bar(
    x=["EU"], y=[variation_range_eu_hourly],
    name="EU",
    showlegend=False,

))
        if click is None :
            selected_countries = all_country
            filtered_df = variation_range_hourly[np.isin(variation_range_hourly.index,selected_countries)]
            x = filtered_df.index
            y = filtered_df.values


            fig.add_trace(go.Bar(
                x=x, y=y,
                hoverlabel = dict(namelength=0),
                showlegend=False,
                marker_color='gray'
            ))

        else :
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in  selected_countries :
                filtered_df = variation_range_hourly[np.isin(variation_range_hourly.index,selected_country)]
                x = filtered_df.index
                y = filtered_df.values


                fig.add_trace(go.Bar(
                    x=x, y=y,
                    hoverlabel = dict(namelength=0),
                    showlegend=False,
                ))


        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        yaxis_title="Capacity factor",
        xaxis_title="Countries",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )

        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == 'Min, Max, Avg monthly capacity factor':

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[],y=[]))
        fig.add_trace(go.Scatter(
        x=["EU"], y=[eu_mean_monthly_cp],
        mode='markers',
        showlegend=False,
        name="EU",
        error_y=dict(
            type='data',
            symmetric=False,
            array=[eu_mean_monthly_cp-eu_min_monthly_cp],
            arrayminus=[eu_max_monthly_cp-eu_mean_monthly_cp])
    ))

        if click is None :
            selected_countries = all_country
            mask = np.isin(min_monthly_cp.index,selected_countries)

            x = min_monthly_cp[mask].index
            y = mean_monthly_cp[mask].values
            error = max_monthly_cp[mask].values-y
            error_min = y-min_monthly_cp[mask].values

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                showlegend=False,
                hoverlabel = dict(namelength=0),
                line=dict(color="gray"),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error,
                    arrayminus=error_min)
            ))

        else :
            selected_countries = [point["location"] for point in click["points"]]
            for selected_country in selected_countries :
                mask = np.isin(min_monthly_cp.index,selected_country)
                x = min_monthly_cp[mask].index
                y = mean_monthly_cp[mask].values
                error = max_monthly_cp[mask].values-y
                error_min = y-min_monthly_cp[mask].values

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    showlegend=False,
                    hoverlabel = dict(namelength=0),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error,
                        arrayminus=error_min)
                ))



        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        yaxis_title="Capacity factor",
        xaxis_title="Countries",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )

        fig.update_yaxes(rangemode="tozero")

        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}

    if fig_choice == "Cumulative days above thresholds":
        


        daily_cp_eu = df_daily_cp[df_daily_cp.year==year].groupby("timestamp").mean().reset_index()
        cum_days_eu = [(daily_cp_eu.capacity_factor > i).mean() for i in np.linspace(0, 1, 100)]


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[],y=[]))
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_days_eu,
                      line_shape='linear', line=dict(color="rgb(187,51,59)"), name="EU"))

        
        
        if click is not None:
            selected_countries = [point["location"] for point in click["points"]]
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year==year]
            for country in selected_countries:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]
                cum_days_dict[country] = [(daily_cp_dict[country]['capacity_factor'] > i).mean() for i in np.linspace(0, 1, 100)]

            for selected_country in selected_countries :
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100),
                              y=cum_days_dict[selected_country], line_shape='linear', name=selected_country))

        else:
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year==year]
            for country in all_country:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]
                cum_days_dict[country] = [(daily_cp_dict[country]['capacity_factor'] > i).mean() for i in np.linspace(0, 1, 100)]

            for selected_country in all_country:
                fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=cum_days_dict[selected_country], line_shape='linear',name=selected_country, line=dict(
                    color='gray'), opacity=0.2, showlegend = False))

        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        xaxis_title="Capacity factor threshold",
        yaxis_title="Proportion of days above threshold",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )

        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {}


    if fig_choice == 'Spatial correlation between daily capacity factor':


        if not click or len(click["points"])!=1 :
            return html.Div("Select one country on the left map to display the correlation between the LWP day distribution of this country and the LWP day distribution of the other European countries", style={"marginTop":"10vh","textAlign":"center"}), {"display": "none"}, {"display": "none"}

        df = daily_cp_df.corr()[click["points"][0]['location']]
        fig =px.choropleth_mapbox(df,
        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
        featureidkey='properties.ISO2',
        locations=df.index,        #column in dataframe
        color=df,  #dataframe
            zoom=1, center = {"lat": 56.4, "lon": 15.0},
            range_color=[0,1.0],
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            opacity = 0.5,
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Correlation',
            ))
        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",

        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )
        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}





    if fig_choice == 'Spatial correlation between LWP day distribution':

        if not click or len(click["points"])!=1 :
            return html.Div("Select one country on the left map to display the correlation between the LWP day distribution of this country and the LWP day distribution of the other European countries", style={"marginTop":"10vh","textAlign":"center"}), {"display": "none"}, {"display": "none"}

        df = daily_LWP_df.corr()[click["points"][0]['location']]
        fig =px.choropleth_mapbox(df,
        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
        featureidkey='properties.ISO2',
        locations=df.index,        #column in dataframe
        color=df,  #dataframe
            zoom=1, center = {"lat": 56.4, "lon": 15.0},
            range_color=[0,0.6],
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            opacity = 0.5,
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Correlation',
            ))
        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",

        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )
        return dcc.Graph(figure=fig,style={'height': '80vh'}), {"display": "none"}, {"display": "none"}




    if fig_choice == 'LWP events':

        if not click or len(click["points"])<1 :
            return [html.Div("Select a region on the left map to display the number of low wind power (LWP) events during the selected year",style={"marginBottom":"10vh","marginTop":"10vh","textAlign":"center"}),
        html.Br(),
        gif.GifPlayer(
        gif='assets/rec.gif',
        still='assets/rec.png',
        autoplay=True,
        )],{"display": "none"}, {"display": "none"}


        x = list(range(1, 9))
        mask = (daily_cp_eu_entire_period.timestamp.dt.year >= year) & (
            daily_cp_eu_entire_period.timestamp.dt.year <= year)
        y = [num_events(daily_cp_eu_entire_period[mask], i) for i in range(1, 9)]
        fig = go.Figure()
        fig2 = None
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.add_trace(go.Bar(x=x, y=y, marker_color="rgb(187,51,59)", name="EU",showlegend=True))

        if click is not None:
            mask = dict()
            for point in click["points"]:
                selected_country = point["location"]
                mask[selected_country] = (daily_cp_dict_entire_period[selected_country].timestamp.dt.year >= year) & (
                    daily_cp_dict_entire_period[selected_country].timestamp.dt.year <= year)
                fig.add_trace(go.Bar(name = selected_country,x=list(range(1, 9)), y=[num_events(daily_cp_dict_entire_period[selected_country][mask[selected_country]], i) for i in range(1, 9)]
                ))

            dfs = [daily_cp_dict_entire_period[selected_country][mask[selected_country]
                                              ].reset_index() for selected_country in mask.keys()]
            df_concat = pd.concat(dfs)
            by_row_index = df_concat.groupby(df_concat.timestamp)
            df_means = by_row_index.mean().reset_index()
            dummy_df = pd.DataFrame({
                "ds": pd.date_range(dfs[0].timestamp.values[0], dfs[0].timestamp.values[-1]),
                "value": (df_means.capacity_factor < 0.1).values.astype(int),

            })

            fig2 = calplot(dummy_df, x='ds', y='value', colorscale=[[0, "rgb(4,204,148)"], [1, "rgb(227,26,28)"]])
            fig2.update_xaxes(fixedrange=True)
            fig2.update_yaxes(fixedrange=True)


            fig2.update_layout(
        title = {
         'text': "Low wind power days in the selected area in {}".format(year),
         'y':0.09, # new
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top' # new
        }

    )



            if len(click["points"]) > 1 :
                fig.add_trace(go.Bar(
                    x=list(range(1, 9)), y=[num_events(df_means, i) for i in range(1, 9)], name="mean capacity<br>factor of selected<br>countries"
                ))

        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        yaxis_title="Number of occurences",
        xaxis_title="Minimum duration of the low wind power event (days)",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )




        return [
            dcc.Graph(figure=fig, style={'height': '50vh'}),
            (dcc.Graph(figure=fig2, style={'marginTop': '1vh'}) if fig2 else None)],  {"display":"none"}, {}

    if fig_choice == 'Daily capacity factor distribution':


        daily_cp_eu = df_daily_cp[df_daily_cp.year==year].groupby("timestamp").mean().reset_index()


        fig = go.Figure()
        fig.add_trace(go.Histogram(x=[],y=[]))
        fig.add_trace(go.Histogram(
            xbins=dict(
            size=0.05),
            name="EU",
            showlegend=True,
            marker=dict(color="rgb(187,51,59)"),
            x=daily_cp_eu.capacity_factor, histnorm='percent'))

        if click is not None:
            selected_countries = [point["location"] for point in click["points"]]
            cum_days_dict = dict()
            daily_cp_dict = dict()
            df_daily_cp_year = df_daily_cp[df_daily_cp.year==year]
            for country in selected_countries:
                daily_cp_dict[country] = df_daily_cp_year[df_daily_cp_year.country == country]
            
            for selected_country in selected_countries:
                fig.add_trace(go.Histogram(x=daily_cp_dict[selected_country].capacity_factor, histnorm='percent', name=selected_country, xbins=dict(size=0.05)))



        fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # title="Min, Max, Avg monthly capacity factor",
        xaxis_title="Capacity factor",
        yaxis_title="Percentage of days",
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        # )
    )

        return dcc.Graph(figure=fig, style={'height': '80vh'}), {"display": "none"}, {}


if __name__ == '__main__':
    app.run_server(port=8088, debug=True, use_reloader=True)