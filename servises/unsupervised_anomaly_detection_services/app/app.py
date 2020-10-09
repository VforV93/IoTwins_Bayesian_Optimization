# Copyright 2020-2020 Thales SIX GTS France
#
# Licensed under the Thales Inner Source Software License:
#
#   Version 1.1, InnerRestricted - Distribution Not Controlled
#
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://ecm.corp.thales/Livelink/livelink.exe?func=ll&objId=162631966&objAction=browse&viewType=1
# See the License for the specific language governing permissions and limitations under the License.


import dash
import dash_table
import dash_upload_components
import dash_core_components as dcc
import dash_html_components as html

from flask import request
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import os
import uuid
import re
import operator
import boto3
import numpy as np
import pandas as pd
import urllib.parse
import dill as pickle

import plotly.express as px

from adtk.detector import PcaAD, OutlierDetector, InterQuartileRangeAD, ThresholdAD, QuantileAD, GeneralizedESDTestAD, \
    PersistAD, LevelShiftAD, VolatilityShiftAD, AutoregressionAD, SeasonalAD, MinClusterDetector
from adtk.data import validate_series

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import base64
import datetime
import io

import cufflinks as cf

from datetime import datetime

from botocore.client import Config

import csv
from FIF import *

header = None

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets)
colors = {
    "graphBackground": "#F5F5F5",
    'background': '#111111',
    'text': '#7fe6ff'
}

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
upload_style = {
    'width': '100%',
    'height': '30px',
    'lineHeight': '30px',
    'borderWidth': '1px',
    'borderStyle': 'solid',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '15px auto'
}

Dictionary_projection = [['gaussian_wavelets', 'gaussian wavelets']
    , ['Dyadic_indicator', 'Dyadic indicator']
    , ['cosinus', 'cosinus']
    , ['Brownian_bridge', 'Brownian bridge ']
    , ['Brownian', 'Brownian ']
    , ['Multiresolution_linear', 'Multiresolution linear']
    , ['Self_local', 'Self local']
    , ['indicator_uniform', 'indicator uniform']
    , ['linear_indicator_uniform', 'linear indicator uniform']
                         ]

Dictionary_anomaly = [['threshold', 'Univariate time series: Anomaly detection based on thresholds']
    , ['quantile', 'Univariate time series: Anomaly detection based on quantile']
    , ['quantile_range', 'Univariate time series:Anomaly detection based on quantile range']
    , ['student_test', 'Univariate time series: Anomaly detection based on Student test']
    , ['persist', 'Univariate time series: Change-point detection based on recent observation']
    , ['level_shift', 'Univariate time series: Change-point detection based on level shift']
    , ['volatibility_shift', 'Univariate time series: Change-point detection based on variance shift']
    , ['seasonal', 'Univariate time series: Change in seasonal behaviour']
    , ['autoregression', 'Univariate time series: Change in autoregressive behaviour']
    , ['mincluster', 'Multivariate time series: Anomaly detection based on clustering']
    , ['outlierDetector', 'Multivariate time series: Anomaly detection based on local outlier factor model']
    , ['regressionDetector', 'Multivariate time series: Anomaly detection by tracking regressive error']
    , ['pcaDetector', 'Multivariate time series: Anomaly detection based on Principal Composant Analysis']

                      ]


# image_filename = 'logo.png' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())


def named_slider(id, min, max, value, step=None, marks=None, slider_type=dcc.Slider, hidden=False):
    div = html.Div([
        html.Div(id=f'name-{id}', hidden=hidden),
        slider_type(
            id=id,
            min=min,
            max=max,
            marks=marks,
            step=step,
            value=value,
        )],
        style={'margin': '25px 5px 30px 0px'},
        hidden=hidden,
        id=f'slidercontainer-{id}'
    )
    return div


@app.callback(Output('div-download-explore', 'children'),
              [Input('embedding-graph', 'selectedData'),
               Input('input-filename', 'value'),
               Input('feature-store', 'data')],
              [State('filename-store', 'data')])
def update_download_link_explore(select_data, user_input_filename, data, original_filename):
    if data and (user_input_filename or original_filename):
        df = pd.DataFrame(data)
        if select_data:
            selected_points = [point['pointIndex'] for point in select_data['points']]
            df = df.loc[selected_points].reindex()
            text = f"Download {len(selected_points)} out of {len(data)} points"
        else:
            text = f'Download all'

        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

        if user_input_filename:
            filename = user_input_filename + '.csv'
        else:
            filename = original_filename + '.csv'

        download_button = html.A(
            text,
            id='download-link-explore',
            download=filename,
            href=csv_string,
            target="_blank",
        )
        return download_button


@app.callback(Output('mapping-store', 'data'),
              [Input('upload-data', 'fileNames')])
def create_file_key_mapping(filenames):
    if filenames is not None:
        user_ip = get_user_ip()
        filepath = TEMP_STORAGE + filenames[-1]
        time_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename, ext = os.path.litext(os.path.basename(filepath))
        key = f'{filename}_{time_now}_{user_ip}.wav'
        return {'key': key, 'filepath': filepath, 'time': time_now}
    else:
        raise PreventUpdate


@app.callback(Output('input-filename', 'value'),
              [Input('upload-data', 'fileNames')])
def update_input_filename(filenames):
    if filenames is not None:
        filename = os.path.splitext(filenames[-1])[0]
        return filename


def parse_data2(contents, filename):
    print(contents)

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), index_col="Date", parse_dates=True, squeeze=True)

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), index_col="Date", parse_dates=True, squeeze=True)
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+', index_col="Date", parse_dates=True,
                squeeze=True)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


def parse_data(contents, filename):
    print(contents)
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=header)
            if header is None:
                name_columns = ["var_" + str(i) for i in df.columns]
                df.columns = name_columns

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


@app.callback(Output('Mygraph', 'figure'),
              [
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename')
              ])
def update_graph(contents, filename):
    fig = {
        'layout': go.Layout(
            plot_bgcolor=colors["graphBackground"],
            paper_bgcolor=colors["graphBackground"])
    }

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df_2 = df.T
        if header is None:
            name_columns_2 = ["serie_" + str(i) for i in df_2.columns]
            df_2.columns = name_columns_2

        df_2["Time"] = [i for i in range(len(df.columns))]
        df_2 = df_2.set_index(df_2.columns[len(df_2.columns) - 1])
        fig['data'] = df_2.iplot(asFigure=True, kind='scatter', mode='lines+markers', size=1)

        return fig['data']

    else:
        raise PreventUpdate


@app.callback(Output('output-data-upload', 'children'),
              [
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename')
              ])
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        if df.shape[0] > 10:
            df_3 = df.iloc[0:9]
        else:
            df_3 = df

        table = html.Div([
            # html.H5(filename),
            dash_table.DataTable(
                data=df_3.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in df_3.columns]
            )  # ,
            # html.Hr()#,
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #    'whiteSpace': 'pre-wrap',
            #    'wordBreak': 'break-all'
            # })
        ])

        return table
    else:
        raise PreventUpdate


#
#
# @app.callback(Output('output-data-upload', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename'),
#                State('upload-data', 'last_modified')])
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children


@app.callback([
    Output('embedding-graph', 'figure'),
    Output('embedding-graph-2', 'figure'),
    Output('embedding-graph-3', 'figure'),
    Output('output-data-upload-4', 'children')

],

    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('apply-button', 'n_clicks')],
    [State('algorithm-dictionary', 'value'),
     State('algorithm-ntree', 'value'),
     State('algorithm-alpha', 'value'),
     State('algorithm-proportion', 'value')
     ])
def train_fif(contents, filename, n_clicks, dictionary, ntrees, alpha, proportion):
    if n_clicks > 0:
        temppath = "fif_" + str(alpha) + "_" + str(proportion) + "_" + str(ntrees) + "_" + dictionary + ".pkl"
        proportion = proportion * 2 / 100
        alpha = alpha / 10
        contents = contents[0]
        filename = filename[0]

        df = parse_data(contents, filename)

        X_train = np.array(df, dtype=float)

        time = np.linspace(0, 1, X_train.shape[1])

        if dictionary == "Dyadic_indicator":
            alpha = 1
        if dictionary == "Multiresolution_linear":
            alpha = 1
        if dictionary == "Self_local":
            alpha = 1
        if dictionary == "indicator_uniform":
            alpha = 1
        if dictionary == "linear_indicator_uniform":
            alpha = 1

        F = FIForest(X_train, ntrees=ntrees, time=time, subsample_size=X_train.shape[0],
                     D=dictionary, innerproduct="auto", alpha=alpha)

        S = F.compute_paths()
        predict = F.predict_label(S, proportion)

        fig = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }
        fig_2 = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }

        fig_3 = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }

        df["serie"] = ["serie_" + str(i) for i in range(len(X_train))]
        df_inter_1 = df[predict == 1]
        df_inter_2 = df[predict != 1]
        column_1 = df_inter_1["serie"]
        column_2 = df_inter_2["serie"]
        df_inter_1 = df_inter_1.loc[:, df_inter_1.columns != "serie"]
        df_inter_2 = df_inter_2.loc[:, df_inter_2.columns != "serie"]
        df_final_2 = df_inter_2.T
        df_final_1 = df_inter_1.T
        df_final_2.columns = column_2
        df_final_1.columns = column_1
        df_final_2["Time"] = [i for i in range(len(df_inter_2.columns))]
        df_final_2 = df_final_2.set_index(df_final_2.columns[len(df_final_2.columns) - 1])
        df_final_1["Time"] = [i for i in range(len(df_inter_1.columns))]
        df_final_1 = df_final_1.set_index(df_final_1.columns[len(df_final_1.columns) - 1])
        fig['data'] = df_final_1.iplot(asFigure=True, kind='scatter', mode='lines+markers', size=1)
        fig_2['data'] = df_final_2.iplot(asFigure=True, kind='scatter', mode='lines+markers', size=1)

        fig_3['data'] = pd.DataFrame(S).iplot(asFigure=True, kind='scatter', size=1)

        S2 = pd.DataFrame()
        S2["serie"] = ["serie_" + str(i) for i in range(len(X_train))]
        S2["score"] = S
        S2 = S2.sort_values(by='score', ascending=False)

        if len(S2) > 13:
            S2 = S2.iloc[0:12, :]

        table = html.Div([

            dash_table.DataTable(
                data=S2.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in S2.columns]
            ),

        ])

        list_pickle = open(temppath, 'wb')
        pickle.dump(F, list_pickle)
        list_pickle.close()

        return fig['data'], fig_2['data'], fig_3['data'], table
    else:
        raise PreventUpdate


@app.callback([
    Output('embedding-graph-test', 'figure'),
    Output('embedding-graph-test-2', 'figure'),
    Output('embedding-graph-test-3', 'figure'),
    Output('output-data-upload-test-5', 'children')

],

    [Input('upload-data-test', 'contents'),
     Input('upload-data-test', 'filename'),
     Input('apply-button-test', 'n_clicks')],
    [State('algorithm-dictionary', 'value'),
     State('algorithm-ntree', 'value'),
     State('algorithm-alpha', 'value'),
     State('algorithm-proportion', 'value'),
     State('algorithm-proportion-2', 'value')
     ])
def test_fif(contents, filename, n_clicks, dictionary, ntrees, alpha, proportion, proportion_2):
    if n_clicks > 0:
        temppath = "fif_" + str(alpha) + "_" + str(proportion) + "_" + str(ntrees) + "_" + dictionary + ".pkl"
        proportion = proportion * 2 / 100
        proportion_2 = proportion_2 * 2 / 100

        alpha = alpha / 10
        contents = contents[0]
        filename = filename[0]

        # temppath = "fif_" + str(alpha) + "_" + proportion + "_" + ntrees + "_" + dictionary + ".pkl"
        list_unpickle = open(temppath, 'rb')

        # load the unpickle object into a variable
        F = pickle.load(list_unpickle)
        list_unpickle.close()

        df = parse_data(contents, filename)

        X_train = np.array(df, dtype=float)

        time = np.linspace(0, 1, X_train.shape[1])

        S = F.compute_paths(X_train)
        predict = F.predict_label(S, proportion_2)

        fig = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }
        fig_2 = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }

        fig_3 = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }

        df["serie"] = ["serie_" + str(i) for i in range(len(X_train))]
        df_inter_1 = df[predict == 1]
        df_inter_2 = df[predict != 1]
        column_1 = df_inter_1["serie"]
        column_2 = df_inter_2["serie"]
        df_inter_1 = df_inter_1.loc[:, df_inter_1.columns != "serie"]
        df_inter_2 = df_inter_2.loc[:, df_inter_2.columns != "serie"]
        df_final_2 = df_inter_2.T
        df_final_1 = df_inter_1.T
        df_final_2.columns = column_2
        df_final_1.columns = column_1
        df_final_2["Time"] = [i for i in range(len(df_inter_2.columns))]
        df_final_2 = df_final_2.set_index(df_final_2.columns[len(df_final_2.columns) - 1])
        df_final_1["Time"] = [i for i in range(len(df_inter_1.columns))]
        df_final_1 = df_final_1.set_index(df_final_1.columns[len(df_final_1.columns) - 1])
        fig['data'] = df_final_1.iplot(asFigure=True, kind='scatter', mode='lines+markers', size=1)
        fig_2['data'] = df_final_2.iplot(asFigure=True, kind='scatter', mode='lines+markers', size=1)

        fig_3['data'] = pd.DataFrame(S).iplot(asFigure=True, kind='scatter', size=1)

        S2 = pd.DataFrame()
        S2["serie"] = ["serie_" + str(i) for i in range(len(X_train))]
        S2["score"] = S
        S2 = S2.sort_values(by='score', ascending=False)

        if len(S2) > 13:
            S2 = S2.iloc[0:12, :]

        table = html.Div([

            dash_table.DataTable(
                data=S2.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in S2.columns]
            ),

        ])

        return fig['data'], fig_2['data'], fig_3['data'], table
    else:
        raise PreventUpdate


@app.callback([
    Output('anomaly-plot', 'figure')],

    [Input('upload-data-anomaly', 'contents'),
     Input('upload-data-anomaly', 'filename'),
     Input('apply-button-anomaly', 'n_clicks')],
    [State('anomaly-objective', 'value'),
     State('threshold-min', 'value'),
     State('threshold-max', 'value'),
     State('quantile-min', 'value'),
     State('quantile-max', 'value'),
     State('range-quantile-coefficient', 'value'),
     State('student-coefficient', 'value'),
     State('cpd-window', 'value'),
     State('cpd-coeff', 'value'),
     State('cpd-side', 'value'),
     State('seasonal-coeff', 'value'),
     State('seasonal-side', 'value'),
     State('autoreg-steps', 'value'),
     State('autoreg-size', 'value'),
     State('autoreg-coeff', 'value')
     ])
def anomaly_detection(contents
                      , filename
                      , n_clicks
                      , anomaly_objective
                      , threshold_min
                      , threshold_max
                      , quantile_min
                      , quantile_max
                      , range_quantile_coeff
                      , student_coeff
                      , cpd_window
                      , cpd_coeff
                      , cpd_side
                      , seasonal_coeff
                      , seasonal_side
                      , autoreg_steps
                      , autoreg_size
                      , autoreg_coeff

                      ):
    if n_clicks > 0:

        contents = contents[0]
        filename = filename[0]

        df = parse_data2(contents, filename)

        print(df)
        s = validate_series(df)

        fig = {
            'layout': go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"])
        }

        if anomaly_objective == 'threshold':
            iqr_ad = ThresholdAD(high=threshold_max, low=threshold_min)
            anomalies = iqr_ad.detect(s)

            test_3 = pd.DataFrame()

            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'quantile':
            iqr_ad = QuantileAD(high=quantile_max, low=quantile_min)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'quantile_range':
            iqr_ad = InterQuartileRangeAD(c=range_quantile_coeff)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'student_test':
            iqr_ad = GeneralizedESDTestAD(alpha=student_coeff)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'persist':
            iqr_ad = PersistAD(c=cpd_coeff, side=cpd_side)
            iqr_ad.window = cpd_window

            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values.astype(np.str)

            test_3 = test_3.query('Anomaly!="nan"')

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'level_shift':
            iqr_ad = LevelShiftAD(c=cpd_coeff, side=cpd_side, window=cpd_window)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values.astype(np.str)

            test_3 = test_3.query('Anomaly!="nan"')

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'volatibility_shift':
            iqr_ad = VolatilityShiftAD(c=cpd_coeff, side=cpd_side, window=cpd_window)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values.astype(np.str)

            test_3 = test_3.query('Anomaly!="nan"')

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'seasonal':
            iqr_ad = SeasonalAD(c=seasonal_coeff, side=seasonal_side)
            print(s.head())
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values.astype(np.str)

            test_3 = test_3.query('Anomaly!="nan"')

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()

        if anomaly_objective == 'autoregression':
            iqr_ad = AutoregressionAD(n_steps=autoreg_steps, step_size=autoreg_size, c=autoreg_coeff)
            anomalies = iqr_ad.fit_detect(s)
            test_3 = pd.DataFrame()
            test_3['Date'] = df.index
            test_3['Serie'] = s.values
            test_3['Anomaly'] = anomalies.values.astype(np.str)

            test_3 = test_3.query('Anomaly!="nan"')

            fig = px.scatter(x=test_3['Date'], y=test_3['Serie'], color=test_3['Anomaly'])
            return fig.show()








    else:
        raise PreventUpdate


def generate_layout():
    session_id = str(uuid.uuid4())
    div = html.Div(
        style={'backgroundColor': 'black'},
        children=[

            html.Div(className="Row", children=[
                html.Div(
                    [
                        # html.Div(className="seven columns", children=[

                        html.Div(

                            [
                                html.P(""),
                                html.H3("TIME SERIES ANOMALY DETECTION"
                                        , className="app__header__title--grey"
                                        , style={'color': "white"}
                                        ),
                                html.P(
                                    "This app uses Functional Isolation Forest to detect abnormal time series, adtk to detect outliers in time series and prophet to forecast time series",
                                    className="app__header__title--grey", style={'color': '#b1b5ba'}
                                ),
                            ],
                            className="app__header__desc",
                        ),
                        # ]),
                        # html.Div(className="five columns", children=[

                        # ]),

                    ],
                    className="app__header",
                ),
            ]),
            dcc.Tabs(id='tabs', children=[

                dcc.Tab(label='Detect abnormal time series', children=[
                    html.Div(
                        className="container",
                        style={
                            'width': '92%',
                            'max-width': 'none',
                            'font-size': '1.5rem',
                            'padding': '10px 10px'
                        },
                        children=[
                            html.Div(className="row", children=[
                                html.H5("TIME SERIES DATASET"
                                        , className="app__header__title--grey"
                                        , style={
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'color': "white"
                                    }
                                        ),
                            ]
                                     ),

                            html.Div([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'color': '#b1b5ba'
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                                dcc.Graph(id='Mygraph'),
                                html.Div(id='output-data-upload')
                            ]),
                            html.Hr(),

                            html.Div(id='functional-forest', style={'display': 'none'}),
                            html.Div(id='score-training', style={'display': 'none'}),
                            html.Div(id='predict-training', style={'display': 'none'}),
                            html.Div(id='dataframe', style={'display': 'none'}),
                            dcc.Store(id='forest', storage_type='memory'),

                            # Body
                            html.Div(className="row", children=[
                                html.H5("TRAINING SET"
                                        , className="app__header__title--grey"
                                        , style={
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'color': 'white'
                                    }
                                        ),
                            ]
                                     ),
                            html.Div(className="row", children=[

                                html.Div(className="eight columns", children=[
                                    html.Div(className="row", children=[
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph'
                                            ),
                                        ]
                                                 ),
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph-2'
                                            ),
                                        ]
                                                 ),
                                    ]
                                             ),
                                    html.P(""),
                                    html.Div(className="row", children=[
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph-3'
                                            ),
                                        ]
                                                 ),
                                        html.Div(className="six columns", children=[
                                            html.Div(id='output-data-upload-4'),
                                        ]
                                                 ),
                                    ]
                                             ),

                                    # html.Div(className='row', children=[
                                    #
                                    #     html.Div(className='seven columns', children=[
                                    #         dcc.Input(id='input-filename', type='text', debounce=True,
                                    #                   placeholder='Filename',
                                    #                   style={'display': 'inline-block', 'width': '300px',
                                    #                          'margin-right': '30px'}),
                                    #         html.Div(id='div-download-explore',
                                    #                  style={'display': 'inline-block', 'margin-right': '60px'}),
                                    #     ]),
                                    #     html.Div(className='five columns', children=[
                                    #         html.Div(id='div-report-selection')
                                    #     ])
                                    # ])
                                ]),
                                html.Div(className="four columns", children=[
                                    html.Div([
                                        html.Button('Apply', id='apply-button', style=upload_style),
                                    ], style={'columnCount': 2, 'color': '#b1b5ba'}),
                                    html.P("Choose the dictionary",
                                           className="app__header__title--grey", style={'color': '#b1b5ba'}),
                                    dcc.Dropdown(
                                        id='algorithm-dictionary',
                                        options=[{'label': label, 'value': value} for value, label in
                                                 Dictionary_projection],
                                        placeholder='Select Dictionary',
                                        value='gaussian_wavelets'
                                    ),
                                    html.P("Choose the size of the Forests",
                                           className="app__header__title--grey", style={'color': '#b1b5ba'}),
                                    named_slider(
                                        id='algorithm-ntree',
                                        min=10,
                                        max=1000,
                                        step=10,
                                        marks={i * 100: f'{i * 100}' for i in range(0, 10)},
                                        value=100
                                    ),
                                    html.P("Choose the scalar products",
                                           className="app__header__title--grey", style={'color': '#b1b5ba'}),
                                    named_slider(
                                        id='algorithm-alpha',
                                        min=0,
                                        max=11,
                                        marks={i: f'{i / 10}' for i in range(0, 11)},

                                        value=5
                                    ),
                                    html.P("Choose the percentage of abnormal time series targeted",
                                           className="app__header__title--grey", style={'color': '#b1b5ba'}),
                                    named_slider(
                                        id='algorithm-proportion',
                                        min=0,
                                        max=20,

                                        marks={i: f'{i * 2}' for i in range(0, 20)},

                                        value=2
                                    ),

                                ]),
                            ]),

                            html.Hr(),

                            html.Div(className="row", children=[
                                html.H5("TESTING SET"
                                        , className="app__header__title--grey"
                                        , style={
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'color': 'white'
                                    }
                                        ),
                            ]
                                     ),

                            html.Div(className="row", children=[

                                html.Div(className="eight columns", children=[
                                    html.Div(className="row", children=[
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph-test'
                                            ),
                                        ]
                                                 ),
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph-test-2'
                                            ),
                                        ]
                                                 ),
                                    ]
                                             ),
                                    html.P(""),
                                    html.Div(className="row", children=[
                                        html.Div(className="six columns", children=[
                                            dcc.Graph(
                                                id='embedding-graph-test-3'
                                            ),
                                        ]
                                                 ),
                                        html.Div(className="six columns", children=[
                                            html.Div(id='output-data-upload-test-5'),
                                        ]
                                                 ),
                                    ]
                                             ),

                                    # html.Div(className='row', children=[
                                    #
                                    #     html.Div(className='seven columns', children=[
                                    #         dcc.Input(id='input-filename-test-2', type='text', debounce=True,
                                    #                   placeholder='Filename',
                                    #                   style={'display': 'inline-block', 'width': '300px',
                                    #                          'margin-right': '30px'}),
                                    #         html.Div(id='div-download-explore-test-2',
                                    #                  style={'display': 'inline-block', 'margin-right': '60px'}),
                                    #     ]),
                                    #     html.Div(className='five columns', children=[
                                    #         html.Div(id='div-report-selection-test-2')
                                    #     ])
                                    # ])
                                ]),
                                html.Div(className="four columns", children=[
                                    html.Div([
                                        dcc.Upload(
                                            id='upload-data-test',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '30px',
                                                'lineHeight': '30px',
                                                'borderWidth': '1px',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                                , 'color': '#b1b5ba'
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=True
                                        ),

                                        html.Button('Apply', id='apply-button-test', style=upload_style),
                                    ], style={'columnCount': 2, 'color': '#b1b5ba'}),

                                    html.P("Choose the percentage of abnormal time series targeted",
                                           className="app__header__title--grey", style={'color': '#b1b5ba'}),
                                    named_slider(
                                        id='algorithm-proportion-2',
                                        min=0,
                                        max=20,

                                        marks={i: f'{i * 2}' for i in range(0, 20)},

                                        value=2
                                    ),

                                ]),
                            ]),
                        ]
                    )
                ]),
                dcc.Tab(label='Detect anomaly in time series', children=[

                    html.Div(id='anomaly-plot', style={'display': 'none'}),

                    html.Div(className="Row", children=[
                        html.Div(className="four columns", children=[

                            dcc.Upload(
                                id='upload-data-anomaly',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '30px',
                                    'lineHeight': '30px',
                                    'borderWidth': '1px',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px', 'color': '#b1b5ba'
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),

                        ]),
                        html.Div(className="four columns", children=[

                            html.P(""),
                            html.P(""),

                            dcc.Dropdown(
                                id='anomaly-objective',
                                options=[{'label': label, 'value': value} for value, label in Dictionary_anomaly],
                                placeholder='Select approach',
                                value='quantile_range'
                            ),
                        ]),
                        html.Div(className="four columns", children=[

                            html.Button('Apply', id='apply-button-anomaly', style={'color': '#b1b5ba'}),
                        ])
                    ]),

                    html.Hr(),
                    html.P(""),

                    html.Div(className="Row", children=[
                        html.Hr(),
                    ]
                             ),

                    html.Div(className="Row", children=[
                        html.H5("Univariate Time Series: Anomaly detection"
                                , className="app__header__title--grey"
                                , style={
                                'textAlign': 'center',
                                'margin': '10px', 'color': 'white'
                            })

                    ]
                             ),
                    html.Div(className="Row", children=[
                        html.Div(className="three columns", children=[
                            html.H6("Threshold"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px', 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="six columns", children=[
                                    html.P("Min value", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='threshold-min', type='number', debounce=True,
                                              value=0.0,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                                html.Div(className="six columns", children=[

                                    html.P("Max value", style={'color': '#b1b5ba'}),

                                    dcc.Input(id='threshold-max', type='number', debounce=True,
                                              value=30.0,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                            ]),

                        ]),

                        html.Div(className="three columns", children=[
                            html.H6("Quantile"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px'
                                    , 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="six columns", children=[
                                    html.P("Min quantile", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='quantile-min', type='number', debounce=True,
                                              value=0.01,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                                html.Div(className="six columns", children=[

                                    html.P("Max quantile", style={'color': '#b1b5ba'}),

                                    dcc.Input(id='quantile-max', type='number', debounce=True,
                                              value=0.99,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                            ]),

                        ]),

                        html.Div(className="three columns", children=[
                            html.H6("Quantile range"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px'
                                    , 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="six columns", children=[
                                    html.P("Coefficient", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='range-quantile-coefficient', type='number', debounce=True,
                                              value=1.5,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                            ]),

                        ]),

                        html.Div(className="three columns", children=[
                            html.H6("Student test"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px', 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="six columns", children=[
                                    html.P("Coefficient", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='student-coefficient', type='number', debounce=True,
                                              value=0.3,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                            ]),

                        ]),

                    ]
                             ),

                    html.Hr(),
                    html.P(""),

                    html.Div(className="Row", children=[
                        html.Hr(),
                        html.P(""),
                    ]
                             ),

                    html.Div(className="Row", children=[
                        html.Hr(),
                        html.P(""),
                    ]
                             ),

                    html.Div(className="Row", children=[
                        html.P(""),
                        html.P(""),
                        html.Hr(),
                        html.P(""),
                        html.H5("Univariate Time Series: Change-point detection"
                                , className="app__header__title--grey"
                                , style={
                                'textAlign': 'center',
                                'margin': '10px', 'color': 'white'
                            })

                    ]
                             ),
                    html.Div(className="Row", children=[
                        html.Div(className="four columns", children=[
                            html.P("Window", style={'color': '#b1b5ba'}),
                            dcc.Input(id='cpd-window', type='number', debounce=True,
                                      value=1,
                                      style={'display': 'inline-block', 'width': '100px',
                                             'margin-right': '30px'})

                        ]),
                        html.Div(className="four columns", children=[
                            html.P("Coefficient", style={'color': '#b1b5ba'}),
                            dcc.Input(id='cpd-coeff', type='number', debounce=True,
                                      value=6.0,
                                      style={'display': 'inline-block', 'width': '100px',
                                             'margin-right': '30px'})

                        ]),
                        html.Div(className="four columns", children=[
                            html.P("Side", style={'color': '#b1b5ba'}),
                            dcc.Dropdown(
                                id='cpd-side',
                                options=[{'label': label, 'value': value} for value, label in [["both", "Both"]
                                    , ["positive", "Positive"]
                                    , ["negative", "Negative"]
                                                                                               ]
                                         ],

                                value='both',
                                style={'width': '200px'}
                            ),

                        ])

                    ]
                             ),

                    html.Hr(),
                    html.P(""),

                    html.Div(className="Row", children=[
                        html.P(""),
                        html.Hr(),

                    ]
                             ),

                    html.Div(className="Row", children=[
                        html.Hr(),
                        html.P(""),
                        html.H5("Univariate Time Series: Behaviour change detection"
                                , className="app__header__title--grey"
                                , style={
                                'textAlign': 'center',
                                'margin': '10px', 'color': 'white'
                            })

                    ]
                             ),

                    html.Div(className="Row", children=[
                        html.Div(className="six columns", children=[
                            html.H6("Seasonal change"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px', 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="six columns", children=[
                                    html.P("Coefficient", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='seasonal-coeff', type='number', debounce=True,
                                              value=6.0,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                                html.Div(className="six columns", children=[

                                    html.P("Side", style={'color': '#b1b5ba'}),

                                    dcc.Dropdown(
                                        id='seasonal-side',
                                        options=[{'label': label, 'value': value} for value, label in [["both", "Both"]
                                            , ["positive", "Positive"]
                                            , ["negative", "Negative"]
                                                                                                       ]
                                                 ],

                                        value='both',
                                        style={'width': '200px'}
                                    ),
                                ]),

                            ]),

                        ]),

                        html.Div(className="six columns", children=[
                            html.H6("Autoregressive change"
                                    , className="app__header__title--grey"
                                    , style={
                                    'textAlign': 'center',
                                    'margin': '10px', 'color': 'white'
                                }
                                    ),

                            html.Div(className="Row", children=[
                                html.Div(className="four columns", children=[
                                    html.P("Nb steps", style={'color': '#b1b5ba'}),
                                    dcc.Input(id='autoreg-steps', type='number', debounce=True,
                                              placeholder=14,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                                html.Div(className="four columns", children=[

                                    html.P("Step size", style={'color': '#b1b5ba'}),

                                    dcc.Input(id='autoreg-size', type='number', debounce=True,
                                              placeholder=24,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),
                                html.Div(className="four columns", children=[

                                    html.P("Coefficient", style={'color': '#b1b5ba'}),

                                    dcc.Input(id='autoreg-coeff', type='number', debounce=True,
                                              value=3.0,
                                              style={'display': 'inline-block', 'width': '100px',
                                                     'margin-right': '30px'}),

                                ]),

                            ]),

                        ]),

                    ]
                             ),

                ]),
                dcc.Tab(label='Forecasting', children=[
                    html.Div(
                        children=html.Div(
                            className="container",
                            style={
                                'width': '95%',
                                'max-width': 'none',
                                'font-size': '1.5rem',
                                'padding': '10px 30px'
                            },
                            children=[
                                html.Div(id='features-container'),
                                html.Div(id='div-download-table')
                            ]
                        )
                    ),
                ]),
                dcc.Tab(label='Help', children=[
                    html.Div(
                        style={
                            'width': '75%',
                            'margin': '30px auto',
                        },
                        children=dcc.Markdown()
                    )
                ]),
            ]),

            html.Div(style={'backgroundColor': 'black'},

                     children=[
                         html.P(""),
                         html.P(""),
                         html.P(""),

                         html.Div(className="one columns", children=[
                             html.P("")
                         ]
                                  ),
                         html.Div(className="two columns", children=[
                             html.Img(width=500,

                                      src=app.get_asset_url("logo.png")
                                      )

                             # html.Img(src='data:image/png;base64,{}'.format(encoded_image)
                             #         , className="app__menu__img"
                             #         )

                         ]
                                  )
                     ])
        ])
    return div


app.layout = generate_layout()

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
