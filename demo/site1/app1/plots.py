import datetime
import glob
import logging
import os

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import pandas as pd
logger = logging.getLogger(__name__)


def plot1d():
    df_cut = pd.read_csv('uploads/test.csv')
    # x_data = np.arange(0, 120, 0.1)
    # trace1 = go.Scatter(
    #     x=x_data,
    #     y=np.sin(x_data)
    # )
    #
    # data = [trace1]
    layout = go.Layout(
        autosize=True,
        # width=900,
        # height=500,
        )
    # )
    # fig = go.Figure(data=data, layout=layout)
    id = 135
    exp_data = df_cut[df_cut["original_id"] == id]
    x_axes = np.arange(0, exp_data.shape[0])
    rrs = exp_data["x"].to_numpy() / 1000
    y_target = exp_data["gb_net_predict"].to_numpy().astype(int)  # [50:]
    # y_target_cut = exp_data["cut_y"].to_numpy()#[50:]
    corr = exp_data["corrected"]
    font_size = 25
    fig = go.Figure([go.Scatter(x=x_axes, y=rrs, name="RR - intervals", marker=dict(color="blue"), line=dict(width=3)),
                     # error_x=dict(type='data', color="green", visible=True)),
                     go.Scatter(x=x_axes, y=y_target, name="detected covid", marker=dict(color="red"), line=dict(width=2.7)),
                     ], layout=layout)  #
    fig.update_layout(title={'text': f"Rec. {id}", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      font=dict(size=font_size),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                  font=dict(size=font_size)),
                      )
    fig.update_xaxes(title_text="Time", title_font=dict(size=font_size), tickfont=dict(size=font_size))
    fig.update_yaxes(title_text="RR-interval / Prediction", title_font=dict(size=font_size), tickfont=dict(size=font_size))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, image_width=2200, image_height=2200)
    # logger.info("Plotting number of points {}.".format(len(x_data)))
    return plot_div


