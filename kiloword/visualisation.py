import numpy as np
from itertools import groupby
import plotly.graph_objects as go

COLORS = {
    "NOUN": "indianred",
   "PROPN": "violet",
   "VERB": "blue",
   "ADV": "darkcyan",
   "ADJ": "darkgreen",
   "ADP": "darkgoldenrod",
   "PRON": "darkviolet",
   "AUX": "dodgerblue",
   "DET": "grey",
    "NUM": "goldenrod",
    "X": "black",
    "SCONJ": "darkorange",
    "CCONJ": "lime",
    "INTJ": "olive"
}

def group_by_labels(labels: list):
    data_labels = [(labels[i], i) for i in range(len(labels))]
    result = {}
    data_labels.sort(key=lambda x: x[0])  # Sort the data based on the key

    for key, group in groupby(data_labels, key=lambda x: x[0]):
        result[key] = [item[1] for item in group]
    return result

def plot_features(features: np.array,
                  vocab: list,
                  color: list,
                  **fig_kwargs):
    n_samples, h = features.shape
    dict_labels = group_by_labels(color)
    assert (h <= 3)
    fig = go.Figure()
    labels_names = list(dict_labels.keys())

    for j, key in enumerate(dict_labels.keys()):
        if key in COLORS.keys():
            color = COLORS[key]
        ids = dict_labels[key]
        # Add trace
        if h == 3:
            fig.add_trace(go.Scatter3d(x=features[ids, 0],
                                     y=features[ids, 1],
                                     z=features[ids, 2],
                                     mode="markers+text",
                                     marker_color=color,
                                     name=key,
                                     text=[vocab[i] for i in ids],
                                     marker_size=8,
                                     textposition="bottom center",
                                     textfont_color=color))
        else:
            fig.add_trace(go.Scatter(x=features[ids, 0],
                                     y=features[ids, 1],
                                     mode="markers+text",
                                     marker_color=color,
                                     name=key,
                                     text=[vocab[i] for i in ids],
                                     marker_size=8,
                                     textposition="bottom center",
                                     textfont_color=color))

    # if h == 3:
    #     for i in range(n_samples):
    #         fig.add_trace(go.Scatter3d(x=[features[i][0]],
    #                                    y=[features[i][1]],
    #                                    z=[features[i][2]],
    #                                    mode="markers+text",
    #                                    marker_color=COLORS[color[i]],
    #                                    name=vocab[i],
    #                                    text=vocab[i],
    #                                    marker_size=8,
    #                                    textposition="bottom center",
    #                                    textfont_color=COLORS[color[i]]))
    # elif h == 2:
    #     for i in range(n_samples):
    #         fig.add_trace(go.Scatter(x=[features[i][0]],
    #                                    y=[features[i][1]],
    #                                    mode="markers+text",
    #                                    marker_color=COLORS[color[i]],
    #                                    name=vocab[i],
    #                                    text=vocab[i],
    #                                    marker_size=8,
    #                                    textposition="bottom center",
    #                                    textfont_color=COLORS[color[i]]))

    fig.update(**fig_kwargs)
    fig.show()
