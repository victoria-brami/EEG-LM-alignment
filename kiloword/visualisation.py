import numpy as np
import plotly.graph_objects as go

COLORS = {
    "PROPN": "pink",
    "VERB": "blue",
    "ADV": "steel",
    "NOUN": "indianred",
    "PRON": "violet",
    "AUX": "cornflowerblue",
    "DET": "grey",
    "ADJ": "forrestgreen",
    "ADP": "",
    "NUM": "goldenrod"
}


def plot_features(features: np.array,
                  vocab: list,
                  color: dict = COLORS,
                  **fig_kwargs):
    n_samples, h = features.shape
    assert (h <= 3)
    fig = go.Figure()

    if h == 3:
        for i in range(n_samples):
            fig.add_trace(go.Scatter3d(x=[features[i][0]],
                                       y=[features[i][1]],
                                       z=[features[i][2]],
                                       mode="markers+text",
                                       marker_color=color[vocab[i]],
                                       name=vocab[i],
                                       text=vocab[i],
                                       marker_size=8,
                                       textposition="bottom center",
                                       textfont_color=color[vocab[i]]))
    elif h == 2:
        for i in range(n_samples):
            fig.add_trace(go.Scatter(x=[features[i][0]],
                                       y=[features[i][1]],
                                       mode="markers+text",
                                       marker_color=color[vocab[i]],
                                       name=vocab[i],
                                       text=vocab[i],
                                       marker_size=8,
                                       textposition="bottom center",
                                       textfont_color=color[vocab[i]]))

    fig.update(**fig_kwargs)
    fig.show()
