import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Circle, Ellipse, RegularPolygon, Polygon


def _prepare_topomap(electrodes, head_radius=0.2, **fig_kwargs):
    fig, ax = plt.subplots(**fig_kwargs)
    left_ear = Ellipse(electrodes[6], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    right_ear = Ellipse(electrodes[26], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    nose = Polygon([electrodes[4] + np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.]),
                     electrodes[15] + np.array([0, 0.02]),
                     electrodes[24]- np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.])],
                   facecolor="none", **fig_kwargs)
    ellipse = Ellipse(electrodes[1], height=head_radius, width=head_radius, facecolor="white", **fig_kwargs)
    ax.add_patch(nose)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    ax.add_patch(ellipse)
    ax.set_aspect("equal")
    return fig, ax

def plot_2d_topomap(coords, values, grid_res=100, cmap="coolwarm", margin=0.01, head_radius=0.15, **fig_kwargs):

    grid_x, grid_y = np.meshgrid(np.linspace(coords.min() - margin, coords.max() + margin, grid_res),
                                 np.linspace(coords.min() - margin, coords.max() + margin, grid_res))
    grid_z = griddata(coords, values, (grid_x, grid_y), method="cubic")

    fig, ax = _prepare_topomap(coords, head_radius=0.2, **fig_kwargs)
    # ellipse = Ellipse(coords[1], height=head_radius, width=head_radius, facecolor="none", **fig_kwargs)
    # ax.add_patch(ellipse)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap=cmap, vmin=-0.3, vmax=0.3)
    plt.colorbar(contour)
    ax.scatter(coords[:, 0], coords[:, 1], c=values, edgecolors="k", cmap=cmap)
    plt.show()
