import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from scipy.interpolate import griddata


def _prepare_kiloword_topomap(electrodes, head_radius=0.2, cols=None, rows=None, size=3,  ax=None, **fig_kwargs):
    left_ear = Ellipse(electrodes[6], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    right_ear = Ellipse(electrodes[26], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    nose = Polygon([electrodes[4] + np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.]),
                    electrodes[15] + np.array([0, 0.02]),
                    electrodes[24] - np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.])],
                   facecolor="none", **fig_kwargs)
    ellipse = Ellipse(electrodes[1], height=head_radius, width=head_radius, facecolor="white", **fig_kwargs)

    if (rows, cols) in [(1, 1), (None, None)]:
        if ax is None:
            fig, ax = plt.subplots(figsize=(size * cols, size * rows), **fig_kwargs)
        ax.add_patch(nose)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
    elif rows == 1 and cols > 1:
        if ax is None:
            fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows), **fig_kwargs)
        for j in range(cols):
            left_ear = Ellipse(electrodes[6], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                               **fig_kwargs)
            right_ear = Ellipse(electrodes[26], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                **fig_kwargs)
            nose = Polygon([electrodes[4] + np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.]),
                            electrodes[15] + np.array([0, 0.02]),
                            electrodes[24] - np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.])],
                           facecolor="none", **fig_kwargs)
            ellipse = Ellipse(electrodes[1], height=head_radius, width=head_radius, facecolor="white", **fig_kwargs)

            ax[j].add_patch(nose)
            ax[j].add_patch(left_ear)
            ax[j].add_patch(right_ear)
            ax[j].add_patch(ellipse)
            ax[j].set_aspect("equal")
    else:
        if ax is None:
            fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows), **fig_kwargs)
        for i in range(rows):
            for j in range(cols):
                left_ear = Ellipse(electrodes[6], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                   **fig_kwargs)
                right_ear = Ellipse(electrodes[26], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                    **fig_kwargs)
                nose = Polygon([electrodes[4] + np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.]),
                                electrodes[15] + np.array([0, 0.02]),
                                electrodes[24] - np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.])],
                               facecolor="none", **fig_kwargs)
                ellipse = Ellipse(electrodes[1], height=head_radius, width=head_radius, facecolor="white", **fig_kwargs)

                ax[i, j].add_patch(nose)
                ax[i, j].add_patch(left_ear)
                ax[i, j].add_patch(right_ear)
                ax[i, j].add_patch(ellipse)
                ax[i, j].set_aspect("equal")
    return fig, ax

def _prepare_ubira_topomap(electrodes, head_radius=0.2, cols=None, rows=None, size=3, axs=None, **fig_kwargs):


    if (rows, cols) in [(1, 1), (None, None)]:
        left_ear = Ellipse(electrodes[8], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
        right_ear = Ellipse(electrodes[25], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                            **fig_kwargs)
        nose = Polygon([electrodes[0] + np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.]),
                        electrodes[34] + np.array([0, 0.02]),
                        electrodes[31] - np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.])],
                       facecolor="none", **fig_kwargs)
        col = "white" if axs is None else "none"
        ellipse = Ellipse(electrodes[23], height=head_radius, width=head_radius, facecolor=col, **fig_kwargs)
        if axs is None:
            fig, ax = plt.subplots(figsize=(size * cols, size * rows), **fig_kwargs)
        else:
            ax = axs
        ax.add_patch(nose)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
    elif rows == 1 and cols > 1:
        if axs is None:
            fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows), **fig_kwargs)
        else:
            ax = axs
        for j in range(cols):
            left_ear = Ellipse(electrodes[8], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                               **fig_kwargs)
            right_ear = Ellipse(electrodes[25], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                **fig_kwargs)
            nose = Polygon([electrodes[0] + np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.]),
                            electrodes[34] + np.array([0, 0.02]),
                            electrodes[31] - np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.])],
                           facecolor="none", **fig_kwargs)
            col = "white" if axs is None else "none"
            ellipse = Ellipse(electrodes[23], height=head_radius, width=head_radius, facecolor=col, **fig_kwargs)
            ax[j].add_patch(nose)
            ax[j].add_patch(left_ear)
            ax[j].add_patch(right_ear)
            ax[j].add_patch(ellipse)
            ax[j].set_aspect("equal")
    else:
        if axs is None:
            fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows), **fig_kwargs)
        else:
            ax = axs
        for i in range(rows):
            for j in range(cols):
                left_ear = Ellipse(electrodes[8], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                   **fig_kwargs)
                right_ear = Ellipse(electrodes[25], height=head_radius / 4, width=head_radius / 8, facecolor="none",
                                    **fig_kwargs)
                nose = Polygon([electrodes[0] + np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.]),
                                electrodes[34] + np.array([0, 0.05]),
                                electrodes[31] - np.array([abs(electrodes[0, 0] - electrodes[34, 0]) / 3, 0.])],
                               facecolor="none", **fig_kwargs)
                col = "white" if axs is None else "none"
                ellipse = Ellipse(electrodes[23], height=head_radius, width=head_radius, facecolor=col, **fig_kwargs)
                if axs is None:
                    ax[i, j].add_patch(nose)
                ax[i, j].add_patch(left_ear)
                ax[i, j].add_patch(right_ear)
                ax[i, j].add_patch(ellipse)
                ax[i, j].set_aspect("equal")
    if axs is not None:
        return ax
    return fig, ax


def _prepare_topomap(electrodes, dataname, cols=None, rows=None, size=3, axs=None, **fig_kwargs):

    if dataname == "kiloword":
        head_radius = np.linalg.norm(electrodes[25, :2] - electrodes[8, :2])
        return _prepare_kiloword_topomap(electrodes, head_radius, cols, rows, size, axs=axs, **fig_kwargs)
    elif dataname == "ubira":
        head_radius = np.linalg.norm(electrodes[25, :2] - electrodes[8, :2])
        return _prepare_ubira_topomap(electrodes, head_radius, cols, rows, size, axs=axs, **fig_kwargs)

def plot_2d_topomap(coords, values, dataname,
                    grid_res=100, cmap="coolwarm",
                    margin=0.01,
                    rows=1, cols=1, size=3,
                    coords_name=None,
                    subfig_name=None,
                    savepath=None,
                    dpi=100,
                    vmin=-0.5,
                    vmax=0.5,
                    title=None,
                    **fig_kwargs):
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.bottom"] = False

    grid_x, grid_y = np.meshgrid(np.linspace(coords.min() - margin, coords.max() + margin, grid_res),
                                 np.linspace(coords.min() - margin, coords.max() + margin, grid_res))

    fig, ax = _prepare_topomap(coords, dataname, rows=rows, cols=cols, size=size, **fig_kwargs)
    list_contours = []

    if (rows, cols) in [(1, 1), (None, None)]:
        print(rows, cols, len(values), len(values[0]), len(grid_x), len(grid_y), len(coords_name), len(subfig_name))
        grid_z = griddata(coords, values, (grid_x, grid_y), method="cubic")

        contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap=cmap, vmin=-0.3, vmax=0.3)
        plt.colorbar(contour)
        ax.scatter(coords[:, 0], coords[:, 1], c=values, edgecolors="k", cmap=cmap)

    elif rows == 1 and cols > 1:
        for j in range(cols):
            grid_z = griddata(coords, values[j], (grid_x, grid_y), method="cubic")
            grid_x, grid_y = np.meshgrid(np.linspace(coords.min() - margin, coords.max() + margin, grid_res),
                                         np.linspace(coords.min() - margin, coords.max() + margin, grid_res))

            contour = ax[j].contourf(grid_x, grid_y, grid_z,  levels=np.linspace(-0.3, 0.3, 21),
                                     cmap=cmap, vmin=-0.3, vmax=0.3)
            list_contours.append(contour)
            ax[j].scatter(coords[:, 0], coords[:, 1], c=values[j], edgecolors="k", cmap=cmap)
            if coords_name is not None:
                for (xi, yi), text in zip(coords, coords_name):
                    ax[j].annotate(text,
                                   xy=(xi, yi), xycoords='data',
                                   xytext=(0., 2.5), textcoords='offset points')
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            if subfig_name is not None:
                ax[j].set_title(subfig_name[j])

        fig.colorbar(list_contours[0], ax=ax, orientation='horizontal', fraction=.1)

        # fig.colorbar(contour, ax=ax.ravel().tolist())
        # fig.tight_layout()
    else:
        for i in range(rows):
            for j in range(cols):
                if j > len(values[i]) - 1:
                    break
                grid_x, grid_y = np.meshgrid(np.linspace(coords.min() - margin, coords.max() + margin, grid_res),
                                             np.linspace(coords.min() - margin, coords.max() + margin, grid_res))

                grid_z = griddata(coords, values[i][j], (grid_x, grid_y), method="cubic")
                contour = ax[i, j].contourf(grid_x, grid_y, grid_z,  levels=np.linspace(vmin, vmax, 25),
                                     cmap=cmap, vmin=vmin, vmax=vmax)
                list_contours.append(contour)

                # Re-plot the ears nose, etc
                n_ax = _prepare_topomap(coords, dataname, rows=1, cols=1, size=size, axs=ax[i, j], **fig_kwargs)

                ax[i, j].scatter(coords[:, 0],
                                 coords[:, 1],
                                 c=values[i][j],
                                 edgecolors="grey",
                                 linewidths=0.5,
                                 cmap=cmap)

                if coords_name is not None:
                    for (xi, yi), text in zip(coords, coords_name):
                        ax[i, j].annotate(text,
                                          xy=(xi, yi), xycoords='data',
                                          xytext=(0.5, 2.5), textcoords='offset points', fontsize=8)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if subfig_name is not None:
                    ax[i, j].set_title(subfig_name[i][j])
        if title is not None:
            fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.colorbar(list_contours[0], ax=ax, orientation='vertical', fraction=.05)



    if savepath is not None:
        plt.savefig(savepath)  # , bbox_inches="tight"
    plt.close()
    # plt.show()
