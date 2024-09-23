import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from scipy.interpolate import griddata


def _prepare_kiloword_topomap(electrodes, head_radius=0.2, cols=None, rows=None, size=3, axs=None, axs_fris=None, **fig_kwargs):
    left_ear = Ellipse(electrodes[6], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    right_ear = Ellipse(electrodes[26], height=head_radius / 4, width=head_radius / 8, facecolor="none", **fig_kwargs)
    nose = Polygon([electrodes[4] + np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.]),
                    electrodes[15] + np.array([0, 0.02]),
                    electrodes[24] - np.array([abs(electrodes[4, 0] - electrodes[15, 0]) / 3, 0.])],
                   facecolor="none", **fig_kwargs)
    ellipse = Ellipse(electrodes[1], height=head_radius, width=head_radius, facecolor="white", **fig_kwargs)

    if (rows, cols) in [(1, 1), (None, None)]:
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
        if axs is None:

            from matplotlib.gridspec import GridSpec
            #fig = plt.figure(figsize=(size * cols, size * (rows + 1)), **fig_kwargs)

            # ax = []
            # for i in range(rows):
            #     ax.append([])
            #     for j in range(cols):
            #         ax[-1].append(fig.add_subplot(spec[i, j]))
            RATIO = 4
            fig, ax = plt.subplots(rows + 1, cols, figsize=(size * cols, size * rows + size / RATIO),
                                   gridspec_kw={'height_ratios': [RATIO,RATIO,1]},
                                   **fig_kwargs)
            # fig.add_subplot(rows + 1, 1, (rows * cols + 1, (rows + 1) * cols))
            #spec = GridSpec(nrows=rows + 1, ncols=cols, figure=fig) #, **fig_kwargs)
            #ax_fris = fig.add_subplot(spec[-1, :])
            ax_fris = fig.add_subplot(rows + 1, 1, rows + 1, adjustable='box',
                                      aspect=12, anchor=(-0.25, 0.15))
            ax_fris.margins(x=0.15, y=0)
            ax_fris.set_xticks([])
            ax_fris.set_yticks([])

        else:
            ax = axs
            ax_fris = axs_fris
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

                ax[i][j].add_patch(nose)
                ax[i][j].add_patch(left_ear)
                ax[i][j].add_patch(right_ear)
                ax[i][j].add_patch(ellipse)
                ax[i][j].set_aspect("equal")
    if axs is not None:
        return ax, ax_fris
    return fig, ax, ax_fris


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

    fig, ax, ax_fris = _prepare_topomap(coords, dataname, rows=rows, cols=cols, size=size, **fig_kwargs)
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

            contour = ax[j].contourf(grid_x, grid_y, grid_z, levels=np.linspace(-0.3, 0.3, 21),
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
                print(i, j)
                contour = ax[i][j].contourf(grid_x, grid_y, grid_z, levels=np.linspace(vmin, vmax, 100),
                                            cmap=cmap, vmin=vmin, vmax=vmax)
                list_contours.append(contour)

                # Re-plot the ears nose, etc
                # n_ax = _prepare_topomap(coords, dataname, rows=1, cols=1, size=size, axs=ax[i, j], **fig_kwargs)

                ax[i][j].scatter(coords[:, 0],
                                 coords[:, 1],
                                 c=values[i][j],
                                 edgecolors="grey",
                                 linewidths=0.5,
                                 cmap=cmap)

                if coords_name is not None:
                    for (xi, yi), text in zip(coords, coords_name):
                        ax[i][j].annotate(text,
                                          xy=(xi, yi), xycoords='data',
                                          xytext=(0.5, 2.5), textcoords='offset points', fontsize=8)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[rows][j].set_xticks([])
                ax[rows][j].set_yticks([])
                if subfig_name is not None:
                    ax[i][j].set_title(subfig_name[i][j])

        # Set the timeline bar
        time_range = [int(title.split(" ")[0]), int(title.split(" ")[2])]
        ax_fris.broken_barh([(0, 1000), (time_range[0], time_range[1] - time_range[0])], (0, 2), facecolors=('lightgray', 'brown'))
        ax_fris.annotate(f'{time_range[0]}ms', xy=(time_range[0], 2), xytext=(0, 10),
                    textcoords="offset points", ha='center', va='bottom', fontsize=14)
        ax_fris.annotate(f'{time_range[1]}ms', xy=(time_range[1], 2), xytext=(0, 10),
                           textcoords="offset points", ha='center', va='bottom', fontsize=14)

        if title is not None:
            fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.colorbar(list_contours[0], ax=ax, orientation='vertical', fraction=.05)

    if savepath is not None:
        plt.savefig(savepath)  # , bbox_inches="tight"
    plt.close()
    # plt.show()
