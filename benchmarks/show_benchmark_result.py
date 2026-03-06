import pathlib

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    gray = (132 / 255, 145 / 255, 158 / 255, 1.0)
    light_gray = (200 / 255, 200 / 255, 203 / 255, 1.0)
    blue = (0 / 255, 90 / 255, 255 / 255, 1.0)
    bar_colors = [gray] * 3 + [blue] * 2
    font_colors = [gray] * 3 + [(0, 0, 0, 1)] * 2

    result_dir = pathlib.Path(__file__).parent / "results"
    filenames = [
        (
            result_dir / "contiguous_data_to_1d_arrays.txt",
            "take contiguous data to $\\bf{{1d}}$ array",
            None,
        ),
        (
            result_dir / "contiguous_data_to_Nd_arrays.txt",
            "take contiguous data to $\\bf{{Nd}}$ array",
            0,
        ),
        (
            result_dir / "discontiguous_data_to_1d_arrays.txt",
            "take $\\bf{{dis}}$contiguous data to $\\bf{{1d}}$ array",
            None,
        ),
        (
            result_dir / "discontiguous_data_to_Nd_arrays.txt",
            "take $\\bf{{dis}}$contiguous data to $\\bf{{Nd}}$ array",
            0,
        ),
    ]

    xticklabel = {
        "numpy_outplace": "out=$\\bf{{np.}}$take(a,i,axis={axis})",
        "numpy_inplace": "$\\bf{{np.}}$take(a,i,axis={axis},out=out)",
        "numba_outplace": "out=$\\bf{{np.}}$take(a,i,axis={axis})\nin $\\bf{{@njit}}$ method",
        "npind_outplace": "out=$\\bf{{npi.}}$take(a,i,axis={axis})",
        "npind_inplace": "$\\bf{{npi.}}$take(a,i,axis={axis},out=out)",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.03, top=0.95, hspace=0.3)
    ax: plt.Axes
    for ax, (fn, desc, axis) in zip(axes.flat, filenames):
        data = np.loadtxt(
            fn, dtype=[("name", np.object_), ("time", np.float64)], delimiter=" "
        )

        ypos = [-2.1, -1.1, 0, 1.1, 2.1]
        texts = [xticklabel[n].format(axis=axis) for n in data["name"]]
        bars = ax.barh(texts, data["time"], color=bar_colors)

        for k in ["bottom", "right", "top"]:
            ax.spines[k].set_color(light_gray)
        ax.spines["left"].set_color(gray)
        ax.tick_params(bottom=False)
        ax.invert_yaxis()

        bar_labels = ax.bar_label(bars, padding=5, fmt="%.1f [ms]")
        for label, c in zip(bar_labels, font_colors):
            label.set_color(c)
            label.set_weight("bold")
            label.set_size(12)
        for label, c in zip(ax.get_yticklabels(), font_colors):
            label.set_color(c)
            label.set_size(12)
        ax.set_title(desc, fontdict=dict(size=16))
        ax.set_xticklabels([])
        ax.grid(axis="x", color=light_gray)
        ax.set_xlim(0, 20)

    assets_dir = pathlib.Path(__file__).parents[1] / "assets"
    fig.savefig(assets_dir / "benchmark_result.png")

    plt.close(fig)
