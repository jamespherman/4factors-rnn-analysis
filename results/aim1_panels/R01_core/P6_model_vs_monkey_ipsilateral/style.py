"""Shared style for the Aim 1 panels (copied into each results/aim1_panels/<ITEM>/ folder so plot scripts stay
self-contained). Colours: first three slots of a CVD-validated categorical palette, plus ink and null grey."""
import matplotlib as mpl

DATA = '#222222'        # recorded data
RESOLVER = '#eb6834'    # resolver (A / AE) prediction or networks; dashed when it is a prediction
PRIORITY = '#2a78d6'    # priority hub (B)
OUTCOME = '#1baf7a'     # outcome hub (C)
NULL = '#d9d9d6'        # null / chance band
MUTED = '#8a8984'       # secondary ink (reference lines, annotations)
ARCH = {'A': RESOLVER, 'AE': RESOLVER, 'B': PRIORITY, 'C': OUTCOME}
PANEL = (2.2, 2.0)      # inches per panel


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.5,
        'lines.markersize': 3.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'figure.dpi': 150})


def save(fig, path_stem):
    """Save <path_stem>.png (300 dpi) and .pdf (editable text)."""
    fig.savefig(f'{path_stem}.png')
    fig.savefig(f'{path_stem}.pdf')
