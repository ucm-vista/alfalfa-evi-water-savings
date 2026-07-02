"""Centralized publication styling for Phase 5 figures.

Defines the Wong 2011 / Okabe-Ito colorblind-safe palette, journal sizing
constants, matplotlib rcParams for publication quality, dual-format save
utility (PNG 300 DPI + PDF), and consistent panel labeling.

All Phase 5 figure code should import from this module rather than defining
style constants inline.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Wong 2011 / Okabe-Ito colorblind-safe palette (8 colors)
# ---------------------------------------------------------------------------
WONG_PALETTE = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

# ---------------------------------------------------------------------------
# Journal sizing constants (inches)
# ---------------------------------------------------------------------------
SINGLE_COL_WIDTH = 3.5   # ~89 mm (single-column figure)
DOUBLE_COL_WIDTH = 7.5   # ~190 mm (double-column figure)
MAX_HEIGHT = 9.0          # typical journal page max

# ---------------------------------------------------------------------------
# Panel labels for multi-panel figures
# ---------------------------------------------------------------------------
PANEL_LABELS = 'abcdefghijklmnop'


def apply_style():
    """Apply publication-quality rcParams globally.

    Sets font sizes, line widths, DPI, colorblind-safe color cycle,
    and TrueType font embedding for journal PDF submission.
    """
    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 1.0,
        'axes.prop_cycle': mpl.cycler(color=WONG_PALETTE),
        'pdf.fonttype': 42,   # TrueType for journal PDF submission
        'ps.fonttype': 42,
    })


def save_pub_figure(fig, name, out_dir, dpi=300):
    """Save figure as both PNG (300 DPI) and PDF to separate subdirectories.

    Creates ``out_dir/png/`` and ``out_dir/pdf/`` if they don't exist,
    saves the figure in both formats, prints file sizes, and closes
    the figure to release memory.

    Args:
        fig: matplotlib Figure to save.
        name: Base filename (without extension).
        out_dir: Root output directory.
        dpi: DPI for PNG output (default 300).
    """
    out_dir = Path(out_dir)
    png_dir = out_dir / 'png'
    pdf_dir = out_dir / 'pdf'
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f'{name}.png'
    pdf_path = pdf_dir / f'{name}.pdf'

    fig.savefig(str(png_path), dpi=dpi, bbox_inches='tight')
    fig.savefig(str(pdf_path), bbox_inches='tight')
    plt.close(fig)

    png_kb = png_path.stat().st_size / 1024
    pdf_kb = pdf_path.stat().st_size / 1024
    print(f"  Saved: {png_path} ({png_kb:.0f} KB)")
    print(f"  Saved: {pdf_path} ({pdf_kb:.0f} KB)")


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Place a consistent panel label on an axes.

    Adds ``(label)`` text (e.g. ``(a)``, ``(b)``) at the specified
    position in axes-relative coordinates.

    Args:
        ax: matplotlib Axes to annotate.
        label: Single character label (e.g. 'a', 'b', 'c').
        x: Horizontal position in axes fraction (default -0.12).
        y: Vertical position in axes fraction (default 1.05).
    """
    ax.text(
        x, y, f'({label})',
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
    )
