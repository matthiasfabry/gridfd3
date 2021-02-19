import matplotlib


pdf_with_latex = {                      # setup matplotlib to use latex for output
    "text.usetex": True,                # use LaTeX to write all text
    "axes.labelsize": 20,               # LaTeX default is 10pt font.
    "font.family": 'serif',
    "font.size": 20,
    "legend.fontsize": 10,               # Make the legend/label fonts
    "xtick.labelsize": 15,               # a little smaller
    "ytick.labelsize": 15,
    "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{siunitx}",  # use utf8 fonts
    "figure.autolayout": True
    }

matplotlib.use("agg")
matplotlib.rcParams.update(pdf_with_latex)
