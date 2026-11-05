"""Display-only axis transformations; annotation measurements stay unchanged."""
import numpy as np

DIMENSIONS = ("RT", "m/z", "Kendrick mass defect")
DIMENSION_TITLES = {"RT": "Retention time (min)", "m/z": "m/z",
                    "Kendrick mass defect": "Kendrick mass defect (CH₂)"}


def kendrick_mass_defect(mz):
    mass = np.asarray(mz, dtype=float) * (14.0 / 14.015650064)
    return np.rint(mass) - mass


def dimension_values(rt, mz, dimension):
    if dimension == "RT":
        return rt
    if dimension == "m/z":
        return mz
    if dimension == "Kendrick mass defect":
        return kendrick_mass_defect(mz)
    raise ValueError(f"Unknown display dimension: {dimension}")


def projected_grid_line(line, x_dimension, y_dimension):
    if line is None:
        return [], []
    rt, mz = (np.asarray([np.nan if value is None else value for value in axis], dtype=float)
              for axis in line)
    x = np.asarray(dimension_values(rt, mz, x_dimension), dtype=float)
    y = np.asarray(dimension_values(rt, mz, y_dimension), dtype=float)
    xs, ys = [], []
    for index, (xvalue, yvalue) in enumerate(zip(x, y)):
        if index and ((x_dimension == "Kendrick mass defect" and abs(xvalue - x[index-1]) > .5)
                      or (y_dimension == "Kendrick mass defect" and abs(yvalue - y[index-1]) > .5)):
            xs.append(None)
            ys.append(None)
        xs.append(float(xvalue) if np.isfinite(xvalue) else None)
        ys.append(float(yvalue) if np.isfinite(yvalue) else None)
    return xs, ys
