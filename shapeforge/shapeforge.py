from scipy import stats

from ._base import CircularInclusion, UnitCellDomain


def get_initialized_inclusions(
    vf: float = 0.5,
    bounds: tuple = (0.0, 0.0, 1.0, 1.0),
    radii_mean: float = 1.5,
    radii_std: float = 0.5,
):
    xlb, ylb, xub, yub = bounds
    domain_area = (xub - xlb) * (yub - ylb)
    inclusion_area = vf * domain_area

    xc_distribution = stats.uniform(xlb, xub - xlb)
    yc_distribution = stats.uniform(ylb, yub - ylb)

    # Ensure the radius distribution is valid (non-negative)
    r_min = max(0, radii_mean - 3 * radii_std)
    r_max = radii_mean + 3 * radii_std
    if r_min < 0:
        raise ValueError("Radius distribution must have non-negative support.")
    if r_max <= 0:
        raise ValueError("Radius distribution must have positive support.")

    radius_distribution = stats.truncnorm(
        a=(r_min - radii_mean) / radii_std,
        b=(r_max - radii_mean) / radii_std,
        loc=radii_mean,
        scale=radii_std,
    )
    inclusion = CircularInclusion(
        xc_distribution=xc_distribution,
        yc_distribution=yc_distribution,
        radius_distribution=radius_distribution,
        name="Circular Inclusion",
    )
    return inclusion.generate(cum_area=inclusion_area)


def undo_overlaps(inclusions: list):
    # Evaluate the cost function and gradients of overlaps
    # Update the positions of the inclusions to minimize overlaps
    return


def make_rve_with_circular_inclusions(
    vf: float = 0.5,
    bounds: tuple = (0.0, 0.0, 1.0, 1.0),
    radii_mean: float = 1.5,
    radii_std: float = 0.5,
) -> UnitCellDomain:
    init_inclusions = get_initialized_inclusions(
        vf, bounds, radii_mean=radii_mean, radii_std=radii_std
    )

    return UnitCellDomain(inclusions=init_inclusions, bounds=bounds)
