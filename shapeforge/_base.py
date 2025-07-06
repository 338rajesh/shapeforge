from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import gbox as gb


class UnitCellDomain:
    def __init__(self, bounds: tuple = (0.0, 0.0, 1.0, 1.0), inclusions=None):
        self._bounds = bounds
        if inclusions is None:
            inclusions = []
        self.inclusions = inclusions

        self.domain = gb.BoundingBox(
            lower_bound=(bounds[0], bounds[1]),
            upper_bound=(bounds[2], bounds[3]),
        )

    @property
    def domain_bounds(self):
        return self._bounds

    def plot(self, fig=None, ax=None, show=False, f_path=None, **kwargs):
        """
        Plot the unit cell domain on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the domain. If None, a new figure and axes will be created.
        kwargs : dict, optional
            Additional keyword arguments for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        self.domain.plot(
            axs=ax, facecolor="lightgrey", edgecolor="k", **kwargs
        )
        for inclusion in self.inclusions:
            inclusion.plot(axs=ax, color="blue", alpha=0.5, **kwargs)
        gb.utils.con_figure(fig, ax)
        if f_path is not None:
            fig.savefig(f_path, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            return fig, ax


class Inclusion:
    """
    Base class for all inclusions in a shape.
    This class is intended to be subclassed for specific inclusion types.
    """

    def __init__(self):
        """
        Initialize the Inclusion with a name and fraction.

        Parameters
        ----------
        name : str, optional
            Name of the inclusion.
        fraction : float, optional
            Fraction of the inclusion in the shape.
        kwargs : dict, optional
            Additional keyword arguments for specific inclusion types.
        """
        pass

    def generate(self, *args, **kwargs):
        """
        Generate the inclusion in the given shape.

        Parameters
        ----------
        shape : Shape
            The shape in which to generate the inclusion.

        Returns
        -------
        gb.Shape
            A GBox shape representing the inclusion.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class CircularInclusion(Inclusion):
    """
    Class for circular inclusions in a shape.
    """

    def __init__(
        self,
        xc_distribution: stats.rv_continuous,
        yc_distribution: stats.rv_continuous,
        radius_distribution: stats.rv_continuous,
        name=None,
    ):
        """
        Initialize the CircularInclusion with center and radius.

        Parameters
        ----------
        center : tuple of float
            Center of the circular inclusion (x, y).
        radius : float
            Radius of the circular inclusion.
        name : str, optional
            Name of the inclusion.
        fraction : float, optional
            Fraction of the inclusion in the shape.
        """
        super().__init__()
        self.name = name or "CircularInclusion"
        self.xc_distribution = xc_distribution
        self.yc_distribution = yc_distribution
        self.radius_distribution = radius_distribution
        self._validate_distributions()

    def _type_err_msg(self, distribution_name):
        return TypeError(
            f"{distribution_name} must be a continuous distribution instance. "
            f"Got {type(getattr(self, distribution_name))} instead."
        )

    def _val_err_msg(self, distribution_name):
        return ValueError(
            f"{distribution_name} must have finite support. "
            f"Got {getattr(self, distribution_name).support()} instead."
        )

    def _validate_distributions(self):
        """
        Validate that the distributions for xc, yc, and radius are all
        instances of rv_continuous.
        """
        valid_distr = (
            stats.rv_continuous,
            stats._distn_infrastructure.rv_continuous_frozen,
        )
        if not isinstance(self.xc_distribution, valid_distr):
            raise self._type_err_msg("xc_distribution")
        if not isinstance(self.yc_distribution, valid_distr):
            raise self._type_err_msg("yc_distribution")
        if not isinstance(self.radius_distribution, valid_distr):
            raise self._type_err_msg("radius_distribution")
        if self.radius_distribution.support()[0] < 0:
            raise self._val_err_msg("radius_distribution")
        if np.isinf(self.xc_distribution.support()).any():
            raise self._val_err_msg("xc_distribution")
        if np.isinf(self.yc_distribution.support()).any():
            raise self._val_err_msg("yc_distribution")
        return True

    def generate(
        self,
        cum_area: Optional[float] = None,
        num_instances: Optional[int] = None,
    ) -> List[gb.CirclesArray]:
        """
        Generate the circular inclusions, following the specified
        distributions for the center and radius. If `cum_area` is provided,
        the number of instances will be adjusted to match the cumulative area.
        If `num_instances` is provided, it will generate that many instances
        without considering the cumulative area. If both are provided,
        a ValueError will be raised, and if neither is provided, a single
        instance will be generated.

        Parameters
        ----------
        cum_area : float, optional
            Cumulative area of the shape. If provided, the number of instances
            will vary to match this area.
        num_instances : int, optional
            Number of instances to generate. It cannot be specified
            together with cum_area.

        Returns
        -------
        List[gb.CirclesArray]
            A list of GBox CirclesArray objects representing the generated
            circular inclusions.
        """
        if cum_area is not None and num_instances is not None:
            raise ValueError("Cannot specify both cum_area and num_instances.")

        elif cum_area is None and num_instances is None:
            radii = self.radius_distribution.rvs(size=(1,))
        elif cum_area is not None:
            pass
            # sample radii to match the cumulative area
            ca = 0.0
            radii = []
            while ca < cum_area:
                radii.append(self.radius_distribution.rvs())
                ca += np.pi * radii[-1] ** 2
            radii = np.array(radii)
        elif num_instances is not None:
            radii = self.radius_distribution.rvs(size=num_instances)

        xc_values = self.xc_distribution.rvs(size=radii.shape)
        yc_values = self.yc_distribution.rvs(size=radii.shape)
        # return gb.CirclesArray(
        #     circles=np.stack((xc_values, yc_values, radii), axis=-1),
        #     initial_capacity=len(radii),
        # )
        return [
            gb.CirclesArray(circles=[gb.Circle(r, (xc, yc))])
            for r, xc, yc in zip(radii, xc_values, yc_values)
        ]
