# ShapeForge

A python package for generating 2D or 3D domain containing geometry shapes of interest, following a specified distribution.

## Installation

```sh
pip install shapeforge
```

## Cell

A cell is an union of a domain and inclusions. At the moment, only rectangular
domains are supported, while inclusions can be of any shape. The shapes can be
described using positional and size parameters.

- **Positional parameters**: These parameters define the position and orientation
  of the shape. A set of `x`, `y`, `z`, $\theta$, and $\phi$ values defined for
  a pivot line, and a pivot point on this line of a arbitrary shape can be
  used to define its position and orientation in space.  
  In cell generation process, these parameters are tuned while the size  
  parameters are fixed.
- **Size parameters**: These parameters define the size of the shape. These can be
  arbitrary, depending on the shape. For example, a sphere can be defined using
  its radius, while a cuboid can be defined using its length, width, and height.
  In cell generation process, these parameters are fixed upfront while the
  positional parameters are tuned to achieve the desired sptail distribution
  of inclusions in the cell domain.

> For flowchart of events involved in cell generation, please refer to
> [Flow of Events](docs/flow_of_events.md).

## Planned features

TODO udpate this list as features are implemented.

- [ ] no boundary conditions
- [ ] non-rectangular domains may be used in the future, but for now they are rectangle
- [ ] all kinds of shapes, that I will take care of, I have a plan for this
- [ ] Instead of writing distributions, I would like to use the standard packages like scipy or any other that is dedicated for distributions
- [ ] output as raw data, for sure. Other optional features like downloading as image, mesh files, can be done later.
- [ ] Yes, metrics are required, that compare generated vs requested.
- [ ] GPU acceleration is surely in plan, but not now.
- [ ] Yes, I am looking for scalable modules


## For Developers

Using [uv](https://docs.astral.sh/uv/) for Python Package and Project management.

### Install dependencies

```bash
uv sync
```

### Add New dependencies

```bash
uv add package==version
```

### Add Dev-Dependency

```bash
uv add --group dev dev-package==version
```

### Run Tests

```bash
uv run --group dev pytest tests/
```

> Check [uv](https://docs.astral.sh/uv/)'s documentation for more details
