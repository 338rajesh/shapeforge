# ShapeForge

A python package for generating 2D or 3D domain containing geometry shapes of interest, following a specified distribution.

## Installation

```sh
pip install shapeforge
```

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
