# EvoDiffusion

This repo contains an implementation of DDIM with the intention of applying it to genetic sequence data.

You can install this package locally by running

```bash
poetry build
pip install <.whl>
```

Alternatively, if you would like to develop this package, please install [poetry](https://python-poetry.org/docs/)

```bash
poetry install
poetry shell
```

You can then run the examples within the poetry shell or run the notebooks with `jupyter notebook`.
Dependencies can be added with `poetry add <package_name>`.
If you would like this only as a development dependency, please add the `--dev` flag.
