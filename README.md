# compaction-1d
finite element code for a one-dimensional compaction model

An example of how to run the code is provided in example.ipynb.

The code relies on [FEniCSx](https://fenicsproject.org). Probably the easiest
way to run the Jupyter notebook is through a [Docker](https://www.docker.com)
container with the command:

`docker run --init -ti -p 8888:8888 -v $(pwd):/home/fenics/shared -w /home/fenics/shared dolfinx/lab:stable`