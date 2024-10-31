# Grid processing toolkit
Grid Processing ToolKit (GPTK) is a python package used for transformation, manipulation, and simulation of electrical distribution grids. It is mostly an extension on power-grid-model; it provides utility functions to easily add, remove, and change components, as well as transforming the topology to a pandapower grid and a networkx graph. 

The main datatype is in `builders.Grid`, which inherits from `core.BaseGrid`. `BaseGrid` is based on the components and datatypes stored in power-grid-model. The relevant information about the topology is stored in `builders.Grid.grid`, which transforms the topology information to a so-called '`grid_dict`'. This `grid_dict` contains all information about the topology in a format very similar to power-grid-model's. All the components described in the grid_dict are defined in `core.py`, which also shows the required attributes per component.

The `grid_dict` is a dictionary, where each key-value pair is a 'component-type' and a child-dictionary. Each sub_dictionary defines the number and characteristics of each of that component in the grid. E.g., a `line` component defines the lines in the grid. The line component is defined by its corresponding child-dictionary. In this child-dictionary, each key is an attribute required for the line, and each value is a list, where the j'th element indicates the value for that attribute for the j'th line. 

Below you see an example of a `grid_dict`. To find all attributes corresponding to the second line, go to `line` and for all attributes look at the 2nd value in the list, so `id=5, from_node=2, to_node=3, r1=0.25`, etc.
```
`builders.Grid.grid = {
    "node": {
        "id": [1, 2, 3],
        "u_rated": [10_000, 10_000, 10_000]
    },
    "line": {
        "id": [4, 5],
        "from_node": [1, 2],
        "to_node": [2, 3],
        "r1": [0.25, 0.25],
        ...,
    },
    "sym_load": {
        "id": [6],
        "node: [3],
        "p_specified: [100],
        ...
    }
    "source": {
        "id": [7],
        "node": [1],
        ...
    }

}
```



To get started, you can use the example topologies defined in `gptk.topologies`, e.g. `gptk.topologies.build_star_grid(...)`.