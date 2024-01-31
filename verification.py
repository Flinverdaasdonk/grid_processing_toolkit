from . import converters

import networkx as nx

def nx_verify(grid):
    graph = converters.NetworkxAdapter()(grid)

    assert nx.is_connected(graph), f"Is AssertionError it means that the grid is actually multiple disconnected grids"

def pgm_verify(grid):
    from power_grid_model import CalculationType, CalculationMethod
    from power_grid_model.validation import assert_valid_input_data

    pgm_input = converters.PGMAdapter()(grid)

    assert_valid_input_data(input_data=pgm_input, calculation_type=CalculationType.power_flow)

def verify_all(grid):
    nx_verify(grid)
    pgm_verify(grid)