from .. import core
from .. import builders
from .. import converters
from .. import gptk_config
from power_grid_model_io.converters import PandaPowerConverter
import pandapower as pp
from pathlib import Path

import numpy as np
import pickle

def build_line_grid(n_loads, u_rated=230, p_specified=1000, q_specified=100):
    if not isinstance(u_rated, list):
        u_rated = [u_rated]*(n_loads+1)

    if not isinstance(p_specified, list):
        p_specified = [p_specified]*(n_loads+1)

    if not isinstance(q_specified, list):
        q_specified = [q_specified]*(n_loads+1)

    
    sources = []
    nodes = []
    lines = []
    sym_loads = []

    for i, n in enumerate(range(n_loads+1)):
        node_id = n*3 + 1

        nodes.append(core.Node(id=node_id, u_rated=u_rated[i]))

        if i == 0:
            sources.append(core.Source(id=node_id-1, node=node_id))

        if i > 0:
            lines.append(core.Line(id=node_id+1, from_node=prev_node_id, to_node=node_id))

            sym_loads.append(core.SymLoad(id=node_id+2, node=node_id, p_specified=p_specified[i], q_specified=q_specified[i]))

        prev_node_id = node_id

    return builders.Grid(node=nodes, line=lines, source=sources, sym_load=sym_loads)

def build_star_grid(n_loads, p_values, q_values, u_rated=230):
    

    if not isinstance(p_values, (np.ndarray, list)):
        p_values = [p_values]*n_loads

    if not isinstance(q_values, (np.ndarray, list)):
        q_values = [q_values]*n_loads

    sources = []
    nodes = []
    lines = []
    sym_loads = []

    source_id = 0
    source_node_id = 1

    sources.append(core.Source(id=source_id, node=source_node_id))
    nodes.append(core.Node(id=source_node_id, u_rated=u_rated))

    for i in range(0, n_loads):
        base_id = (i+1)*3

        node_id = base_id
        sym_load_id = base_id + 1
        line_id = base_id + 2

        nodes.append(core.Node(id=node_id, u_rated=u_rated))
        sym_loads.append(core.SymLoad(id=sym_load_id, node=node_id, p_specified=p_values[i], q_specified=q_values[i]))
        lines.append(core.Line(id=line_id, from_node=source_node_id, to_node=node_id))


    return builders.Grid(node=nodes, line=lines, source=sources, sym_load=sym_loads)

def build_wide_star_grid(n_spokes, n_nodes_per_spoke, p_values, q_values, u_rated=230):
    if not isinstance(p_values, (np.ndarray, list)):
        p_values = [p_values]*n_spokes*n_nodes_per_spoke

    if not isinstance(q_values, (np.ndarray, list)):
        q_values = [q_values]*n_spokes*n_nodes_per_spoke

    assert len(p_values) == n_spokes*n_nodes_per_spoke
    assert len(q_values) == n_spokes*n_nodes_per_spoke

    sources = []
    nodes = []
    lines = []
    sym_loads = []

    source_id = 0
    source_node_id = 1

    sources.append(core.Source(id=source_id, node=source_node_id))
    nodes.append(core.Node(id=source_node_id, u_rated=u_rated))

    for i in range(0, n_spokes):
        for j in range(0, n_nodes_per_spoke):
            base_id = (i*n_nodes_per_spoke + j + 1)*3

            node_id = base_id
            sym_load_id = base_id + 1
            line_id = base_id + 2

            nodes.append(core.Node(id=node_id, u_rated=u_rated))
            sym_loads.append(core.SymLoad(id=sym_load_id, node=node_id, p_specified=p_values[i*n_nodes_per_spoke + j], q_specified=q_values[i*n_nodes_per_spoke + j]))
            
            if j == 0:
                # this node is connected to the source
                lines.append(core.Line(id=line_id, from_node=source_node_id, to_node=node_id))
            else:
                # this node is connected to the previous node in the spoke
                lines.append(core.Line(id=line_id, from_node=base_id-3, to_node=node_id))


    return builders.Grid(node=nodes, line=lines, source=sources, sym_load=sym_loads)



def sloppy_case33bw(dump=False):
    # load the case from pandapower
    pp_network = pp.networks.case33bw()

    # convert it to pgm
    converter = PandaPowerConverter()
    pgm_input, _ = converter.load_input_data(pp_network)

    # for some reason after conversion there's thrice as many sym_loads; so I crop of those
    pgm_input["sym_load"] = pgm_input["sym_load"][:32]


    # these are nans in the original pgm_input
    pgm_input["line"]["tan1"] = np.array([0]*len(pgm_input["line"]["tan1"]), dtype=pgm_input["line"]["tan1"].dtype)


    gridBuilder = builders.GridBuilder()
    grid = gridBuilder.from_pgm_input(pgm_input)

    if dump:
        with open(gptk_config.CASE33BW_PATH, "wb") as f:
            return pickle.dump(grid, f)
        
    return grid

def scale_grid_attributes(grid, scalers=None):
    if scalers is None:
        scalers = []

    for sub_dict in scalers:
        component_name = sub_dict["component_name"]
        component_attribute = sub_dict["component_attribute"]
        source_values = grid.grid[component_name][component_attribute]

        factor = sub_dict["factor"]

        if isinstance(factor, (int, float)):
            factor = [factor]*len(source_values)
        elif isinstance(factor, (list, np.ndarray)):
            assert len(factor) == len(source_values)
        else:
            raise NotImplementedError(f"type(factor)={type(factor)}")
            

        new_values = [f*v for f, v in zip(factor, source_values)]

        grid.list_update_component_values(component_name, component_attribute, new_values)

    return grid

def quick_case33bw(scalers=[]):
    with open(gptk_config.CASE33BW_PATH, "rb") as f:
        grid = pickle.load(f)
    
    grid = builders.GridBuilder().from_grid_dict(grid)

    grid = scale_grid_attributes(grid, scalers)

    return grid


def load_and_convert_pp_net(case_name, scalers=None, test_pgm_power_flow=True, replace_gens_with_loads_bool=True, merge_multiple_pq_busses_per_node_bool=True):
    # tested for case30, case33bw, and case_illinois200
    pp_net = getattr(pp.networks, case_name)()

    # verify that there are no components in the pp_net for which the conversion to pgm_net has not been implemented yet 
    not_implemented_components = check_for_not_implemented_components(pp_net)
    if any(not_implemented_components.values()):
        for k, v in not_implemented_components.items():
            if v:
                print(f"There are {k} in {case_name}")
        print("Exiting this function")
        return False
    
    grid_dict = converters.PPConverter().flexible_convert_from_pandapower(pp_net)
    grid = builders.GridBuilder(replace_gens_with_loads_bool=replace_gens_with_loads_bool, merge_multiple_pq_busses_per_node_bool=merge_multiple_pq_busses_per_node_bool).from_grid_dict(grid_dict)

    grid = scale_grid_attributes(grid, scalers)

    # if you want to validate
    if test_pgm_power_flow:
        grid.load_flow()

    return grid

def load_stored_pgm_net(case_name, scalers):
    fn = Path(gptk_config.SAVED_NETWORKS_FOLDER) / case_name
    fn = fn.withsuffix('.pkl')
    with open(fn, "rb") as f:
        grid_dict = pickle.load(f)
    
    grid = builders.GridBuilder().from_grid_dict(grid_dict)

    grid = scale_grid_attributes(grid, scalers)

    return grid

def save_grid_dict(grid, case_name, overwrite_existing_grid=False):
    grid_dict = grid.grid
    fn = Path(gptk_config.SAVED_NETWORKS_FOLDER) / case_name
    fn = fn.withsuffix('.pkl')

    if not overwrite_existing_grid:
        if fn.exists():
            entries = [f for f in fn.parent.iterdir() if f.stem.startswith(case_name)]
            num = len(entries) + 1
            fn = fn.parent / f"{fn.stem}_{num}{fn.suffix}"


    with open(fn, "wb") as f:
        pickle.dump(grid_dict, f)  

    return


def check_for_not_implemented_components(net):
    # List of components to check
    components = [
        'asymmetric_load',
        'motor',
        'asymmetric_sgen',
        'storage',
        'switch',
        'svc',
        'trafo3w',
        'ward',
        'xward'
    ]

    # Dictionary to hold the presence status of each component
    component_presence = {component: False for component in components}
    
    # Check each component
    for component in components:
        # Check if the component is present and has entries
        if component in net and not net[component].empty:
            component_presence[component] = True

    return component_presence 