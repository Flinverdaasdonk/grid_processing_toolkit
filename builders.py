from . import core
from . import converters
from . import decorators

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from 

class Grid(core.BaseGrid):
    def __init__(self, node, line, source, sym_load, link=[], sym_gen=[], shunt=[], transformer=[]):
        super().__init__(node=node, line=line, source=source, sym_load=sym_load, link=link, transformer=transformer, sym_gen=sym_gen, shunt=shunt)
        self.grid_version = 0


    @property
    @decorators.lazy_evaluation_with_caching
    def grid(self):
        return converters.DictionaryConverter().convert(self)
    
    @property
    @decorators.lazy_evaluation_with_caching
    def pgm_format(self):
        return converters.PGMConverter().convert(self)
    
    @property
    @decorators.lazy_evaluation_with_caching
    def pp_format(self):
        return converters.PPConverter().convert(self)
    

    @property
    @decorators.lazy_evaluation_with_caching
    def graph(self):
        return converters.NetworkxConverter().convert(self)

    @decorators.lazy_evaluation_with_caching
    def load_flow(self):
        from_node_to_children_features = ["u_pu", "u", "u_angle"]
        from_children_to_node_features = ["i", "s", "pf"]

        _output = converters.PGMConverter().load_flow(grid=self)
        output = {}

        ### PART 1:
        # convert _output to a dict, instead of the default pgm_format which uses named ndarrays
        for component in _output.keys():
            component_dict = {}

            for attr in _output[component].dtype.names:
                component_dict[attr] = list(_output[component][attr])

            output[component] = component_dict

        ### PART 2: Make sure that all children also get u, u_pu, and u_angle
        for component in self.nodechild_types:
            attached_node_ids = self.grid[component]["node"]
            indices = [self.grid["node"]["id"].index(attached_node_id) for attached_node_id in attached_node_ids]

            for feature in from_node_to_children_features:
                output[component][feature] = [output["node"][feature][idx] for idx in indices]


        ### PART 3: Make sure that all nodes also get i, s, pf
        default_value = 0
        child_component_indices = {}

        for component in self.nodechild_types:
            attached_to_nodes = self.grid[component]["node"]
            indices = [self.grid["node"]["id"].index(attached_to_node) for attached_to_node in attached_to_nodes]
            child_component_indices[component] = indices

        n_nodes = len(self.grid["node"]["id"])

        for feature in from_children_to_node_features:
            values = [default_value]*n_nodes

            for component, indices in child_component_indices.items():
                for idx, value in zip(indices, output[component][feature]):
                    values[idx] = value

            output["node"][feature] = values

        ### PART 4: Add the node_ids to the child childs
        for component in self.nodechild_types:
            output[component]["node"] = self.grid[component]["node"]

        output["node"]["node"] = self.grid["node"]["id"]

        return output
    
    @decorators.lazy_evaluation_with_caching
    def short_circuit(self):
        output = converters.PPConverter().short_circuit(grid=self)

        return output

    def draw_grid(self, node_colors=None, ax=None):
        return GridArtist(self).draw_grid(node_colors, ax)
    
    def draw_controllables(self, controllable_ids, ax=None):
        return GridArtist(self).draw_controllables(controllable_ids=controllable_ids, ax=ax)

    @decorators.lazy_evaluation_with_caching
    def compute_additional_characteristics(self):
        return GridInspector(self)()

    @property
    @decorators.lazy_evaluation_with_caching
    def additional_characteristics(self):
        return self.compute_additional_characteristics()
    
    @property
    @decorators.lazy_evaluation_with_caching
    def all_grid_characteristics(self):
        grid = self.grid
        additional_characteristics = self.additional_characteristics

        for k, v in grid.items():
            grid[k] = {**grid[k], **additional_characteristics[k]}

        return grid

    def node_is_pq_bus(self):
        return GridInspector(grid=self).determine_if_node_is_pq_bus()

    

class GridBuilder:
    """
    Used to build the Grid from other formats (such as pgm, grid_dict)
    """
    def __init__(self, replace_gens_with_loads_bool=True, merge_multiple_pq_busses_per_node_bool=True):
        self.replace_gens_with_loads_bool = replace_gens_with_loads_bool
        self.merge_multiple_pq_busses_per_node_bool = merge_multiple_pq_busses_per_node_bool
        pass

    def from_grid_dict(self, grid_dict):
        component_dict = {"node": core.Node,
                        "line": core.Line,
                        "source": core.Source,
                        "sym_load": core.SymLoad,
                        "shunt": core.Shunt,
                        "transformer": core.Transformer,
                        "sym_gen": core.SymGen}
        
        if self.replace_gens_with_loads_bool:
            grid_dict = self.replace_gens_with_loads(grid_dict)

        if self.merge_multiple_pq_busses_per_node_bool:
            grid_dict = self.merge_multiple_pq_busses_per_node(grid_dict)
        

        components = {}

        for k in grid_dict.keys():
            if k in component_dict:
                components[k] = generalized_component_builder(component_dict[k], **grid_dict[k]) 
            elif k in ["three_winding_transformer", "asym_load", "asym_gen"]:
                raise NotImplementedError(f"component={k} not implemented")
            else:
                print(f"Warning: component {k} is not implemented, but we'll continue anyway")

        return Grid(**components)
    
    def replace_gens_with_loads(self, grid_dict):
        component = "sym_gen"
        if component in grid_dict.keys():

            n_components = len(grid_dict[component]["id"])

            if n_components > 0:
                flip_sign_for_attributes = ["p_specified", "q_specified"]
                sym_load_core_attributes = ['id', 'node', 'status', 'p_specified', 'q_specified', 'type']


            for n in range(n_components):

                init_attrs = {k: grid_dict[component][k][n] for k in sym_load_core_attributes}
                new_attrs = {k: -v if k in flip_sign_for_attributes else v for k, v in init_attrs.items()}

                for k, v in new_attrs.items():
                    if "sym_load" not in grid_dict.keys():
                        grid_dict["sym_load"] = {k: np.array([]) for k in new_attrs.keys()}
                    
                    if k not in grid_dict["sym_load"]:
                        grid_dict["sym_load"][k] = np.array([])

                    grid_dict["sym_load"][k] = np.append(grid_dict["sym_load"][k], v)

                
            del grid_dict[component]  # remove all sym_gens

        return grid_dict

    def merge_multiple_pq_busses_per_node(self, grid_dict):
        pq_bus_types = ["sym_load", "sym_gen"]
        pq_bus_types = [pqbt for pqbt in pq_bus_types if pqbt in grid_dict.keys()]

        # PART 1: Verify if there are more than one pq_busses connected to this node
        for node_id in grid_dict["node"]["id"]: # for all nodes in the grid
            total_p = 0
            total_q = 0
            n_connections = 0

            # go over the possible pq_bus_types
            for pq_bus_type in pq_bus_types:
                for i, connected_to_node_id in enumerate(grid_dict[pq_bus_type]["node"]):
                    if node_id == connected_to_node_id:
                        n_connections += 1
                        # new_id = grid_dict[pq_bus_type]["id"][i]

                        if pq_bus_type == "sym_load":
                            p = grid_dict[pq_bus_type]["p_specified"][i]
                            q = grid_dict[pq_bus_type]["q_specified"][i]

                        elif pq_bus_type == "sym_gen":
                            p = -grid_dict[pq_bus_type]["p_specified"][i]
                            q = -grid_dict[pq_bus_type]["q_specified"][i]

                        else:
                            raise NotImplementedError
                        
                        total_p += p
                        total_q += q

            # PART 2: If there are 2 or more connected pq_busses, delete them all and replace them with one merged sym_load
            if n_connections >= 2: # there are multiple pq busses connected to this node:
                # delete all the connected busses:

                first = True
                for pq_bus_type in pq_bus_types:
                    connected_indices = [i for i, _id in enumerate(grid_dict[pq_bus_type]["node"]) if _id == node_id]
                    connected_indices.sort() # sort them from lowest to highest
                    connected_indices.reverse() # reverse the order; this way we can delete each index from the end backwards, which makes accounting easier

                    
                    for idx in connected_indices:
                        if first:
                            first = False
                            _id = grid_dict[pq_bus_type]["id"][idx]
                            _type =  grid_dict[pq_bus_type]["type"][idx]

                        for k, v in grid_dict[pq_bus_type].items():
                            grid_dict[pq_bus_type][k] = np.delete(grid_dict[pq_bus_type][k], idx)


                new_attrs = {"id": _id, "node": node_id, "status": 1, "p_specified": total_p, "q_specified": total_q, "type": _type}
                for k, v in new_attrs.items():
                    if "sym_load" not in grid_dict.keys():
                        grid_dict["sym_load"] = {k: np.array([]) for k in new_attrs.keys()}
                    
                    grid_dict["sym_load"][k] = np.append(grid_dict["sym_load"][k], v)

        return grid_dict
    
    def from_pgm_input(self, pgm_input):
        grid_dict = self.pgm_input_to_grid_dict(pgm_input)
        return self.from_grid_dict(grid_dict)

    def pgm_input_to_grid_dict(self, pgm_input):
        grid_dict = {}

        for component_name in pgm_input.keys():
            component_dict = {}

            for attribute in pgm_input[component_name].dtype.names:
                component_dict[attribute] = pgm_input[component_name][attribute]

            grid_dict[component_name] = component_dict

        return grid_dict


class GridInspector:
    """
    Grid inspectors adds additional features to a grid, such as the node degree, node local loading, etc.
    """
    def __init__(self, grid, max_edge_depth=0, max_node_depth=1) -> None:
        self.grid = grid
        self.max_edge_depth = max_edge_depth
        self.max_node_depth = max_node_depth

        self.apply_per_source_methods = ["add_shortest_path_to_source", "add_impedance_magnitude_over_shortest_path_to_source", "add_reactance_over_shortest_path_to_source", "add_resistance_over_shortest_path_to_source"]
        self.utility_methods = ["find_surrounding_edges", "return_component_sign", "compute_additional_characteristics", "flexible_characteristic_adder", "find_target_order", "find_neighbors_at_depth", "find_edges_at_depth", "match_edge_nodes_to_line_id", "reorder_values", "reorder_dictionary_values_to_list", "determine_to_what_each_node_is_connected", "determine_if_node_is_pq_bus", "match_nodes_to_edge_id"]
        self.ignore_these_methods = ["add_pagerank", "add_katz_centrality", "add_node_degree", "add_real_power", "add_reactive_power", "add_local_p_ratio", "add_avg_local_loading", "add_max_local_loading"]
        
        default_methods = [m for m in dir(self) if callable(getattr(self, m)) and m.startswith("__") and m.endswith("__")]

        self.all_excluded_methods = self.apply_per_source_methods + self.utility_methods + self.ignore_these_methods + default_methods

    def __call__(self):
        return self.compute_additional_characteristics()
    
    def compute_additional_characteristics(self):
        """ For all grid components ('node', 'line', 'sym_load', etc.), compute all the additional characteristics and returns this as a dict"""
        self.graph = self.grid.graph
        self.pgm_format = self.grid.pgm_format
        self.pgm_output = self.grid.load_flow()
        self.short_circuit_output = self.grid.short_circuit()
        # self.source_object = grid
        
        
        additional_characteristics = {k: {} for k in self.grid.grid.keys()}

        for component in additional_characteristics.keys():
            if component not in self.grid.edge_types:
                additional_characteristics[component] = self.flexible_characteristic_adder(component=component)


        return additional_characteristics
    
    def flexible_characteristic_adder(self, component):
        all_methods = [attr for attr in dir(self) if callable(getattr(self, attr))] # find all callable methods in the object
        methods = [m for m in all_methods if m not in self.all_excluded_methods] # exclude methods

        additional_characteristics = {}
        for method in methods:
            key, values = getattr(self, method)(component)
            if not values is False:
                additional_characteristics[key] = values

        for method in self.apply_per_source_methods:
            for source_node in self.grid.grid["source"]["node"]:
                key, values = getattr(self, method)(component, source_node)
                if not values is False:
                    additional_characteristics[key] = values
        
        return additional_characteristics
    
    def find_target_order(self, component):
        """ Depending on the component type, it finds the right target order
        """

        if component == "node":
            target_order = self.grid.grid["node"]["id"]
        elif component in self.grid.nodechild_types:
            target_order = self.grid.grid[component]["node"]
        else:
            raise NotImplementedError(f"For {target_order} I didn't specify how to find the target order yet")
        
        return target_order
    
    def add_closeness_centrality(self, component):
        allowed_components = self.grid.node_types
        key = "closeness_centrality"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.closeness_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_betweenness_centrality(self, component):
        allowed_components = self.grid.node_types
        key = "betweenness_centrality"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.betweenness_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_pagerank(self, component):
        allowed_components = self.grid.node_types
        key = "pagerank"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.pagerank(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_katz_centrality(self, component):
        allowed_components = self.grid.node_types
        key = "katz_centrality"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.katz_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_information_centrality(self, component):
        allowed_components = self.grid.node_types
        key = "information_centrality"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)


        edge_attrs = {}
        for fn, tn, r1, x1 in zip(self.grid["line"]["from_node"], self.grid["line"]["to_node"], 
                                  self.grid["line"]["r1"], self.grid["line"]["x1"]):
            z = (r1**2 + x1**2)**0.5  # compute the impedance magnitude
            edge_attrs[(fn, tn)] = z  # add it to the edge dictionary

        graph = self.grid.graph
        nx.set_edge_attributes(graph, edge_attrs, 'z_mag')

        values = nx.information_centrality(self.graph, weight="z_mag")

        max_v = max(values.values())
        min_v = min(values.values())
        values = {node: (v - min_v)/(max_v - min_v) for node, v in values.items()} # normalize

        values = self.reorder_dictionary_values_to_list(values, target_order=target_order)
        return key, values
    

    def add_node_degree(self, component):
        allowed_components = self.grid.node_types
        key = "node_degree"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = dict(self.graph.degree())
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_degree_centrality(self, component):
        allowed_components = self.grid.node_types
        key = "degree_centrality"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.degree_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_short_circuit_resistance(self, component):
        allowed_components = self.grid.node_types
        key = "short_circuit_resistance"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        values = dict(self.short_circuit_output["rk_ohm"])
        values = self.reorder_dictionary_values_to_list(values, target_order=target_order)
        return key, values

    def add_short_circuit_reactance(self, component):
        allowed_components = self.grid.node_types
        key = "short_circuit_reactance"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        values = dict(self.short_circuit_output["xk_ohm"])
        values = self.reorder_dictionary_values_to_list(values, target_order=target_order)
        return key, values

    def add_short_circuit_impedance_magnitude(self, component):
        allowed_components = self.grid.node_types
        key = "short_circuit_impedance_magnitude"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        x_values = dict(self.short_circuit_output["xk_ohm"])
        r_values = dict(self.short_circuit_output["rk_ohm"])
        
        Z_mag_dict = {}
        for _id, x in x_values.items():
            r = r_values[_id]
            Z = complex(r, x)
            Z_mag = abs(Z)
            Z_mag_dict[_id] = Z_mag

        values = self.reorder_dictionary_values_to_list(Z_mag_dict, target_order=target_order)
        return key, values

    def add_shortest_path_to_source(self, component, source_node):
        allowed_components = self.grid.node_types
        key = f"shortest_path_to_source_{source_node}"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        dictionary = nx.shortest_path_length(self.graph, source=source_node)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_resistance_over_shortest_path_to_source(self, component, source_node):
        allowed_components = self.grid.node_types
        key = f"resistance_over_shortest_path_to_source_{source_node}"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        # paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}

        for k, v in paths_to_source.items():
            r_tot = 0
            for n1, n2 in zip(v[:-1], v[1:]):
                edge_id, _component = self.match_nodes_to_edge_id(edge=(n1, n2))
                edge_idx = list(self.pgm_format[_component]["id"]).index(edge_id)
                r_tot += self.grid.core_components[_component][edge_idx].r1 # self.pgm_format[component]["r1"][edge_idx]
            paths_to_source[k] = r_tot
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values

    def add_reactance_over_shortest_path_to_source(self, component, source_node):
        allowed_components = self.grid.node_types
        key = f"reactance_over_shortest_path_to_source_{source_node}"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        # paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}

        for k, v in paths_to_source.items():
            x_tot = 0
            for n1, n2 in zip(v[:-1], v[1:]):
                edge_id, component = self.match_nodes_to_edge_id(edge=(n1, n2))
                edge_idx = list(self.pgm_format[component]["id"]).index(edge_id)
                x_tot += self.grid.core_components[component][edge_idx].x1 #self.pgm_format[component]["x1"][edge_idx]
            paths_to_source[k] = x_tot
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values

    def add_impedance_magnitude_over_shortest_path_to_source(self, component, source_node):
        allowed_components = self.grid.node_types
        key = f"impedance_magnitude_over_shortest_path_to_source_{source_node}"

        if component not in allowed_components:
            return key, False

        target_order = self.find_target_order(component)

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        # paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}

        for k, v in paths_to_source.items():
            z_tot = 0
            for n1, n2 in zip(v[:-1], v[1:]):
                edge_id, component = self.match_nodes_to_edge_id(edge=(n1, n2))
                edge_idx = list(self.pgm_format[component]["id"]).index(edge_id)
                r = self.grid.core_components[component][edge_idx].r1 #  self.pgm_format[component]["r1"][edge_idx]
                x = self.grid.core_components[component][edge_idx].x1 # self.pgm_format[component]["x1"][edge_idx]
                z = abs(complex(r, x))
                z_tot += z 
            paths_to_source[k] = z_tot
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values
    
    def return_component_sign(self, component):
        # returns the sign corresponding to that specific component (loads are +, generators are -)
        if component in ["sym_gen", "source"]:
            sign = -1
        elif "gen" in component:
            raise NotImplementedError
        else:
            sign = 1

        return sign

    def add_real_power(self, component):
        allowed_components = self.grid.node_types
        key = "real_power"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        
        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["p"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_reactive_power(self, component):
        allowed_components = self.grid.node_types

        key = "reactive_power"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["q"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_apparant_power(self, component):
        allowed_components = self.grid.node_types

        key = "apparant_power"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["s"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_current(self, component):
        allowed_components = self.grid.node_types

        key = "current"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["i"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_u_pu(self, component):
        allowed_components = self.grid.node_types

        key = "u_pu"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["u_pu"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_u(self, component):
        allowed_components = self.grid.node_types

        key = "u"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["u"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_u_angle(self, component):
        allowed_components = self.grid.node_types

        key = "u_angle"

        if component not in allowed_components:
            return key, False

        sign = self.return_component_sign(component=component)
        

        target_order = self.find_target_order(component)

        dictionary = {_id: sign*p for _id, p in zip(self.pgm_output[component]["node"], self.pgm_output[component]["u_angle"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_local_p_ratio(self, component, depth=1):
        allowed_components = self.grid.node_types
        assert depth > 0, f"For d=0, there are no neighbors (d=1 yields first neighbors)"
        key = f"avg_local_node_p_ratio_d{depth}"       
        power_key = "s"

        if component not in allowed_components:
            return key, False
        
        target_order = self.find_target_order(component)
        dictionary = {}

        if component == "node":
            node_ids = self.grid.grid[component]["id"]  
        else:
            node_ids = self.grid.grid[component]["node"]

        for node_id in node_ids:

            node_idx = list(self.pgm_output[component]["node"]).index(node_id)
            node_power = self.return_component_sign(component) * self.pgm_output[component][power_key][node_idx]


            neighboring_nodes = self.find_neighbors_at_depth(graph=self.graph, node=node_id, depth=depth)
            neighboring_powers = []
            for neighbor in neighboring_nodes:
                for component_type in self.grid.nodechild_types:
                    if neighbor in self.grid.grid[component_type]["node"]:
                        idx = self.grid.grid[component_type]["node"].index(neighbor)
                        neighboring_power = self.return_component_sign(component_type) * self.pgm_output[component_type][power_key][idx]
                        neighboring_powers.append(neighboring_power)
                        break
                    else:
                        # if we get here, it means the for loop ended without breaking, which means the neghboring node is not attached to a specific nodechild_type, but it is only a 'node'
                        idx = self.grid.grid["node"]["id"].index(neighbor)
                        neighboring_powers.append(self.pgm_output["node"][power_key][idx])

            # after we coll
            if len(neighboring_powers) == 0 or sum(neighboring_powers) == 0:
                ratio = 0
            else:
                ratio = node_power / (sum(neighboring_powers)/len(neighboring_powers))

            dictionary[node_id] = ratio

        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def find_neighbors_at_depth(self, graph, node, depth):
        """

        Depth 0 would be: no one.
        Depth 1 would be: Node A's neighbors
        Depth 2 would be: Node A's neighbors + their neighbors
        etcetera
            """
        visited = set()
        queue = deque([(node, 0)])  # (node, depth)
        
        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth > depth:
                break  # Stop if the current depth exceeds the specified depth
            visited.add(current_node)
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
        
        return visited - {node}  # Exclude the original node

    def find_edges_at_depth(self, graph, node, depth):
        """

        Depth 0 would be: Edges connected to A
        Depth 1 would be: Edges connected to A + connected to A's neighbors
        Depth 2 would be: Edges connected to A + connected to A's neighbors + connected to their neighbors
        etcetera
            """

        visited_nodes = set()
        visited_edges = set()
        queue = deque([(node, 0)])  # (node, depth)
        
        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth > depth:
                break  # Stop if the current depth exceeds the specified depth
            visited_nodes.add(current_node)
            for neighbor in graph.neighbors(current_node):
                edge = (current_node, neighbor) if current_node < neighbor else (neighbor, current_node)
                if neighbor not in visited_nodes and edge not in visited_edges:
                    queue.append((neighbor, current_depth + 1))
                    visited_edges.add(edge)  # Moved this line outside the if block
        
        return visited_edges

    def match_nodes_to_edge_id(self, edge):
        edge_dict = self.grid.edge_dict
        for _id, fn, tn, component in zip(edge_dict["id"], edge_dict["from_node"], edge_dict["to_node"], edge_dict["component"]):
            if (fn == edge[0] and tn == edge[1]) or (fn == edge[1] and tn == edge[0]):
                return _id, component
        else:
            raise AssertionError(f"No edge found with from_node={fn} and to_node={tn} or the other way around") 

    def reorder_values(self, source_order, source_values,  target_order):
        # verify that both orders only contains unique value
        assert len(set(source_order)) == len(source_order)
        assert len(set(target_order)) == len(target_order)
        assert len(source_values) == len(source_order)

        target_values = [source_values[source_order.index(t)] for t in target_order if t in source_order]
        return target_values

    def reorder_dictionary_values_to_list(self, source_dictionary, target_order):
        return self.reorder_values(list(source_dictionary.keys()), list(source_dictionary.values()), target_order)
    
    def determine_to_what_each_node_is_connected(self):
        possible_components = self.grid.nodechild_types
        connected_to = []

        for node_id in self.grid.grid["node"]["id"]:
            for component in possible_components:
                if node_id in self.grid.grid[component]["node"]:
                    connected_to.append(component)
                    break
            else:
                connected_to.append(None)
            
        return connected_to

    def determine_if_node_is_pq_bus(self):
        """For each node check if it is connected to a PQ bus (either sym_load or sym_gen)"""
        is_pq_bus = []

        for node_id in self.grid.grid["node"]["id"]:
            node_is_pq = int(any(node_id in self.grid.grid[component]["node"] for component in self.grid.pq_bus_types))

            is_pq_bus.append(node_is_pq)
        
        return is_pq_bus
    
    def find_surrounding_edges(self, component, depth):      
        all_surrounding_edges = []
        target_order = self.find_target_order(component)

        for node_id in self.grid["node"]["id"]:
            edges = self.find_edges_at_depth(graph=self.graph, node=node_id, depth=depth)
            surrounding_edges = [self.match_nodes_to_edge_id(edge) for edge in edges]
            all_surrounding_edges.append(surrounding_edges)
        
        all_surrounding_line_ids = self.reorder_values(source_order=self.grid["node"]["id"], source_values=all_surrounding_line_ids,  target_order=target_order)
        return all_surrounding_line_ids

    def add_avg_local_loading(self, component, depth=0):
        allowed_components = self.grid.edge_types

        key = f"avg_local_loading_d{depth}"

        if component not in allowed_components:
            return key, False      

        # after this, the all_surrounding_line_ids are already in the target_order belonging to 'component'
        all_surrounding_line_ids = self.find_surrounding_edges(component, depth)

        avg_local_loadings = []
        for ids, edge_type in all_surrounding_line_ids:
            loading = 0
            for _id in ids:
                idx = list(self.pgm_output[edge_type]["id"]).index(_id)
                loading += self.pgm_output[edge_type]["loading"][idx]

            avg_local_loadings.append(loading/len(ids))

        return key, avg_local_loadings
    
    def add_max_local_loading(self, component, depth=0):
        allowed_components = self.grid.edge_types

        key = f"max_local_loading_d{depth}"

        if component not in allowed_components:
            return key, False      

        # after this, the all_surrounding_line_ids are already in the target_order belonging to 'component'
        all_surrounding_line_ids = self.find_surrounding_edges(component, depth)

        max_local_loadings = []
        for ids, edge_type in all_surrounding_line_ids:
            loadings = []
            for _id in ids:
                idx = list(self.pgm_output[edge_type]["id"]).index(_id)
                loadings.append(self.pgm_output[edge_type]["loading"][idx])

            max_local_loadings.append(max(loadings))

        return key, max_local_loadings


class DepreciatedGridInspector:
    """
    Grid inspector adds additional features to a grid, such as the degree and local loading for each node.
    """
    def __init__(self, grid, max_edge_depth=0, max_node_depth=1) -> None:
        """
        All the functions with the prefix 'add' should:
            - return a key describing the feature name, 
            - and a value containing a list, which is in the same order as how that feature is in the 'grid' dictionary
        """
        self.grid = grid.grid
        self.max_edge_depth = max_edge_depth
        self.max_node_depth = max_node_depth

        self.graph = grid.graph
        self.pgm_format = grid.pgm_format
        self.pgm_output = grid.load_flow()
        self.short_circuit_output = grid.short_circuit()


    def __call__(self):
        return self.compute_additional_characteristics()

    def compute_additional_characteristics(self):
        additional_characteristics = {k: {} for k in self.grid.keys()}

        additional_characteristics["node"] = self.compute_additional_node_characteristics()
        # additional_characteristics["sym_load"] = self.compute_additional_sym_load_characteristics()
        additional_characteristics["line"] = self.compute_additional_line_characteristics()

        return additional_characteristics
    
    def reorder_values(self, source_order, source_values,  target_order):
        # verify that both orders only contains unique value
        assert len(set(source_order)) == len(source_order)
        assert len(set(target_order)) == len(target_order)
        assert len(source_values) == len(source_order)

        target_values = [source_values[source_order.index(t)] for t in target_order if t in source_order]
        return target_values

    def reorder_dictionary_values_to_list(self, source_dictionary, target_order):
        return self.reorder_values(list(source_dictionary.keys()), list(source_dictionary.values()), target_order)
    
    def compute_general_node_characteristics(self, target_order):
        general_characteristics = {}

        key, values = self.add_closeness_centrality(target_order)
        general_characteristics[key] = values

        key, values = self.add_node_degree(target_order)
        general_characteristics[key] = values

        key, values = self.add_betweenness_centrality(target_order)
        general_characteristics[key] = values

        key, values = self.add_katz_centrality(target_order)
        general_characteristics[key] = values

        key, values = self.add_pagerank(target_order)
        general_characteristics[key] = values

        key, values = self.add_degree_centrality(target_order)
        general_characteristics[key] = values

        key, values = self.compute_initial_power(target_order)
        general_characteristics[key] = values

        key, values = self.compute_initial_current(target_order)
        general_characteristics[key] = values

        key, values = self.compute_initial_pu_voltage(target_order)
        general_characteristics[key] = values

        key, values = self.compute_short_circuit_resistance(target_order)
        general_characteristics[key] = values

        key, values = self.compute_short_circuit_reactance(target_order)
        general_characteristics[key] = values

        key, values = self.compute_short_circuit_impedance_magnitude(target_order)
        general_characteristics[key] = values

        ## Handle several sources
        for source in self.grid["source"]["node"]:
            key, values = self.add_shortest_path_to_source(target_order, source_node=source)
            general_characteristics[key] = values

            key, values = self.add_resistance_over_shortest_path_to_source(target_order, source_node=source)
            general_characteristics[key] = values

            key, values = self.add_reactance_over_shortest_path_to_source(target_order, source_node=source)
            general_characteristics[key] = values

            key, values = self.add_impedance_magnitude_over_shortest_path_to_source(target_order, source_node=source)
            general_characteristics[key] = values

        ## Handle any depth
        for depth in range(self.max_edge_depth + 1):
            key, values = self.compute_avg_local_loading(target_order, depth=depth)
            general_characteristics[key] = values

            key, values = self.compute_max_local_loading(target_order, depth=depth)
            general_characteristics[key] = values

        ## Handle any depth (except for 0 which would mean there are no neighbors)
        for depth in range(1, self.max_node_depth + 1):
            key, values = self.compute_local_p_ratio(target_order, depth=depth)
            general_characteristics[key] = values


        return general_characteristics

    def compute_initial_power(self, target_order):
        # remove source nodes from target order,since initial power isn't specified for this node
        target_order = [node_id for node_id in target_order if node_id not in self.pgm_format["source"]["node"]]

        key = f"initial_power"

        dictionary = {_id: p for _id, p in zip(self.pgm_format["sym_load"]["node"], self.pgm_format["sym_load"]["p_specified"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def compute_initial_pu_voltage(self, target_order):
        key = f"initial_pu_voltage"

        lf = self.pgm_output

        dictionary = {_id: p for _id, p in zip(lf["node"]["id"], lf["node"]["u_pu"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
            
    def compute_initial_current(self, target_order):
        # remove source nodes from target order,since initial power isn't specified for this node
        target_order = [node_id for node_id in target_order if node_id not in self.pgm_format["source"]["node"]]

        key = f"initial_current"

        dictionary = {_id: p for _id, p in zip(self.pgm_format["sym_load"]["node"], self.pgm_output["sym_load"]["i"])}
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def compute_surrounding_line_ids(self, target_order, depth):
        all_surrounding_line_ids = []

        for node_id in self.grid["node"]["id"]:
            edges = self.find_edges_at_depth(graph=self.graph, node=node_id, depth=depth)
            line_ids = [self.match_edge_nodes_to_line_id(edge) for edge in edges]
            all_surrounding_line_ids.append(line_ids)
        
        all_surrounding_line_ids = self.reorder_values(source_order=self.grid["node"]["id"], source_values=all_surrounding_line_ids,  target_order=target_order)
        return all_surrounding_line_ids
    

    def depreciated_compute_surrounding_line_ids(self, target_order, depth=1):
        """For each node, find the index of the surrounding lines"""
        if depth != 1:
            raise NotImplementedError
        
        # yes should be node, not sym_load, because in the graph it uses the node_ids, not the sym_load ids
        all_surrounding_line_ids = []
        for node_id in self.grid["node"]["id"]:
            surrounding_line_ids = []
            surrounding_edges = list(self.graph.edges(node_id))
            for edge in surrounding_edges:
                _id = self.match_edge_nodes_to_line_id(edge)
                surrounding_line_ids.append(_id)
            
            all_surrounding_line_ids.append(surrounding_line_ids)
        
        all_surrounding_line_ids = self.reorder_values(source_order=self.grid["node"]["id"], source_values=all_surrounding_line_ids,  target_order=target_order)
        return all_surrounding_line_ids
                        
    def compute_avg_local_loading(self, target_order, depth):
        key = f"avg_local_loading_d{depth}"
        all_surrounding_line_ids = self.compute_surrounding_line_ids(target_order, depth)

        avg_local_loadings = []
        for ids in all_surrounding_line_ids:
            loading = 0
            for _id in ids:
                idx = list(self.pgm_output["line"]["id"]).index(_id)
                loading += self.pgm_output["line"]["loading"][idx]

            avg_local_loadings.append(loading/len(ids))

        return key, avg_local_loadings
                  
    def compute_max_local_loading(self, target_order, depth):
        key = f"max_local_loading_d{depth}"
        all_surrounding_line_ids = self.compute_surrounding_line_ids(target_order, depth)

        max_local_loadings = []
        for ids in all_surrounding_line_ids:
            loadings = []
            for _id in ids:
                idx = list(self.pgm_output["line"]["id"]).index(_id)
                loadings.append(self.pgm_output["line"]["loading"][idx])

            max_local_loadings.append(max(loadings))

        return key, max_local_loadings
    
    def compute_local_p_ratio(self, target_order, depth):
        assert depth > 0, f"For d=0, there are no neighbors (d=1 yields first neighbors)"
        key = f"avg_local_node_p_ratio_d{depth}"       

        dictionary = {}

        for node_id in self.pgm_format["sym_load"]["node"]:
            node_idx = list(self.pgm_format["sym_load"]["node"]).index(node_id)
            node_p = self.pgm_format["sym_load"]["p_specified"][node_idx]


            neighboring_nodes = self.find_neighbors_at_depth(graph=self.graph, node=node_id, depth=depth)
            ps = []
            for neighbor in neighboring_nodes:
                if neighbor not in self.pgm_format["source"]["node"]:
                    idx = list(self.pgm_format["sym_load"]["node"]).index(neighbor)
                    ps.append(self.pgm_format["sym_load"]["p_specified"][idx])

            if len(ps) == 0:

                ratio = 0
            else:
                ratio = node_p / (sum(ps)/len(ps))

            dictionary[node_id] = ratio

        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    

    def compute_additional_node_characteristics(self):
        additional_node_characteristics = {"is_pq_bus": self.determine_if_node_is_pq_bus()}
        additional_node_characteristics = {"connected_to": self.determine_to_what_each_node_is_connected()}
        target_order = self.grid["node"]["id"]

        general_node_characteristics = self.compute_general_node_characteristics(target_order=target_order)

        additional_node_characteristics = {**additional_node_characteristics, **general_node_characteristics}
        return additional_node_characteristics
    
    def determine_to_what_each_node_is_connected(self):
        possible_components = self.grid.nodechild_types
        connected_to = []

        for node_id in self.grid["node"]["id"]:
            for component in possible_components:
                if node_id in self.grid[component]["node"]:
                    connected_to.append(component)
                    break
            else:
                connected_to.append(None)
            
        return connected_to

    def determine_if_node_is_pq_bus(self):
        """For each node check if it is connected to a PQ bus (either sym_load or sym_gen)"""
        is_pq_bus = []
        pq_bus_like_components = ["sym_load", "sym_gen"]

        for node_id in self.grid["node"]["id"]:
            node_is_pq = 0
            for component in pq_bus_like_components:
                if node_id in self.grid[component]["node"]:
                    node_is_pq = 1
            
            is_pq_bus.append(node_is_pq)
        
        return is_pq_bus

    def add_closeness_centrality(self, target_order):
        key = "closeness_centrality"
        dictionary = nx.closeness_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_betweenness_centrality(self, target_order):
        key = "betweenness_centrality"
        dictionary = nx.betweenness_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_pagerank(self, target_order):
        key = "pagerank"
        dictionary = nx.pagerank(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_katz_centrality(self, target_order):
        key = "katz_centrality"
        dictionary = nx.katz_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values

    def add_node_degree(self, target_order):
        key = "node_degree"
        dictionary = dict(self.graph.degree())
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_degree_centrality(self, target_order):
        key = "degree_centrality"
        dictionary = nx.degree_centrality(self.graph)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_shortest_path_to_source(self, target_order, source_node):
        key = f"shortest_path_to_source_{source_node}"

        dictionary = nx.shortest_path_length(self.graph, source=source_node)
        values = self.reorder_dictionary_values_to_list(dictionary, target_order=target_order)
        return key, values
    
    def add_resistance_over_shortest_path_to_source(self, target_order, source_node):
        key = f"resistance_over_shortest_path_to_source_{source_node}"

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}
        
        for k, v in paths_to_source.items():
            r_tot = 0
            assert len(v) >= 2, f"This should always be the case since v is the list of nodes between A and B, including A and B (if not you're pointing to yourself)"
            for n1, n2 in zip(v[:-1], v[1:]):
                line_id = self.match_edge_nodes_to_line_id(edge=(n1, n2))
                line_idx = list(self.pgm_format["line"]["id"]).index(line_id)
                r_tot += self.pgm_format["line"]["r1"][line_idx]
            
            paths_to_source[k] = r_tot
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values
    
    def add_reactance_over_shortest_path_to_source(self, target_order, source_node):
        key = f"reactance_over_shortest_path_to_source_{source_node}"

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}

        for k, v in paths_to_source.items():
            x_tot = 0
            assert len(v) >= 2, f"This should always be the case since v is the list of nodes between A and B, including A and B (if not you're pointing to yourself)"
            for n1, n2 in zip(v[:-1], v[1:]):
                line_id = self.match_edge_nodes_to_line_id(edge=(n1, n2))
                line_idx = list(self.pgm_format["line"]["id"]).index(line_id)
                x_tot += self.pgm_format["line"]["r1"][line_idx]
            
            paths_to_source[k] = x_tot
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values

    def add_impedance_magnitude_over_shortest_path_to_source(self, target_order, source_node):
        key = f"impedance_magnitude_over_shortest_path_to_source_{source_node}"

        paths_to_source = nx.shortest_path(self.graph, source=source_node)
        paths_to_source = {k: v for k, v in paths_to_source.items() if k not in self.pgm_format["source"]["node"]}

        for k, v in paths_to_source.items():
            z_tot = 0
            assert len(v) >= 2, f"This should always be the case since v is the list of nodes between A and B, including A and B (if not you're pointing to yourself)"
            for n1, n2 in zip(v[:-1], v[1:]):
                line_id = self.match_edge_nodes_to_line_id(edge=(n1, n2))
                line_idx = list(self.pgm_format["line"]["id"]).index(line_id)
                z_tot += self.pgm_format["line"]["r1"][line_idx]
            
            paths_to_source[k] = abs(z_tot)
        values = self.reorder_dictionary_values_to_list(paths_to_source, target_order=target_order)
        return key, values

    def compute_short_circuit_resistance(self, target_order):
        key = f"short_circuit_resistance"

        values = dict(self.short_circuit_output["rk_ohm"])
        values = self.reorder_dictionary_values_to_list(values, target_order=target_order)
        return key, values

    def compute_short_circuit_reactance(self, target_order):
        key = f"short_circuit_reactance"

        values = dict(self.short_circuit_output["xk_ohm"])
        values = self.reorder_dictionary_values_to_list(values, target_order=target_order)
        return key, values

    def compute_short_circuit_impedance_magnitude(self, target_order):
        key = f"short_circuit_impedance_magnitude"

        x_values = dict(self.short_circuit_output["xk_ohm"])
        r_values = dict(self.short_circuit_output["rk_ohm"])
        
        Z_mag_dict = {}

        for _id, x in x_values.items():
            r = r_values[_id]
            Z = complex(r, x)
            Z_mag = abs(Z)
            Z_mag_dict[_id] = Z_mag

        values = self.reorder_dictionary_values_to_list(Z_mag_dict, target_order=target_order)
        return key, values


        
    def compute_additional_sym_load_characteristics(self):
        additional_sym_load_characteristics = {}
        target_order = self.grid["sym_load"]["node"]

        general_node_characteristics = self.compute_general_node_characteristics(target_order=target_order)

        additional_sym_load_characteristics = {**additional_sym_load_characteristics, **general_node_characteristics}
        return additional_sym_load_characteristics

    def compute_additional_line_characteristics(self):
        additional_line_characteristics = {}
        target_order = self.grid["line"]["id"]

        # no characteristics to add yet

        return additional_line_characteristics

    def find_neighbors_at_depth(self, graph, node, depth):
        """

        Depth 0 would be: no one.
        Depth 1 would be: Node A's neighbors
        Depth 2 would be: Node A's neighbors + their neighbors
        etcetera
            """
        visited = set()
        queue = deque([(node, 0)])  # (node, depth)
        
        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth > depth:
                break  # Stop if the current depth exceeds the specified depth
            visited.add(current_node)
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
        
        return visited - {node}  # Exclude the original node

    def find_edges_at_depth(self, graph, node, depth):
        """

        Depth 0 would be: Edges connected to A
        Depth 1 would be: Edges connected to A + connected to A's neighbors
        Depth 2 would be: Edges connected to A + connected to A's neighbors + connected to their neighbors
        etcetera
            """

        visited_nodes = set()
        visited_edges = set()
        queue = deque([(node, 0)])  # (node, depth)
        
        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth > depth:
                break  # Stop if the current depth exceeds the specified depth
            visited_nodes.add(current_node)
            for neighbor in graph.neighbors(current_node):
                edge = (current_node, neighbor) if current_node < neighbor else (neighbor, current_node)
                if neighbor not in visited_nodes and edge not in visited_edges:
                    queue.append((neighbor, current_depth + 1))
                    visited_edges.add(edge)  # Moved this line outside the if block
        
        return visited_edges

    def match_edge_nodes_to_line_id(self, edge):
        for _id, fn, tn in zip(self.grid["line"]["id"], self.grid["line"]["from_node"], self.grid["line"]["to_node"]):
            if (fn == edge[0] and tn == edge[1]) or (fn == edge[1] and tn == edge[0]):
                return _id
        else:
            raise AssertionError(f"No edge found with from_node={fn} and to_node={tn} or the other way around") 


class GridArtist:
    def __init__(self, grid) -> None:
        self.grid = grid


    def draw_grid(self, node_colors=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        graph = self.grid.graph
        node_labels = nx.get_node_attributes(graph, 'label')
        edge_labels = nx.get_edge_attributes(graph, 'label')
        
        if node_colors is None:
            node_colors = {}
            for _id in self.grid.grid["node"]["id"]:
                if _id in self.grid.grid["source"]["node"]:
                    color = "blue"
                elif _id in self.grid.grid["sym_load"]["node"]:
                    color = "red"
                elif _id in self.grid.grid["sym_gen"]["node"]:
                    color = "green"
                else:
                    color = "skyblue"

                node_colors[_id] = color

            # node_colors = [node_colors[node] for node in graph.nodes()]

        # Compute the layout for our nodes
        pos = nx.kamada_kawai_layout(graph)

        # Draw the graph
        nx.draw(graph, pos, with_labels=False, node_size=700, node_color=[node_colors[_id] for _id in graph.nodes()],  font_size=10, font_weight='bold', ax=ax)

        # Draw node labels
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax)

        # Draw edge labels
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_weight='bold', ax=ax)

        return ax

    def draw_controllables(self, controllable_ids, ax=None):
        # initialize all nodes as uncontrollable
        node_colors = {_id: "orange" for _id in self.grid.grid["node"]["id"]}

        # if they're controllable, make them blue
        for _id in controllable_ids:
            assert _id in node_colors.keys()
            node_colors[_id] = "blue"


        return self.draw_grid(node_colors=node_colors, ax=ax)




def generalized_component_builder(component_class, **kwargs):
    n_components = len(kwargs["id"])

    components = []
    for i in range(n_components):
        component_arguments = {k: v[i] for k, v in kwargs.items()}
        components.append(component_class(**component_arguments))
    
    return components 
