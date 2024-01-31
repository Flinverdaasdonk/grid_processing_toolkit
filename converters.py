"""
Does NOT import any modules from the networkprocessingtoolkit package into this script (to prevent importErrors). 
Only external modules.

Author: Flin Verdaasdonk
"""

import networkx as nx
import pandapower as pp
import numpy as np
import pandapower.shortcircuit as sc
import power_grid_model as pgm
from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model import CalculationType
from power_grid_model.validation import assert_valid_input_data
from . import decorators

# from power_grid_model import PowerGridModel, CalculationMethod, CalculationType

class Converter:
    def __init__(self):
        pass

    def convert(self, grid):
        raise NotImplementedError
    
    def __call__(self, grid):
        return self.convert(grid)
    

class NetworkxConverter(Converter):
    def __init__(self):
        super().__init__()
        

    def convert(self, grid):
        graph = nx.Graph()

        node_labels = {}
        for node_id in grid.grid["node"]["id"]:
            graph.add_node(node_id)
            node_labels[node_id] = node_id

        nx.set_node_attributes(graph, node_labels, 'label')

        edge_labels = {}
        for et in grid.edge_types:
            if et in grid.grid.keys():
                for edge_id, from_node, to_node in zip(grid.grid[et]["id"], grid.grid[et]["from_node"], grid.grid[et]["to_node"]):
                    graph.add_edge(from_node, to_node)
                    edge_labels[(from_node, to_node)] = edge_id

        nx.set_edge_attributes(graph, edge_labels, 'label')

        return graph
    

class PGMConverter(Converter):
    def __init__(self):
        super().__init__()

    def convert(self, grid):
        # see https://power-grid-model.readthedocs.io/en/stable/examples/Power%20Flow%20Example.html

        input_data = {}

        for k, v in grid.grid.items():
            n_items = len(v["id"])
            component = pgm.initialize_array("input", k, n_items)
            
            for sub_k, sub_v in v.items():
                component[sub_k] = sub_v

            input_data[k] = component

        return input_data
    
    def assert_validity(self, pgm_input_data):
        assert_valid_input_data(pgm_input_data, calculation_type=CalculationType.power_flow, symmetric=True)
    
    def load_flow(self, grid, method=None):
        if method is None:
            method = pgm.CalculationMethod.newton_raphson

        pgm_input = self.convert(grid)
        model = pgm.PowerGridModel(pgm_input)

        output_data = model.calculate_power_flow(
            symmetric=True,
            error_tolerance=1e-8,
            max_iterations=20,
            calculation_method=method)
        
        return output_data
    
    def convert_from_pgm_input(self, pgm_input):
        dictionary = {}

        for key in pgm_input.keys():
            component = {}

            for attribute in pgm_input[key].dtype.names:
                component[attribute] = pgm_input[key][attribute]

            dictionary[key] = component
        return dictionary
        

class DictionaryConverter(Converter):
    def __init__(self):
        super().__init__()

    def convert(self, grid):
        # see https://power-grid-model.readthedocs.io/en/stable/examples/Power%20Flow%20Example.html

        dic = {}
        for component_name, component_list in grid.core_components.items():
            if len(component_list) == 0:
                continue

            core_component_attributes = component_list[0].core_attributes.keys()

            # verify that all components of this type have the same attributes
            assert all(set(foo.core_attributes.keys()) == set(core_component_attributes) for foo in component_list)

            new_format = {}

            for component in component_list:
                for attr in core_component_attributes:
                    if not attr in new_format.keys():
                        new_format[attr] = []

                    new_format[attr].append(getattr(component, attr))
                    
            dic[component_name] = new_format

        return dic
        

class PPConverter(Converter):
    def __init__(self, fix_transformers=True):
        super().__init__()
        self.from_pp_net_PQ_ratio = 0 # used in the 'convert_from_pp_net' functionality to set Q as a ratio of P
        self.fix_transformers = fix_transformers

    def convert(self, grid):
        return self.convert_to_pp_network(grid)
    
    def cycle(self, grid):
        # convert from A to B to A, to verify that the results wil be the same.
        pp_net = self.convert_to_pp_network(grid)
        regrid = self.flexible_convert_from_pandapower(pp_net)
        return regrid
    
    def convert_to_pp_network(self, grid):
        # the keys I've implemented
        implemented_keys = ["node", "line", "source", "sym_load", "sym_gen", "shunt", "transformer",]

        # keys in grid that I haven't implemented
        remaining_keys = [k for k in grid.grid.keys() if k not in implemented_keys]

        # validate that there are no elements in any of these remanin keys
        for k in remaining_keys:
            assert len(grid.grid[k]) == 0 or len(grid.grid[k]["ïd"]) == 0, f"AssertionError for key={k}; \n grid.grid[k]={grid.grid[k]}"

        input_data = grid.grid

        # initialize an empty grid
        pp_network = pp.create_empty_network()

        ### BUSSES/NODES
        # Process node data and create buses in pandapower
        for i, node_id in enumerate(input_data['node']['id']):
            pp.create_bus(pp_network, vn_kv=input_data['node']['u_rated'][i] / 1e3, index=node_id)

        ### LINES
        # Process line data and create lines in pandapower
        for i, line_id in enumerate(input_data['line']['id']):
            from_bus = input_data['line']['from_node'][i]
            to_bus = input_data['line']['to_node'][i]
            r1 = input_data["line"]["r1"][i]
            x1 = input_data["line"]["x1"][i]
            c1 = input_data["line"]["c1"][i]
            i_n  = input_data["line"]["i_n"][i]
            length_km = 1  # Assuming a length of 1 km; adjust as necessary

            std_name = f"line_{i}"
            
            pp.create_std_type(pp_network, name=std_name, element="line", data={"r_ohm_per_km": r1, "x_ohm_per_km": x1, "c_nf_per_km": c1*1e9, "max_i_ka": i_n/1e3})

            pp.create_line(pp_network, from_bus, to_bus, length_km, std_type=std_name, index=line_id)

        ### LINK
        if "link" in input_data.keys():
            for i, link_id in enumerate(input_data['link']['id']):
                from_bus = input_data['link']['from_node'][i]
                to_bus = input_data['link']['to_node'][i]
                std_name = f"link_{i}"
                
                pp.create_std_type(pp_network, name=std_name, element="line", data={"r_ohm_per_km": 1/10e6, "x_ohm_per_km": 0, "c_nf_per_km": 0, "max_i_ka": 1e6})

                pp.create_line(pp_network, from_bus, to_bus, length_km=1, std_type=std_name, index=link_id)

        ### TRANSFORMER
        if "transformer" in input_data.keys():
            for i, transformer_id in enumerate(input_data['transformer']['id']):
                # print("Warning! Not sure if TRANSFORMER is implemented correctly; verify with network conversions")

                from_bus = input_data['transformer']['from_node'][i]
                to_bus = input_data['transformer']['to_node'][i]

                from_bus_idx = input_data["node"]["id"].index(from_bus)
                to_bus_idx = input_data["node"]["id"].index(to_bus)

                from_bus_voltage = input_data["node"]["u_rated"][from_bus_idx]
                to_bus_voltage = input_data["node"]["u_rated"][to_bus_idx]

                hv_bus = from_bus if from_bus_voltage > to_bus_voltage else to_bus
                lv_bus = from_bus if from_bus_voltage <= to_bus_voltage else to_bus

                hv_bus_idx = input_data["node"]["id"].index(hv_bus)
                lv_bus_idx = input_data["node"]["id"].index(lv_bus)

                pp.create_transformer_from_parameters(net=pp_network, 
                                                    name=f"transformer_{i}",
                                                    hv_bus=hv_bus, 
                                                    lv_bus=lv_bus,
                                                    sn_mva = input_data["transformer"]["sn"][i] / 1e6,
                                                    vn_hv_kv = input_data["node"]["u_rated"][hv_bus_idx],
                                                    vn_lv_kv = input_data["node"]["u_rated"][lv_bus_idx],
                                                    vkr_percent = 100 * input_data["transformer"]["pk"][i] / input_data["transformer"]["sn"][i], # Calculated according to pandapower docs (see Note)
                                                    vk_percent = 100 * input_data["transformer"]["uk"][i], 
                                                    pfe_kw = input_data["transformer"]["pk"][i] / 1e3,
                                                    i0_percent = input_data["transformer"]["pk"][i] / input_data["transformer"]["sn"][i], # rated current isn't an arg for pgm models, so I base it off rated power
                                                    vector_group = "Dyn", # No clue; something about Delta Y connection, but no clue what I should pick
                                                    vk0_percent = 100 * 0.15, # No clue
                                                    vkr0_percent = 100 * 0.15,  # No clue
                                                    mag0_percent = 100 * 0.15,  # No clue
                                                    mag0_rx = 10, # No clue
                                                    si0_hv_partial = 100 * 0.5, # No clue
                                                    index=transformer_id)



        ### SYM LOAD
            # Process symmetric load data and create loads in pandapower
        if "sym_load" in input_data.keys():
            for i, node_id in enumerate(input_data['sym_load']['node']):
                sym_load_id = input_data["sym_load"]["id"][i]
                p_specified = input_data['sym_load']['p_specified'][i]
                q_specified = input_data['sym_load']['q_specified'][i]

                pp.create_load(pp_network, node_id, p_mw=p_specified / 1e6, 
                                q_mvar=q_specified / 1e6, index=sym_load_id)

        ### SYM_GEN
        if "sym_gen" in input_data.keys():
            for i, node_id in enumerate(input_data['sym_gen']['node']):
                sym_gen_id = input_data["sym_gen"]["id"][i]
                p_specified = input_data['sym_gen']['p_specified'][i]
                q_specified = input_data['sym_gen']['q_specified'][i]

                default_sn_mva = 100e6
                default_k = 0.1

                pp.create_sgen(pp_network, 
                               node_id, 
                               p_mw=p_specified / 1e6, 
                               q_mvar=q_specified / 1e6, 
                               sn_mva=default_sn_mva, 
                               k=default_k,
                               index=sym_gen_id)
                

        ### SHUNT
        if "shunt" in input_data.keys():
            for i, node_id in enumerate(input_data['shunt']['node']):
                # print("Warning! Not sure if SHUNT is implemented correctly; verify with network conversions")
                shunt_id = input_data["shunt"]["id"][i]
                p_specified = grid.load_flow()["shunt"]["p"][i]
                q_specified = grid.load_flow()["shunt"]["q"][i]
                #p_specified = input_data['shunt']['p_specified'][i]
                #q_specified = input_data['shunt']['q_specified'][i]

                pp.create.create_shunt(net=pp_network, 
                                    bus=node_id, 
                                    p_mw=p_specified/1e6, 
                                    q_mvar=q_specified/1e6, 
                                    index=shunt_id)

            
        ### SOURCE
            # Process source data and create external grid in pandapower
        if "source" in input_data.keys():
            for i, node_id in enumerate(input_data['source']['node']):
                u_ref = input_data['source']['u_ref'][i]
                source_id = input_data["source"]["id"][i]
                pp.create_ext_grid(pp_network, node_id, vm_pu=u_ref, index=source_id)

        return pp_network
    
    def short_circuit(self, grid):
        # some random default values suggested by chatgpt
        grid.pp_format.ext_grid['s_sc_max_mva'] = 1000
        grid.pp_format.ext_grid['rx_max'] = 0.1

        # calculate short circuit
        sc.calc_sc(grid.pp_format)

        return grid.pp_format.res_bus_sc
    
    @decorators.suppress_print
    @decorators.suppress_warnings
    def flexible_convert_from_pandapower(self, pp_net, assert_validity=True):
        """
        This simply uses the power_grid_model_io pandapower converter
        However, that converter breaks when the pp_net has 'generators', which are PV-busses which pgm doesn't support yet
        So I've hacked something together the handles the PV-busses seperately

    
        """
        different_name_dict = {"sym_gen": "sgen", "source": "ext_grid", "node": "bus", "sym_load": "load", "transformer": "trafo"}

        pp_net = self.replace_pp_gen_with_sgen(pp_net) # this also runs fine 
        pgm_input, extra_info = PandaPowerConverter().load_input_data(pp_net)

        # this is a hack; replace all nans in tan1 with 0
        pgm_input["line"]["tan1"][np.isnan(pgm_input["line"]["tan1"])] = 0

        # this is a hack; sometimes after the conversion for some reason the symloads are copied thrice. This fixes that
        for pgm_name in pgm_input.keys():
            pp_name = different_name_dict.get(pgm_name, pgm_name)
            if len(pgm_input[pgm_name]["id"]) != len(pp_net[pp_name]):
                # print(f"There are thrice as many {pgm_name}'s after conversion from pp")
                assert len(pgm_input[pgm_name]["id"]) == 3*len(pp_net[pp_name]), f"Which is what happens during the conversion"
                pgm_input[pgm_name] = pgm_input[pgm_name][:len(pp_net[pp_name])]

        

        if 'transformer' in pgm_input.keys() and self.fix_transformers:
            eps = 1e-6
            # common error for transformers: tap_pos not between min and max
            pgm_input["transformer"]["tap_min"][np.isnan(pgm_input["transformer"]["tap_min"])] = 0
            pgm_input["transformer"]["tap_max"][np.isnan(pgm_input["transformer"]["tap_max"])] = 1
            pgm_input["transformer"]["tap_pos"][np.isnan(pgm_input["transformer"]["tap_pos"])] = 1

            pgm_input["transformer"]["tap_pos"] = np.clip(pgm_input["transformer"]["tap_pos"], pgm_input["transformer"]["tap_min"], pgm_input["transformer"]["tap_max"])
            pgm_input["transformer"]["tap_size"][np.isnan(pgm_input["transformer"]["tap_size"])] = 0
            pgm_input["transformer"]["tap_size"] = np.clip(pgm_input["transformer"]["tap_size"], 0, None)

            # common error for transformers: uk_min not between 0 and 1
            pgm_input["transformer"]["uk_min"][np.isnan(pgm_input["transformer"]["uk_min"])] = 0.1 - eps
            pgm_input["transformer"]["uk_min"] = np.clip(pgm_input["transformer"]["uk_min"], 0 + eps, 1 - eps)
            

            # common error for transformers: uk_max not between 0 and 1
            pgm_input["transformer"]["uk_max"][np.isnan(pgm_input["transformer"]["uk_max"])] = 0.1 + eps
            pgm_input["transformer"]["uk_max"] = np.clip(pgm_input["transformer"]["uk_max"], 0 + eps, 1 - eps)
            
            # common error for transformers: 'i0' is not greater than (or equal to) p0/sn for 4 transformers.
            pgm_input["transformer"]["i0"][np.isnan(pgm_input["transformer"]["i0"])] = 0.5
            pgm_input["transformer"]["i0"] = np.clip(pgm_input["transformer"]["i0"], pgm_input["transformer"]["p0"]/pgm_input["transformer"]["sn"], 1 - eps)   

            # common error for transformers: 'pk' is not greater than (or equal to) zero
            pgm_input["transformer"]["pk"] = np.clip(pgm_input["transformer"]["pk"], 0, None)  

            # common error for transformers: 'pk_min' is not greater than (or equal to) zero
            pgm_input["transformer"]["pk_min"] = np.clip(pgm_input["transformer"]["pk_min"], 0, None)  

            # common error for transformers: 'pk_max' is not greater than (or equal to) zero
            pgm_input["transformer"]["pk_max"] = np.clip(pgm_input["transformer"]["pk_max"], 0, None)

            # common error is that, for transformers, 'uk' not between 0, 1
            pgm_input["transformer"]["uk"][np.isnan(pgm_input["transformer"]["uk"])] = 0.1
            pgm_input["transformer"]["uk"] = np.clip(pgm_input["transformer"]["uk"], 0+eps, 1-eps)
            
            # common error is that, for transformers, is not greater than (or equal to) pk/sn
            pgm_input["transformer"]["uk"] = np.clip(pgm_input["transformer"]["uk"], pgm_input["transformer"]["pk"]/pgm_input["transformer"]["sn"], None)
            

        if assert_validity:  
            PGMConverter().assert_validity(pgm_input)       


        dictionary = PGMConverter().convert_from_pgm_input(pgm_input)
        return dictionary
    
    def replace_pp_gen_with_sgen(self, pp_net):
        # Replace 'gen' components with 'sgen' components
        
        # run load-flow; required to find the right q_mvar per gen
        pp.runpp(pp_net, numba=False)



        for gen_idx, gen in pp_net.gen.iterrows():


            # Create sgen component
            pp.create_sgen(pp_net, bus=gen.bus, p_mw=gen.p_mw, q_mvar=pp_net.res_gen["q_mvar"][gen_idx], name=gen.name)


        # Drop all generators from the network
        pp_net.gen.drop(pp_net.gen.index, inplace=True)


        for res_table in [t for t in pp_net.keys() if t.startswith('res_')]:
            del pp_net[res_table]

        return pp_net