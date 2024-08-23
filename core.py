"""
The base objects required for network processing.

Based on https://power-grid-model.readthedocs.io/en/stable/user_manual/components.html

Author: Flin Verdaasdonk, 29-9-2023
"""

from power_grid_model import LoadGenType
import numpy as np
import inspect

class BaseGrid:
    def __init__(self, node, line, source, sym_load, link=[], transformer=[], sym_gen=[], shunt=[]):
        self.core_components = {"node": node, "line": line, "link": link, "source": source, "sym_load": sym_load, 'transformer': transformer, 'sym_gen': sym_gen, "shunt": shunt}
        self.all_edge_types = ["line", "link", "transformer"]
        self.all_loadgen_types = ["sym_load", "sym_gen"]
        self.all_nodechild_types = self.loadgen_types + ["source", "shunt"]
        self.all_node_types = self.nodechild_types + ["node"]
        self.all_pq_bus_types = ["sym_load", "sym_gen"]
        assert all([k in self.all_node_types + self.all_edge_types + self.all_loadgen_types + self.all_nodechild_types for k in self.core_components.keys()]), f"One of the core components isn't categorized"
        assert all([isinstance(self.core_components[k], list) for k in self.core_components.keys()])

        self.grid_version = 0 # Increment in update_component_value; used by the lazy_evaluation_with_caching decorator to simply return the cached version if the grid_version is equal to the version from the last time the decorator was called

    @property
    def edge_types(self):
        method_name = inspect.currentframe().f_code.co_name  # this returns the name of the current method (e.g. 'edge_types')
        available_types = [t for t in getattr(self, f"all_{method_name}") if len(self.core_components[t]) > 0] # make a list containing the types actually in the object
        # so if there are no 'transformers' in this Grid, edge_types no longer contains transformers
        return available_types

    @property
    def loadgen_types(self):
        method_name = inspect.currentframe().f_code.co_name  # this returns the name of the current method (e.g. 'edge_types')
        available_types = [t for t in getattr(self, f"all_{method_name}") if len(self.core_components[t]) > 0]
        return available_types

    @property
    def nodechild_types(self):
        method_name = inspect.currentframe().f_code.co_name  # this returns the name of the current method (e.g. 'edge_types')
        available_types = [t for t in getattr(self, f"all_{method_name}") if len(self.core_components[t]) > 0]
        return available_types

    @property
    def node_types(self):
        method_name = inspect.currentframe().f_code.co_name  # this returns the name of the current method (e.g. 'edge_types')
        available_types = [t for t in getattr(self, f"all_{method_name}") if len(self.core_components[t]) > 0]
        return available_types

    @property
    def pq_bus_types(self):
        method_name = inspect.currentframe().f_code.co_name  # this returns the name of the current method (e.g. 'edge_types')
        available_types = [t for t in getattr(self, f"all_{method_name}") if len(self.core_components[t]) > 0]
        return available_types


    def update_component_value(self, component_name, component_idx, component_attribute, new_value):
        component = self.core_components[component_name][component_idx]
        
        if getattr(component, component_attribute) == new_value:
            # if the new value is the same as the old value, there is no need to update
            return # do nothing
        
        else:
            # if the new value is different as the old value
            self.grid_version += 1 # Increment in update_component_value; used by the lazy_evaluation_with_caching decorator to simply return the cached version if the grid_version is equal to the version from the last time the decorator was called
            setattr(component, component_attribute, new_value) # update the value
            return

    def list_update_component_values(self, component_name, component_attribute, value_list):
        assert len(value_list) == len(self.core_components[component_name])

        if list(value_list) == self.grid[component_name][component_attribute]:
            # no need to update if we're trying to update it to the same value
            return

        for idx, value in enumerate(value_list):
            self.update_component_value(component_name=component_name, component_idx=idx, component_attribute=component_attribute, new_value=value)
    
    def remove_component(self, component_name, component_idx):
        self.grid_version += 1
        del self.core_components[component_name][component_idx]
        return

    def add_component(self, component_name, component):
        self.grid_version += 1
        self.core_components[component_name].append(component)
        return

    @property
    def edge_dict(self):
        edges = {"id": [], "from_node": [], "to_node": [], "component": [], "from_status": [], "to_status": []}

        for edge_type in self.edge_types:
            for edge in self.core_components[edge_type]:
                for k in edges.keys():
                    if k == "component":
                        edges[k].append(edge_type)
                    else:
                        edges[k].append(getattr(edge, k))
                    
        return edges

### COMPONENTS AND DERIVATIVES

class Component:
    def __init__(self, id) -> None:
        self.core_attributes = {"id": id}
        self.assign_core_attributes()

    def assign_core_attributes(self):
        for k, v in self.core_attributes.items():
            setattr(self, k, v)

class Node(Component):
    def __init__(self, id, u_rated) -> None:
        super().__init__(id)
        
        self.core_attributes = {**self.core_attributes, "u_rated": u_rated}

        self.assign_core_attributes()

class Branch(Component):
    def __init__(self, id, from_node, to_node, from_status=1, to_status=1) -> None:
        super().__init__(id)

        self.core_attributes = {**self.core_attributes, "from_node": from_node, "to_node": to_node, "from_status": from_status, "to_status": to_status}
        self.assign_core_attributes()

class Line(Branch):
    def __init__(self, id, from_node, to_node, from_status=1, to_status=1, r1=0.25, x1=0.02, c1=1e-6, i_n=1_000, tan1=0.0, r0=np.nan, x0=np.nan, c0=np.nan, tan0=np.nan) -> None:


        super().__init__(id, from_node, to_node, from_status, to_status)

        self.core_attributes = {**self.core_attributes, "r1": r1, "x1": x1, "c1": c1, "i_n": i_n, "tan1": tan1, "r0": r0, "x0": x0, "c0": c0, "tan0": tan0}
        self.assign_core_attributes()

class Link(Branch):
    def __init__(self, id, from_node, to_node, to_status=1, from_status=1):

        super().__init__(id, from_node, to_node, from_status, to_status)
        self.assign_core_attributes()
        self.r1 = 1/1e6    
        self.x1 = 0    

class TestBranch(Branch):
    def __init__(self, id, from_node, to_node, to_status=1, from_status=1, foo=0):

        super().__init__(id, from_node, to_node, from_status, to_status)
        self.core_attributes = {**self.core_attributes, "foo": foo}
        
        self.assign_core_attributes()
        #self.r1 = 1/1e6        

class Transformer(Branch):
    def __init__(self, id, from_node, to_node, to_status, from_status, u1, u2, sn, uk, pk, i0, p0, winding_from, winding_to, 
             clock, tap_side, tap_pos, tap_min, tap_max, tap_nom, tap_size, uk_min, uk_max, pk_min, pk_max, r_grounding_from, x_grounding_from, r_grounding_to, x_grounding_to):
        super().__init__(id, from_node, to_node, from_status, to_status)
        self.core_attributes = {**self.core_attributes, "u1": u1, "u2": u2, "sn": sn, "uk": uk, "pk": pk, "i0": i0, "p0": p0, "winding_from": winding_from, "winding_to": winding_to, "clock": clock, 
                                "tap_side": tap_side, "tap_pos": tap_pos, "tap_min": tap_min, "tap_max": tap_max, "tap_nom": tap_nom, "tap_size": tap_size, "uk_min": uk_min, "uk_max": uk_max, "pk_min": pk_min, "pk_max": pk_max, 
                                "r_grounding_from": r_grounding_from, "x_grounding_from": x_grounding_from, "r_grounding_to": r_grounding_to, "x_grounding_to": x_grounding_to}
        
        self.assign_core_attributes()
        self.r1 = 0  # self.core_attributes["p0"] / (self.core_attributes["i0"]**2)
        self.x1 = 0

class Appliance(Component):
    def __init__(self, id, node, status=1) -> None:
        super().__init__(id)
        self.core_attributes = {**self.core_attributes, "node": node, "status": status}
        
        self.assign_core_attributes()
    
class Source(Appliance):
    def __init__(self, id, node, status=1, u_ref=1.0, u_ref_angle=0, sk=np.nan, rx_ratio=np.nan, z01_ratio=np.nan) -> None:
        super().__init__(id, node, status)
        self.core_attributes = {**self.core_attributes, "u_ref": u_ref, "u_ref_angle": u_ref_angle, "sk": sk, "rx_ratio": rx_ratio, "z01_ratio": z01_ratio}
        self.assign_core_attributes()

class LoadGenLike(Appliance):
    def __init__(self, id, node, p_specified, q_specified, status=1) -> None:
        super().__init__(id, node, status)
        self.core_attributes = {**self.core_attributes, "p_specified": p_specified, "q_specified": q_specified}
        self.assign_core_attributes()

class SymLoad(LoadGenLike):
    def __init__(self, id, node, p_specified, q_specified, status=1, type=None) -> None:

        super().__init__(id, node, p_specified, q_specified, status)
        if type is None or type == 0:
             # maybe look at https://github.com/PowerGridModel/power-grid-model/blob/3dc1552aa5ae3b5c7877ae495452f84d32a79392/power_grid_model_c/power_grid_model/include/power_grid_model/enum.hpp#L13
           
            type = LoadGenType.const_power
        elif type == 1:
            type = LoadGenType.const_impedance
        elif type == 2:
            type = LoadGenType.const_current
        elif isinstance(type, LoadGenType):
            pass
        else:
            raise NotImplementedError(f"type={type}, type(type)={type(type)}")
        

        self.core_attributes = {**self.core_attributes, "p_specified": p_specified, "q_specified": q_specified, "type": type}
        self.assign_core_attributes()

class SymGen(LoadGenLike):
    def __init__(self, id, node, p_specified, q_specified, status=1, type=None) -> None:

        super().__init__(id, node, p_specified, q_specified, status)
        if type is None or type == 0:
             # maybe look at https://github.com/PowerGridModel/power-grid-model/blob/3dc1552aa5ae3b5c7877ae495452f84d32a79392/power_grid_model_c/power_grid_model/include/power_grid_model/enum.hpp#L13
           
            type = LoadGenType.const_power
        elif type == 1:
            type = LoadGenType.const_impedance
        elif type == 2:
            type = LoadGenType.const_current
        elif isinstance(type, LoadGenType):
            pass
        else:
            raise NotImplementedError(f"type={type}, type(type)={type(type)}")
        

        self.core_attributes = {**self.core_attributes, "p_specified": p_specified, "q_specified": q_specified, "type": type}
        self.assign_core_attributes()

class Shunt(Appliance):
        def __init__(self, id, node, g1, b1, g0=0, b0=0 ,status=1) -> None:
            super().__init__(id, node, status)
            self.core_attributes = {**self.core_attributes, "g1": g1, "b1": b1, "g0":g0, "b0": b0}
            self.assign_core_attributes()    


