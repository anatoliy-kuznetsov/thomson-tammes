import time
import sys
import itertools
import numpy as np
import pyomo.environ as pyo
import gurobipy as gp
from dataclasses import dataclass

@dataclass
class LinearInequality:
    """
    Convention: a_x * x + a_y * y + a_z * z <= RHS
    """
    x_coefficient: float
    y_coefficient: float
    z_coefficient: float
    right_hand_side: float

def matlab_code_for_inequality(inequality: LinearInequality, variable_bounds: dict, inequality_index: int, resolution: int) -> str:
    if inequality.z_coefficient is None:
        return ""
    code = f"x{inequality_index} = linspace({variable_bounds['x'][0]}, {variable_bounds['x'][1]}, {resolution});\n"
    code += f"y{inequality_index} = linspace({variable_bounds['y'][0]}, {variable_bounds['y'][1]}, {resolution});\n"
    code += f"[X{inequality_index}, Y{inequality_index}] = meshgrid(x{inequality_index}, y{inequality_index});\n"
    code += f"""surf(X{inequality_index}, Y{inequality_index}, ({inequality.right_hand_side} - ({inequality.x_coefficient}) * X{inequality_index} - ({inequality.y_coefficient}) * Y{inequality_index}) ./ ({inequality.z_coefficient}), 'edgecolor', 'none', 'facecolor', 'g', 'facealpha', 0.5);\n"""
    return code

class SpherePartitioning:
    def __init__(self, free_electron_count: int, minimum_distance: float):
        self.free_electron_count = free_electron_count
        self.minimum_distance = minimum_distance
        self.generate_partition(use_new_formulation=True)
        self.enumerate_regions()
        self.inequalities = {}
        self.variable_bounds = {}
        self.inequalities_and_bounds_from_partition()
        self.considered_assignments = {}

    def generate_partition(self, use_new_formulation: bool):
        """
        One electron will always be fixed to the "north pole". Generates a partition of the
        remaining sphere's surface in rows of latitude-longitude "rectangles" with the exception of
        the region near the "south pole", which is also a spherical cap. The rows are guaranteed to have
        the same length, except for the last row. For example, the Mercator projection of three rows with
        a row length of 5 looks like this:
        ---------------------------------------------------
        |         North cap (doesn't count as a row)      |
        ---------------------------------------------------
        |         |         |         |         |         |
        ---------------------------------------------------
        |         |         |         |         |         |
        ---------------------------------------------------
        |     South cap (counts as a row with length 1)   |
        ---------------------------------------------------
        Note that the last row is the south cap and has a length of 1, but the other rows have the nominal
        row length.
        
        We will use an optimization solver to solve a sequence of problems increasing the number of rows.
        In each problem, we seek to minimize the row length. This is a heuristic method for generating a
        small number of regions, which will help control the combinatorics of electron assignment.
        """
        # upper polar angle of the top row
        phi_start = np.arccos((2 - (self.minimum_distance ** 2)) / 2)
        # bottom polar angle of the bottom row
        phi_end = np.pi - np.arcsin(self.minimum_distance / 2)
        row_count = 1
        solved = False
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1)
        env.start()
        while not solved:
            if use_new_formulation:
                model = pyo.ConcreteModel()
                model.i = pyo.Set(initialize=[i for i in range(row_count)])
                model.phi = pyo.Var(model.i, domain=pyo.Reals, bounds=(0, np.pi))
                model.theta = pyo.Var(domain=pyo.Reals, bounds=(0, np.pi / 2))
                model.z = pyo.Var(domain=pyo.Integers)
                def cell_diameter_bound(model, i):
                    # phi_lower_bound = phi_start + sum(model.phi[j] for j in range(i))
                    return pyo.sqrt( 4 - pyo.cos(model.theta - model.phi[i]) - 2 * pyo.cos(model.phi[i]) - pyo.cos(model.theta + model.phi[i])
                                        + pyo.cos(model.theta - model.phi[i] - 2 * (phi_start + sum(model.phi[j] for j in range(i)))) 
                                        - 2 * pyo.cos(model.phi[i] + 2 * (phi_start + sum(model.phi[j] for j in range(i))))
                                        + pyo.cos(model.theta + model.phi[i] + 2 * (phi_start + sum(model.phi[j] for j in range(i))))
                                        ) <= self.minimum_distance * np.sqrt(2)
                model.cell_diameter_bound = pyo.Constraint(model.i, rule=cell_diameter_bound)
                def polar_angle_span(model):
                    return phi_start + sum(model.phi[i] for i in range(row_count)) == phi_end
                model.polar_angle_span = pyo.Constraint(rule=polar_angle_span)
                def equal_rows(model):
                    return model.z * model.theta == 2 * np.pi
                model.equal_rows = pyo.Constraint(rule=equal_rows)
                def z_lower_bound(model):
                    return model.z >= 4
                model.z_lower_bound = pyo.Constraint(rule=z_lower_bound)
                def obj(model):
                    return model.z
                model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
                model.pprint()
                opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
                opt.options["solver"] = "scip"
                results = opt.solve(model, tee=True, keepfiles=False, symbolic_solver_labels=True, add_options=['Option optcr = 0;'])
                solved = (results.solver.termination_condition == pyo.TerminationCondition.optimal)
                if solved:
                    self.row_count = row_count + 1
                    self.row_length = int(pyo.value(model.obj))
                    self.row_starts = [phi_start]
                    for i in range(row_count):
                        self.row_starts.append(self.row_starts[i] + pyo.value(model.phi[i]))
                    return
                row_count += 1
            else:
                model = gp.Model(name="partition", env=env)
                phi = model.addVars(row_count, lb=0, ub=(phi_end - phi_start), vtype=gp.GRB.CONTINUOUS, name="phi")
                theta = model.addVar(lb=0, ub=(np.pi / 2), vtype=gp.GRB.CONTINUOUS, name="theta")
                a1 = model.addVars(row_count, lb=-(4 * np.pi), ub=(4 * np.pi), vtype=gp.GRB.CONTINUOUS, name="a1")
                a2 = model.addVars(row_count, lb=-(4 * np.pi), ub=(4 * np.pi), vtype=gp.GRB.CONTINUOUS, name="a2")
                a3 = model.addVars(row_count, lb=-(4 * np.pi), ub=(4 * np.pi), vtype=gp.GRB.CONTINUOUS, name="a3")
                a4 = model.addVars(row_count, lb=-(4 * np.pi), ub=(4 * np.pi), vtype=gp.GRB.CONTINUOUS, name="a4")
                a5 = model.addVars(row_count, lb=-(4 * np.pi), ub=(4 * np.pi), vtype=gp.GRB.CONTINUOUS, name="a5")
                b1 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b1")
                b2 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b2")
                b3 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b3")
                b4 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b4")
                b5 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b5")
                b6 = model.addVars(row_count, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="b6")
                model.addConstrs(a1[i] == theta - phi[i] for i in range(row_count))
                model.addConstrs(a2[i] == theta + phi[i] for i in range(row_count))
                model.addConstrs(a3[i] == theta - phi[i] - 2 * (phi_start + sum(phi[j] for j in range(i))) for i in range(row_count))
                model.addConstrs(a4[i] == phi[i] + 2 * (phi_start + sum(phi[j] for j in range(i))) for i in range(row_count))
                model.addConstrs(a5[i] == theta + phi[i] + 2 * (phi_start + sum(phi[j] for j in range(i))) for i in range(row_count))
                for i in range(row_count):
                    model.addGenConstrCos(a1[i], b1[i])
                    model.addGenConstrCos(phi[i], b2[i])
                    model.addGenConstrCos(a2[i], b3[i])
                    model.addGenConstrCos(a3[i], b4[i])
                    model.addGenConstrCos(a4[i], b5[i])
                    model.addGenConstrCos(a5[i], b6[i])
                c = model.addVars(row_count, lb=0, ub=(2 * (self.minimum_distance ** 2)), vtype=gp.GRB.CONTINUOUS, name="c")
                model.addConstrs(c[i] == 4 - b1[i] - 2 * b2[i] - b3[i] + b4[i] - 2 * b5[i] + b6[i] for i in range(row_count))
                row_length = model.addVar(lb=3, ub=1000, vtype=gp.GRB.INTEGER, name="row_length")
                model.addConstr(row_length * theta == 2 * np.pi)
                model.addConstr(phi_start + sum(phi[i] for i in range(row_count)) == phi_end)
                model.setObjective(row_length, sense=gp.GRB.MINIMIZE)
                gp.setParam("LogToConsole", 1)
                gp.setParam("FuncNonlinear", 1)
                gp.setParam("Threads", 1)
                model.optimize()
                solved = (model.status == gp.GRB.OPTIMAL)
                if solved:
                    self.row_count = row_count + 1
                    self.row_length = int(model.getObjective().getValue())
                    self.row_starts = [phi_start]
                    for i in range(row_count):
                        self.row_starts.append(self.row_starts[i] + model.getVarByName(f"phi[{i}]").getAttr("X"))
                    return
                row_count += 1

    def enumerate_regions(self):
        self.regions = [(i, j) for i in range(self.row_count - 1) for j in range(self.row_length)]
        self.regions.append((self.row_count - 1, 0))

    def inequalities_and_bounds_from_partition(self):
        azimuthal_extent = 2 * np.pi / self.row_length
        for region in self.regions:
            self.variable_bounds[region] = {}
            self.inequalities[region] = []
            azimuthal_range = (region[1] * azimuthal_extent, (region[1] + 1) * azimuthal_extent)
            if region[0] < self.row_count - 1:
                polar_range = (self.row_starts[region[0]], self.row_starts[region[0] + 1])
            else:
                # bottom row: x and y bounds not based on angles, and no inequalities to write
                z_upper = np.cos(self.row_starts[self.row_count - 1]) # fact: z_upper < 0 for every N > 2
                self.variable_bounds[region]["z"] = (-1, z_upper)
                largest_radius = np.sqrt(1 - z_upper ** 2)
                self.variable_bounds[region]["x"] = (-largest_radius, largest_radius)
                self.variable_bounds[region]["y"] = (-largest_radius, largest_radius)
                return
            # bounds on variables
            self.variable_bounds[region]["z"] = (np.cos(polar_range[1]), np.cos(polar_range[0]))
            if polar_range[1] < np.pi / 2:
                # northern hemisphere
                innermost_radius = np.sin(polar_range[0])
                outermost_radius = np.sin(polar_range[1])
            elif polar_range[0] > np.pi / 2:
                # southern hemisphere
                innermost_radius = np.sin(polar_range[1])
                outermost_radius = np.sin(polar_range[0])
            else:
                # includes equator
                innermost_radius = np.sin(polar_range[0]) if abs(polar_range[0] - np.pi / 2) > abs(polar_range[1] - np.pi / 2) else np.sin(polar_range[1])
                outermost_radius = 1
            x_endpoints = (innermost_radius * np.cos(azimuthal_range[0]), innermost_radius * np.cos(azimuthal_range[1]), 
                           outermost_radius * np.cos(azimuthal_range[0]), outermost_radius * np.cos(azimuthal_range[1]))
            y_endpoints = (innermost_radius * np.sin(azimuthal_range[0]), innermost_radius * np.sin(azimuthal_range[1]), 
                           outermost_radius * np.sin(azimuthal_range[0]), outermost_radius * np.sin(azimuthal_range[1]))
            # both azimuthal endpoints are in [0, 2pi]
            x_upper_bound = outermost_radius if np.equal(azimuthal_range[0], 0) or np.equal(azimuthal_range[1], 2 * np.pi) else max(x_endpoints)
            x_lower_bound = -outermost_radius if azimuthal_range[0] <= np.pi <= azimuthal_range[1] else min(x_endpoints)
            self.variable_bounds[region]["x"] = (x_lower_bound, x_upper_bound)
            y_upper_bound = outermost_radius if azimuthal_range[0] <= np.pi / 2 <= azimuthal_range[1] else max(y_endpoints)
            y_lower_bound = -outermost_radius if azimuthal_range[0] <= 3 * np.pi / 2 <= azimuthal_range[1] else min(y_endpoints)
            self.variable_bounds[region]["y"] = (y_lower_bound, y_upper_bound)
            self.inequalities[region].append(
                calculate_inner_approximation(polar_range, azimuthal_range)
            )
            self.inequalities[region].extend(
                angle_ranges_to_inequalities(azimuthal_range)
            )
            # clean up numerics
            for inequality in self.inequalities[region]:
                inequality.x_coefficient = 0 if abs(inequality.x_coefficient) < 1e-12 else inequality.x_coefficient
                inequality.y_coefficient = 0 if abs(inequality.y_coefficient) < 1e-12 else inequality.y_coefficient
                inequality.z_coefficient = 0 if abs(inequality.z_coefficient) < 1e-12 else inequality.z_coefficient
                inequality.right_hand_side = 0 if abs(inequality.right_hand_side) < 1e-12 else inequality.right_hand_side

    def consider_assignment(self, assignment: tuple[tuple[int]], putative_minimum_energies: dict, subproblem_solver: str):
        model = pyo.ConcreteModel()
        model.i = pyo.Set(initialize=[i for i in range(self.free_electron_count)])
        def get_x_bounds(model, i):
            return (self.variable_bounds[assignment[i]]["x"][0], self.variable_bounds[assignment[i]]["x"][1])
        model.x = pyo.Var(model.i, domain=pyo.Reals, bounds=get_x_bounds)
        def get_y_bounds(model, i):
            if i == 0:
                return (0, 0)
            return (self.variable_bounds[assignment[i]]["y"][0], self.variable_bounds[assignment[i]]["y"][1])
        model.y = pyo.Var(model.i, domain=pyo.Reals, bounds=get_y_bounds)
        def get_z_bounds(model, i):
            return (self.variable_bounds[assignment[i]]["z"][0], self.variable_bounds[assignment[i]]["z"][1])
        model.z = pyo.Var(model.i, domain=pyo.Reals, bounds=get_z_bounds)
        interaction_set = [(i, j) for i in range(self.free_electron_count) for j in range(i + 1, self.free_electron_count)]
        model.d = pyo.Var(interaction_set, domain=pyo.NonNegativeReals, bounds=(self.minimum_distance ** 2, 4))
        def sphere_surface(model, i):
            return model.x[i] ** 2 + model.y[i] ** 2 + model.z[i] ** 2 == 1
        model.sphere_surface = pyo.Constraint(model.i, rule=sphere_surface)
        model.geometric_constraints = pyo.ConstraintList()
        for index, region in enumerate(assignment):
            for inequality in self.inequalities[region]:
                if inequality.z_coefficient is None:
                    model.geometric_constraints.add(expr = (inequality.x_coefficient) * model.x[index] + (inequality.y_coefficient) * model.y[index] <= inequality.right_hand_side)
                else:
                    model.geometric_constraints.add(expr = (inequality.x_coefficient) * model.x[index] + (inequality.y_coefficient) * model.y[index] + (inequality.z_coefficient) * model.z[index] <= inequality.right_hand_side)
        def sq_distance_def(model, i, j):
            return model.d[i,j] == 2 - 2 * (model.x[i] * model.x[j] + model.y[i] * model.y[j] + model.z[i] * model.z[j])
        model.distances = pyo.Constraint(interaction_set, rule=sq_distance_def)
        def z_relation(model, i):
            if i == 0:
                return pyo.Constraint.Skip
            return model.z[0] >= model.z[i]
        model.z_relation = pyo.Constraint(model.i, rule=z_relation)
        def total_energy(model):
            return sum(1 / pyo.sqrt(2 - 2 * model.z[i]) for i in range(self.free_electron_count)) + sum(
                       1 / pyo.sqrt(model.d[i,j]) 
                       for i in range(self.free_electron_count) for j in range(i + 1, self.free_electron_count))
        model.obj = pyo.Objective(rule=total_energy, sense=pyo.minimize)
        model.pprint()
        if subproblem_solver == "baron":
            opt = pyo.SolverFactory('baron', executable='/usr/local/gamsexp/baron')
            results = opt.solve(model, options={'maxtime': -1, 'allowipopt': 0}, tee=True, keepfiles=False, symbolic_solver_labels=False)
        elif subproblem_solver == "scip":
            opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
            opt.options["solver"] = "scip"
            results = opt.solve(model, tee=True, keepfiles=True, symbolic_solver_labels=True, add_options=['Option optcr = 0;', f'GAMS_MODEL.cutoff={putative_minimum_energies[self.free_electron_count + 1]}'])
        elif subproblem_solver == "gurobi":
            opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
            results = opt.solve(model, tee=True, add_options=['Option optcr = 0;', 'Option nlp=gurobi;', 'Option threads=1;'])
        
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            self.considered_assignments[frozenset(assignment)] = np.Inf
        elif results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            self.considered_assignments[frozenset(assignment)] = pyo.value(model.obj)
            for i in range(self.free_electron_count):
                print(f"(x{i}, y{i}, z{i}): ({model.x[i].value}, {model.y[i].value}, {model.z[i].value})")
        else:
            exit(f"Solve failed for assignment {assignment} of N = {self.free_electron_count + 1} with status {results.solver.status} and termination condition {results.solver.termination_condition}")
    
    def consider_assignment_tammes(self, assignment: tuple[tuple[int]]):
        model = pyo.ConcreteModel()
        model.i = pyo.Set(initialize=[i for i in range(self.free_electron_count)])
        def get_x_bounds(model, i):
            return (self.variable_bounds[assignment[i]]["x"][0], self.variable_bounds[assignment[i]]["x"][1])
        model.x = pyo.Var(model.i, domain=pyo.Reals, bounds=get_x_bounds)
        def get_y_bounds(model, i):
            if i == 0:
                return (0, 0)
            return (self.variable_bounds[assignment[i]]["y"][0], self.variable_bounds[assignment[i]]["y"][1])
        model.y = pyo.Var(model.i, domain=pyo.Reals, bounds=get_y_bounds)
        def get_z_bounds(model, i):
            return (self.variable_bounds[assignment[i]]["z"][0], self.variable_bounds[assignment[i]]["z"][1])
        model.z = pyo.Var(model.i, domain=pyo.Reals, bounds=get_z_bounds)
        interaction_set = [(i, j) for i in range(self.free_electron_count) for j in range(i + 1, self.free_electron_count)]
        model.d = pyo.Var(interaction_set, domain=pyo.NonNegativeReals, bounds=(self.minimum_distance ** 2, 4))
        def sphere_surface(model, i):
            return model.x[i] ** 2 + model.y[i] ** 2 + model.z[i] ** 2 == 1
        model.sphere_surface = pyo.Constraint(model.i, rule=sphere_surface)
        model.geometric_constraints = pyo.ConstraintList()
        for index, region in enumerate(assignment):
            for inequality in self.inequalities[region]:
                if inequality.z_coefficient is None:
                    model.geometric_constraints.add(expr = (inequality.x_coefficient) * model.x[index] + (inequality.y_coefficient) * model.y[index] <= inequality.right_hand_side)
                else:
                    model.geometric_constraints.add(expr = (inequality.x_coefficient) * model.x[index] + (inequality.y_coefficient) * model.y[index] + (inequality.z_coefficient) * model.z[index] <= inequality.right_hand_side)
        def sq_distance_def(model, i, j):
            return model.d[i,j] == 2 - 2 * (model.x[i] * model.x[j] + model.y[i] * model.y[j] + model.z[i] * model.z[j])
        model.distances = pyo.Constraint(interaction_set, rule=sq_distance_def)
        def z_relation(model, i):
            if i == 0:
                return pyo.Constraint.Skip
            return model.z[0] >= model.z[i]
        model.z_relation = pyo.Constraint(model.i, rule=z_relation)
        model.minimal_sq_distance = pyo.Var(domain=pyo.NonNegativeReals)
        def minimal_sq_distance_def(model, i, j):
            return model.minimal_sq_distance <= model.d[i,j]
        model.minimal_sq_distance_def = pyo.Constraint(interaction_set, rule=minimal_sq_distance_def)
        def sq_distance_from_north_pole_def(model, i):
            return model.x[i] * model.x[i] + model.y[i] * model.y[i] + (model.z[i] - 1) * (model.z[i] - 1) >= model.minimal_sq_distance
        model.sq_distance_from_north_pole_def = pyo.Constraint(model.i, rule=sq_distance_from_north_pole_def)
        def objective(model):
            return model.minimal_sq_distance
        model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
        model.pprint()
        # opt = pyo.SolverFactory('baron', executable='/usr/local/gamsexp/baron')
        # results = opt.solve(model, options={'maxtime': -1, 'allowipopt': 0}, tee=True, keepfiles=False, symbolic_solver_labels=False)
        # scip
        opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
        opt.options["solver"] = "scip"
        results = opt.solve(model, tee=False, symbolic_solver_labels=True, add_options=['Option optcr = 0;'])
        # gurobi
        # opt = pyo.SolverFactory('gurobi')
        # opt.options["threads"] = 1
        # opt.options["mipgap"] = 0
        # opt.options["nonconvex"] = 2
        # opt.options["timelimit"] = 1e9
        # results = opt.solve(model, tee=True)
        if results.solver.termination_condition in (pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded):
            self.considered_assignments[frozenset(assignment)] = np.Inf
        elif results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            self.considered_assignments[frozenset(assignment)] = np.sqrt(pyo.value(model.obj))
            for i in range(self.free_electron_count):
                print(f"(x{i}, y{i}, z{i}): ({model.x[i].value}, {model.y[i].value}, {model.z[i].value})")
        else:
            exit(f"Solve failed for assignment {assignment} of N = {self.free_electron_count + 1} with status {results.solver.status} and termination condition {results.solver.termination_condition}")

def angles_to_point(azimuthal_angle: float, polar_angle: float) -> tuple[float]:
    """
    Assumes unit radius
    """
    x = np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = np.cos(polar_angle)
    return x, y, z

def calculate_inner_approximation(polar_range, azimuthal_range):
    """
    Assumes unit radius
    """
    phi_1 = polar_range[0]
    phi_2 = polar_range[1]
    theta_1 = azimuthal_range[0]
    theta_2 = azimuthal_range[1]
    x_coefficient = ((np.cos(phi_1) - np.cos(phi_2))*(np.sin(theta_1) - np.sin(theta_2))) / (np.sin(theta_1 - theta_2) * np.sin(phi_1 - phi_2))
    y_coefficient = -((np.cos(theta_1) - np.cos(theta_2))*(np.cos(phi_1) - np.cos(phi_2))) / (np.sin(theta_1 - theta_2) * np.sin(phi_1 - phi_2))
    z_coefficient = -np.cos((phi_1 + phi_2) / 2) / np.cos((phi_1 - phi_2) / 2)
    return LinearInequality(x_coefficient=x_coefficient, y_coefficient=y_coefficient, z_coefficient=z_coefficient, right_hand_side=-1)

def angle_ranges_to_inequalities(azimuthal_range):
    inequalities = []
    # inequality for start of azimuthal range
    if np.equal(azimuthal_range[0], np.pi / 2):
        inequalities.append(
            LinearInequality(x_coefficient=1, y_coefficient=0, z_coefficient=0, right_hand_side=0)
        )
    elif np.equal(azimuthal_range[0], 3 * np.pi / 2):
        inequalities.append(
            LinearInequality(x_coefficient=-1, y_coefficient=0, z_coefficient=0, right_hand_side=0)
        )
    elif azimuthal_range[0] >= 3 * np.pi / 2 or azimuthal_range[0] <= np.pi / 2:
        inequalities.append(
            LinearInequality(x_coefficient=np.tan(azimuthal_range[0]), y_coefficient=-1, z_coefficient=0, right_hand_side=0)
        )
    else:
        inequalities.append(
            LinearInequality(x_coefficient=-np.tan(azimuthal_range[0]), y_coefficient=1, z_coefficient=0, right_hand_side=0)
        )
    # inequality for end of azimuthal range
    if np.equal(azimuthal_range[1], np.pi / 2):
        inequalities.append(
            LinearInequality(x_coefficient=-1, y_coefficient=0, z_coefficient=0, right_hand_side=0)
        )
    elif np.equal(azimuthal_range[1], 3 * np.pi / 2):
        inequalities.append(
            LinearInequality(x_coefficient=1, y_coefficient=0, z_coefficient=0, right_hand_side=0)
        )
    elif azimuthal_range[1] >= 3 * np.pi / 2 or azimuthal_range[1] <= np.pi / 2:
        inequalities.append(
            LinearInequality(x_coefficient=-np.tan(azimuthal_range[1]), y_coefficient=1, z_coefficient=0, right_hand_side=0)
        )
    else:
        inequalities.append(
            LinearInequality(x_coefficient=np.tan(azimuthal_range[1]), y_coefficient=-1, z_coefficient=0, right_hand_side=0)
        )
    return inequalities

def assignment_already_considered(assignment: tuple[tuple[int]], sphere_partitioning: SpherePartitioning) -> bool:
    """
    Checks whether an equivalent assignment of electrons to regions has already been considered
    """
    for i in range(1, sphere_partitioning.row_length):
        """
        Symmetry is given by addition modulo row length for all rows except the last
        """
        shifted_assignment = [(row, (column + i) % sphere_partitioning.row_length) for (row, column) in assignment if row < sphere_partitioning.row_count - 1]
        if (sphere_partitioning.row_count - 1, 0) in assignment:
            shifted_assignment.append((sphere_partitioning.row_count - 1, 0))
        if frozenset(shifted_assignment) in sphere_partitioning.considered_assignments:
            return True
    reversed_assignment = [(row, sphere_partitioning.row_length - 1 - column) for (row, column) in assignment if row < sphere_partitioning.row_count - 1]
    if (sphere_partitioning.row_count - 1, 0) in assignment:
        reversed_assignment.append((sphere_partitioning.row_count - 1, 0))
    for i in range(sphere_partitioning.row_length):
        shifted_reversed_assignment = [(row, (column + i) % sphere_partitioning.row_length) for (row, column) in reversed_assignment if row < sphere_partitioning.row_count - 1]
        if (sphere_partitioning.row_count - 1, 0) in reversed_assignment:
            shifted_reversed_assignment.append((sphere_partitioning.row_count - 1, 0))
        if frozenset(shifted_reversed_assignment) in sphere_partitioning.considered_assignments:
            return True
        
    return False

def calculate_minimum_distance(electron_count: int, putative_minimum_energies: dict, budget_seconds: float) -> float:
    """
    Uses OBBT within the given budget to compute a valid lower bound on every inter-electron distance
    """
    
    """
    One lower bound can be obtained if the global minimum for N - 1 electrons is known.
    The interactions of N electrons can be partitioned into interactions among the first N - 1 electrons,
    and N - 1 interactions between those and the last electron. The sum of these energies must be no
    greater than the putative minimum energy for N electrons. The value 0.5 is a lower bound for
    the reciprocal of the distance of two points on the unit sphere.
    """
    max_interaction_energy = putative_minimum_energies[electron_count] - putative_minimum_energies[electron_count - 1] - 0.5 * (electron_count - 2)
    lower_bound = 1 / (max_interaction_energy ** 2)
    model = pyo.ConcreteModel()
    model.i = pyo.Set(initialize=[i for i in range(electron_count - 1)])
    def get_x_bounds(model, i):
        if i == 0:
            return (0, 1)
        return (-1, 1)
    model.x = pyo.Var(model.i, domain=pyo.Reals, bounds=get_x_bounds)
    def get_y_bounds(model, i):
        if i == 0:
            return (0, 0)
        return (-1, 1)
    model.y = pyo.Var(model.i, domain=pyo.Reals, bounds=get_y_bounds)
    def get_z_bounds(model, i):
        return (-1, 1)
    model.z = pyo.Var(model.i, domain=pyo.Reals, bounds=get_z_bounds)
    interaction_set = [(i, j) for i in range(electron_count - 1) for j in range(i + 1, electron_count - 1)]
    model.d = pyo.Var(interaction_set, domain=pyo.NonNegativeReals, bounds=(lower_bound, 4))
    def sphere_surface(model, i):
        return model.x[i] ** 2 + model.y[i] ** 2 + model.z[i] ** 2 == 1
    model.sphere_surface = pyo.Constraint(model.i, rule=sphere_surface)
    def optimality(model):
        return sum(1 / pyo.sqrt(2 - 2 * model.z[i]) for i in range(electron_count - 1)) + sum(
            1 / pyo.sqrt(model.d[i,j]) for i in range(electron_count - 1) for j in range(i + 1, electron_count - 1)) <= putative_minimum_energies[electron_count]
    model.optimality = pyo.Constraint(rule=optimality)
    def sq_distance_def(model, i, j):
        return model.d[i,j] == 2 - 2 * (model.x[i] * model.x[j] + model.y[i] * model.y[j] + model.z[i] * model.z[j])
    model.distances = pyo.Constraint(interaction_set, rule=sq_distance_def)
    def z_ordering(model, i):
        if i == 0:
            return pyo.Constraint.Skip
        return model.z[i] <= model.z[i - 1]
    model.z_ordering = pyo.Constraint(model.i, rule=z_ordering)
    def obj(model):
        return 2 - 2 * model.z[0]
    model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
    model.pprint()
    # baron
    # opt = pyo.SolverFactory('baron', executable='/usr/local/gamsexp/baron')
    # start_time = time.time()
    # results = opt.solve(model, options={'maxtime': budget_seconds}, tee=True, keepfiles=True, symbolic_solver_labels=True)
    # end_time = time.time()
    # scip
    # opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
    # opt.options["solver"] = "scip"
    # start_time = time.time()
    # results = opt.solve(model, tee=True, keepfiles=True, symbolic_solver_labels=True, add_options=['Option optcr = 0;', f'Option reslim={budget_seconds};'])
    # end_time = time.time()
    # gurobi
    opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
    start_time = time.time()
    results = opt.solve(model, tee=True, add_options=['Option optcr = 0;', f'Option reslim={budget_seconds};', 'Option nlp=gurobi;', 'Option threads=1;'])
    end_time = time.time()
    if results.solver.termination_condition == pyo.TerminationCondition.optimal and end_time - start_time < budget_seconds:
        return np.sqrt(pyo.value(model.obj))
    elif results.solver.status == pyo.SolverStatus.ok:
        return np.sqrt(results.problem.lower_bound)
    else:
        exit(f"OBBT failed for N = {free_electron_count} with status {results.solver.status} and termination condition {results.solver.termination_condition}")

if __name__ == "__main__":
    putative_minimum_energies = {
        2: 0.5,
        3: 1.732050808,
        4: 3.674234614,
        5: 6.474691495,
        6: 9.985281374,
        7: 14.452977414,
        8: 25.759986531
    }
    electron_count = int(sys.argv[1])
    subproblem_solver = sys.argv[2]
    print(f"{electron_count} electrons")
    lower_bound = calculate_minimum_distance(electron_count=electron_count, putative_minimum_energies=putative_minimum_energies, budget_seconds=(40*(electron_count)))
    free_electron_count = electron_count - 1
    a = SpherePartitioning(free_electron_count=free_electron_count, minimum_distance=lower_bound)
    print(f"rows = {a.row_count}, row length = {a.row_length}, row starts = {a.row_starts}")
    unique_assignments = 0
    total_assignments = 0
    for assignment in itertools.combinations(a.regions, free_electron_count):
        total_assignments += 1
        if assignment_already_considered(assignment, a):
            continue
        unique_assignments += 1
        a.consider_assignment(assignment, putative_minimum_energies, subproblem_solver)
        print(f"Considered {assignment} (unique assignment #{unique_assignments})")
    print(f"{unique_assignments} assignments out of {total_assignments} were unique")
    for assignment in a.considered_assignments:
        if np.isinf(a.considered_assignments[assignment]):
            print(f"{assignment}: suboptimal")
        else:
            print(f"{assignment}: {a.considered_assignments[assignment]}")