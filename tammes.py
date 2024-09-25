import sys
import itertools
import thomson
import time
import tammes_putative_optima
import numpy as np
import pyomo.environ as pyo

def time_standard_formulation_solve(point_count: int, solver: str, minimum_distance: float, time_limit_seconds: float) -> tuple[float, float]:
    """
    Returns time to solve standard Tammes formulation and value of best solution found
    """
    model = pyo.ConcreteModel()
    model.i = pyo.Set(initialize=[i for i in range(point_count)])
    def get_x_bounds(model, i):
        if i == 0:
            return (0, 0)
        if i == 1:
            return (0, 1)
        return (-1, 1)
    model.x = pyo.Var(model.i, domain=pyo.Reals, bounds=get_x_bounds)
    def get_y_bounds(model, i):
        if i in (0, 1):
            return (0, 0)
        return (-1, 1)
    model.y = pyo.Var(model.i, domain=pyo.Reals, bounds=get_y_bounds)
    def get_z_bounds(model, i):
        if i == 0:
            return (1, 1)
        return (-1, 1)
    model.z = pyo.Var(model.i, domain=pyo.Reals, bounds=get_z_bounds)
    interaction_set = [(i, j) for i in range(point_count) for j in range(i + 1, point_count)]
    model.d = pyo.Var(interaction_set, domain=pyo.NonNegativeReals, bounds=(minimum_distance ** 2, 4))
    def sphere_surface(model, i):
        return model.x[i] ** 2 + model.y[i] ** 2 + model.z[i] ** 2 == 1
    model.sphere_surface = pyo.Constraint(model.i, rule=sphere_surface)
    def sq_distance_def(model, i, j):
        return model.d[i,j] == 2 - 2 * (model.x[i] * model.x[j] + model.y[i] * model.y[j] + model.z[i] * model.z[j])
    model.distances = pyo.Constraint(interaction_set, rule=sq_distance_def)
    def z_ordering(model, i):
        if i == 0:
            return pyo.Constraint.Skip
        return model.z[i] <= model.z[i - 1]
    model.z_ordering = pyo.Constraint(model.i, rule=z_ordering)
    model.minimal_sq_distance = pyo.Var(domain=pyo.NonNegativeReals)
    def minimal_sq_distance_def(model, i, j):
        return model.minimal_sq_distance <= model.d[i,j]
    model.minimal_sq_distance_def = pyo.Constraint(interaction_set, rule=minimal_sq_distance_def)
    def obj(model):
        return model.minimal_sq_distance
    model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)
    # model.pprint()
    start_time = time.time()
    try:
        if solver == "gurobi":
            opt = pyo.SolverFactory('gurobi')
            opt.options["threads"] = 1
            opt.options["mipgap"] = 0
            opt.options["nonconvex"] = 2
            opt.options["timelimit"] = time_limit_seconds
            opt.solve(model, tee=True)
        else:
            opt = pyo.SolverFactory('gams', executable='/usr/local/bin/gams')
            opt.solve(model, tee=True, add_options=['Option optcr = 0;', f'Option reslim={time_limit_seconds};', f'Option nlp={solver};', 'Option threads=1;'])
        end_time = time.time()
        return (end_time - start_time, np.sqrt(pyo.value(model.obj)))
    except:
        end_time = time.time()
        return (end_time - start_time, np.inf)

if __name__ == "__main__":
    point_count = int(sys.argv[1])
    free_point_count = point_count - 1
    a = thomson.SpherePartitioning(free_electron_count=free_point_count, minimum_distance=(tammes_putative_optima.putatively_optimal_distances[point_count]))
    print(f"for {point_count} points: rows = {a.row_count}, row length = {a.row_length}, row starts = {a.row_starts}")
    unique_assignments = 0
    total_assignments = 0
    for assignment in itertools.combinations(a.regions, free_point_count):
        total_assignments += 1
        if thomson.assignment_already_considered(assignment, a):
            continue
        unique_assignments += 1
        a.consider_assignment_tammes(assignment)
        print(f"Considered {assignment} (unique assignment #{unique_assignments})")
    print(f"{unique_assignments} assignments out of {total_assignments} were unique")
    best_objective_value = 0
    for assignment in a.considered_assignments:
        if np.isinf(a.considered_assignments[assignment]):
            print(f"{assignment}: infeasible")
        else:
            print(f"{assignment}: {a.considered_assignments[assignment]}")
            best_objective_value = max(best_objective_value, a.considered_assignments[assignment])
    print(f"Best objective value found: {best_objective_value:.8f} (putative optimum: {best_objective_value:.8f})")