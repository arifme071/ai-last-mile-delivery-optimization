import gurobipy as gp

print("Gurobi version:", gp.gurobi.version())

model = gp.Model("test_model")

x = model.addVar(lb=0, ub=10, name="x")

model.setObjective(x, gp.GRB.MAXIMIZE)

model.optimize()

if model.status == gp.GRB.OPTIMAL:
    print("Optimal x =", x.X)
    print("License and solver are working correctly.")