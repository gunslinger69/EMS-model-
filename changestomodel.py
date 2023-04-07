# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:21:28 2023

@author: mky14
"""


import pandas as pd
import numpy as np
import csv
import gurobipy as gp
from gurobipy import GRB
import os

#load datasets
#load excelfiles of stations with ID,a nd number of ambulance vehilces 
def read_candidates_sites(candidates_csv_path):
    df = pd.read_csv(candidates_csv_path)
    df["NumberOfAmbulances"] = df["NumberOfAmbulances"].fillna(value=0)
    return df
#load excelfiles of demand points, including ID and Number of Ambulance vehicles
def read_demands(demand_csv_path):
    df = pd.read_csv(demand_csv_path)
    df["CallFreq"] = df["CallFreq"].fillna(value=0)
    return df
# load cost matrix including facility id,demand id, drive time
def create_od_matrix(od_csv_path, time_threshold):
    df = pd.read_csv(od_csv_path)
    df["DriveTime"] = df["DriveTime"].fillna(value=0)
    
    # Print the DriveTime distribution
    print(df["DriveTime"].describe())

    # Update the time_threshold value
    threshold = time_threshold
    # Use the updated threshold value to create the coverage matrix
    df["covered"] = np.where(df["DriveTime"] <= threshold, 1, 0)
    
    # Create pivot table for OD matrix  
    pivot = df.pivot_table(index = "StationID",columns = "DemandID", values=[ "covered", "DriveTime"])
    print("Time Threshold:", time_threshold)
    print("Updated Coverage Matrix:\n", pivot)
    return pivot

def calculate_coverage_without_optimization(demand_csv, candidates_csv, odmatrix_csv, time_threshold):
    # Read input data for the model
    demands = read_demands(demand_csv)
    candidates = read_candidates_sites(candidates_csv)
    coverage_matrix = create_od_matrix(odmatrix_csv, time_threshold)

    covered_demand = 0
    total_demand = demands["CallFreq"].sum()

    # Use a set to store the covered demand points
    covered_demand_points = set()

    for i in demands["DemandID"]:
        for j in candidates["StationID"]:
            if coverage_matrix.loc[j, ("covered", i)] == 1:
                # Only add the demand if it is not already in the set
                if i not in covered_demand_points:
                    covered_demand += demands.loc[demands["DemandID"] == i, "CallFreq"].item()
                    covered_demand_points.add(i)

    percentage_covered_demand = (covered_demand / total_demand) * 100

    print(f"Total covered demand: {covered_demand}")
    print(f"Percentage of covered demand: {percentage_covered_demand}%")
    
def mcmclp_additive_model(demand_csv, candidates_csv, odmatrix_csv, time_threshold, unit_car_capacity, maximal_cars_per_site, output_csv, total_added_cars, scenario, fixed_ambulances=False, additive_mode=0, gurobi_method=-1):
  
    # Reading the input data for this model
    demands = read_demands(demand_csv)
    candidates = read_candidates_sites(candidates_csv)
    coverage_matrix = create_od_matrix(odmatrix_csv, time_threshold)
    print("Coverage Matrix:\n", coverage_matrix)
    print("Demands:\n", demands)
    print("Candidates:\n", candidates)

    # Create a Gurobi model
    gurobi_model = gp.Model("EMS vehicles")
    
    # Set the Gurobi method parameter
    gurobi_model.setParam("Method", gurobi_method)
    
    # Define the decision variables
    # Decision variable 1 - percentage of demand i covered by facility j
    y_i_j_vars = gurobi_model.addVars(demands["DemandID"], candidates["StationID"], ub=1, vtype=gp.GRB.CONTINUOUS, name="y")

    # Decision variable 2 - Number of cars added at a station
    x_j_vars = gurobi_model.addVars(candidates["StationID"], vtype=gp.GRB.INTEGER, name="x") 
    
    # Add constraints
    # Constraint 1 - allocated demand should not exceed the capacity of the facility
    for j in candidates["StationID"]:
        gurobi_model.addConstr(gp.quicksum(y_i_j_vars[i, j] * demands.loc[demands["DemandID"] == i, "CallFreq"].item() for i in demands["DemandID"]) <= unit_car_capacity * candidates.loc[candidates["StationID"] == j, "NumberOfAmbulances"].item())
     
    # Constraint 2 - total number of cars at each site should not exceed the capacity of the facility
    for j in candidates["StationID"]:
        num_existing_cars = candidates.loc[candidates["StationID"] == j, "NumberOfAmbulances"].item()
        num_total_cars = num_existing_cars + x_j_vars[j]

        gurobi_model.addConstr(num_total_cars >= 0)
        gurobi_model.addConstr(num_total_cars - maximal_cars_per_site <= x_j_vars[j])
        gurobi_model.addConstr(gp.quicksum(y_i_j_vars[i, j] * demands.loc[demands["DemandID"] == i, "CallFreq"].item() for i in demands["DemandID"]) <= unit_car_capacity * num_total_cars)

    # Constraint 3 - total number of EMS vehicles added should be equal to total_added_cars
    gurobi_model.addConstr(gp.quicksum(x_j_vars[j] for j in candidates["StationID"]) == total_added_cars)

    # Constraint 4 - allocated demand at i should not exceed 100%
    for i in demands["DemandID"]:
        gurobi_model.addConstr(gp.quicksum(y_i_j_vars[i, j] for j in candidates["StationID"]) == 1)
        
    # Constraint 5 - number of added cars at each site should not exceed maximal_cars_per_site
    if scenario == 2:
        
        for j in candidates["StationID"]:
            
            gurobi_model.addConstr(x_j_vars[j] <= maximal_cars_per_site)
            
    # Constraint 5.1 - no additional vehicles allowed for scenario 1
    if scenario == 1:
        
        for j in candidates["StationID"]:
            gurobi_model.addConstr(x_j_vars[j] == 0)
        

    
        #objective function:
    
    total_covered_demand = gp.quicksum(y_i_j_vars[i, j] * demands.loc[demands["DemandID"] == i, "CallFreq"].item()
                                for i in demands["DemandID"] for j in candidates["StationID"]
                                if coverage_matrix.loc[j, ("covered", i)] == 1)

    
    gurobi_model.setObjective(total_covered_demand, gp.GRB.MAXIMIZE)
    
    print("---- Constraints ----")
    gurobi_model.update()
    for c in gurobi_model.getConstrs():
        print(c.ConstrName, c)

    # Solve the model
    gurobi_model.optimize()
    
    #check for infeasibility
    if gurobi_model.status == gp.GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("infeasible_scenario_0.ilp")
    
    # Retrieve the solution
    for var in gurobi_model.getVars():
        print(f"{var.VarName}: {var.X}")    
    total_cars = 0
    for j in candidates["StationID"]:
        num_added_cars = round(x_j_vars[j].X)
        total_cars += num_added_cars
        if (num_added_cars > 0):
            print("{0} vehicles added: {1!s}".format(j, num_added_cars))
            print([y_i_j_vars[i,j].X for i in demands["DemandID"]])
            print("total cars added: {0!s}".format(total_cars))
            
     # Retrieve the solution
    covered_demand = 0
    for i in demands["DemandID"]:
         for j in candidates["StationID"]:
             if y_i_j_vars[i, j].X == 1:
                 covered_demand += demands.loc[demands["DemandID"] == i, "CallFreq"].item()
                 print(f"y[{i}, {j}]: {y_i_j_vars[i, j].X}")
                 
    total_demand = demands["CallFreq"].sum()
    percentage_covered_demand = (covered_demand / total_demand) * 100
    print(f"Total covered demand: {covered_demand}")
    print(f"Percentage of covered demand: {percentage_covered_demand}%")
    
if __name__=="__main__":
    print("hello")
    #running the model
    input_data_folder = r"C:\Users\mky14\Documents\csvfiles_mky_thesis"
    demand_csv = os.path.join(input_data_folder, "accidents_final.csv")
    candidates_csv = os.path.join(input_data_folder, "ambulances_final.csv")
    odmatrix_csv = os.path.join(input_data_folder, "od_final.csv")
    output_csv = os.path.join(input_data_folder, "output.csv")

    time_threshold = 12
    unit_car_capacity = 30
    maximal_cars_per_site = 3
   #USe scenarios varibale to run the model in different scenarios
    #0- current distribution
    #1= ambulances are relocated
    #2= current+ added ambulances
    scenario = 1
    
    if scenario == 1:
        
        mcmclp_additive_model(demand_csv, candidates_csv, odmatrix_csv, time_threshold, unit_car_capacity, maximal_cars_per_site, output_csv, total_added_cars=0, scenario=1, additive_mode=0, gurobi_method=-1)
        
    elif scenario == 2:
            
            for i in range(0, 5):
                
                mcmclp_additive_model(demand_csv, candidates_csv, odmatrix_csv, time_threshold, unit_car_capacity, maximal_cars_per_site, output_csv, total_added_cars=i, scenario=2, additive_mode=1, gurobi_method=-1)

            
    elif scenario == 0:
        calculate_coverage_without_optimization(demand_csv, candidates_csv, odmatrix_csv, time_threshold)
