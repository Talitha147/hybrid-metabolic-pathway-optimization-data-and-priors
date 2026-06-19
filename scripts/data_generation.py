from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.simulated_dbtl.dbtl import DesignBuildTestLearnCycle
from jaxkineticmodel.parameter_estimation.training import TrainingData
import jax.numpy as jnp
import pandas as pd
import jax
import numpy as np
from pathlib import Path

def generate_training_data(filepath, ts, parameter_perturbations, model_name, samples: int):
  
    # load model from file_path
    model = SBMLModel(filepath)
    kinetic_model = jax.jit(model.get_kinetic_model())

    model = SBMLModel(filepath)

    # ts = jnp.linspace(0,10,2000)
    # target = "m4"

    dbtl_cycle = DesignBuildTestLearnCycle(model=model,
                                        parameters=model.parameters,
                                        initial_conditions=model.y0,
                                        timespan=ts,
                                        target=[])

    # parameter_perturbations = {'A_Vmax': [0.2, 0.5, 1, 1.5, 2],
    #                         'B_Vmax': [1.1, 1.6, 1.3],
    #                         'C_Vmax': [1, 2, 3]}

    dbtl_cycle.design_establish_library_elements(parameter_perturbations=parameter_perturbations)

    _ = dbtl_cycle.design_assign_positions(n_positions=6)
    _ = dbtl_cycle.design_assign_probabilities(probabilities_per_position=None)
    strain_designs = dbtl_cycle.design_generate_strains(samples=samples)


    perturbed_strains = pd.DataFrame()
    for k, strain_p in enumerate(strain_designs): 
        perturbed_strains[k] = strain_p


    perturbed_strains.to_csv(f"data/{model_name}_strain_designs.csv")

    metabolites = model.species_names
    y0 = model.y0
    num_metabolites = len(metabolites)
    dfs = [pd.DataFrame() for x in range(num_metabolites)]

    for k in perturbed_strains.keys():
        params = dict(perturbed_strains[k])
        ys = kinetic_model(ts, y0, params)
        ys = jnp.where(jnp.abs(ys) < 1e-9, 0.0, ys)

        ys = pd.DataFrame(ys, columns=model.species_names)
        for i in range(num_metabolites):
            dfs[i][k] = ys[metabolites[i]] 

    for i in range(num_metabolites):
        dfs[i].insert(0, "time", ts)
        dfs[i].to_csv(f"data/{model_name}_{metabolites[i]}.csv", index=False)
    return 




def load_data_from_csvs(
    csv_dir,
    metabolites,
    model_name
):
    csv_dir = Path(csv_dir)

    param_csv = f"{model_name}_strain_designs.csv"
 
    df0 = pd.read_csv(csv_dir / f"{model_name}_{metabolites[0]}.csv")
    ts = jnp.array(df0["time"].values)

    strain_cols = [c for c in df0.columns if c != "time"]
    n_strains = len(strain_cols)
    n_timepoints = len(ts)
    n_metabolites = len(metabolites)

  
    ys = jnp.zeros((n_strains, n_timepoints, n_metabolites))

    for m_idx, met in enumerate(metabolites):
        df = pd.read_csv(csv_dir / f"{model_name}_{met}.csv")
        for s_idx, strain in enumerate(strain_cols):
            ys = ys.at[s_idx, :, m_idx].set(df[strain].values)

  
    param_df = pd.read_csv(csv_dir / param_csv, header=None)

    
    param_names = param_df.iloc[:, 0]
    param_values = param_df.iloc[1:, 1:]

    param_values = param_values.apply(pd.to_numeric)
    names = param_names.iloc[1:].reset_index(drop=True)

    # shape = (n_strains, n_params)
    params = jnp.array(param_values.values.T)
    return ts, ys, params, names
