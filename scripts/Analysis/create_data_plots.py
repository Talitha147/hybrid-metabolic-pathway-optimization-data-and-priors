from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from scripts.data_generation import load_data_from_csvs

def main():
    # Load model
    filepath = 'models/sbml_models/pCA_model_changed_S.xml'
    sbmlModel = SBMLModel(filepath)
    kinetic_model = sbmlModel.get_kinetic_model()
    
    # First plot from pCA_model.ipynb
    S = sbmlModel._get_stoichiometric_matrix() 
    ts = jnp.linspace(0, 40, 40).round()
    ts_7 = jnp.linspace(0, 40, 8).round()

    ys_7 = kinetic_model(ts_7, sbmlModel.y0, sbmlModel.parameters)
    ys = kinetic_model(ts, sbmlModel.y0, sbmlModel.parameters)

    ys = jnp.where(jnp.abs(ys) < 1e-9, 0.0, ys)

    ys = pd.DataFrame(ys, columns=S.index)
    ys_7 = pd.DataFrame(ys_7, columns=S.index)

    key_map = {
        "substrate": "Substrate",
        "c_biomass": "Biomass",
        "co2": "CO2",
        "e4p": "E4P",
        "pep": "PEP",
        "dhap": "DHAP",
        "epsp": "EPSP",
        "pcoumaric_acid": "p-coumaric acid",
    }

    plt.figure(figsize=(6, 4))
    for key in ys.keys():
        plt.scatter(ts_7, ys_7[key], label=key_map[key])
        plt.plot(ts, ys[key], linestyle="--")
     
  

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (M)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig('Figures/original_trajectories.png')
    plt.close()
    print("Saved original_trajectories.png")

    # Second plot: trajectories for only pca of the first 50 strains
    ts_data, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir="dataset/data",
        metabolites=sbmlModel.species_names,
        model_name="pCA_model",
    )

    plt.figure(figsize=(6, 4))
    # pCA is the last metabolite (index -1)
    plt.plot(ts_data, ys_total[:50, :, -1].T, linestyle="-")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (M)")
    plt.tight_layout()
    plt.savefig('Figures/pca_50_strains.png')
    plt.close()
    print("Saved pca_50_strains.png")

if __name__ == "__main__":
    main()
