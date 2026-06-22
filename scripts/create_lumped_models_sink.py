import sys
from pathlib import Path
import os

# Add local packages to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "Packages" / "jaxkineticmodel"))

from jaxkineticmodel.building_models import JaxKineticModelBuild as build
from matplotlib import colors
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as mechanisms
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
import os
from jaxkineticmodel.kinetic_mechanisms import JaxKineticModifiers as modifier
import jax.numpy as jnp
import numpy as np
import jax
import pandas as pd
import matplotlib.pyplot as plt
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter
jax.config.update("jax_enable_x64", True)



# now biomass, from the internal carbon source species
mech_biomass = mechanisms.Jax_MM_Irrev_Uni(substrate="substrate",
                                  vmax="vmax_v1",
                                  km_substrate="km_v1_gluc", )
mech_biomass.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_biomass.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))
v_biomass = build.Reaction(
    name="v_biomass",
    species=["substrate", "c_biomass", "co2"],
    stoichiometry=[-7, 42, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_biomass, )



# Respiration
mech_respiration = mechanisms.Jax_MM_Irrev_Uni(substrate='substrate',
                                               vmax="vmax_respiration",
                                               km_substrate="km_respiration_A")
mech_respiration.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_respiration.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_respiration = build.Reaction(
    name="v_respiration",
    species=["substrate", "co2", 'c_biomass'],
    stoichiometry=[-1, 6, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_respiration,
)


# PPP
mech_ppp = mechanisms.Jax_MM_Irrev_Uni(
    substrate='substrate',
    vmax="vmax_ppp",
    km_substrate="km_ppp_substrate",
)
mech_ppp.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_ppp.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_ppp = build.Reaction(
    name = "v_ppp",
    species=["substrate", "e4p", 'c_biomass'],
    stoichiometry=[-2, 3, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_ppp,
)


# Glycolysis
mech_glyc = mechanisms.Jax_MM_Irrev_Uni(
    substrate='substrate',
    vmax="vmax_glyc",
    km_substrate="km_glyc_substrate",
)
mech_glyc.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_glyc.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_glyc = build.Reaction(
    name = "v_glyc",
    species=["substrate", "pep", 'c_biomass'],
    stoichiometry=[-1, 2, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_glyc,
)


# Shikimate pathway
mech_shiki_step1 = mechanisms.Jax_MM_Irrev_Bi(
    substrate1='pep',
    substrate2='e4p',
    vmax ="vmax_shiki_step1",
    km_substrate1="km_shiki1_pep",
    km_substrate2="km_shiki1_e4p",
)
mech_shiki_step1.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_shiki_step1.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_shiki_step1 = build.Reaction(
    name = "v_shiki_step1",
    species=["e4p", "pep", 'dhap','c_biomass'],
    stoichiometry=[-1, -1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=mech_shiki_step1,
)


# Shikimate pathway step 2
mech_shiki_step2 = mechanisms.Jax_MM_Irrev_Bi(
    substrate1='pep',
    substrate2='dhap',
    vmax ="vmax_shiki_step2",
    km_substrate1="km_shiki2_pep",
    km_substrate2="km_shiki2_dhap",
)
mech_shiki_step2.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_shiki_step2.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_shiki_step2 = build.Reaction(
    name = "v_shiki_step2",
    species=["pep", "dhap", 'epsp','c_biomass'],
    stoichiometry=[-1, -1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=mech_shiki_step2,
)


# Product formation
mech_product = mechanisms.Jax_MM_Irrev_Uni(
    substrate='epsp',
    vmax="vmax_product",
    km_substrate="km_product_epsp",
)

mech_product.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_product.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_product = build.Reaction(
    name = "v_product",
    species=["epsp", "pcoumaric_acid",'c_biomass'],
    stoichiometry=[-1, 1, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_product,
)


##sinks
mech_vsink1 = mechanisms.Jax_MM_Sink(substrate="e4p",
                             v_sink="vmax_sink1_e4p",
                             km_sink="km_sink1_e4p")
mech_vsink1.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_vsink1.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_sink1 = jkm.Reaction(
    name="v_sink_e4p",
    species=['e4p','c_biomass'],
    stoichiometry=[-1,0],
    compartments=['c','c'],
    mechanism=mech_vsink1,)

mech_vsink2 = mechanisms.Jax_MM_Sink(substrate="pep",
                             v_sink="vmax_sink2_pep",
                             km_sink="km_sink2_pep")
mech_vsink2.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_vsink2.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_sink2 = jkm.Reaction(
    name="v_sink_pep",
    species=['pep','c_biomass'],
    stoichiometry=[-1,0],
    compartments=['c','c'],
    mechanism=mech_vsink2,)

mech_vsink3 = mechanisms.Jax_MM_Sink(substrate="epsp",
                             v_sink="vmax_sink3_epsp",
                             km_sink="km_sink3_epsp")
mech_vsink3.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_vsink3.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_sink3 = jkm.Reaction(
    name="v_sink_epsp",
    species=['epsp','c_biomass'],
    stoichiometry=[-1,0],
    compartments=['c','c'],
    mechanism=mech_vsink3,)




#Lumped reactions


#shikimate 2 + product

#NOTE: the parameters here need to have the same name as the original model, but the flux 
# that is calculated with them is not actually used, since we are masking this reaction.

mech_shiki_2_product = mechanisms.Jax_MM_Irrev_Bi(
    substrate1='pep',
    substrate2='dhap',
    vmax ="vmax_shiki_step2",
    km_substrate1="km_shiki2_pep",
    km_substrate2="km_shiki2_dhap",
)
mech_shiki_2_product.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_shiki_2_product.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_shiki_2_product = build.Reaction(
    name = "v_shiki_2_product",
    species=["pep", "dhap", 'pcoumaric_acid', 'co2', 'c_biomass'],
    stoichiometry=[-1, -1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c', 'c'],
    mechanism=mech_shiki_2_product,
)


#shikimate 1 + product


mech_shiki_1_product = mechanisms.Jax_MM_Irrev_Bi(
    substrate1='pep',
    substrate2='e4p',
    vmax ="vmax_shiki_step1",
    km_substrate1="km_shiki1_pep",
    km_substrate2="km_shiki1_e4p",
)
mech_shiki_1_product.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_shiki_1_product.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_shiki_1_product = build.Reaction(
    name = "v_shiki_1_product",
    species=["pep", "e4p", 'pcoumaric_acid', 'co2', 'c_biomass'],
    stoichiometry=[-2, -1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c', 'c'],
    mechanism=mech_shiki_1_product,
)


# dhap -> product

mech_dhap_product = mechanisms.Jax_MM_Irrev_Uni(
    substrate='dhap',
    vmax ="vmax_product",
    km_substrate="km_product_epsp",
)
mech_dhap_product.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_dhap_product.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_dhap_product = build.Reaction(
    name = "v_dhap_product",
    species=["dhap", "pcoumaric_acid", 'co2', 'c_biomass'],
    stoichiometry=[-1, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=mech_dhap_product,
)



# substrate -> dhap

mech_substrate_dhap = mechanisms.Jax_MM_Irrev_Uni(
    substrate='substrate',
    vmax ="vmax_shiki_step1",
    km_substrate="km_shiki1_pep",
)
mech_substrate_dhap.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_substrate_dhap.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_substrate_dhap = build.Reaction(
    name = "v_substrate_dhap",
    species=["substrate", "dhap",'c_biomass'],
    stoichiometry=[-3, 1, 0],
    compartments=['c', 'c', 'c'],
    mechanism=mech_substrate_dhap,
)



# substrate -> product

mech_substrate_product = mechanisms.Jax_MM_Irrev_Uni(
    substrate='substrate',
    vmax ="vmax_product",
    km_substrate="km_product_epsp",
)
mech_substrate_product.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_substrate_product.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_substrate_product = build.Reaction(
    name = "v_substrate_product",
    species=["substrate", "pcoumaric_acid", 'co2', 'c_biomass'],
    stoichiometry=[-3, 1, 1, 0],
    compartments=['c', 'c', 'c', 'c'],
    mechanism=mech_substrate_product,
)


# substrate -> product no co2

mech_substrate_product_no_co2 = mechanisms.Jax_MM_Irrev_Uni(
    substrate='substrate',
    vmax ="vmax_product",
    km_substrate="km_product_epsp",
)
mech_substrate_product_no_co2.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
# mech_substrate_product_no_co2.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_substrate_product_no_co2 = build.Reaction(
    name = "v_substrate_product_no_co2",
    species=["substrate", "pcoumaric_acid"],
    stoichiometry=[-3, 1],
    compartments=['c', 'c'],
    mechanism=mech_substrate_product_no_co2,
)


# New sinks for lumped models
# Again reused some existing parameter names, but aren't used to calculate the flux as these reactions are always masked
# avoided creating new parameters that would otherwise affect what the model trains on. 
mech_vsink_substrate = mechanisms.Jax_MM_Sink(substrate="substrate",
                             v_sink="vmax_v1",
                             km_sink="km_v1_gluc")
mech_vsink_substrate.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_vsink_substrate.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_sink_substrate = jkm.Reaction(
    name="v_sink_substrate",
    species=['substrate','c_biomass'],
    stoichiometry=[-1,0],
    compartments=['c','c'],
    mechanism=mech_vsink_substrate,)

mech_vsink_substrate_no_biomass = mechanisms.Jax_MM_Sink(substrate="substrate",
                             v_sink="vmax_v1",
                             km_sink="km_v1_gluc")

v_sink_substrate_no_biomass = jkm.Reaction(
    name="v_sink_substrate",
    species=['substrate'],
    stoichiometry=[-1],
    compartments=['c'],
    mechanism=mech_vsink_substrate_no_biomass,)

mech_vsink_dhap = mechanisms.Jax_MM_Sink(substrate="dhap",
                             v_sink="vmax_shiki_step2",
                             km_sink="km_shiki2_dhap")
mech_vsink_dhap.add_modifier(modifier.BasicDivision(symbol="a", scaling="scale"))
mech_vsink_dhap.add_modifier(modifier.BasicMultiplication(symbol="c_biomass", scaling="scale"))

v_sink_dhap = jkm.Reaction(
    name="v_sink_dhap",
    species=['dhap','c_biomass'],
    stoichiometry=[-1,0],
    compartments=['c','c'],
    mechanism=mech_vsink_dhap,)



parameters_original = {
    "vmax_v1": 0.3/40,
    "km_v1_gluc": 0.01,
    "a": 1.6,
    "scale": 1,  # is the broth volume
    "vmax_respiration": 0.35,
    "km_respiration_A": 0.03,
    "vmax_ppp": 0.1/2,
    "km_ppp_substrate": 0.02/2,
    "vmax_glyc": 0.1,
    "km_glyc_substrate": 0.02,
    "km_shiki1_pep": 0.02,
    "km_shiki1_e4p": 0.02,
    "vmax_shiki_step1": 0.1,
    "vmax_shiki_step2": 0.1,
    "km_shiki2_pep": 0.02,
    "km_shiki2_dhap": 0.02,
    "vmax_product": 0.1,
    "km_product_epsp": 0.02,
    "vmax_sink2_pep":0.1,
    "vmax_sink1_e4p": 0.1,
    "km_sink2_pep": 0.02,
    "km_sink1_e4p": 0.02,
    "vmax_sink3_epsp":0.1,
    "km_sink3_epsp":0.02,
}

# construct model object
reactions_original = [v_biomass, v_respiration,
             v_ppp, v_glyc,
             v_shiki_step1,
             v_shiki_step2,
             v_product,
             v_sink1,
             v_sink2,
             v_sink3]


reactions_lumped_1 = [v_biomass, v_respiration, v_substrate_product, v_sink_substrate]

reactions_lumped_2 = [v_biomass, v_respiration, v_substrate_dhap, v_dhap_product, v_sink_substrate, v_sink_dhap]

reactions_lumped_3 = [v_biomass, v_respiration, v_ppp, v_glyc, v_shiki_step1, v_shiki_2_product, v_sink1, v_sink2, v_sink_dhap]



# 20 g/L = 0.111 mol/L
y0_original = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0, #e4p
                0.0224056, #pep
                0.002, #dhap (guessed)
                0.001, #epsp (guessed)
                0 #pca
                ])

y0_lumped_1 = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0 #pca
                ])

y0_lumped_2 = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0.002, #dhap (guessed)
                0 #pca
                ])

y0_lumped_3 = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0, #e4p
                0.0224056, #pep
                0.002, #dhap (guessed)
                0 #pca
                ])


def make_lumped_model(y0, parameters,reactions, output_dir, model_name):

    compartment_values = {'c': 1}
    kmodel = build.JaxKineticModelBuild(reactions, compartment_values)

    kmodel_sim = build.NeuralODEBuild(kmodel)
    ts = jnp.linspace(0, 50, 1000)
    print(kmodel.species_names)

    sbml = SBMLExporter(model=kmodel_sim)
    sbml.export(initial_conditions=y0,
                parameters=parameters,
                output_file=f"{output_dir}/{model_name}_sink.xml")



output_dir = Path("models/pCA_model")
output_dir.mkdir(parents=True, exist_ok=True)

make_lumped_model(y0_lumped_1, parameters_original,reactions_lumped_1, output_dir, "pCA_model_lumped_1_part_1")
make_lumped_model(y0_lumped_2, parameters_original,reactions_lumped_2, output_dir, "pCA_model_lumped_2_part_1")
make_lumped_model(y0_lumped_3, parameters_original,reactions_lumped_3, output_dir, "pCA_model_lumped_3_part_1")


# Question 3 part 2

reactions_lumped_3_part_2 = [v_biomass, v_respiration, v_ppp, v_glyc, v_shiki_step1, v_shiki_2_product, v_sink1, v_sink2, v_sink_dhap]

reactions_lumped_2_part_2 = [v_biomass, v_respiration, v_ppp, v_glyc, v_shiki_1_product, v_sink1, v_sink2]

reactions_lumped_1_part_2 = [v_substrate_product_no_co2, v_sink_substrate_no_biomass]




y0_lumped_1_part_2 = jnp.array([0.111,  # 'c_A',
                0 #pca
                ])

y0_lumped_2_part_2 = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0, #e4p
                0.0224056, #pep
                0 #pca
                ])

y0_lumped_3_part_2 = jnp.array([0.111,  # 'c_A',
                0.01,  # 'c_biomass'
                0,  # co2
                0, #e4p
                0.0224056, #pep
                0.002, #dhap (guessed)
                0 #pca
                ])


make_lumped_model(y0_lumped_1_part_2, parameters_original, reactions_lumped_1_part_2, output_dir, "pCA_model_lumped_1_part_2")
make_lumped_model(y0_lumped_2_part_2, parameters_original, reactions_lumped_2_part_2, output_dir, "pCA_model_lumped_2_part_2")
make_lumped_model(y0_lumped_3_part_2, parameters_original, reactions_lumped_3_part_2, output_dir, "pCA_model_lumped_3_part_2")