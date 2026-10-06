# Simulation of the data capture and processing of the sun for the interferometry project
# Joshua Smith 
# 2026 - EEE4022S

using Plots

sample_rate = 2.5e6 # MSPS, set as it would be on the Pluto



s1 = random(ComplexF16, 1000) # random complex signal