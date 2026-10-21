# PlutoIF: the shared code of the FINAL programs (pluto_interferometer.jl, stress_test.jl, viewer.jl), so all three
# configure the radio, correlate and read files the same way. Use it as
#     include(joinpath(@__DIR__, "src", "PlutoIF.jl")); const P = PlutoIF
module PlutoIF

using AdalmPluto, FFTW, Statistics, Dates, Printf, Random
const AP = AdalmPluto

include("config.jl")      # constants, Config, GeoCfg, Block, Record, key=value parsing
include("dsp.jl")         # conversion, health, FX correlator, band averages, tone fit, fits
include("visfile.jl")     # .vis / .meta files and the follower used by the viewer
include("radio.jl")       # the Pluto: configure, read back, stream
include("fake.jl")        # a stand-in radio for tests without hardware
include("pipeline.jl")    # capture thread + processing thread; synchronous capture
include("geometry.jl")    # Sun position, baseline geometry, fringe stopping
include("sim.jl")         # simulated observation through the same correlator (sim mode)

end
