# Interferometer
This repo contains all notes and ancillary information regarding my work for EEE4022S research project


The aim of this project is to develop an Inteferometer using the Analog Devices Adalm Pluto SDR to 

## pluto_interferometer.jl (single-file pipeline)
One Julia file that takes the Pluto from power-on to timestamped visibilities. Run from this folder with `julia -t 3 pluto_interferometer.jl <mode> [key=value ...]`.

| Mode | Hardware | Purpose |
|---|---|---|
| `selftest` | no | Checks the correlator against a simulated source, the dropped-sample monitor, the file format, the Bessel function and the antenna inputs |
| `check` | yes | Connects, sets and **reads back** every setting, captures briefly, reports health, spectral peaks and correlations |
| `rate` | yes | Sustained capture; real sample rate against the host clock (`tone=true` also watches for dropped samples) |
| `run` | yes | Capture, FX correlator, timestamped visibility records to disk |
| `show` | no | Summarises a `.vis` file |
| `sim` | no | Simulated drift-scan observation through the same correlator (Sun disk, point, Gaussian; optional HI line, DC offset, instrument phase; dish diameter sets area and beam) |



## arc_test.jl and arc_plot.jl (over-the-air arc test)
A tone source is moved along an arc around two receive antennas on RX1 and RX2; the RX1-RX2 phase follows the exact path difference to the two antennas. `julia arc_test.jl predict f_lo=2.45e9 B_m=0.25 R_m=3 angles=-60:3:60` plans the marks (straight-line distance from the broadside mark, phase per step, points per fringe). `julia arc_test.jl capture f_lo=2.45e9 gain1=20 gain2=20 B_m=0.25 R_m=3 tag=A` prompts for each angle, takes broadside references for the instrument-phase drift, and shows each point against the prediction as it goes. `julia arc_test.jl fit tag=A` removes the drift, fits B, the bar offset and the instrument phase, checks the sign and the pass criteria, and writes CSVs for `julia arc_plot.jl tag=A` (PDF panels and a summary PNG). `julia arc_test.jl selftest` runs synthetic data through the fit. Uses pluto_interferometer.jl's radio and tone functions (pluto_interferometer.jl in this folder or in `Previous/`). Refuses 1400-1427 MHz (protected band). Procedure: vault note `Design/Arc Test Procedure.md`.
