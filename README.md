# Interferometer
This repo contains all notes and ancillary information regarding my work for EEE4022S research project


The aim of this project is to develop an Inteferometer using the Analog Devices Adalm Pluto SDR to 

## V1.jl (single-file pipeline)
One Julia file that takes the Pluto from power-on to timestamped visibilities. Run from this folder with `julia -t 3 V1.jl <mode> [key=value ...]`.

| Mode | Hardware | Purpose |
|---|---|---|
| `selftest` | no | Checks the correlator against a simulated source, the dropped-sample monitor, the file format, the Bessel function and the antenna inputs |
| `check` | yes | Connects, sets and **reads back** every setting, captures briefly, reports health, spectral peaks and correlations |
| `rate` | yes | Sustained capture; real sample rate against the host clock (`tone=true` also watches for dropped samples) |
| `run` | yes | Capture, FX correlator, timestamped visibility records to disk |
| `show` | no | Summarises a `.vis` file |
| `sim` | no | Simulated drift-scan observation through the same correlator (Sun disk, point, Gaussian; optional HI line, DC offset, instrument phase; dish diameter sets area and beam) |

Notes and design reasoning are in the Obsidian vault (`Design/V1 Code Explained.md`, `V1.md`). Visibility convention: V = x1 * conj(x2); positive phase means the signal reaches channel 2 later. Keep captures outside the OneDrive folder (default `C:/PlutoCaptures`).
