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

## bt_capture.jl (2.4 GHz Bluetooth capture)
Records the 2.4 GHz band at 2.5 Msps to disk for post-processing: `julia -t 3 bt_capture.jl` (10 minutes of BLE advertising channel 38, RX1, raw int16 I/Q). Options `channel=37|38|39`, `freq=`, `minutes=`, `rx=1|2|both`, `gain=`, `out_dir=`; `julia bt_capture.jl look path=<file>.cs16` summarises a finished file. About 6 GB per stored channel for 10 minutes. Details in the vault note `Design/Bluetooth Capture.md`.

## tone and delayfit (splitter test)
`julia -t 3 V1.jl tone tag=A tn_sweep="-1000:200:1000"` measures a generator tone on both channels (coherence, phase, spur check) and prompts for each sweep frequency; `julia V1.jl delayfit tag=A tn_tag2=B` fits the delay from the phase slope and compares it with the known cable change. Procedure: vault note `Design/Splitter Test Procedure.md`.

## arc_test.jl and arc_plot.jl (over-the-air arc test)
A tone source is moved along an arc around two receive antennas on RX1 and RX2; the RX1-RX2 phase follows the exact path difference to the two antennas. `julia arc_test.jl predict f_lo=2.45e9 B_m=0.25 R_m=3 angles=-60:3:60` plans the marks (straight-line distance from the broadside mark, phase per step, points per fringe). `julia arc_test.jl capture f_lo=2.45e9 gain1=20 gain2=20 B_m=0.25 R_m=3 tag=A` prompts for each angle, takes broadside references for the instrument-phase drift, and shows each point against the prediction as it goes. `julia arc_test.jl fit tag=A` removes the drift, fits B, the bar offset and the instrument phase, checks the sign and the pass criteria, and writes CSVs for `julia arc_plot.jl tag=A` (PDF panels and a summary PNG). `julia arc_test.jl selftest` runs synthetic data through the fit. Uses V1.jl's radio and tone functions (V1.jl in this folder or in `Previous/`). Refuses 1400-1427 MHz (protected band). Procedure: vault note `Design/Arc Test Procedure.md`.

## bt_process.jl (decode BLE advertising packets from a bt_capture file)
`julia bt_process.jl path=<file>_rx1.cs16 channel=38` finds bursts, demodulates GFSK, syncs on the BLE access address, checks the CRC-24 and lists advertisers (writes `<file>_packets.csv`). `julia bt_process.jl selftest` runs a synthetic test. The whitening and CRC conventions were validated on real over-the-air packets. Details: vault note `Design/Bluetooth Capture.md`.

## bt_fft.jl (spectra of a bt_capture file)
`julia bt_fft.jl path=<file>_rx1.cs16 [n=1000000 start=<sample>]` writes an average and max-hold spectrum, a waterfall and a close-up of the strongest decoded packet as PNGs (plus the spectrum as CSV). `n` and `start` (or `start_s`, `dur_s`) select a portion; the whole file takes about 3 minutes. See the vault note `Design/Bluetooth Capture.md`.
