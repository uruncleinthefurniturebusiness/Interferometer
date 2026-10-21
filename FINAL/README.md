# FINAL: host software for the two-channel ADALM-Pluto interferometer

Three programs that share one core (`src/`), so the radio is configured, the samples correlated and the files read the same way everywhere.

| File | What it does |
|---|---|
| `pluto_interferometer.jl` | The host processing: `check`, `run` (capture → FX correlator → `.vis` / `.meta` / `.csv`, with fringe stopping when `target=sun`), `fake` (same pipeline from a stand-in radio), `show`, `selftest`, `sim` (simulated observation of a source on the baseline, checked against its own expectation) |
| `viewer.jl` | Live display in its own process: fringes (Re ρ, Im ρ, \|ρ\| and phase against time, raw and fringe-stopped), visibility amplitude and phase spectra, auto spectra, waterfall |
| `stress_test.jl` | Characterises the receiving Pluto: `readback`, `rate`, `floor`, `levels`, `coherence`, `soak`, then `report` writes `results_table.tex` and `fig_*.pdf` |
| `src/PlutoIF.jl` | The shared module: `config.jl`, `dsp.jl` (FX correlator, tone fit, fits), `visfile.jl`, `radio.jl`, `fake.jl`, `pipeline.jl` (capture + processing threads), `geometry.jl` (Sun, baseline, fringe stopping), `sim.jl` (the `sim` source models and sample generator) |

Convention everywhere: V = x₁·x₂\*; a positive phase means the signal reaches RX2 later.

## Observing (or demonstrating)

```
julia -t 3 pluto_interferometer.jl  # menu: the modes, the current settings and set key=value from a prompt
julia -t 3 pluto_interferometer.jl check addr=usb:1.2.5
julia -t 3 pluto_interferometer.jl run addr=usb:1.2.5 duration_s=3600 tag=sun1 target=sun baseline_m=20 baseline_az_deg=35
julia viewer.jl                      # second terminal: follows the newest run in C:/PlutoCaptures
```

Without hardware: `julia -t 3 pluto_interferometer.jl fake duration_s=600 fake_period_s=60`, then `julia viewer.jl`.
`julia viewer.jl path=<file>.vis` replays a finished run; `snapshot=pic.png` saves one picture instead of opening a window.

Simulated observation (AT-02, and the dish prediction): `julia -t 3 pluto_interferometer.jl sim baseline_m=20 baseline_az_deg=35 sim_dec_deg=-10`, or with the dish values `sim_area_m2=2.31 sim_flux_jy=5e5 sim_tsys_k=150` in place of `sim_snr_db`. It writes `PI_<time>_SIM.vis`, `.meta` and `.truth.csv` (expected and measured ρ per record) and ends with PASS or FAIL. The `sim_...` settings are in `src/sim.jl`; the baseline, latitude and source diameter (`diam_arcmin`) are the same keys as for `run`.

Files go to `C:/PlutoCaptures` (outside OneDrive). One record per second, 65 576 bytes each (about 236 MB per hour).

## Stress-test session (about 1 hour)

The Pluto+ only transmits, from its own laptop with `Real/tx_source.jl`. The script prints each Pluto+ command and waits for Enter.

```
julia stress_test.jl plan                                  # the session plan and the Pluto+ commands
julia -t 3 stress_test.jl readback addr=usb:1.2.5          # 1 min smoke test first
julia -t 3 stress_test.jl all addr=usb:1.2.5               # everything, then the report
julia stress_test.jl report out=C:/PlutoCaptures/STRESS_<time>
```

- Receive-only tests at 1420 MHz; Pluto+ tests with the receiver LO at 869.0 MHz, tone 869.2 MHz (Pluto+ TX LO 868.9 MHz, so its leakage and image fall away from the measured bins), noise 869.0 ± 0.75 MHz. Everything transmitted stays in 868–870 MHz.
- Set-up for the 869 MHz tests: both receive antennas on the bar, the Pluto+ antenna at broadside 2–3 m away, same height, nobody moving.
- Each test can be run on its own, and again; `report` uses the latest run of each test. Every value is in `summary.tsv`, the raw numbers in the per-test CSV files.
- `fake=true` runs the whole session against the stand-in radio. Use it to check the code, never for report numbers.

### What it cannot claim
- No splitter: delay, phase and RX1/RX2 level matching are over the air, so they include the antennas, the geometry (about zero at broadside) and reflections.
- No calibrated source: levels are in dBFS and dB relative, not dBm. The "LO offset" is the difference between the Pluto's and the Pluto+'s references, not an absolute error.
- Antennas removed means open ports, not 50 Ω terminations.

## Changes from `Real/pluto_interferometer.jl`
- The tracking loops are now set on explicitly (1) and checked in the read-back, instead of being left as found.
- Before each measurement, the stale samples in the Pluto's buffers are read out (`flush!`). While the host is not reading (a prompt, a gain change, moving the source), up to 0.8 s of old data waits there. `Real/arc_test.jl` discards only 2 blocks, so up to 6 stale blocks (0.6 s of a 5 s point) can come from when the source was being moved.
- The tone fit skips ±2.5 kHz around DC, so a weak tone is not mistaken for the LO leakage.
- The `.meta` file is written at the start of a run (so the viewer can open it at once) and completed at the end.
- `sim` moved here (2026-10-21) with the same physics and output; its baseline, latitude and diameter keys are now `baseline_m`, `baseline_az_deg`, `lat_deg` and `diam_arcmin` (were `sim_baseline_m`, `sim_az_deg`, `sim_lat_deg`, `sim_diam_arcmin`), and the baseline has no default. For the same settings and seed it gives the same records as the `Real/` version.
- `tone`, `delayfit` and `rate` stay in `Real/`; `rate` and the delay measurement are now part of `stress_test.jl`.
