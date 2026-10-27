# Monte Carlo of the arc test at 868 MHz indoors, to set the pass criteria for `arc_test.jl fit`.
# Replaces the 2450.2 MHz Monte Carlo of 2026-10-15 (arc_mc.py, never saved), with the same error sizes and the same
# indoor multipath. Each case simulates n sessions with random set-up errors, drift and multipath, removes the drift
# and fits them with the functions of Real/arc_test.jl (so the fit is exactly the one the real data goes through), and
# records the residual rms, the fitted B against the tape, the bar offset and the leave-one-out drift check.
# Faults are injected to check that the criteria catch them.
#
# Usage (from the Simulation folder):  julia -t auto arc_mc.jl [n=300]
# Writes figures/arc_mc_<time>.csv (every session) and .png (distributions).
# Joshua Smith
# 2026 - EEE4022S

include(joinpath(@__DIR__, "..", "Real", "arc_test.jl"))   # fit_arc, pred_phase_deg, unwrap_deg, interp_clamped, wrap180, C_LIGHT

using Plots, Random, Printf, Statistics, Dates
Plots.gr()
default(markerstrokewidth=0)

# Var declarations

# Test set-up
const f_tone = 868.2e6           # Hz, the Pluto+ tone
const λ = C_LIGHT / f_tone       # m (34.5 cm)
const h = 1.2                    # m, antennas and source above the floor
const t_step = 36.0              # s per capture (move, step away, 5 s capture): references every 5 are 3.6 min apart
const ref_every = 5              # broadside reference after every N positions (also at the start and the end)
const phase_noise = 1.0          # deg rms per capture (tone SNR >= 30 dB; mostly the mount and people moving)
n_runs = 300                     # sessions per case

# Set-up errors, as in the 2450.2 MHz Monte Carlo
const σ_tape = 0.002             # m, tape value of B given to the fit
const σ_R = 0.01                 # m, true arc radius against the value given to the fit
const σ_mark = 0.015             # m, placing the source on a mark (rms distance, any direction)
const σ_centre = 0.01            # m, arc centre against the bar midpoint (each axis)
const σ_rot = 1.0                # deg, bar rotation (broadside against the 0 mark)
const drift_mean = -5.8          # deg/min, measured 2026-10-06
const drift_sd = 2.0             # deg/min, session to session
const wobble_max = 8.0           # deg, amplitude of a slow wobble on top (period 10 to 30 min)

# Indoor multipath: a floor reflection plus point scatterers (walls, furniture, metal) around the set-up
const floor_in = (0.2, 0.4)      # floor reflection coefficient range
const n_scat = 4                 # scatterers
const scat_amp = (0.1, 0.3)      # scatterer reflection coefficient range
const scat_r = (1.5, 5.0)        # m, scatterer distance from the bar midpoint, any direction
const scat_z = (0.0, 2.5)        # m, scatterer height

# Layouts: B as captured on 2026-10-24, and the smaller layout for a confined space
layouts = [(name = "B 0.69 m, R 3 m, -60:3:60", B = 0.69, R = 3.0, th = collect(-60.0:3.0:60.0)),
           (name = "B 0.69 m, R 2 m, -45:3:45", B = 0.69, R = 2.0, th = collect(-45.0:3.0:45.0))]

# Cases: multipath on or off, and an injected fault (all faults indoors)
cases = [(name = "Geometry and drift only",      multipath = false, fault = :none),
         (name = "Indoors",                      multipath = true,  fault = :none),
         (name = "Fault: tape value 10% wrong",  multipath = true,  fault = :tape),
         (name = "Fault: 90° phase jump mid-run", multipath = true,  fault = :jump),
         (name = "Fault: arc radius 5 cm off",   multipath = true,  fault = :radius)]

for a in ARGS
    k, v = split(a, '='; limit = 2)
    k == "n" ? (global n_runs = parse(Int, v)) : error("Unknown option '$k' (only n=)")
end

uni(rng, r) = r[1] + (r[2] - r[1]) * rand(rng)
dist(a, b) = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2)

# Field at an antenna from the source: the direct path, the floor image 2h below, and each scatterer
# (reflection coefficient k, path source -> scatterer -> antenna). Amplitude 1/path length.
function field(s, xa, Γ, scat)
    a = (xa, 0.0, h)
    d = dist(s, a)
    E = cis(-2π * d / λ) / d
    if Γ != 0
        di = dist((s[1], s[2], -h), a)
        E -= Γ * cis(-2π * di / λ) / di
    end
    for (p, k) in scat
        L = dist(s, p) + dist(p, a)
        E += k * cis(-2π * L / λ) / L
    end
    return E
end

# One simulated session: the capture order of arc_test.jl capture, then its drift removal and fit
function session(rng, L, c)
    B = L.B
    Bt = c.fault == :tape ? 1.1B : B + σ_tape * randn(rng)            # tape value given to the fit
    Rt = L.R + σ_R * randn(rng) + (c.fault == :radius ? 0.05 : 0.0)   # true radius; the fit is given L.R
    cx, cy = σ_centre * randn(rng), σ_centre * randn(rng)
    rot = deg2rad(σ_rot * randn(rng))
    Γ = c.multipath ? uni(rng, floor_in) : 0.0
    scat = c.multipath ? [(let r = uni(rng, scat_r), az = 2π * rand(rng)
                               (r * sin(az), r * cos(az), uni(rng, scat_z)) end,
                          uni(rng, scat_amp) * cis(2π * rand(rng))) for _ in 1:n_scat] : []
    rate = drift_mean + drift_sd * randn(rng)
    wob = wobble_max * rand(rng); per = uni(rng, (600.0, 1800.0)); wph = 2π * rand(rng)
    φ0 = 360 * rand(rng)

    sched = Tuple{Symbol, Float64}[(:ref, 0.0)]
    for (j, th) in enumerate(L.th)
        push!(sched, (:point, th))
        (j % ref_every == 0 || j == length(L.th)) && push!(sched, (:ref, 0.0))
    end
    t = (0:length(sched)-1) .* t_step
    t_jump = t[end] / 2

    ph = map(eachindex(sched)) do i
        th = sched[i][2]
        e = σ_mark / sqrt(2) .* (randn(rng), randn(rng))
        x = cx + Rt * sind(th) + e[1]; y = cy + Rt * cosd(th) + e[2]
        s = (x * cos(rot) + y * sin(rot), -x * sin(rot) + y * cos(rot), h)     # into the bar's frame
        geo = rad2deg(angle(field(s, B / 2, Γ, scat) * conj(field(s, -B / 2, Γ, scat))))
        inst = φ0 + rate * t[i] / 60 + wob * sin(2π * t[i] / per + wph) + (c.fault == :jump && t[i] > t_jump ? 90.0 : 0.0)
        wrap180(geo + inst + phase_noise * randn(rng))
    end

    ir = findall(s -> s[1] == :ref, sched); ip = findall(s -> s[1] == :point, sched)
    tr = Float64.(t[ir]); pr = unwrap_deg(ph[ir])
    ph_c = [wrap180(ph[i] - interp_clamped(tr, pr, Float64(t[i]))) for i in ip]
    fp = fit_arc(L.th, ph_c, L.R, λ, Bt, 5.0)
    loo = [wrap180(pr[k] - (pr[k-1] + (pr[k+1] - pr[k-1]) * (tr[k] - tr[k-1]) / (tr[k+1] - tr[k-1]))) for k in 2:length(pr)-1]
    return (rms = fp.rms, berr = 100 * abs(fp.B - Bt) / Bt, theta0 = abs(fp.theta0),
            loo = sqrt(mean(abs2, loo)), refgap = maximum(abs, diff(pr)))
end

q95(x) = quantile(x, 0.95)

@printf("Arc-test Monte Carlo: %.1f MHz (λ = %.1f cm), %d sessions per case, %d threads\n\n", f_tone / 1e6, 100λ, n_runs, Threads.nthreads())
results = Dict{Tuple{Int, Int}, Vector{NamedTuple}}()
for (li, L) in enumerate(layouts), (ci, c) in enumerate(cases)
    out = Vector{NamedTuple}(undef, n_runs)
    Threads.@threads for r in 1:n_runs
        out[r] = session(MersenneTwister(1000 * li + 100_000 * ci + r), L, c)
    end
    results[(li, ci)] = out
end

# Criteria from the indoor distribution: each AT-04 limit is kept unless the indoor 95th percentile is above it, in which
# case it is raised to that percentile, rounded up (residual to 5 deg, B to 1 %, bar offset to 1 deg, leave-one-out to
# 5 deg). Then how often each case passes all four.
const at04 = (rms = 10.0, b = 5.0, theta0 = 4.0, loo = 10.0)     # AT-04 limits (outdoors)
stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
figdir = joinpath(@__DIR__, "figures"); mkpath(figdir)
panels = []
for (li, L) in enumerate(layouts)
    ind = results[(li, 2)]
    rms_lim = max(at04.rms, 5 * ceil(q95([r.rms for r in ind]) / 5))
    b_lim = max(at04.b, ceil(q95([r.berr for r in ind])))
    t0_lim = max(at04.theta0, ceil(q95([r.theta0 for r in ind])))
    loo_lim = max(at04.loo, 5 * ceil(q95([r.loo for r in ind]) / 5))
    passes(r) = r.rms <= rms_lim && r.berr <= b_lim && r.theta0 <= t0_lim && r.loo <= loo_lim
    println("Layout: $(L.name) ($(round(L.B / λ, digits=2)) λ, $(round(2L.B^2 / λ, digits=2)) m plane-wave distance)")
    println("| Case | Residual rms, median / 95th pct | |B_fit - B_tape|, median / 95th pct | Bar offset 95th pct | Leave-one-out 95th pct | Passes all four |")
    println("|---|---|---|---|---|---|")
    for (ci, c) in enumerate(cases)
        x = results[(li, ci)]
        rms = [r.rms for r in x]; be = [r.berr for r in x]
        @printf("| %s | %.1f° / %.1f° | %.1f %% / %.1f %% | %.1f° | %.1f° | %.0f %% |\n", c.name, median(rms), q95(rms),
                median(be), q95(be), q95([r.theta0 for r in x]), q95([r.loo for r in x]), 100 * count(passes, x) / length(x))
    end
    @printf("Proposed indoor criteria: residual rms <= %.0f°, |B_fit - B_tape| <= %.0f %%, bar offset <= %.0f°, leave-one-out <= %.0f°; ",
            rms_lim, b_lim, t0_lim, loo_lim)
    @printf("references more than 90° apart in %.1f %% of indoor sessions\n\n", 100 * count(r -> r.refgap > 90, ind) / length(ind))

    p1 = plot(xlabel = "Residual rms (°)", ylabel = "Sessions", title = L.name, legend = :topright, titlefontsize = 10)
    p2 = plot(xlabel = "|B_fit - B_tape| (%)", ylabel = "Sessions", title = L.name, legend = :topright, titlefontsize = 10)
    for (ci, c) in enumerate(cases)
        stephist!(p1, [r.rms for r in results[(li, ci)]], bins = 0:1:60, label = c.name, lw = 2)
        stephist!(p2, [r.berr for r in results[(li, ci)]], bins = 0:0.5:15, label = c.name, lw = 2)
    end
    vline!(p1, [rms_lim], ls = :dash, color = :black, label = "limit")
    vline!(p2, [b_lim], ls = :dash, color = :black, label = "limit")
    push!(panels, p1, p2)
end

open(joinpath(figdir, "arc_mc_$(stamp).csv"), "w") do io
    println(io, "layout,case,run,rms_deg,b_err_pct,theta0_abs_deg,loo_rms_deg,max_ref_step_deg")
    for (li, L) in enumerate(layouts), (ci, c) in enumerate(cases), (r, x) in enumerate(results[(li, ci)])
        @printf(io, "\"%s\",\"%s\",%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", L.name, c.name, r, x.rms, x.berr, x.theta0, x.loo, x.refgap)
    end
end
savefig(plot(panels..., layout = (length(layouts), 2), size = (1600, 900), margin = 6Plots.mm), joinpath(figdir, "arc_mc_$(stamp).png"))
println("Wrote figures/arc_mc_$(stamp).csv and .png")
