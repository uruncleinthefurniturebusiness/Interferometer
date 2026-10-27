# Simulation of the bench arc test (Real/arc_test.jl): the Pluto+ tone is moved along an arc around the two
# receive antennas on the bar, captured on RX1 and RX2 of the Pluto and correlated as in capture_sim.jl.
# It is the bench version of the Sun drifting across the dishes. The difference from capture_sim.jl: the source
# is a single point a few metres away, so the delay uses the exact distance to each antenna (no plane wave).
# Test numbers from the Arc Test Procedure note; change them to match the bench.
# Joshua Smith
# 2026 - EEE4022S

using Plots, Random, FFTW, Dates, Printf
Plots.gr()  # Use GR backend for plotting
default(markerstrokewidth=0)   # small markers otherwise show only their black outline
Random.seed!(1234) # Seed for reproducibility

# Var declarations

# Signal Stuff
c = 299792458.0         # m/s, the speed of light
sample_rate = 2.5e6     # samples/s per channel, as set on the receiving Pluto
f_LO = 868.0e6          # Hz, receiver LO
f_offset = 200e3        # Hz, where the Pluto+ tone is put in baseband (clear of the DC spike)
f_ref_err = 3e3         # Hz, the Pluto+ tone error: its reference is independent of the receiver's (a few to tens of kHz)
f_tone = f_LO + f_offset + f_ref_err   # Hz, the RF frequency actually transmitted
λ = c / f_tone          # m, wavelength (34.5 cm at 868 MHz)
N_samples = 2^14        # samples per capture (the real capture is 5 s = 1.25e7 samples; this keeps the run quick)
tone_snr_db = 5.0       # dB, tone power over the receiver noise power across the whole band, per channel, at R
dc_leak = 0.5           # LO leakage (the DC spike), shared by both channels, amplitude in units of the noise rms
dc_guard = 2.5e3        # Hz, bins within this of DC are left out, as in FINAL

# Geometry (bar midpoint at the origin, RX1 at (+B/2, 0), RX2 at (-B/2, 0), source at (R sin θ, R cos θ);
# θ = 0 is broadside and positive θ is toward RX1, as in Real/arc_test.jl)
baseline = 0.25         # m, antenna spacing B, centre to centre
R_arc = 3.0             # m, arc radius from the midpoint of the bar
theta_marks = -60:3:60  # deg, the marks on the ground
mark_err = 0.01         # m, rms error placing the source on a mark (along the arc)
theta0_bar = 1.0        # deg, the bar's pointing offset: broadside is not exactly at the 0 mark (the fit finds it)
h = 1.2                 # m, height of the antennas and the source above the ground
gamma_ground = -0.1     # effective ground reflection coefficient, after the antenna patterns (0 switches it off)

# Instrument: RX1-RX2 phase from the cables and the PLLs, drifting during the session
phi0 = 40.0             # deg, instrument phase at the start
drift_rate = -5.8       # deg/min, measured on the 60 s run of 2026-10-06
drift_walk = 1.0        # deg/sqrt(min), random walk on top of the linear drift
t_step = 36.0           # s per capture (move the source, step away, 5 s capture): references every 5 are 3.6 min apart
ref_every = 5           # broadside reference after every N positions (also at the start and the end)

# Pass criteria (Arc Test Procedure note, outdoors)
pass_rms_deg = 10.0     # residual rms after the fit
pass_b_pct = 5.0        # fitted B within this % of the tape value
pass_rho = 0.98         # tone coherence at every position

# Path difference d2 - d1 (m), exact, as path_diff in Real/arc_test.jl
function path_diff(theta_deg, B, R)
    s = sind(theta_deg)
    return sqrt(R^2 + R * B * s + B^2 / 4) - sqrt(R^2 - R * B * s + B^2 / 4)
end
wrap180(x) = mod(x + 180, 360) - 180

d_far = 2 * baseline^2 / λ
@printf("Tone %.4f MHz, λ = %.2f cm, B = %.3f m = %.2f λ, R = %.1f m (plane-wave distance 2B²/λ = %.2f m, R is %.1f times that)\n",
        f_tone / 1e6, 100λ, baseline, baseline / λ, R_arc, d_far, R_arc / d_far)
@printf("Fringes over the arc: %.1f; phase change at broadside %.1f deg per deg of angle (%.1f points per fringe)\n",
        (path_diff(last(theta_marks), baseline, R_arc) - path_diff(first(theta_marks), baseline, R_arc)) / λ,
        360 * baseline / λ * π / 180, 360 / (360 * baseline / λ * π / 180 * step(theta_marks)))

# The capture order: a broadside reference, then the marks with a reference after every ref_every of them
schedule = Tuple{Symbol, Float64}[(:ref, 0.0)]
for (j, th) in enumerate(theta_marks)
    push!(schedule, (:point, th))
    (j % ref_every == 0 || j == length(theta_marks)) && push!(schedule, (:ref, 0.0))
end
N_cap = length(schedule)
t_cap = (0:N_cap-1) .* t_step                                   # s, time of each capture
phi_inst = phi0 .+ drift_rate .* t_cap ./ 60 .+
           cumsum([0.0; drift_walk .* sqrt(t_step / 60) .* randn(N_cap - 1)])   # deg, instrument phase at each capture
println("$(N_cap) captures ($(length(theta_marks)) positions, $(count(s -> s[1] == :ref, schedule)) broadside references), ",
        "$(round(last(t_cap) / 60, digits=1)) min")

# Stage 1: simulate IQ1 and IQ2 for one capture
# The tone reaches each antenna directly and off the ground (a longer path, from the image source 2h below).
# Downconverted with the shared LO, a path d gives the baseband phase -2π d/λ, so x1*conj(x2) has phase
# 2π(d2 - d1)/λ: positive when the signal reaches RX2 later. The instrument phase is put on RX2.
A_tone = sqrt(10^(tone_snr_db / 10))
t = (0:N_samples-1) ./ sample_rate
dc = dc_leak * cis(2π * rand())

function capture(theta_mark, phi_deg)
    th = theta_mark + rad2deg(mark_err / R_arc * randn()) - theta0_bar      # angle from the bar's broadside
    src = (R_arc * sind(th), R_arc * cosd(th))
    a = map((baseline / 2, -baseline / 2)) do xa                            # RX1, RX2
        d = hypot(src[1] - xa, src[2])
        dr = hypot(d, 2h)
        A_tone * (R_arc / d * cis(-2π * d / λ) + gamma_ground * R_arc / dr * cis(-2π * dr / λ))
    end
    tone = cis.(2π * (f_tone - f_LO) .* t)
    x1 = a[1] .* tone .+ randn(ComplexF64, N_samples) .+ dc
    x2 = a[2] .* tone .* cis(-deg2rad(phi_deg)) .+ randn(ComplexF64, N_samples) .+ dc
    return x1, x2, abs2.(a)
end

# Stage 2: process a capture. FFT both, multiply by the conjugate, and read the cross spectrum at the tone
# (phase and tone coherence, 7 bins as in arc_test.jl) and over the whole band without DC (band coherence,
# the SNR/(1+SNR) dilution capture_sim.jl checks).
win_hann = 0.5 .- 0.5 .* cos.(2π .* (0:N_samples-1) ./ N_samples)
f_bb = fftshift(fftfreq(N_samples, sample_rate))
band = abs.(f_bb) .> dc_guard

function process(x1, x2)
    X1 = fftshift(fft(win_hann .* x1))
    X2 = fftshift(fft(win_hann .* x2))
    S = X1 .* conj.(X2)
    P1 = abs2.(X1); P2 = abs2.(X2)
    kt = argmax(ifelse.(band, P1, 0.0))
    w = (kt - 3):(kt + 3)
    ρ_tone = sum(S[w]) / sqrt(sum(P1[w]) * sum(P2[w]))
    ρ_band = sum(S[band]) / sqrt(sum(P1[band]) * sum(P2[band]))
    noise = copy(band); noise[w] .= false
    snr_bin = 10log10(min(sum(P1[w]), sum(P2[w])) / (7 * sum(P1[noise]) / count(noise)))
    return (phase = rad2deg(angle(ρ_tone)), ρ_tone = abs(ρ_tone), ρ_band = abs(ρ_band), f = f_LO + f_bb[kt],
            snr = snr_bin, P1 = P1)
end

meas = Vector{NamedTuple}(undef, N_cap)
rho_band_exp = zeros(N_cap)
x1_show = zeros(ComplexF64, N_samples)           # first broadside capture, for the spectrum plot
for i in 1:N_cap
    x1, x2, pw = capture(schedule[i][2], phi_inst[i])
    meas[i] = process(x1, x2)
    rho_band_exp[i] = sqrt(pw[1] * pw[2]) / sqrt((pw[1] + 1) * (pw[2] + 1))
    i == 1 && (x1_show .= x1)
end

is_ref = [s[1] == :ref for s in schedule]
ip = findall(.!is_ref); ir = findall(is_ref)
theta_p = [schedule[i][2] for i in ip]
phase_raw = [m.phase for m in meas]

# Drift removal, as in arc_test.jl fit: unwrap the broadside references in time and interpolate them at each point
ref_unw = phase_raw[ir[1]] .+ cumsum([0.0; wrap180.(diff(phase_raw[ir]))])
interp(tq, tx, y) = [let k = clamp(searchsortedlast(tx, x), 1, length(tx) - 1)
                         y[k] + (y[k+1] - y[k]) * (x - tx[k]) / (tx[k+1] - tx[k]) end for x in tq]
phase_cor = wrap180.(phase_raw[ip] .- interp(t_cap[ip], t_cap[ir], ref_unw))

# Leave-one-out: predict each interior reference from its two neighbours
loo = [wrap180(ref_unw[k] - (ref_unw[k-1] + (ref_unw[k+1] - ref_unw[k-1]) *
       (t_cap[ir[k]] - t_cap[ir[k-1]]) / (t_cap[ir[k+1]] - t_cap[ir[k-1]]))) for k in 2:length(ir)-1]

# Fit B, the bar offset θ0 and a constant phase, exact distances, at the measured tone frequency (as arc_test.jl fit)
λ_meas = c / (sum(m.f for m in meas) / N_cap)
model(th, B, th0) = 360 .* path_diff.(th .- th0, B, R_arc) ./ λ_meas
function fit_arc(th, ψ)
    best = (rms = Inf, B = NaN, th0 = NaN, φc = NaN)
    for B in range(0.8baseline, 1.2baseline, step=1e-4), th0 in -5:0.05:5
        r = ψ .- model(th, B, th0)
        φc = rad2deg(angle(sum(cis.(deg2rad.(r)))))
        rms = sqrt(sum(abs2, wrap180.(r .- φc)) / length(r))
        rms < best.rms && (best = (rms = rms, B = B, th0 = th0, φc = φc))
    end
    return best
end
fit = fit_arc(theta_p, phase_cor)
resid = wrap180.(phase_cor .- fit.φc .- model(theta_p, fit.B, fit.th0))
B_err_pct = 100 * (fit.B - baseline) / baseline
near_far = maximum(abs, 360 .* (path_diff.(theta_marks, baseline, R_arc) .- baseline .* sind.(theta_marks)) ./ λ)

@printf("Tone SNR in the 7-bin window: %.1f to %.1f dB; tone |ρ| min %.4f (pass ≥ %.2f)\n",
        minimum(m.snr for m in meas), maximum(m.snr for m in meas), minimum(m.ρ_tone for m in meas), pass_rho)
@printf("Band |ρ| (no DC): mean %.4f, expected SNR/(1+SNR) %.4f\n",
        sum(m.ρ_band for m in meas) / N_cap, sum(rho_band_exp) / N_cap)
@printf("Drift: %.1f deg over the session; leave-one-out rms %.1f deg (pass ≤ 10)\n",
        ref_unw[end] - ref_unw[1], sqrt(sum(abs2, loo) / max(length(loo), 1)))
@printf("Fit: B = %.4f m (tape %.3f m, %+.2f %%, pass ≤ %.0f %%), θ0 = %+.2f deg (true %+.2f), phase %+.1f deg; rms %.2f deg (pass ≤ %.0f)\n",
        fit.B, baseline, B_err_pct, pass_b_pct, fit.th0, theta0_bar, fit.φc, fit.rms, pass_rms_deg)
@printf("Exact against far-field phase: at most %.2f deg over the arc\n", near_far)
pass = fit.rms <= pass_rms_deg && abs(B_err_pct) <= pass_b_pct && minimum(m.ρ_tone for m in meas) >= pass_rho
println(pass ? "PASS" : "FAIL")


# Plot 1: the layout from above (the antennas are B apart at the centre, the marks on the arc)
θd = range(-90, 90, length=181)
p1 = plot(R_arc .* sind.(θd), R_arc .* cosd.(θd), color=:gray, ls=:dot, label="arc, R = $(R_arc) m",
    aspect_ratio=:equal, xlabel="x (m)", ylabel="y (m)", title="Arc test layout (top view)")
scatter!(p1, R_arc .* sind.(theta_marks), R_arc .* cosd.(theta_marks), ms=3, label="marks ($(step(theta_marks))° steps)")
plot!(p1, [0, R_arc * sind(theta0_bar)], [0, R_arc * cosd(theta0_bar)], ls=:dash, color=:black, label="broadside (bar offset θ0)")
scatter!(p1, [baseline / 2, -baseline / 2], [0, 0], ms=5, marker=:utriangle, label="RX1 (+B/2), RX2 (-B/2)")

# Plot 2: the fringes, cosine of the drift-corrected phase against the exact prediction
θf = range(first(theta_marks), last(theta_marks), length=1201)
pred_fit = model(θf, fit.B, fit.th0) .+ fit.φc
p2 = plot(θf, cosd.(pred_fit), label="exact, fitted", lw=2, ls=:dash,
    xlabel="Angle θ (°)", ylabel="cos(phase)", title="Fringes (real part of the visibility)")
scatter!(p2, theta_p, cosd.(phase_cor), label="simulated", ms=3)

# Plot 3: wrapped phase, as measured (instrument phase and drift included) and drift-corrected
wrapped_line(y) = (w = wrap180.(y); w[[false; abs.(diff(w)) .> 180]] .= NaN; w)
p3 = plot(θf, wrapped_line(pred_fit), label="exact, fitted", lw=2, ls=:dash,
    xlabel="Angle θ (°)", ylabel="Phase (°)", ylims=(-180, 180), title="RX1 - RX2 phase (wrapped)")
scatter!(p3, theta_p, wrap180.(phase_raw[ip]), label="raw (instrument phase + drift)", ms=2, color=:gray)
scatter!(p3, theta_p, phase_cor, label="drift-corrected", ms=3)

# Plot 4: unwrapped phase against the exact and far-field predictions with the tape B
meas_unw = model(theta_p, fit.B, fit.th0) .+ resid
p4 = plot(θf, model(θf, baseline, 0.0), label="exact, tape B", lw=2,
    xlabel="Angle θ (°)", ylabel="Phase (°)", title=@sprintf("Unwrapped phase: fitted B = %.4f m (%+.2f %%)", fit.B, B_err_pct))
plot!(p4, θf, 360 .* baseline .* sind.(θf) ./ λ_meas, label="far field, tape B", ls=:dash)
scatter!(p4, theta_p, meas_unw, label="simulated (constant phase removed)", ms=3)

# Plot 5: residuals after the fit
p5 = scatter(theta_p, resid, ms=3, legend=false, xlabel="Angle θ (°)", ylabel="Residual (°)",
    title=@sprintf("Residuals: rms %.2f° (pass ≤ %.0f°)", fit.rms, pass_rms_deg))
hline!(p5, [-pass_rms_deg, 0, pass_rms_deg], ls=:dash, color=:gray)

# Plot 6: the broadside references, the instrument phase they track, and the interpolation used for the correction
tm = t_cap ./ 60
p6 = plot(tm, phi_inst .- phi_inst[1] .+ ref_unw[1], label="true instrument phase (shifted)", lw=2,
    xlabel="Time (min)", ylabel="Phase (°)", title="Broadside references and drift")
plot!(p6, tm[ir], ref_unw, label="interpolation", ls=:dash)
scatter!(p6, tm[ir], ref_unw, label="references", ms=4)

# Plot 7: coherence at every capture, tone and band
p7 = scatter(theta_p, [meas[i].ρ_tone for i in ip], label="tone |ρ|", ms=3, ylims=(0, 1.05),
    xlabel="Angle θ (°)", ylabel="|ρ|", title="Coherence")
scatter!(p7, theta_p, [meas[i].ρ_band for i in ip], label="band |ρ| (no DC)", ms=3)
plot!(p7, theta_p, rho_band_exp[ip], label="SNR/(1+SNR)", ls=:dash)
hline!(p7, [pass_rho], ls=:dot, color=:gray, label="tone pass")

# Plot 8: spectrum of RX1 at the first broadside reference (the tone at +offset, the DC spike at 0)
p8 = plot(f_bb ./ 1e3, 10log10.(meas[1].P1 ./ maximum(meas[1].P1)), legend=false,
    xlabel="Baseband frequency (kHz)", ylabel="Power (dB)", title="RX1 spectrum, broadside")
vline!(p8, [f_offset / 1e3], ls=:dash, color=:gray)


# Combine plots and explicitly force display
plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(1920, 1600), margin=8Plots.mm)
display(plt)

# Save image locally as a fallback
stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
figdir = joinpath(@__DIR__, "figures")   # next to the script, wherever you run it from
mkpath(figdir)
name = "arc_B$(baseline)m_R$(R_arc)m_$(round(f_tone / 1e6, digits=1))MHz_$(stamp).png"
savefig(plt, joinpath(figdir, name))

# The simulated points, for the overlay on the real run: julia arc_plot.jl tag=A sim=<this file>
# (phase with the instrument phase removed, unwrapped along the arc, as meas_unwrapped_deg - phi0 in arc_test.jl fit)
simfile = joinpath(figdir, replace(name, ".png" => "_points.csv"))
open(simfile, "w") do io
    println(io, "theta_deg,phase_unwrapped_deg,rho,rho_band")
    for (k, i) in enumerate(ip)
        @printf(io, "%.3f,%.3f,%.5f,%.5f\n", theta_p[k], meas_unw[k], meas[i].ρ_tone, meas[i].ρ_band)
    end
end
println("Simulated points: $(simfile)")

# Prevents terminal scripts from closing the plot window immediately
println("Press [ENTER] to exit...")
readline()
