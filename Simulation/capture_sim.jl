# Simulation of the data capture and processing of the sun for the interferometry project
# Ideally, this would be run on the Pluto SDR, but for now it is a simulation of the data capture and processing.
# Joshua Smith
# 2026 - EEE4022S

using Plots, Random, FFTW, Dates
include("signal_shift_function 1.jl")  # Include the signal shifting function
Plots.gr()  # Use GR backend for plotting
Random.seed!(1234) # Seed for reproducibility

# Var declarations

# Signal Stuff
sample_rate = 2.5e6 # samples/s, set as it would be on the Pluto with the two Rx channels
f_LO = 868e6         # Hz, centre frequency the Pluto is tuned to (the LO); the Sun is broadband noise around it
c = 299792458.0   # m/s, the speed of light
N_samples = 1000    # Number of samples per capture

# Antenna Stuff
baseline = 0.69     # m, the distance between the two antennas 
dish_diameter = 3  # m, the diameter of the dish antennas

# Sun
au = 149597870700 # m, astronomical unit (distance from Earth to Sun)

# Far-field check: the delay formula below (B*sin(theta)/c) assumes a flat wavefront across the
# baseline, valid when the source is much farther than 2B^2/lambda (~4 km on 20 m). The Sun's
# distance then drops out and only its direction matters.
d_far = 2 * baseline^2 / (c / f_LO)
println("Far field: Sun at 1 AU is $(round(au / d_far, sigdigits=2)) times the plane-wave distance ",
        "2B²/λ = $(round(d_far / 1e3, digits=2)) km, so the plane-wave delay is valid")

# The Sun as an extended source, not a point.
# Why: at each snapshot along its path, the radio emission comes from the whole disk, and every
# part of the disk radiates independently (incoherent noise). Each part sits at a slightly
# different angle, so it reaches the two antennas with a slightly different delay. The disk
# (~0.6 deg at 1.4 GHz) is about as wide as one fringe (~0.6 deg on 20 m), so the fringes
# from different parts of the disk partly cancel when they add up. A point-source Sun would
# give full-contrast fringes (|V| constant); the real Sun gives much weaker ones. Modelling the
# disk as many independent emitters reproduces this from first principles, and the result can be
# checked against the van Cittert-Zernike prediction for a uniform disk, 2*J1(x)/x.
sun_diam = deg2rad(0.6)   # rad, radio diameter at 1.4 GHz (assumption: larger than the 0.53 deg optical disk; find a source)
R_sun = sun_diam / 2
n_grid = 41               # grid points across the diameter; spacing ~0.03 deg, well below one fringe
grid = range(-R_sun, R_sun, length=n_grid)
# Fixed grid of emitters filling the disk (a fixed grid avoids the scatter random positions would add).
# Only the offset along the baseline direction changes the delay, so keep just that coordinate.
sub_offsets = [x for x in grid, y in grid if x^2 + y^2 <= R_sun^2]
N_sub = length(sub_offsets)

# Earth (not used in this proof of concept, kept for the full simulation)
T_e = 24*60*60      # One solar day in seconds (gives the Sun's hour-angle rate of 15 deg/h)
f_e = 1/T_e         # Frequency of earth rotation in Hz
ω_e = 2π*f_e        # Angular frequency of earth rotation in rad/s
R_e = 6371000       # m, radius of the earth


# Angular sweep range (-3° to +3°). One fringe is asin(c/(f_LO*B)) near broadside (~0.605°),
# so use samples_per_fringe points per fringe (at least 10 are needed to see it working).
theta_fringe = asin(c / (f_LO * baseline))
samples_per_fringe = 20
dtheta = theta_fringe / samples_per_fringe
theta_max = deg2rad(3.0)  # 0,0523599 rad, 3 deg in radians
N_pos = 2 * round(Int, theta_max / dtheta) + 1
theta_range = range(-theta_max, theta_max, length=N_pos)

println("One fringe = $(round(rad2deg(theta_fringe), digits=3)) deg; $(N_pos) positions, ",
        "$(round(rad2deg(step(theta_range)), digits=4)) deg per step ",
        "(~$(round(rad2deg(step(theta_range)) / (15/3600), digits=1)) s of Sun motion per step)")

# Stage 1: simulate IQ1 and IQ2, one column per sky position
IQ1 = Matrix{ComplexF64}(undef, N_samples, N_pos)
IQ2 = similar(IQ1)

SNR_linear = 0.5 # Total solar signal power (whole disk) relative to SDR noise power

# Keep the Sun-only signals of one snapshot for plotting (plots 5 and 6). A quarter fringe off
# broadside, where the fringe phase between the antennas is 90 degrees.
i_show = argmin(abs.(theta_range .- theta_fringe / 4))
sun1_show = zeros(ComplexF64, N_samples)
sun2_show = zeros(ComplexF64, N_samples)

for (i, theta) in enumerate(theta_range)
    sun1 = zeros(ComplexF64, N_samples)
    sun2 = zeros(ComplexF64, N_samples)

    # Each grid point is an independent emitter carrying an equal share of the Sun's power
    for d_off in sub_offsets
        # Geometric delay between antennas for this part of the disk
        delta_tau = (baseline * sin(theta + d_off)) / c

        part_signal = sqrt(SNR_linear / N_sub) .* randn(ComplexF64, N_samples)
        sun1 .+= part_signal

        # "circular": the delay is a tiny fraction of a sample, so no samples should be zeroed
        shifted_part = shift_sampled_signal(part_signal, delta_tau, 1/sample_rate, "circular")
        sun2 .+= shifted_part .* exp(-2π * im * f_LO * delta_tau)
    end

    n1 = randn(ComplexF64, N_samples)
    n2 = randn(ComplexF64, N_samples)

    IQ1[:, i] = sun1 .+ n1
    IQ2[:, i] = sun2 .+ n2

    if i == i_show
        sun1_show .= sun1
        sun2_show .= sun2
    end
end

# Stage 2: process each position. FFT both, multiply by the conjugate, IFFT, zero lag (the delay (≤ 67 ns) is under one sample (400 ns).)
C_max = Vector{ComplexF64}(undef, N_pos)   # complex visibility, normalised so |C| <= 1
#peak_lag = Vector{Int}(undef, N_pos)       # index of the correlation peak (1 = zero lag)

for i in 1:N_pos
    iq1 = IQ1[:, i]
    iq2 = IQ2[:, i]

    # 1. Frequency domain conversion
    F1 = fft(iq1)
    F2 = fft(iq2)

    # 2. Conjugate multiplication & Inverse FFT (Cross-correlation)
    corr_time = ifft(F1 .* conj.(F2))

    # 3. Peak visibility extraction
    idx = 1 #val, idx = findmax(abs.(corr_time))
    #peak_lag[i] = idx
    C_max[i] = corr_time[idx] / sqrt(sum(abs2, iq1) * sum(abs2, iq2)) # zero lag = Σ x1·conj(x2) (Parseval); divide by channel powers so |C| ≤ 1.
end 

# Self-check against the analytic answer: x1*conj(x2) has phase +2*pi*f_LO*tau
expected = exp.(2im * π * f_LO * baseline .* sin.(theta_range) ./ c)
C_rot = C_max .* conj.(expected)          # each snapshot with the predicted fringe phase removed (used for fix 6 below)
phase_err_deg = rad2deg.(angle.(C_rot))

# Expected magnitude for the disk (van Cittert-Zernike): the receiver noise limits the correlation
# to SNR/(1+SNR), and the disk's partial cancellation multiplies that by 2*J1(x)/x.
# J1 from its integral definition, J1(x) = (1/pi) * integral over 0..pi of cos(t - x*sin(t)) dt
besselj1_int(x) = sum(cos(t - x * sin(t)) for t in range(0, π, length=2001)) / 2000
x_disk = π * baseline * sun_diam / (c / f_LO)
V_disk = 2 * besselj1_int(x_disk) / x_disk
V_expected = SNR_linear / (1 + SNR_linear) * V_disk

println("Sun modelled as $(N_sub) emitters over a $(round(rad2deg(sun_diam), digits=2)) deg disk")
#println("Correlation peak at zero lag in $(count(==(1), 1)) of $(N_pos) positions")
println("Visibility magnitude: min $(round(minimum(abs.(C_max)), digits=4)), max $(round(maximum(abs.(C_max)), digits=4)), ",
        "mean $(round(sum(abs.(C_max)) / N_pos, digits=4)); expected $(round(V_expected, digits=4)) ",
        "(disk factor $(round(V_disk, digits=3)), point source would give $(round(SNR_linear / (1 + SNR_linear), digits=4)))")
println("Phase error vs analytic: max $(round(maximum(abs.(phase_err_deg)), digits=3)) deg, ",
        "mean $(round(sum(phase_err_deg) / N_pos, digits=3)) deg")

# FIX 6: COHERENT (UNBIASED) CHECK OF |V| (added 2026-10-07)
# Problem: the "mean" magnitude printed above averages abs.(C_max). Noise adds in a random direction
# but a magnitude can never be negative, so noise never cancels and the average reads high (Rician
# bias): about |V| + sigma^2/(2|V|) = 0.067 instead of 0.063 here, ~6% high at this SNR.
# Fix (fringe stopping): C_rot (above) removes the predicted fringe phase from every snapshot, so the
# signal points along the positive real axis in all of them while the noise stays random. Averaging
# the complex values then lets the noise cancel (shrinks as 1/sqrt(N_pos)) instead of adding.
# - Real part   = unbiased estimate of V (negative if the disk were past the first null of 2*J1(x)/x)
# - Imag part   = pure noise, should be ~0 (sanity check)
# This also tests the phase: if the predicted phase were wrong (sign, baseline, delay), the snapshots
# would not line up and the average would collapse towards zero. The old magnitude average hid that.
V_coh = sum(C_rot) / N_pos                    # coherent average over all positions
σ_coh = 1 / sqrt(2 * N_samples * N_pos)       # expected noise on each component of V_coh (~0.0016 here)
println("Coherent V (fringe-stopped): real $(round(real(V_coh), digits=4)) ± $(round(σ_coh, digits=4)) ",
        "(expected $(round(V_expected, digits=4))), imaginary $(round(imag(V_coh), digits=4)) (should be ~0)")
        

# Plot 1: Interference Fringes (Phase Angle), simulated against the analytic curve
p1 = plot(rad2deg.(theta_range), angle.(expected),
    label="analytic",
    ls=:dash,
    lw=2,
    xlabel="Angle (°)",
    ylabel="Phase (rad)",
    title="Interferometer Fringe Pattern"
    
)
scatter!(p1, rad2deg.(theta_range), angle.(C_max), label="simulated", ms=2)

# Plot 2: Spectrum of the fringe pattern over angle.
# The fringe oscillates at B/lambda cycles per radian, so the peak sits at the baseline length.
d_theta = step(theta_range)                                        # Angular step size (radians)
spatial_freqs = fftshift(fftfreq(N_pos, 1/d_theta))                # cycles per radian
baseline_axis = spatial_freqs .* (c / f_LO)                        # metres
spec = abs.(fftshift(fft(C_max)))
B_est = baseline_axis[argmax(spec)]
bin_width = (c / f_LO) / (N_pos * d_theta)                         # metres per FFT bin
println("Spectrum peak at B = $(round(B_est, digits=2)) m (true $(baseline) m, bin width $(round(bin_width, digits=2)) m)")

p2 = plot(baseline_axis, spec,
    xlims=(-60, 60),
    xlabel="Baseline length (m)",
    ylabel="Magnitude",
    title="Fringe-Frequency Spectrum (peak at the baseline length)",
    legend=false
)
vline!(p2, [baseline], ls=:dash)

p3 = plot(rad2deg.(theta_range), abs.(C_max),
    xlabel="Angle (°)",
    ylabel="Magnitude",
    title="Visibility Magnitude vs Angle",
    legend=false
)
hline!(p3, [V_expected], ls=:dash)   # van Cittert-Zernike prediction for the disk

# Plot 4: real part of the visibility, the classic cosine fringes
p4 = plot(rad2deg.(theta_range), real.(C_max),
    xlabel="Angle (°)",
    ylabel="Re(visibility)",
    title="Fringes (real part of the visibility)",
    legend=false
)

# Plots 5 and 6: the signals themselves at one snapshot (first n_show samples, real part = I)
n_show = 60
t_us = (0:n_show-1) ./ sample_rate .* 1e6     # time in microseconds (0.4 us per sample)
theta_show = round(rad2deg(theta_range[i_show]), digits=3)

# The Sun-only signals at the two antennas are not copies of each other: the disk is resolved, so
# they correlate only by the disk factor (not 1), at the fringe phase for this angle.
ρ_sun = sum(sun1_show .* conj.(sun2_show)) / sqrt(sum(abs2, sun1_show) * sum(abs2, sun2_show))
println("Snapshot at $(theta_show) deg: Sun-only correlation |ρ| = $(round(abs(ρ_sun), digits=3)) ",
        "(disk factor $(round(V_disk, digits=3)), noise ~$(round(1 / sqrt(2 * N_samples), digits=3))), ",
        "phase $(round(rad2deg(angle(ρ_sun)), digits=1)) deg (predicted $(round(rad2deg(angle(expected[i_show])), digits=1)))")

# Plot 5: the Sun's signal as it reaches each antenna, before the receiver adds its own noise
p5 = plot(t_us, real.(sun1_show[1:n_show]),
    label="antenna 1",
    marker=:circle,
    ms=2,
    xlabel="Time (µs)",
    ylabel="I (Re)",
    title="Sun signal at each antenna, θ = $(theta_show)°"
)
plot!(p5, t_us, real.(sun2_show[1:n_show]), label="antenna 2", marker=:circle, ms=2)

# Plot 6: what the Pluto captures, the same Sun signal plus independent receiver noise per channel
p6 = plot(t_us, real.(IQ1[1:n_show, i_show]),
    label="RX1",
    marker=:circle,
    ms=2,
    xlabel="Time (µs)",
    ylabel="I (Re)",
    title="Captured signal (Sun + receiver noise), θ = $(theta_show)°"
)
plot!(p6, t_us, real.(IQ2[1:n_show, i_show]), label="RX2", marker=:circle, ms=2)


# Combine plots and explicitly force display
plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1920, 1080), margin=8Plots.mm)
display(plt)

# Save image locally as a fallback
stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")      # e.g. 2026-10-08_181530
figdir = joinpath(@__DIR__, "figures")   # next to the script, wherever you run it from
mkpath(figdir)                           # creates the folder if it doesn't exist
name = "sun_B$(baseline)m_N$(N_samples)_grid$(n_grid)_$(stamp).png"
savefig(plt, joinpath(figdir, name))

# Prevents terminal scripts from closing the plot window immediately
println("Press [ENTER] to exit...")
readline()