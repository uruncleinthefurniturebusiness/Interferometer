using FFTW
using Statistics
using Random

# =================================================================
# Interferometer simulator -- synthetic two-channel IQ data with a
# KNOWN ground-truth visibility, for validating the correlator
# pipeline independent of any real hardware/dish access.
#
# Physical model: a single point source in the far field, received
# at two antennas separated by a baseline. A source at angle theta
# off the baseline's broadside direction imposes a geometric delay
#     tau = B * sin(theta) / c
# between the two antennas. See Theory/van Cittert-Zernike Theorem.md
# for why this delay is exactly what ties a measured visibility phase
# to a source's sky position.
#
# For a point source, the true complex visibility has magnitude 1 (a
# point source is never resolved) and phase = -2*pi*f_RF*tau, i.e.
#     V_true = exp(-j*2*pi*f_center*tau)
# This becomes a direct check: does the ACTUAL correlator code recover
# this magnitude/phase from noisy synthetic data?
#
# Two distinct effects of the delay are modeled, not just one:
#   1. The bulk carrier-phase rotation exp(-j*2*pi*f_center*tau) --
#      this IS the fringe phase, the dominant effect at this baseline
#      scale (tau ~ 5 ns for a 20 m baseline at 15 deg, vs. f_center
#      ~1420 MHz -- a tiny delay rotates this a lot).
#   2. A fractional-sample delay of the baseband ENVELOPE itself,
#      applied via a frequency-domain phase ramp (exact for a
#      bandlimited signal) -- negligible at this baseline scale
#      (tau << 1/sample_rate) but implemented properly so the
#      simulator stays correct if extended to longer baselines later.
#
# Output uses the same ComplexF32, normalized representation that
# captureDualRX! produces, so the same correlator code already used
# on real Pluto captures can run unmodified against this simulator's
# output -- this is what makes it a genuine pipeline validation
# rather than a separate, parallel implementation that could hide a
# bug of its own.
# =================================================================

const C_LIGHT = 2.99792458e8

"""
    simulate_point_source(; f_center, sample_rate, n_samples, baseline_m, theta_deg, snr_db, seed=nothing)

Generates two phase-coherent complex baseband streams (rx1, rx2) as if
captured by a two-element interferometer observing a single point
source at angle `theta_deg` off the baseline's broadside direction.

Returns `(rx1, rx2, tau, V_true)`: `tau` is the injected geometric
delay in seconds, `V_true` is the analytically expected complex
visibility -- the ground truth the correlator output should be
checked against.
"""
function simulate_point_source(; f_center::Float64, sample_rate::Float64, n_samples::Int,
                                   baseline_m::Float64, theta_deg::Float64, snr_db::Float64,
                                   seed::Union{Int,Nothing}=nothing)
    seed === nothing || Random.seed!(seed)

    tau = baseline_m * sind(theta_deg) / C_LIGHT

    # Unit-power baseband source signal. Astronomical sources are
    # noise-like at RF -- there's no coherent carrier to exploit the
    # way the TX-tone bench test does -- so this is complex white
    # Gaussian noise, not a tone.
    s = ComplexF32.(randn(ComplexF64, n_samples)) 

    # Effect 2: fractional-sample envelope delay via frequency-domain
    # phase ramp (Fourier shift theorem, exact for a bandlimited signal).
    S = fft(s)
    freqs = fftfreq(n_samples, sample_rate)
    envelope_delay_ramp = exp.(-2im * pi .* freqs .* tau)
    s_delayed = ifft(S .* envelope_delay_ramp)

    # Effect 1: the bulk carrier-phase rotation -- the actual fringe
    # phase, dominant at this baseline scale.
    carrier_phase = exp(-2im * pi * f_center * tau)
    s2 = ComplexF32.(s_delayed .* carrier_phase)

    # Independent receiver noise per channel, scaled to the requested
    # SNR relative to the source signal power. This is a convenience
    # dial for exercising the correlator at different noise levels,
    # not a physically-derived number -- see the still-open
    # radiometer-equation noise budget (To Do.md) for that.
    sig_power = mean(abs2, s)
    noise_power = sig_power / (10^(snr_db / 10))
    n1 = ComplexF32.(sqrt(noise_power) .* randn(ComplexF64, n_samples))
    n2 = ComplexF32.(sqrt(noise_power) .* randn(ComplexF64, n_samples))

    rx1 = s  .+ n1
    rx2 = s2 .+ n2

    V_true = ComplexF32(exp(-2im * pi * f_center * tau))

    return rx1, rx2, tau, V_true
end

"""
    correlate(rx1, rx2)

Normalized zero-lag complex cross-correlation -- the same quantity
already computed against real hardware captures (see
dual_channel_capture.jl's own sanity check and
dual_channel_radio_station_test.jl's zero-lag/lag-search version).
Reused here unmodified so this is a genuine validation of the actual
processing code, not a separate implementation that could hide a bug.
"""
function correlate(rx1::Vector{ComplexF32}, rx2::Vector{ComplexF32})
    return sum(rx1 .* conj.(rx2)) / sqrt(sum(abs2, rx1) * sum(abs2, rx2))
end

# =================================================================
# Validation run
# =================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    f_center    = 1420.0e6
    sample_rate = 2.7e6          # matches the measured operating rate -- see Hardware Setup.md
    n_samples   = round(Int, sample_rate * 1.0)   # 1s of data
    baseline_m  = 20.0
    theta_deg   = 15.0
    snr_db      = 10.0

    rx1, rx2, tau, V_true = simulate_point_source(; f_center, sample_rate, n_samples,
                                                     baseline_m, theta_deg, snr_db, seed=1)

    V_measured = correlate(rx1, rx2)

    # At finite SNR the zero-lag normalized correlation is a BIASED
    # (attenuated) estimate of true source coherence, because the
    # normalization divides by TOTAL power (signal+noise), not just
    # signal power: E[|V_measured|] -> SNR/(1+SNR) as n_samples grows,
    # not 1 -- a real effect real calibration has to correct for
    # (measuring/subtracting the system-noise contribution), not a
    # simulator bug. Printed here so the expected value is explicit
    # rather than mysteriously "not quite 1".
    snr_linear = 10^(snr_db / 10)
    expected_bias = snr_linear / (1 + snr_linear)

    println("Baseline: $(baseline_m) m, source at $(theta_deg) deg off broadside, SNR = $(snr_db) dB")
    println("Injected geometric delay: $(tau*1e9) ns")
    println()
    println("True visibility:      |V|=$(round(abs(V_true), digits=4)), phase=$(round(rad2deg(angle(V_true)), digits=2)) deg")
    println("Expected |V| at this SNR (noise-bias limited): $(round(expected_bias, digits=4))")
    println("Measured visibility:  |V|=$(round(abs(V_measured), digits=4)), phase=$(round(rad2deg(angle(V_measured)), digits=2)) deg")
    println()
    phase_error = rad2deg(angle(V_measured) - angle(V_true))
    println("Phase error vs. ground truth: $(round(phase_error, digits=2)) deg")
    println(abs(phase_error) < 5 ? "PASS -- correlator recovers the injected fringe phase within 5 deg." :
                                    "CHECK -- phase error larger than expected, investigate before trusting real captures.")
end
