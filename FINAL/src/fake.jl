# A stand-in for the Pluto with the same interface as Radio, for testing the programs without hardware.
# Model per channel, in ADC counts (full scale +-2048): late-stage noise (2.4 counts rms, independent of gain), front-end
# noise that scales with gain, a small LO leakage at DC, and one optional source common to both channels:
#   :tone   a tone at tone_bb Hz with an image at image_dbc, RX2 delayed by tau_inst and turned by phi_inst
#   :noise  band-limited noise (noise_bw_frac of fs), RX2 delayed by tau_inst and turned by phi_inst
#   :sky    full-band noise whose RX1-RX2 phase turns once per sky_period_s (a fringe)
# Levels are given at gain 0 dB in dBFS and scale with the gain; the result is clipped to 12 bits.

mutable struct FakeRadio
    fs::Float64
    lo::Float64
    g1::Float64
    g2::Float64
    bufsize::Int
    kernel_buffers::Int
    n::Int
    t0::Float64
    realtime::Bool
    rng::MersenneTwister
    src::Symbol
    tone_bb::Float64
    tone_dbfs0::Float64
    noise_dbfs0::Float64
    noise_bw_frac::Float64
    sky_snr_db::Float64
    sky_period_s::Float64
    sky_phase_deg::Float64
    tau_inst::Float64
    phi_inst_deg::Float64
    image_dbc::Float64
    late_counts::Float64
    fe_counts0::Float64
    dc_counts::ComplexF64
    lost_every::Int          # > 0: skip 7 samples before every lost_every-th block (tests the lost-sample detection)
    nblocks::Int
    track::NTuple{3,Int}     # quadrature, RF DC, BB DC tracking (stored only)
end

function FakeRadio(cfg::Config; realtime::Bool = true, seed::Int = 1)
    return FakeRadio(cfg.fs, cfg.f_lo, cfg.gain1, cfg.gain2, 0, cfg.kernel_buffers, 0, time(), realtime, MersenneTwister(seed),
                     :sky, 200e3, -75.0, -75.0, 0.6, cfg.fake_snr_db, cfg.fake_period_s, cfg.fake_phase_deg,
                     1.5e-9, 25.0, -38.0, 2.4, 0.04, ComplexF64(3.0, 1.5), 0, 0, (1, 1, 1))
end

function configure!(r::FakeRadio, cfg::Config)
    validate(cfg)
    r.fs = cfg.fs; r.lo = cfg.f_lo; r.g1 = cfg.gain1; r.g2 = cfg.gain2
    r.track = ntuple(i -> (v = (cfg.quad_track, cfg.rf_dc_track, cfg.bb_dc_track)[i]; v >= 0 ? v : r.track[i]), 3)
    return r
end
set_lo!(r::FakeRadio, f::Float64) = (r.lo = round(f); Int(r.lo))
set_gain!(r::FakeRadio, g1::Real, g2::Real) = (r.g1 = round(g1); r.g2 = round(g2); (r.g1, r.g2))
read_back(r::FakeRadio) = (fs1 = round(Int, r.fs), fs2 = round(Int, r.fs), bw1 = round(Int, r.fs), bw2 = round(Int, r.fs), lo = round(Int, r.lo),
                           gain1 = r.g1, gain2 = r.g2, mode1 = "manual", mode2 = "manual", port1 = "A_BALANCED", port2 = "A_BALANCED",
                           quad = (r.track[1], r.track[1]), rfdc = (r.track[2], r.track[2]), bbdc = (r.track[3], r.track[3]))
function start_stream!(r::FakeRadio, cfg::Config)
    _, r.bufsize = block_geometry(cfg)
    r.n = 0; r.t0 = time(); r.nblocks = 0
    return r
end
stop_stream!(r::FakeRadio) = r
close_radio!(r::FakeRadio) = nothing
flush!(r::FakeRadio) = (r.realtime && (r.n = round(Int, (time() - r.t0) * r.fs)); r)

function open_fake(cfg::Config; realtime::Bool = true, verbose::Bool = true)
    r = FakeRadio(cfg; realtime)
    rb = read_back(r)
    verbose && println("FAKE radio (no hardware): ", r.src, " source, LO ", r.lo / 1e6, " MHz, ", r.fs / 1e6, " Msps")
    start_stream!(r, cfg)
    return r, rb
end

amp_counts(dbfs, gain) = 2048 * 10^((dbfs + gain) / 20)

function delayed(s::Vector{ComplexF64}, fs, f_lo, tau, phi_deg)
    S = fft(s)
    return ifft(S .* cis.(-2pi .* (f_lo .+ fftfreq(length(s), fs)) .* tau)) .* cis(-deg2rad(phi_deg))
end

function read_block!(r::FakeRadio, idx::Int; qfull::Bool = false)
    n = r.bufsize; rng = r.rng
    r.nblocks += 1
    r.lost_every > 0 && r.nblocks % r.lost_every == 0 && (r.n += 7)
    t = (r.n .+ (0:n-1)) ./ r.fs
    a1 = 10^(r.g1 / 20); a2 = 10^(r.g2 / 20)
    x1 = r.late_counts .* randn(rng, ComplexF64, n) .+ (r.fe_counts0 * a1) .* randn(rng, ComplexF64, n) .+ r.dc_counts
    x2 = r.late_counts .* randn(rng, ComplexF64, n) .+ (r.fe_counts0 * a2) .* randn(rng, ComplexF64, n) .+ 0.8 * r.dc_counts
    if r.src == :tone
        A1 = amp_counts(r.tone_dbfs0, r.g1); A2 = 0.9 * amp_counts(r.tone_dbfs0, r.g2)
        ph = 2pi * (r.lo + r.tone_bb) * r.tau_inst + deg2rad(r.phi_inst_deg)
        gim = 10^(r.image_dbc / 20)
        x1 .+= A1 .* (cis.(2pi * r.tone_bb .* t .+ 0.3) .+ gim .* cis.(-2pi * r.tone_bb .* t .+ 1.1))
        x2 .+= A2 .* (cis.(2pi * r.tone_bb .* t .+ 0.3 .- ph) .+ gim .* cis.(-2pi * r.tone_bb .* t .- 0.4))
    elseif r.src == :noise
        z = randn(rng, ComplexF64, n)
        Z = fft(z); Z[abs.(fftfreq(n)) .> r.noise_bw_frac / 2] .= 0; s = ifft(Z)
        s ./= sqrt(mean(abs2, s))
        s2 = delayed(s, r.fs, r.lo, r.tau_inst, r.phi_inst_deg)
        x1 .+= amp_counts(r.noise_dbfs0, r.g1) .* s
        x2 .+= amp_counts(r.noise_dbfs0, r.g2) .* s2
    elseif r.src == :sky
        s = randn(rng, ComplexF64, n)
        p1 = r.late_counts^2 + (r.fe_counts0 * a1)^2; p2 = r.late_counts^2 + (r.fe_counts0 * a2)^2
        k = 10^(r.sky_snr_db / 20)
        turn = 2pi * (r.n / r.fs) / r.sky_period_s + deg2rad(r.sky_phase_deg)
        x1 .+= (k * sqrt(p1)) .* s
        x2 .+= (k * sqrt(p2) * cis(-turn)) .* s
    end
    r.n += n
    if r.realtime
        dt = r.t0 + r.n / r.fs - time()
        dt > 0 && sleep(dt)
    end
    q(v) = Int16.(clamp.(round.(v), -2047, 2047))
    return Block(idx, now(UTC), time_ns(), q(real.(x1)), q(imag.(x1)), q(real.(x2)), q(imag.(x2)), false, qfull)
end
