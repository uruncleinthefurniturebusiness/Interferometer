# Simulated observation (`sim` mode of pluto_interferometer.jl), moved here from Real/pluto_interferometer.jl. No hardware.
#
# Generates the two channels a pair of dishes would deliver for a chosen source as the Earth rotates, and runs them
# through the same FX correlator (consume!) as a capture, writing the same .vis records.
#   * the source is complex Gaussian noise, common to both channels with weight equal to the source's normalised
#     visibility; the rest of the source power and all receiver noise are independent per channel
#   * source-to-noise power ratio per channel is T_A/T_S, so the expected correlation is a2*v/(1+a2) before the delay
#     phase, a2 = T_A/T_S, v = |V_norm| * beam factor
#   * channel 2 sees the common part delayed by the geometric delay (carrier phase + fractional-sample envelope delay)
#   * convention as everywhere: V = x1*conj(x2); positive phase = the signal reaches RX2 later
# The baseline, site latitude and source diameter come from GeoCfg (baseline_m, baseline_az_deg, lat_deg, diam_arcmin).

Base.@kwdef mutable struct SimCfg
    sim_model::String          = "sun"     # point | disk | gaussian | sun (disk with optional limb brightening)
    sim_limb::Float64          = 0.0       # sun model: I(r) = 1 + limb*(r/a)^8
    sim_snr_db::Float64        = -20.0     # T_A/T_S per channel for the TOTAL source flux (dB)
    sim_flux_jy::Float64       = NaN       # physical override: with an area and sim_tsys_k, T_A/T_S = S*A_e/2k/T_S
    sim_area_m2::Float64       = NaN       # effective area A_e (m^2); or give sim_dish_m and sim_eff
    sim_tsys_k::Float64        = NaN
    sim_dec_deg::Float64       = 0.0       # source declination
    sim_start_ha_h::Float64    = -3.0      # starting hour angle (hours)
    sim_hours::Float64         = 6.0       # simulated duration
    sim_step_s::Float64        = 120.0     # simulated time between records (each record uses t_int seconds of data)
    sim_min_alt_deg::Float64   = 10.0      # skip records with the source below this altitude
    sim_hpbw_deg::Float64      = 0.0       # > 0: fixed dishes at the meridian with this primary-beam FWHM
    sim_dish_m::Float64        = NaN       # dish diameter: effective area (with sim_eff) and, for fixed dishes, the beamwidth
    sim_eff::Float64           = 0.55      # aperture efficiency
    sim_beam_k::Float64        = 1.15      # HPBW = k * lambda / D
    sim_track::Bool            = true      # true: dishes track the source (no beam loss); false: fixed at the meridian
    sim_phase_off_deg::Float64 = 0.0       # instrument phase added to the measured phase
    sim_gain2_db::Float64      = 0.0       # RX2 gain relative to RX1
    sim_dc::Float64            = 0.0       # shared DC offset amplitude, in units of the noise rms (LO leakage)
    sim_dc_phase_deg::Float64  = 0.0
    sim_line::Bool             = false     # add an HI-like spectral line
    sim_line_v_kms::Float64    = 0.0       # line velocity (radio convention, relative to rest)
    sim_line_fwhm_kms::Float64 = 10.0
    sim_line_amp::Float64      = 0.05      # peak line power relative to the receiver-noise PSD
    sim_line_vis::Float64      = 0.0       # normalised visibility of the line (0 = resolved out: autos only)
    sim_seed::Int              = 1
end

# Visibility (not normalised) of a uniform disk of angular radius a (rad) at projected baseline q (wavelengths).
disk_vis(a, q) = (x = 2pi * a * q; x < 1e-9 ? pi * a^2 : pi * a^2 * 2 * besselj1(x) / x)

# Normalised visibility of the chosen source model at projected baseline q (wavelengths). Real, can be negative.
function model_vis(sc::SimCfg, diam_arcmin::Float64, q::Float64)
    th = deg2rad(diam_arcmin / 60)                   # diameter (disk, sun) or FWHM (gaussian), rad
    m = sc.sim_model
    m == "point"    && return 1.0
    m == "disk"     && return disk_factor(diam_arcmin, q)
    m == "gaussian" && return exp(-(pi * th * q)^2 / (4 * log(2)))
    if m == "sun"                                    # stacked concentric disks
        a = th / 2; N = 64; num = 0.0; den = 0.0
        for i in 1:N
            a_in = a * (i - 1) / N; a_out = a * i / N
            I = 1.0 + sc.sim_limb * (((a_in + a_out) / 2) / a)^8
            num += I * (disk_vis(a_out, q) - disk_vis(a_in, q))
            den += I * pi * (a_out^2 - a_in^2)
        end
        return num / den
    end
    error("sim_model must be point, disk, gaussian or sun")
end

sim_line_center(sc::SimCfg, f_lo) = HI_REST * (1 - sc.sim_line_v_kms / (C_LIGHT / 1e3)) - f_lo
sim_line_width(sc::SimCfg) = sc.sim_line_fwhm_kms / (C_LIGHT / 1e3) * HI_REST

# Effective area A_e = eff*pi*D^2/4, primary-beam FWHM = k*lambda/D, and whether the dishes are fixed or tracking.
function sim_antenna(sc::SimCfg, f_lo::Float64)
    lam = C_LIGHT / f_lo
    area = isfinite(sc.sim_area_m2) ? sc.sim_area_m2 :
           (isfinite(sc.sim_dish_m) ? sc.sim_eff * pi * sc.sim_dish_m^2 / 4 : NaN)
    hpbw = sc.sim_hpbw_deg > 0 ? sc.sim_hpbw_deg :
           (isfinite(sc.sim_dish_m) ? rad2deg(sc.sim_beam_k * lam / sc.sim_dish_m) : 0.0)
    return (area = area, hpbw_deg = hpbw, fixed = !sc.sim_track || sc.sim_hpbw_deg > 0)
end

# One block of synthetic samples for a fixed geometry, in units of the receiver-noise rms (unit complex noise per channel).
function sim_block(rng, n::Int, sp, idx::Int, utc::DateTime)::Block
    cw() = randn(rng, ComplexF32, n)
    C  = (sp.a * sp.sv) .* cw()                      # source component common to both channels
    I1 = (sp.a * sp.si) .* cw()                      # source power that is not common (resolved structure)
    I2 = (sp.a * sp.si) .* cw()
    if sp.gline !== nothing                          # spectral line: coloured noise, common part weighted by its visibility
        colored(w) = ifft(fft(w) .* sp.gline)
        C  .+= sp.sqvl .* colored(cw())
        I1 .+= sp.sqnl .* colored(cw())
        I2 .+= sp.sqnl .* colored(cw())
    end
    x1 = C .+ I1 .+ cw()
    Cd = ifft(fft(C) .* sp.ramp) .* sp.carrier       # channel 2 sees the common component delayed by tau_g
    x2 = (Cd .+ I2 .+ cw()) .* sp.gain2
    if sp.dc != 0
        x1 .+= sp.dc1; x2 .+= sp.dc2
    end
    cl(v) = Int16.(clamp.(round.(v .* sp.scale), -2047, 2047))
    return Block(idx, utc, time_ns(), cl(real.(x1)), cl(imag.(x1)), cl(real.(x2)), cl(imag.(x2)), false, false)
end

# Expected band-average rho over the bins the measurement keeps (white continuum plus the optional line).
function expected_band_rho(cfg::Config, sc::SimCfg, a2, v, tau, theta, neg::Bool)
    n = cfg.nfft; fb = bb_freqs(n, cfg.fs); keep = band_keep(n, cfg.dc_mask)
    ell = sc.sim_line ? sc.sim_line_amp .* exp.(-4 * log(2) .* ((fb .- sim_line_center(sc, cfg.f_lo)) ./ sim_line_width(sc)) .^ 2) : zeros(n)
    ph = 2pi .* (cfg.f_lo .+ fb) .* tau .+ theta .+ (neg ? pi : 0.0)
    cross = (a2 * v .+ ell .* sc.sim_line_vis) .* cis.(ph)
    return sum(cross[keep]) / sum((1 + a2) .+ ell[keep])
end

# Run the simulated observation. on_result(rec, truth) runs on the processing thread for each record, with the
# expected values (hour angle, altitude, u, v, w, tau, visibility, beam, rho_exp). Returns the consistency check.
function simulate_observation(cfg::Config, geo::GeoCfg, sc::SimCfg; on_result = nothing, verbose::Bool = true)
    validate(cfg)
    isfinite(geo.baseline_m) && isfinite(geo.baseline_az_deg) || error("sim needs the baseline: baseline_m= and baseline_az_deg=")
    nsegb, bufsize = block_geometry(cfg)
    fs = cfg.fs
    per_int = max(1, round(Int, cfg.t_int * fs / bufsize))
    blk_secs = bufsize / fs
    rng = MersenneTwister(sc.sim_seed)
    ant = sim_antenna(sc, cfg.f_lo)
    ant.fixed && ant.hpbw_deg <= 0 && error("Fixed dishes need a beamwidth: set sim_dish_m or sim_hpbw_deg")
    physical = all(isfinite, (sc.sim_flux_jy, ant.area, sc.sim_tsys_k))
    a2 = physical ? sc.sim_flux_jy * 1e-26 * ant.area / (2 * 1.380649e-23) / sc.sim_tsys_k : 10^(sc.sim_snr_db / 10)
    theta = deg2rad(sc.sim_phase_off_deg)
    dec = deg2rad(sc.sim_dec_deg)
    nrec = floor(Int, sc.sim_hours * 3600 / sc.sim_step_s) + 1
    utc0 = now(UTC)

    w = cfg.window == "hann" ? 0.5 .- 0.5 .* cos.(2pi .* (0:cfg.nfft-1) ./ cfg.nfft) : ones(cfg.nfft)
    sigma_comp = sqrt(sum(w .^ 4)) / (sum(w .^ 2) * sqrt(2 * per_int * nsegb))     # noise per component of band-average rho
    geom(tsec) = (H = deg2rad(15 * sc.sim_start_ha_h) + OMEGA_E * tsec; g = baseline_uvw(geo, cfg.f_lo, H, dec); merge(g, (H = H,)))

    if verbose
        @printf("Simulated observation: model %s, %.1f arcmin, T_A/T_S = %.4g (%.1f dB), baseline %.1f m az %.0f deg, dec %.1f, lat %.1f\n",
                sc.sim_model, geo.diam_arcmin, a2, 10log10(a2), geo.baseline_m, geo.baseline_az_deg, sc.sim_dec_deg, geo.lat_deg)
        @printf("  %d record times over %.1f h (step %.0f s), %.2f s of data each, band-average rho noise per component about %.2e\n",
                nrec, sc.sim_hours, sc.sim_step_s, per_int * blk_secs, sigma_comp)
        println("  antenna: ", isfinite(ant.area) ? @sprintf("A_e = %.2f m^2 (%.0f Jy/K); ", ant.area, 2761.3 / ant.area) : "A_e not set; ",
                ant.fixed ? @sprintf("fixed dishes, HPBW %.2f deg", ant.hpbw_deg) : "tracking dishes (no beam loss)")
        g0 = geom(0.0); g1 = geom(1.0)
        @printf("  start: |u,v| = %.1f wavelengths, fringe rate about %.2f mHz (period %.0f s)\n",
                hypot(g0.u, g0.v), 1e3 * abs(g1.w - g0.w), 1 / max(abs(g1.w - g0.w), 1e-12))
    end

    f = fftfreq(bufsize, fs)
    gline = sc.sim_line ? Float32.(sqrt(sc.sim_line_amp) .* exp.(-2 * log(2) .* ((f .- sim_line_center(sc, cfg.f_lo)) ./ sim_line_width(sc)) .^ 2)) : nothing
    truths = NamedTuple[]; tlock = ReentrantLock(); irec = Ref(0)
    errs = Float64[]; errs_sig = Float64[]; ratios = Float64[]; snr_hi = Bool[]

    function on_record(rec::Record)
        irec[] += 1
        tr = lock(() -> truths[irec[]], tlock)
        rho = band_rho(rec, band_keep(cfg.nfft, cfg.dc_mask))
        err = wrap_pi(angle(rho) - angle(tr.rho_exp))
        if abs(tr.rho_exp) > sigma_comp              # the phase means something only when the expected signal exceeds the noise
            push!(errs, abs(err)); push!(errs_sig, abs(err) / (sigma_comp / abs(tr.rho_exp)))
            push!(ratios, abs(rho) / abs(tr.rho_exp)); push!(snr_hi, abs(tr.rho_exp) / sigma_comp > 6)
        end
        on_result === nothing || on_result(rec, tr, rho, err, sigma_comp / max(abs(tr.rho_exp), 1e-12))
    end

    q = Channel{Block}(cfg.queue_depth)
    task = Threads.@spawn consume!(q, cfg, fs, per_int, nothing, on_record)
    bind(q, task)
    nskipped = 0
    try
        for k in 0:nrec-1
            tsec = k * sc.sim_step_s
            g = geom(tsec)
            if g.alt < deg2rad(sc.sim_min_alt_deg)
                nskipped += 1; continue
            end
            vn = model_vis(sc, geo.diam_arcmin, hypot(g.u, g.v))
            beam = ant.fixed ? exp(-4 * log(2) * (g.H * cos(dec) / deg2rad(ant.hpbw_deg))^2) : 1.0
            v = abs(vn) * beam; neg = vn < 0
            rho_exp = expected_band_rho(cfg, sc, a2, v, g.tau, theta, neg)
            lock(tlock) do
                push!(truths, (H = g.H, alt = g.alt, u = g.u, v = g.v, w = g.w, tau = g.tau, vnorm = abs(vn), beam = beam, rho_exp = rho_exp))
            end
            sigma_t = sqrt(1 + a2 + sc.sim_dc^2)
            sp = (a = Float32(sqrt(a2)), sv = Float32(sqrt(v)), si = Float32(sqrt(1 - v)),
                  ramp = ComplexF32.(cis.(-2pi .* f .* g.tau)),
                  carrier = ComplexF32(cis(-(2pi * cfg.f_lo * g.tau + theta + (neg ? pi : 0.0)))),
                  gline = gline, sqvl = Float32(sqrt(sc.sim_line_vis)), sqnl = Float32(sqrt(1 - sc.sim_line_vis)),
                  gain2 = Float32(10^(sc.sim_gain2_db / 20)), dc = sc.sim_dc,
                  dc1 = ComplexF32(sc.sim_dc), dc2 = ComplexF32(sc.sim_dc * cis(deg2rad(sc.sim_dc_phase_deg))),
                  scale = Float32(250.0 / sqrt(sigma_t^2 / 2)))
            for j in 1:per_int
                put!(q, sim_block(rng, bufsize, sp, (j - 1) * bufsize, utc0 + Millisecond(round(Int, 1000 * (tsec + j * blk_secs)))))
                Threads.nthreads() == 1 && yield()
            end
        end
    finally
        close(q)
    end
    summary = fetch(task)

    isempty(errs) && return (records = summary.records, skipped = nskipped, a2 = a2, pass = false,
                             med_err_deg = NaN, med_err_sigma = NaN, med_ratio = NaN)
    hi = findall(snr_hi)
    med_ratio = isempty(hi) ? NaN : median(ratios[hi])
    med_sig = median(errs_sig)
    pass = med_sig < 1.2 && (isnan(med_ratio) || abs(med_ratio - 1) < 0.1)
    return (records = summary.records, skipped = nskipped, a2 = a2, pass = pass,
            med_err_deg = rad2deg(median(errs)), med_err_sigma = med_sig, med_ratio = med_ratio)
end
