# =====================================================================
# arc_test.jl -- over-the-air arc test for the two-channel Pluto interferometer
#
# A tone source (generator + antenna) is moved along an arc around two receive antennas on a rigid bar, connected to
# Pluto RX1 and RX2. The RX1-RX2 phase follows the path difference d2 - d1, so it rotates as the source moves: fringes
# against angle. Procedure, layout and pass criteria: vault note Design/Arc Test Procedure.md.
#
# Usage (from the Interferometer folder):
#   julia arc_test.jl predict  f_lo=2.45e9 B_m=0.25 R_m=3 angles=-60:3:60          plan: chords, phases, points per fringe
#   julia arc_test.jl capture  f_lo=2.45e9 gain1=20 gain2=20 B_m=0.25 R_m=3 angles=-60:3:60 tag=A      (hardware)
#   julia arc_test.jl fit      tag=A                                                 fit B, theta0, phi0; pass criteria
#   julia arc_test.jl selftest                                                       synthetic data through the fit
# Figures: julia arc_plot.jl tag=A. Any pluto_interferometer.jl option (f_lo, gain1, gain2, nfft, out_dir, addr, ...) is accepted.
#
# Geometry (as in the procedure note): bar midpoint at the origin, RX1 at (+B/2, 0), RX2 at (-B/2, 0), source at
# (R sin(theta), R cos(theta)); theta = 0 is broadside, positive theta is toward RX1. Exact distances, no far-field
# approximation:  d1,2 = sqrt(R^2 -+ R B sin(theta) + B^2/4),  phase = 360 (d2 - d1) / lambda + phi0  (degrees).
# Convention as in pluto_interferometer.jl: V = x1*conj(x2); positive phase = the signal reaches RX2 later (source on the RX1 side).
# =====================================================================

include(joinpath(@__DIR__, "..", "FINAL", "src", "PlutoIF.jl"))     # the radio, correlator and helpers shared with FINAL
using .PlutoIF: Config, C_LIGHT, wrap180, wrap_pi, unwrap_deg, circ_stats, parse_list, apply_kv!, open_verified, close_radio!,
                read_block!, block_geometry, FX, accumulate!, finalize!, detect_fullscale, to_complex, chan_stats, tone_fit, band_keep
using Dates, Printf, Statistics, Random, LinearAlgebra

Base.@kwdef mutable struct ArcCfg
    B_m::Float64            = NaN          # antenna spacing, centre to centre, measured with a tape (m)
    R_m::Float64            = NaN          # arc radius from the midpoint of the bar (m)
    angles::String          = "-60:3:60"   # source angles (deg): start:step:stop or a comma list; positive = toward RX1
    ref_every::Int          = 10           # broadside reference after every N positions (also at the start and the end)
    secs::Float64           = 5.0          # capture time per position (s)
    file::String            = ""           # results CSV (default <out_dir>/arc_results.csv)
    f_hz::Float64           = NaN          # predict: source frequency (Hz); default f_lo + 200 kHz
    warn_deg::Float64       = 45.0         # capture: flag a point further than this from the prediction (deg)
    theta0_max_deg::Float64 = 5.0          # fit: search range for the bar's pointing offset theta0 (deg)
    drift::Bool             = true         # fit: remove the instrument-phase drift by interpolating the broadside references
    pass_b_pct::Float64     = 5.0          # pass: fitted B within this % of the tape value (or 3 sigma, whichever is larger)
    pass_rms_deg::Float64   = 20.0         # pass: residual rms below this (deg)
    skip_blocks::Int        = 2            # capture: blocks discarded before each point
end

const ARC_USAGE = """
arc_test.jl modes: predict | capture | fit | selftest
  julia arc_test.jl predict  f_lo=2.45e9 B_m=0.25 R_m=3 angles=-60:3:60
  julia arc_test.jl capture  f_lo=2.45e9 gain1=20 gain2=20 B_m=0.25 R_m=3 angles=-60:3:60 tag=A [secs=5 ref_every=10]
  julia arc_test.jl fit      [tag=A file=... drift=true pass_b_pct=5 pass_rms_deg=20]
  julia arc_test.jl selftest
Set the generator to f_lo + 100 to 300 kHz (clear of the DC spike). Never 1400-1427 MHz: that band is protected."""

function parse_arc_cli(args::Vector{String})
    isempty(args) && (println(ARC_USAGE); exit(0))
    cfg = Config(quad_track = -1, rf_dc_track = -1, bb_dc_track = -1)      # tracking loops left as found, as in the original arc test
    ac = ArcCfg()
    apply_kv!(args[2:end], ac, cfg)
    return args[1], cfg, ac
end

# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
# Path difference d2 - d1 (m), exact, for a source at angle theta (deg) on an arc of radius R around antennas B apart.
function path_diff(theta_deg::Real, B::Real, R::Real)
    s = sind(theta_deg)
    return sqrt(R^2 + R * B * s + B^2 / 4) - sqrt(R^2 - R * B * s + B^2 / 4)
end
pred_phase_deg(theta_deg, B, R, lam) = 360 * path_diff(theta_deg, B, R) / lam
farfield_phase_deg(theta_deg, B, lam) = 360 * B * sind(theta_deg) / lam
chord_cm(theta_deg, R) = 200R * sind(abs(theta_deg) / 2)      # straight-line distance from the theta = 0 mark (cm)

function check_geometry(ac::ArcCfg)
    isfinite(ac.B_m) && ac.B_m > 0 || error("Give B_m= (antenna spacing, centre to centre, from a tape), e.g. B_m=0.25")
    isfinite(ac.R_m) && ac.R_m > ac.B_m || error("Give R_m= (arc radius from the bar midpoint, larger than B_m), e.g. R_m=3")
    return nothing
end

function check_band(f::Float64)
    1400e6 <= f <= 1427e6 && error("$(f / 1e6) MHz: the arc test transmits, and 1400-1427 MHz is a protected passive band " *
                                   "(ITU RR 5.340). Use an ISM frequency, e.g. f_lo=2.45e9 with the generator at 2.4502 GHz.")
    ism = (433.05e6 <= f <= 434.79e6) || (868e6 <= f <= 870e6) || (2400e6 <= f <= 2483.5e6)
    ism || @warn "$(f / 1e6) MHz is not in 433.05-434.79, 868-870 or 2400-2483.5 MHz. Check the ICASA limits before transmitting."
    return nothing
end

# ---------------------------------------------------------------------
# predict: plan the positions before marking them out
# ---------------------------------------------------------------------
function arc_predict(cfg::Config, ac::ArcCfg)
    check_geometry(ac)
    f = isfinite(ac.f_hz) ? ac.f_hz : cfg.f_lo + 200e3
    check_band(f)
    lam = C_LIGHT / f; B = ac.B_m; R = ac.R_m
    th = parse_list(ac.angles)
    nf = 2B^2 / lam
    slope = 360 * B / lam * pi / 180                     # deg of phase per deg of angle at broadside
    swing = (pred_phase_deg(maximum(th), B, R, lam) - pred_phase_deg(minimum(th), B, R, lam)) / 360
    @printf("Source %.4f MHz, wavelength %.2f cm; B = %.3f m = %.2f wavelengths; R = %.2f m\n", f / 1e6, 100lam, B, B / lam, R)
    @printf("Plane-wave distance 2B^2/lambda = %.2f m (R is %.1f times that)%s\n", nf, R / nf,
            R < nf ? "  <-- near field: allowed (the fit uses exact distances), but measure R carefully" : "")
    @printf("Fringes over %+.0f to %+.0f deg: %.1f. Phase change per degree of angle at broadside: %.1f deg\n",
            minimum(th), maximum(th), swing, slope)
    if length(th) > 1
        stp = minimum(abs.(diff(th)))
        ppf = 360 / (slope * stp)
        @printf("Step %.1f deg = %.1f cm of arc: %.1f points per fringe at broadside%s\n", stp, 100R * deg2rad(stp), ppf,
                ppf < 6 ? "  <-- too few, use a smaller step" : "")
        @printf("A 1 cm error in a mark is %.2f deg of angle, %.1f deg of phase at broadside\n", rad2deg(0.01 / R), slope * rad2deg(0.01 / R))
    end
    println("\n theta(deg)  chord from 0 mark(cm)  d2-d1(mm)  phase exact(deg)  wrapped  far-field(deg)  exact-farfield(deg)")
    for t in th
        pd = path_diff(t, B, R); pe = 360pd / lam; pf = farfield_phase_deg(t, B, lam)
        @printf(" %+9.1f  %20.1f  %9.2f  %16.1f  %7.1f  %14.1f  %19.1f\n", t, chord_cm(t, R), 1e3pd, pe, wrap180(pe), pf, pe - pf)
    end
    abs(f - cfg.f_lo) < 50e3 && @warn "The source is within 50 kHz of the LO: put it 100 to 300 kHz away (DC spike)."
    abs(f - cfg.f_lo) > 0.4cfg.fs && @warn "The source is outside the central 80% of the capture band."
    return nothing
end

# ---------------------------------------------------------------------
# capture: one point per angle, broadside references for the instrument-phase drift
# ---------------------------------------------------------------------
const ARC_HEADER = "utc,tag,kind,theta_deg,R_m,B_m,f_lo_hz,f_bb_hz,f_true_hz,phase_deg,phase_std_deg,rho,rho_expected," *
                   "rho_band,rho_band_phase_deg,rho_band_expected,snr1_db,snr2_db,gain1_db,gain2_db,blocks,clip"

arc_file(cfg::Config, ac::ArcCfg) = isempty(ac.file) ? joinpath(cfg.out_dir, "arc_results.csv") : ac.file

# The tone phase (RX1 - RX2) and the tone-window coherence at one position, plus the band-average coherence that the
# low-signal run compares with SNR/(1+SNR), and a flag for a tone fit that locked onto the DC spike.
function measure_point(r, cfg::Config, rb, secs::Float64; skip_blocks::Int = 2)
    nsegb, bufsize = block_geometry(cfg)
    fs = Float64(rb.fs1); flo = Float64(rb.lo)
    nblk = max(2, ceil(Int, secs * fs / bufsize))
    for _ in 1:skip_blocks; read_block!(r, 0); end
    fx = FX(cfg.nfft, nsegb, cfg.window)
    ph = Float64[]; fts = Float64[]; fsc = 0.0f0; clip = 0.0
    for _ in 1:nblk
        b = read_block!(r, 0)
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        clip = max(clip, s1.clip, s2.clip)
        f1, p1 = tone_fit(x1, fs); _, p2 = tone_fit(x2, fs)
        push!(fts, f1); push!(ph, rad2deg(wrap_pi(p1 - p2)))     # both phases at the same sample: frequency error cancels
        accumulate!(fx, x1, x2); fx.blocks += 1
        fx.pw1 += s1.power; fx.pw2 += s2.power
    end
    fx.t_first = fx.t_last = now(UTC)
    rec = finalize!(fx)
    fbb = median(fts)
    binw = fs / cfg.nfft
    kt = clamp(round(Int, fbb / binw) + cfg.nfft ÷ 2 + 1, 5, cfg.nfft - 4)
    win = (kt - 3):(kt + 3)
    band = band_keep(cfg.nfft, cfg.dc_mask)                 # band average: all but DC
    noise = copy(band); noise[win] .= false                 # noise: also without the tone
    P1 = Float64.(rec.P1); P2 = Float64.(rec.P2)
    n1 = 7 * median(P1[noise]); n2 = 7 * median(P2[noise])
    t1 = max(sum(P1[win]) - n1, 1e-30); t2 = max(sum(P2[win]) - n2, 1e-30)
    rho = sum(rec.V[win]) / sqrt(sum(P1[win]) * sum(P2[win]))
    B1 = sum(P1[band]); B2 = sum(P2[band])
    rho_band = sum(rec.V[band]) / sqrt(B1 * B2)
    cs = circ_stats(ph)
    return (f_bb = fbb, f_true = flo + fbb, f_lo = flo, phase = cs.mean, phase_std = cs.std,
            rho = abs(rho), rho_exp = sqrt(t1 * t2) / sqrt((t1 + n1) * (t2 + n2)),
            rho_band = abs(rho_band), rho_band_phase = rad2deg(angle(rho_band)), rho_band_exp = sqrt(t1 * t2) / sqrt(B1 * B2),
            snr1_db = 10log10(t1 / n1), snr2_db = 10log10(t2 / n2), nblk = nblk, clip = clip,
            near_dc = abs(fbb) < (cfg.dc_mask + 4) * binw)
end

function write_arc_row(file, utc, tag, kind, theta, R, B, g1, g2, m)
    open(file, "a") do io
        @printf(io, "%s,%s,%s,%.3f,%.4f,%.4f,%.0f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.2f,%.5f,%.2f,%.2f,%.1f,%.1f,%d,%.6f\n",
                string(utc), tag, kind, theta, R, B, m.f_lo, m.f_bb, m.f_true, m.phase, m.phase_std, m.rho, m.rho_exp,
                m.rho_band, m.rho_band_phase, m.rho_band_exp, m.snr1_db, m.snr2_db, g1, g2, m.nblk, m.clip)
    end
end

function take_point(r, cfg::Config, ac::ArcCfg, rb, file, kind, theta, ref)
    m = measure_point(r, cfg, rb, ac.secs; ac.skip_blocks)
    write_arc_row(file, now(UTC), cfg.tag, kind, theta, ac.R_m, ac.B_m, rb.gain1, rb.gain2, m)
    @printf("  %s theta %+.1f: phase %.1f deg (sd %.1f, %d blocks), tone %+.1f kHz, SNR %.1f / %.1f dB, |rho| %.4f (expected %.4f), band |rho| %.4f (expected %.4f)\n",
            kind, theta, m.phase, m.phase_std, m.nblk, m.f_bb / 1e3, m.snr1_db, m.snr2_db, m.rho, m.rho_exp, m.rho_band, m.rho_band_exp)
    if kind == "arc" && isfinite(ref)
        p = pred_phase_deg(theta, ac.B_m, ac.R_m, C_LIGHT / m.f_true)
        d = wrap180(m.phase - ref - p)
        @printf("  relative to the last reference %+.1f deg, predicted %+.1f deg (wrapped %+.1f), difference %+.1f deg%s\n",
                wrap180(m.phase - ref), p, wrap180(p), d, abs(d) > ac.warn_deg ? "   <-- check the mark, the source height, reflections (r repeats)" : "")
    end
    m.clip > 1e-3 && println("  WARNING: clipping. Lower gain1/gain2 or the generator level, or move the source back.")
    m.near_dc && println("  WARNING: the tone fit is at the DC spike. Put the generator 100 to 300 kHz away from f_lo.")
    min(m.snr1_db, m.snr2_db) < 10 && println("  WARNING: tone under 10 dB above the noise on a channel. Check the generator, antennas and frequency.")
    m.phase_std > 20 && println("  NOTE: phase scatter over 20 deg during the capture: something moved, or the tone is weak.")
    return m
end

function prompt_ref(which::String)
    @printf("\nBroadside reference (%s): put the source at theta = 0, step away, then Enter   [s = skip]: ", which)
    return strip(readline()) != "s"
end

function arc_capture(cfg::Config, ac::ArcCfg)
    check_geometry(ac); check_band(cfg.f_lo)
    angles = parse_list(ac.angles)
    isempty(cfg.tag) && (cfg.tag = Dates.format(now(), "mmdd-HHMM"))
    file = arc_file(cfg, ac)
    isdir(dirname(file)) || mkpath(dirname(file))
    isfile(file) || open(io -> println(io, ARC_HEADER), file, "w")
    nref = 2 + (ac.ref_every > 0 ? (length(angles) - 1) ÷ ac.ref_every : 0)
    @printf("\nArc test '%s': %d positions from %+.1f to %+.1f deg, R = %.2f m, B = %.3f m, %.0f s per point, %d broadside references\n",
            cfg.tag, length(angles), minimum(angles), maximum(angles), ac.R_m, ac.B_m, ac.secs, nref)
    @printf("About %.0f minutes if moving the source takes 30 s. Results appended to %s\n", (length(angles) + nref) * (ac.secs + 30) / 60, file)
    println("Generator at f_lo + 100 to 300 kHz, lowest output level. Keep people and metal out of the space between source and antennas.")
    r, rb = open_verified(cfg)
    last_ref = NaN; first_ref = NaN; since_ref = 0; npts = 0
    try
        if prompt_ref("start")
            m = take_point(r, cfg, ac, rb, file, "ref", 0.0, NaN)
            last_ref = first_ref = m.phase
        end
        stop = false
        for (k, th) in enumerate(angles)
            while true
                @printf("\nNext: theta = %+.1f deg, %.1f cm (straight line) from the theta = 0 mark. Enter = measure, a number = measure at that angle instead, s = skip, q = stop: ",
                        th, chord_cm(th, ac.R_m))
                a = strip(readline())
                a == "q" && (stop = true; break)
                a == "s" && break
                thm = isempty(a) ? th : tryparse(Float64, a)
                thm === nothing && (println("  Not understood."); continue)
                take_point(r, cfg, ac, rb, file, "arc", thm, last_ref)
                npts += 1; since_ref += 1
                print("  Enter = next position, r = repeat this one: ")
                if strip(readline()) == "r"
                    continue
                end
                break
            end
            stop && break
            if ac.ref_every > 0 && since_ref >= ac.ref_every && k < length(angles)
                if prompt_ref("drift check")
                    m = take_point(r, cfg, ac, rb, file, "ref", 0.0, NaN)
                    isfinite(last_ref) && @printf("  reference moved %+.1f deg since the previous reference\n", wrap180(m.phase - last_ref))
                    last_ref = m.phase
                    isfinite(first_ref) || (first_ref = m.phase)
                end
                since_ref = 0
            end
        end
        if prompt_ref("end")
            m = take_point(r, cfg, ac, rb, file, "ref", 0.0, NaN)
            isfinite(first_ref) && @printf("  reference moved %+.1f deg over the session\n", wrap180(m.phase - first_ref))
        end
    finally
        close_radio!(r)
    end
    @printf("\n%d positions measured. Next: julia arc_test.jl fit tag=%s%s\n", npts, cfg.tag, isempty(ac.file) ? "" : " file=$(ac.file)")
    return nothing
end

# ---------------------------------------------------------------------
# fit: exact-geometry model, drift removed with the broadside references
# ---------------------------------------------------------------------
function read_arc_rows(path::String)
    rows = NamedTuple[]
    for line in eachline(path)
        (startswith(line, "utc,") || isempty(strip(line))) && continue
        c = split(line, ',')
        length(c) >= 22 || continue
        p(i) = parse(Float64, c[i])
        push!(rows, (utc = DateTime(String(c[1])), tag = String(c[2]), kind = String(c[3]), theta = p(4), R = p(5), B = p(6),
                     f_true = p(9), phase = p(10), std = p(11), rho = p(12), rho_exp = p(13), rho_band = p(14),
                     rho_band_exp = p(16), snr1 = p(17), snr2 = p(18)))
    end
    return rows
end

function interp_clamped(tx::Vector{Float64}, px::Vector{Float64}, t::Float64)
    (length(tx) == 1 || t <= tx[1]) && return px[1]
    t >= tx[end] && return px[end]
    k = searchsortedlast(tx, t)
    return px[k] + (t - tx[k]) / (tx[k+1] - tx[k]) * (px[k+1] - px[k])
end

# Fit phase = phi0 + s * 360 (d2 - d1)(theta + theta0; B) / lambda with s = +1 (s = -1 for the sign check): minimise the
# circular residual over a grid in B and theta0 (phi0 is solved exactly at each grid point), refine, then get the
# uncertainties from the Jacobian.
function fit_arc(th::Vector{Float64}, ph::Vector{Float64}, R::Float64, lam::Float64, Bg::Float64, t0max::Float64; s::Float64 = 1.0)
    function score(B, t0)
        z = zero(ComplexF64)
        @inbounds for i in eachindex(th)
            z += cis(deg2rad(ph[i] - s * pred_phase_deg(th[i] + t0, B, R, lam)))
        end
        return 1 - abs(z) / length(th), rad2deg(angle(z))
    end
    nt = max(1, round(Int, 2t0max / 0.05)) + 1
    Bs = range(0.5Bg, 1.5Bg, length = 1001); ts = range(-t0max, t0max, length = nt)
    best = (Inf, Bg, 0.0)
    for B in Bs, t0 in ts
        c = score(B, t0)[1]
        c < best[1] && (best = (c, B, t0))
    end
    dB = step(Bs); dt = nt > 1 ? step(ts) : 0.0
    for _ in 1:3
        for B in range(best[2] - dB, best[2] + dB, length = 21), t0 in range(best[3] - dt, best[3] + dt, length = 21)
            c = score(B, t0)[1]
            c < best[1] && (best = (c, B, t0))
        end
        dB /= 10; dt /= 10
    end
    cost, B, t0 = best
    phi0 = score(B, t0)[2]
    model = phi0 .+ s .* pred_phase_deg.(th .+ t0, B, R, lam)
    res = wrap180.(ph .- model)
    n = length(th); hB = 1e-5; ht = 1e-3
    J = hcat(s .* (pred_phase_deg.(th .+ t0, B + hB, R, lam) .- pred_phase_deg.(th .+ t0, B - hB, R, lam)) ./ (2hB),
             s .* (pred_phase_deg.(th .+ t0 .+ ht, B, R, lam) .- pred_phase_deg.(th .+ t0 .- ht, B, R, lam)) ./ (2ht),
             ones(n))
    C = n > 3 ? (sum(abs2, res) / (n - 3)) .* inv(J' * J) : fill(NaN, 3, 3)
    return (B = B, theta0 = t0, phi0 = phi0, model = model, res = res, rms = sqrt(mean(abs2, res)), cost = cost,
            sB = sqrt(max(C[1, 1], 0.0)), st0 = sqrt(max(C[2, 2], 0.0)))
end

function analyze_arc(rows, ac::ArcCfg)
    arc = filter(r -> r.kind == "arc", rows)
    refs = sort(filter(r -> r.kind == "ref", rows), by = r -> r.utc)
    length(arc) >= 4 || error("Need at least 4 arc positions; found $(length(arc)).")
    t_start = minimum(r.utc for r in rows)
    tsec(r) = Dates.value(r.utc - t_start) / 1e3
    R = median([r.R for r in arc]); Bt = median([r.B for r in arc])
    f = median([r.f_true for r in arc]); lam = C_LIGHT / f
    th = [r.theta for r in arc]; ta = tsec.(arc); ph = [r.phase for r in arc]
    tr = Float64.(tsec.(refs)); pr = unwrap_deg([r.phase for r in refs])
    use_drift = ac.drift && !isempty(refs)
    ph_c = use_drift ? wrap180.(ph .- [interp_clamped(tr, pr, t) for t in ta]) : ph
    fp = fit_arc(th, ph_c, R, lam, Bt, ac.theta0_max_deg)
    fm = fit_arc(th, ph_c, R, lam, Bt, ac.theta0_max_deg; s = -1.0)
    return (fit = fp, flipped = fm, R = R, Bt = Bt, f = f, lam = lam, th = th, ta = ta, ph_c = ph_c, tr = tr, pr = pr,
            arc = arc, use_drift = use_drift)
end

function report_arc(a, ac::ArcCfg, tag::String)
    fp = a.fit; fm = a.flipped
    @printf("\nArc test '%s': %d positions, %d broadside references; source %.4f MHz (wavelength %.2f cm); R = %.3f m, B (tape) = %.4f m\n",
            tag, length(a.th), length(a.tr), a.f / 1e6, 100a.lam, a.R, a.Bt)
    if isempty(a.tr)
        println("No broadside references: the instrument phase is fitted as a constant and its drift cannot be removed.")
    else
        d = diff(a.pr)
        @printf("References: phase change over the session %+.1f deg in %.1f min", a.pr[end] - a.pr[1], (a.tr[end] - a.tr[1]) / 60)
        isempty(d) || @printf("; largest change between consecutive references %.1f deg", maximum(abs.(d)))
        println(a.use_drift ? " (drift removed by interpolating the references)" : " (drift NOT removed: drift=false)")
        any(abs.(d) .> 30) && println("  WARNING: the instrument phase moved more than 30 deg between references; take them more often (ref_every=).")
    end
    dB = fp.B - a.Bt
    @printf("Fit: B = %.4f +- %.4f m (tape %.4f m, difference %+.1f mm = %+.1f%%); bar offset theta0 = %+.2f +- %.2f deg; phi0 = %.1f deg\n",
            fp.B, fp.sB, a.Bt, 1e3dB, 100dB / a.Bt, fp.theta0, fp.st0, fp.phi0)
    k = argmax(abs.(fp.res))
    @printf("Residuals: rms %.1f deg, largest %+.1f deg at theta %+.1f deg; fringes over the arc %.1f\n",
            fp.rms, fp.res[k], a.th[k], (maximum(fp.model) - minimum(fp.model)) / 360)
    sign_ok = fp.cost <= fm.cost
    @printf("Sign: fit quality %.3f with the V1 convention, %.3f with the opposite sign: %s\n", 1 - fp.cost, 1 - fm.cost,
            sign_ok ? "correct (positive phase with the source on the RX1 side)" : "REVERSED: RX1 and RX2 swapped, or positive angles marked toward RX2")
    @printf("Coherence at the tone: median |rho| %.4f (expected from the SNRs %.4f)\n",
            median([r.rho for r in a.arc]), median([r.rho_exp for r in a.arc]))
    lo = filter(i -> a.arc[i].rho_band_exp < 0.8, eachindex(a.arc))
    if !isempty(lo)
        @printf("Low-signal points (%d): band-average |rho| against SNR/(1+SNR)\n", length(lo))
        for i in lo
            r = a.arc[i]
            @printf("  theta %+6.1f  measured %.4f  expected %.4f  difference %+.4f\n", r.theta, r.rho_band, r.rho_band_exp, r.rho_band - r.rho_band_exp)
        end
    end
    passB = abs(dB) <= max(3fp.sB, ac.pass_b_pct / 100 * a.Bt)
    passR = fp.rms <= ac.pass_rms_deg
    println("\nPass criteria (fix them in the Arc Test Procedure before taking data):")
    @printf("  B within max(3 sigma, %.1f%%) of the tape value: %s\n", ac.pass_b_pct, passB ? "PASS" : "FAIL")
    @printf("  residual rms under %.0f deg: %s\n", ac.pass_rms_deg, passR ? "PASS" : "FAIL")
    @printf("  sign: %s\n", sign_ok ? "PASS" : "FAIL")
    println("\n theta(deg)  t(min)  phase-ref(deg)  model(deg)  residual(deg)   |rho|  band|rho|  expected")
    for i in eachindex(a.th)
        r = a.arc[i]
        @printf(" %+9.1f  %6.1f  %14.1f  %10.1f  %13.1f  %6.4f  %9.4f  %8.4f\n",
                a.th[i], a.ta[i] / 60, a.ph_c[i], wrap180(fp.model[i]), fp.res[i], r.rho, r.rho_band, r.rho_band_exp)
    end
    return (passB = passB, passR = passR, sign_ok = sign_ok)
end

function write_fit_files(a, tag::String, file::String, verdict)
    base = splitext(file)[1] * "_" * tag
    fp = a.fit
    open(base * "_fit_points.csv", "w") do io
        println(io, "kind,theta_deg,t_min,phase_corr_deg,model_deg,meas_unwrapped_deg,farfield_deg,residual_deg,rho,rho_exp,rho_band,rho_band_exp,snr1_db,snr2_db")
        for i in eachindex(a.th)
            r = a.arc[i]
            ff = fp.phi0 + farfield_phase_deg(a.th[i] + fp.theta0, fp.B, a.lam)
            @printf(io, "arc,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f\n", a.th[i], a.ta[i] / 60, a.ph_c[i],
                    fp.model[i], fp.model[i] + fp.res[i], ff, fp.res[i], r.rho, r.rho_exp, r.rho_band, r.rho_band_exp, r.snr1, r.snr2)
        end
        for j in eachindex(a.tr)
            @printf(io, "ref,0,%.4f,%.3f,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n", a.tr[j] / 60, a.pr[j] - a.pr[1])
        end
    end
    open(base * "_fit_params.csv", "w") do io
        println(io, "key,value")
        for (k, v) in (("tag", tag), ("n_arc", length(a.th)), ("n_ref", length(a.tr)), ("f_hz", a.f), ("lambda_m", a.lam),
                       ("R_m", a.R), ("B_tape_m", a.Bt), ("B_fit_m", fp.B), ("B_sigma_m", fp.sB), ("theta0_deg", fp.theta0),
                       ("theta0_sigma_deg", fp.st0), ("phi0_deg", fp.phi0), ("rms_deg", fp.rms), ("drift_removed", a.use_drift),
                       ("pass_B", verdict.passB), ("pass_rms", verdict.passR), ("sign_ok", verdict.sign_ok))
            println(io, k, ",", v)
        end
    end
    println("\nWrote ", base, "_fit_points.csv and ", base, "_fit_params.csv\nFigures: julia arc_plot.jl path=", base, "_fit_points.csv")
    return base
end

function arc_fit(cfg::Config, ac::ArcCfg)
    file = arc_file(cfg, ac)
    isfile(file) || error("No results file at $file (pass file=...)")
    rows = read_arc_rows(file)
    isempty(rows) && error("No rows in $file")
    tag = isempty(cfg.tag) ? rows[end].tag : cfg.tag
    rows = filter(r -> r.tag == tag, rows)
    isempty(rows) && error("No rows with tag '$tag' in $file")
    a = analyze_arc(rows, ac)
    v = report_arc(a, ac, tag)
    write_fit_files(a, tag, file, v)
    return nothing
end

# ---------------------------------------------------------------------
# selftest: synthetic arc data with a known answer through the same CSV, fit and report
# ---------------------------------------------------------------------
function arc_selftest(ac_in::ArcCfg)
    println("arc_test.jl selftest: synthetic arc data (known B, bar offset, instrument phase, drift, noise) through the fit")
    rng = MersenneTwister(11)
    Btrue = 0.250; Btape = 0.252; R = 3.0; f = 2.4502e9; lam = C_LIGHT / f
    t0 = 1.3; phi0 = 37.0; noise = 5.0
    drift(t) = 12.0 * t / 600 + 8.0 * sin(2pi * t / 1800)          # deg: a slow ramp plus a wobble
    file = isempty(ac_in.file) ? joinpath(mktempdir(), "arc_results.csv") : ac_in.file    # file= keeps the output for arc_plot.jl
    open(io -> println(io, ARC_HEADER), file, "w")
    t_start = DateTime(2026, 10, 11, 10, 0, 0)
    for (tag, s) in (("selftest", 1.0), ("selftest_flipped", -1.0))
        t = 0.0
        function add(kind, th)
            phase = s * (phi0 + drift(t) + pred_phase_deg(th + t0, Btrue, R, lam)) + noise * randn(rng)
            m = (f_lo = 2.45e9, f_bb = f - 2.45e9, f_true = f, phase = wrap180(phase), phase_std = 2.0, rho = 0.999, rho_exp = 0.999,
                 rho_band = 0.5, rho_band_phase = 0.0, rho_band_exp = 0.5, snr1_db = 30.0, snr2_db = 30.0, nblk = 50, clip = 0.0)
            write_arc_row(file, t_start + Millisecond(round(Int, 1000t)), tag, kind, th, R, Btape, 20, 20, m)
            t += 35.0
        end
        add("ref", 0.0)
        for (i, th) in enumerate(-60.0:3.0:60.0)
            add("arc", th)
            i % 10 == 0 && add("ref", 0.0)
        end
        add("ref", 0.0)
    end
    rows = read_arc_rows(file)
    ac = ArcCfg(B_m = Btape, R_m = R)
    a = analyze_arc(filter(r -> r.tag == "selftest", rows), ac)
    v = report_arc(a, ac, "selftest")
    write_fit_files(a, "selftest", file, v)
    b = analyze_arc(filter(r -> r.tag == "selftest_flipped", rows), ac)
    okB = abs(a.fit.B - Btrue) < 0.004
    okT = abs(a.fit.theta0 - t0) < max(0.5, 3a.fit.st0)
    okR = 0.5noise < a.fit.rms < 2noise
    okS = v.sign_ok
    okF = b.flipped.cost < b.fit.cost
    @printf("\nSelftest: B %.4f m (true %.4f, tape %.4f) %s; theta0 %+.2f deg (true %+.2f) %s; rms %.1f deg (noise %.1f) %s; sign %s; reversed data flagged %s\n",
            a.fit.B, Btrue, Btape, okB ? "ok" : "WRONG", a.fit.theta0, t0, okT ? "ok" : "WRONG", a.fit.rms, noise, okR ? "ok" : "WRONG",
            okS ? "ok" : "WRONG", okF ? "ok" : "NO")
    pass = okB && okT && okR && okS && okF
    println(pass ? "SELFTEST PASS" : "SELFTEST FAIL")
    return pass
end

# ---------------------------------------------------------------------
function arc_main(args::Vector{String})
    mode, cfg, ac = parse_arc_cli(args)
    mode == "predict"  ? arc_predict(cfg, ac) :
    mode == "capture"  ? arc_capture(cfg, ac) :
    mode == "fit"      ? arc_fit(cfg, ac) :
    mode == "selftest" ? (arc_selftest(ac) || exit(1)) :
    error("Unknown mode '$mode'.\n" * ARC_USAGE)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    arc_main(ARGS)
end
