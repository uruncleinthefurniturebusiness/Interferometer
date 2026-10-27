# =====================================================================
# sky_test.jl (FINAL) -- analysis of an over-the-air "sky" test: the Pluto+ radiates broadband noise as an artificial
# radio source and pluto_interferometer.jl records it exactly as it would record the sky.
#
#     julia sky_test.jl off=<off.vis> static=<static.vis> [transit=<transit.vis>] [B_m=0.69 out=<folder>]
#
# Three recordings with `pluto_interferometer.jl run` (same LO and gains):
#   off      source off: the receiver noise alone (sets the noise bias and the predicted coherence)
#   static   noise source fixed at broadside: visibility spectrum, coherence against the SNR prediction, instrumental
#            delay and phase, phase stability against the thermal-noise prediction
#   transit  noise source carried slowly along an arc across the baseline: the fringes a drifting sky source makes,
#            the phase turning and the delay changing with angle
# Writes sky_*.pdf/png and sky_summary.txt next to the static file (or in out=).
# Convention as the correlator: V = x1 conj(x2); positive phase = the signal reaches RX2 later (source on RX1 side).
# =====================================================================

include(joinpath(@__DIR__, "src", "PlutoIF.jl"))
const P = PlutoIF
using Printf, Statistics, Dates, Plots

Base.@kwdef mutable struct SkyCfg
    off::String = ""
    static::String = ""
    transit::String = ""
    out::String = ""
    B_m::Float64 = 0.69               # baseline, centre to centre (m)
    bw_frac::Float64 = 0.6            # tx_source.jl bw_frac of the noise (sets the band analysed)
    edge::Float64 = 0.9               # use this fraction of the noise band
    tmax_ns::Float64 = 300.0          # delay search range
end

const C_LIGHT = P.C_LIGHT

load(path) = (v = P.load_vis(path); merge(v, (gain = (v.meta["gain1_db"], v.meta["gain2_db"]),)))

avg(recs, f) = mean(Float64.(getfield(r, f)) for r in recs)
avgV(recs) = mean(ComplexF64.(r.V) for r in recs)
tsec(recs) = [r.t_start - recs[1].t_start for r in recs]

# Band-average visibility of one record with a delay model removed.
band_vis(r, band, model) = sum(ComplexF64.(r.V[band]) .* conj.(model[band])) / sqrt(sum(Float64, r.P1[band]) * sum(Float64, r.P2[band]))

function main(args)
    sc = SkyCfg(); P.apply_kv!(args, sc)
    (isempty(sc.off) || isempty(sc.static)) && error("Give off=<file>.vis and static=<file>.vis")
    out = isempty(sc.out) ? dirname(sc.static) : sc.out; mkpath(out)
    gr(); default(fontfamily = "Computer Modern", framestyle = :box, grid = true, titlefontsize = 10, guidefontsize = 9,
                  legendfontsize = 7, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, markerstrokewidth = 0)
    lines = String[]
    say(s) = (println(s); push!(lines, s))
    sv(p, name) = (savefig(p, joinpath(out, name * ".pdf")); savefig(p, joinpath(out, name * ".png")); println("Wrote ", joinpath(out, name * ".pdf")))

    off = load(sc.off); st = load(sc.static)
    (off.nfft == st.nfft && off.fs == st.fs && off.f_lo == st.f_lo && off.gain == st.gain) ||
        @warn "off and static runs differ in nfft, rate, LO or gain: the noise subtraction assumes they match"
    n = st.nfft; fs = st.fs; f_rf = st.f_lo; lam = C_LIGHT / f_rf
    fbb = P.bb_freqs(n, fs); keep = P.band_keep(n, st.dc_mask)
    band = [k for k in 1:n if keep[k] && abs(fbb[k]) <= sc.edge * sc.bw_frac * fs / 2]
    nseg = median(Float64(r.nseg) for r in st.recs)
    say("Sky test, $(Dates.format(now(), "yyyy-mm-dd HH:MM")): LO $(f_rf / 1e6) MHz (lambda $(round(lam, digits = 4)) m), $(fs / 1e6) Msps, gains $(st.gain), B = $(sc.B_m) m ($(round(sc.B_m / lam, digits = 2)) lambda)")
    say("Band: $(length(band)) bins of $(round(fs / n, digits = 1)) Hz over +-$(round(sc.edge * sc.bw_frac * fs / 2e3)) kHz, DC +-$(st.dc_mask) bins removed")
    say("Records: off $(length(off.recs)), static $(length(st.recs)) ($(round(nseg)) FFT segments each)")

    # ---------------- static: spectrum, coherence, delay
    N1 = avg(off.recs, :P1); N2 = avg(off.recs, :P2)
    P1 = avg(st.recs, :P1); P2 = avg(st.recs, :P2); V = avgV(st.recs)
    S1 = P1 .- N1; S2 = P2 .- N2
    rho = V ./ sqrt.(P1 .* P2)
    pred = sqrt.(max.(S1, 0) .* max.(S2, 0)) ./ sqrt.(P1 .* P2)
    snr1 = median(10log10.(max.(S1[band], 1e-30) ./ N1[band])); snr2 = median(10log10.(max.(S2[band], 1e-30) ./ N2[band]))
    d = P.delay_search(fbb[band], rho[band], abs2.(rho[band]); tmax_ns = sc.tmax_ns)
    model = cis.(deg2rad(d.phi0_deg) .+ 2pi .* fbb .* d.tau_ns .* 1e-9)
    eta_bin = median(abs.(rho[band]) ./ pred[band])
    vb = band_vis((V = V, P1 = P1, P2 = P2), band, model)
    pb = sum(sqrt.(max.(S1[band], 0) .* max.(S2[band], 0))) / sqrt(sum(P1[band]) * sum(P2[band]))
    grp = 64; ng = length(band) ÷ grp
    gph = [rad2deg(angle(sum(rho[band[(j-1)*grp+1:j*grp]] .* conj.(model[band[(j-1)*grp+1:j*grp]])))) for j in 1:ng]
    say("")
    say("STATIC (source at broadside)")
    say(@sprintf("  SNR per bin (median)                 %.1f / %.1f dB (RX1 / RX2)", snr1, snr2))
    say(@sprintf("  Band visibility |rho|                %.4f, predicted from the SNR %.4f  -> efficiency %.3f", abs(vb), pb, abs(vb) / pb))
    say(@sprintf("  Per-bin efficiency (median)          %.3f", eta_bin))
    say(@sprintf("  Instrumental delay                   %.2f +- %.2f ns (%.4f samples; geometry at broadside ~0)", d.tau_ns, d.tau_err_ns, d.tau_ns * 1e-9 * fs))
    say(@sprintf("  Instrumental phase at the LO         %.1f deg", d.phi0_deg))
    say(@sprintf("  Bandpass phase ripple                %.2f deg rms (%d-bin groups)", sqrt(mean(abs2, gph)), grp))

    # per-record time series
    t = tsec(st.recs)
    vr = [band_vis(r, band, model) for r in st.recs]
    ph = P.unwrap_deg(rad2deg.(angle.(vr)))
    fl = P.fit_line(t ./ 60, ph)
    neff = length(band) * nseg                      # independent samples per record in the band
    amp = mean(abs.(vr))
    sig_th = rad2deg(sqrt((1 - amp^2) / (2neff)) / amp)   # thermal scatter of the phase of a band average
    say(@sprintf("  |rho| per record                     mean %.4f, rms scatter %.4f", amp, std(abs.(vr))))
    say(@sprintf("  Phase drift                          %.3f deg/min", fl.b))
    say(@sprintf("  Phase scatter (drift removed)        %.3f deg rms; thermal-noise prediction %.3f deg", std(fl.res), sig_th))
    lags = unique(round.(Int, exp.(range(0, log(max(2, length(t) ÷ 2)), length = 12))))
    sf = [(l * median(diff(t)), P.phase_change_rms(ph, l)) for l in lags if l < length(ph)]
    for (lg, v) in sf[[1, cld(length(sf), 2), end]]
        say(@sprintf("  Phase change over %5.0f s             %.3f deg rms", lg, v))
    end

    fk = fbb[band] ./ 1e3
    p1 = plot(xlabel = "Baseband frequency (kHz)", ylabel = "|ρ|", title = "Visibility amplitude (source at broadside)", legend = :bottom)
    plot!(p1, fk, pred[band], lw = 2, label = "predicted from the autos, SNR/(1+SNR)")
    scatter!(p1, fk, abs.(rho[band]), ms = 1.5, label = "measured")
    p2 = plot(xlabel = "Baseband frequency (kHz)", ylabel = "Phase (deg)", title = @sprintf("Visibility phase: delay %.2f ns", d.tau_ns), legend = :topleft)
    scatter!(p2, fk, rad2deg.(angle.(rho[band])), ms = 1.5, label = "measured")
    plot!(p2, fk, rad2deg.(angle.(model[band])), lw = 2, label = "delay + phase fit")
    p3 = plot(fbb ./ 1e3, [10log10.(P1 ./ n) 10log10.(P2 ./ n) 10log10.(N1 ./ n)],          # dBFS per bin, as P.dbfs(P / nfft)
              label = ["RX1 source on" "RX2 source on" "RX1 source off"], xlabel = "Baseband frequency (kHz)", ylabel = "dBFS per bin", title = "Auto spectra", legend = :bottom)
    sv(plot(p3, p1, p2, layout = (1, 3), size = (1300, 380)), "sky_static_spectrum")

    q1 = plot(t ./ 60, abs.(vr), xlabel = "Time (min)", ylabel = "|ρ|", label = "", title = "Band visibility amplitude")
    hline!(q1, [pb], ls = :dash, label = "predicted")
    q2 = plot(t ./ 60, ph, xlabel = "Time (min)", ylabel = "Phase (deg)", label = "", title = "Band visibility phase (delay removed)")
    q3 = plot(first.(sf), last.(sf), xscale = :log10, marker = :circle, xlabel = "Lag (s)", ylabel = "deg rms", label = "measured",
              title = "Phase change against time lag")
    hline!(q3, [sig_th], ls = :dash, label = "thermal noise")
    sv(plot(q1, q2, q3, layout = (1, 3), size = (1300, 360)), "sky_static_time")

    # ---------------- transit: fringes and delay tracking
    if !isempty(sc.transit)
        tr = load(sc.transit)
        tt = tsec(tr.recs)
        inst = cis.(2pi .* fbb .* d.tau_ns .* 1e-9)                      # remove the instrumental delay, keep the phase
        vt = [band_vis(r, band, inst) for r in tr.recs]
        on = [sum(Float64, r.P1[band]) > 2sum(N1[band]) && sum(Float64, r.P2[band]) > 2sum(N2[band]) for r in tr.recs]   # source on (>3 dB rise)
        pht = P.unwrap_deg(rad2deg.(angle.(vt)))
        geo_ph = pht .- d.phi0_deg                                       # geometric phase = measured - instrumental
        sin_ph = clamp.(geo_ph ./ 360 .* lam ./ sc.B_m, -1, 1)
        taus = Float64[]
        for r in tr.recs
            rr = ComplexF64.(r.V[band]) ./ sqrt.(Float64.(r.P1[band]) .* Float64.(r.P2[band]))
            push!(taus, P.delay_search(fbb[band], rr, abs2.(rr); tmax_ns = sc.tmax_ns).tau_ns - d.tau_ns)
        end
        sin_tau = clamp.(C_LIGHT .* taus .* 1e-9 ./ sc.B_m, -1, 1)
        sel = findall(on)
        turns = isempty(sel) ? NaN : (maximum(geo_ph[sel]) - minimum(geo_ph[sel])) / 360
        say("")
        say("TRANSIT (source carried along the arc)")
        say("  Records $(length(tr.recs)), with the source on $(length(sel))")
        if !isempty(sel)
            k = sel
            say(@sprintf("  Fringe phase range                   %.0f deg = %.2f turns", 360turns, turns))
            say(@sprintf("  Angle from the phase                 %.1f to %.1f deg (broadside = the static phase)", asind(minimum(sin_ph[k])), asind(maximum(sin_ph[k]))))
            say(@sprintf("  Geometric delay from the spectrum    %.2f to %.2f ns (max possible B/c = %.2f ns)", minimum(taus[k]), maximum(taus[k]), 1e9sc.B_m / C_LIGHT))
            fl2 = P.fit_line(sin_ph[k], sin_tau[k])
            say(@sprintf("  sin(angle) from delay vs from phase  slope %.3f, rms difference %.3f", fl2.b, std(sin_tau[k] .- sin_ph[k])))
            say(@sprintf("  |rho| during the transit             median %.4f (static %.4f)", median(abs.(vt[k])), amp))
        end
        r1 = plot(tt ./ 60, real.(vt), label = "Re ρ", xlabel = "Time (min)", ylabel = "ρ (band average)", title = "Fringes as the source crosses the baseline")
        plot!(r1, tt ./ 60, imag.(vt), label = "Im ρ"); plot!(r1, tt ./ 60, abs.(vt), lw = 2, color = :black, label = "|ρ|")
        r2 = plot(tt ./ 60, geo_ph, label = "", xlabel = "Time (min)", ylabel = "Geometric phase (deg)", title = "Fringe phase (unwrapped, instrument removed)")
        r3 = plot(tt[sel] ./ 60, asind.(sin_ph[sel]), label = "from the fringe phase", xlabel = "Time (min)", ylabel = "Source angle (deg)",
                  title = "Source angle: phase against delay")
        scatter!(r3, tt[sel] ./ 60, asind.(sin_tau[sel]), ms = 2, label = "from the delay (unambiguous)")
        sv(plot(r1, r2, r3, layout = (1, 3), size = (1300, 360)), "sky_transit")
    end
    open(joinpath(out, "sky_summary.txt"), "w") do io
        println(io, "off = ", sc.off, "\nstatic = ", sc.static, "\ntransit = ", sc.transit, "\n")
        foreach(l -> println(io, l), lines)
    end
    println("Wrote ", joinpath(out, "sky_summary.txt"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
