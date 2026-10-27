# =====================================================================
# at03_source.jl (FINAL) -- AT-03 source go/no-go for the arc test: the Pluto+ tone on both channels at gain 20.
#
#     julia -t 3 at03_source.jl on  addr=usb:X.Y.Z [secs=60 out=...]    tone on: SNR, clipping, Pluto+ image, frequency, steadiness
#     julia -t 3 at03_source.jl off addr=usb:X.Y.Z [out=...]            tone off: nothing left at the tone frequency
#
# Pluto+ as for the arc test: julia tx_source.jl tone freq=869.2e6 offset=300e3 atten=0 (receiver LO 869.0 MHz).
# Uses stress_test.jl's tone measurement. Results go to <out>/at03_on.csv, at03_off.csv and at03_summary.txt.
# =====================================================================

include(joinpath(@__DIR__, "stress_test.jl"))
using Dates, Printf, Statistics

Base.@kwdef mutable struct At03Cfg
    secs::Float64 = 60.0
    out::String = "C:/PlutoCaptures/AT03"
end

# One 1 s measurement: the tone (stress_test.jl's tone_metrics) and the Pluto+ image at TX LO - offset.
function measure(r, c, sc, fs, keep, n)
    cp = P.capture(r, c, fs, P.blocks_for(c, fs, 1.0); tone = true)
    m = tone_metrics(cp.rec, cp.tones, fs, sc, keep)
    rec = cp.rec
    w = P.tone_bins(n, fs, source_lines(sc, m.fb)[4])
    nb1 = median(rec.P1[keep]); nb2 = median(rec.P2[keep])
    img1 = P.dbfs(P.bins_power(rec.P1, w, nb1)) - m.t1; img2 = P.dbfs(P.bins_power(rec.P2, w, nb2)) - m.t2
    below = sum(rec.P1[w]) < 2length(w) * nb1 && sum(rec.P2[w]) < 2length(w) * nb2
    return (m = m, img1 = img1, img2 = img2, below = below)
end

function main(args)
    mode = isempty(args) ? "" : args[1]
    mode in ("on", "off") || error("Usage: julia -t 3 at03_source.jl on|off addr=usb:X.Y.Z")
    c = P.Config(); sc = StressCfg(); ac = At03Cfg()
    P.apply_kv!(args[2:end], c, ac)
    c.f_lo = sc.f_tx_lo; c.gain1 = c.gain2 = 20
    mkpath(ac.out)
    r, rb = P.open_verified(c; verbose = false)
    fs = Float64(rb.fs1); n = c.nfft; keep = P.band_keep(n, c.dc_mask); fbb = P.bb_freqs(n, fs)
    out = String[]
    say(s) = (println(s); push!(out, s))
    try
        if mode == "on"
            rows = Any[]
            for i in 1:round(Int, ac.secs)
                x = measure(r, c, sc, fs, keep, n); m = x.m
                push!(rows, (i, rb.lo + m.fb, m.snr1, m.snr2, m.clip1, m.clip2, x.img1, x.img2, x.below ? 1 : 0, m.rho))
                @printf("  %2d s: tone at %.4f MHz, SNR %.1f / %.1f dB, clip %.1e / %.1e, Pluto+ image %.1f / %.1f dBc%s, |rho| %.4f\n",
                        i, (rb.lo + m.fb) / 1e6, m.snr1, m.snr2, m.clip1, m.clip2, x.img1, x.img2, x.below ? " (below noise)" : "", m.rho)
            end
            csv_write(joinpath(ac.out, "at03_on.csv"), ["sec", "f_tone_hz", "snr1_db", "snr2_db", "clip1", "clip2", "image1_dbc", "image2_dbc",
                      "image_below_noise", "rho"], rows)
            f = [x[2] for x in rows]; snr = minimum(min(x[3], x[4]) for x in rows); clip = maximum(max(x[5], x[6]) for x in rows)
            img = maximum(max(x[7], x[8]) for x in rows); below = all(x -> x[9] == 1, rows)
            ppm = (mean(f) - sc.tone_hz) / sc.tone_hz * 1e6
            say("AT-03 tone on, $(Dates.format(now(), "yyyy-mm-dd HH:MM")), $(length(rows)) s at gain 20, receiver LO $(rb.lo / 1e6) MHz, set tone $(sc.tone_hz / 1e6) MHz")
            say(@sprintf("  Tone SNR (7 bins), lowest of both channels     %.1f dB          (>= 30 dB)  %s", snr, pf(snr >= 30)))
            say(@sprintf("  Clipping, largest                              %.1e         (< 1e-3)    %s", clip, pf(clip < 1e-3)))
            say(@sprintf("  Pluto+ image, largest                          %s  (<= -20 dBc) %s", below ? @sprintf("below noise (< %.1f dBc)", img) : @sprintf("%.1f dBc", img), pf(img <= -20)))
            say(@sprintf("  Tone frequency (mean)                          %.4f MHz = %+.1f Hz = %+.2f ppm from the set value (|ppm| <= 25 + Pluto+ tolerance)  %s",
                         mean(f) / 1e6, mean(f) - sc.tone_hz, ppm, pf(abs(ppm) <= 25)))
            say(@sprintf("  Tone frequency over %d s, peak to peak          %.1f Hz          (<= 1 kHz)  %s", length(rows), maximum(f) - minimum(f), pf(maximum(f) - minimum(f) <= 1e3)))
            say(@sprintf("  Tone |rho|, lowest                             %.4f", minimum(x[10] for x in rows)))
            open(io -> println(io, rb.lo + mean(f) - rb.lo), joinpath(ac.out, "at03_tone_hz.txt"), "w")
        else
            ft = isfile(joinpath(ac.out, "at03_tone_hz.txt")) ? parse(Float64, readline(joinpath(ac.out, "at03_tone_hz.txt"))) : sc.tone_hz + 16e3
            fb = ft - rb.lo
            rows = Any[]
            for i in 1:5
                rec = P.capture(r, c, fs, P.blocks_for(c, fs, 1.0)).rec
                win = findall(k -> keep[k] && abs(fbb[k] - fb) <= 5e3, 1:n)          # +-5 kHz around where the tone was
                nb1 = median(rec.P1[keep]); nb2 = median(rec.P2[keep])
                pk1 = 10log10(maximum(rec.P1[win]) / nb1); pk2 = 10log10(maximum(rec.P2[win]) / nb2)
                push!(rows, (i, pk1, pk2))
                @printf("  %d s: largest bin within 5 kHz of %.4f MHz: %.1f / %.1f dB above the median noise\n", i, ft / 1e6, pk1, pk2)
            end
            csv_write(joinpath(ac.out, "at03_off.csv"), ["sec", "peak1_db_above_median", "peak2_db_above_median"], rows)
            pk = maximum(max(x[2], x[3]) for x in rows)
            say("AT-03 tone off, $(Dates.format(now(), "yyyy-mm-dd HH:MM")): largest bin within 5 kHz of the tone $(round(pk, digits = 1)) dB above the median noise (absent if < 10 dB, the level of a single noise peak)  $(pf(pk < 10))")
        end
    finally
        P.close_radio!(r)
    end
    open(io -> foreach(l -> println(io, l), out), joinpath(ac.out, "at03_summary.txt"), mode == "on" ? "w" : "a")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
