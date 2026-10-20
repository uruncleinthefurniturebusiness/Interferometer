# =====================================================================
# capture_868.jl -- quick two-channel Pluto capture of a signal-generator tone at 868 MHz
#
# Usage (from the Interferometer folder):
#   julia capture_868.jl [f_tone=868e6 offset=200e3 secs=2 gain1=20 gain2=20 plot=true save_raw=true]
# Any pluto_interferometer.jl option (addr, fs, nfft, rx_port, out_dir, tag, ...) is also accepted.
#
# The LO is set to f_tone - offset, so the tone lands at +offset in baseband, clear of the DC spike
# (the Pluto's own clock error moves it by up to about +-20 kHz at 868 MHz).
# Reports per-channel level and clipping, the tone's true frequency, its SNR, and the RX1-RX2 coherence
# and phase at the tone (V1 convention: V = x1*conj(x2), positive phase = the signal reaches RX2 later).
# Also the RX1-RX2 cross-correlation: cross-spectrum magnitude and phase against frequency, and the correlation
# against lag (inverse FFT of the cross spectrum). Writes raw cf32 per channel and a PNG of all panels to out_dir.
# =====================================================================

include(joinpath(@__DIR__, "pluto_interferometer.jl"))     # Config, open_verified, read_block!, FX, tone_fit, ...

using Plots

Base.@kwdef mutable struct CapCfg
    f_tone::Float64 = 868.0e6     # generator frequency (Hz)
    offset::Float64 = 200e3       # where the tone sits in baseband (Hz); LO = f_tone - offset
    secs::Float64   = 2.0         # capture length (s)
    search::Float64 = 60e3        # look for the tone within +-this of the offset (Hz); covers the clock error
    min_snr_db::Float64 = 20.0    # tone counts as found if its fine-FFT bin is this far above the median nearby
    plot::Bool      = true        # save a PNG of the spectra, coherence, samples and phase against time
end

# Strongest tone between fmin and fmax (baseband Hz), so a stronger signal elsewhere (the DC spike, a spur) is
# ignored. Fine FFT of the first L samples, parabolic peak, then refined from the phase advance between the two
# halves (as V1's tone_fit). Returns frequency, phase, amplitude (full scale) and peak-to-median SNR (dB).
function tone_fit_in(x::AbstractVector{ComplexF32}, fs::Float64, fmin::Float64, fmax::Float64; L::Int = 2^18)
    L = min(L, length(x))
    s = ComplexF64.(@view x[1:L])
    p = abs2.(fft(s))
    fb = fftfreq(L, fs)
    ks = findall(f -> fmin <= f <= fmax, fb)
    k = ks[argmax(p[ks])]
    a = log(p[mod1(k - 1, L)] + 1e-300); b = log(p[k] + 1e-300); c = log(p[mod1(k + 1, L)] + 1e-300)
    d = 0.5 * (a - c) / (a - 2b + c)
    f = fb[k] + (isfinite(d) ? d : 0.0) * fs / L
    n = collect(0.0:L-1)
    h = L ÷ 2
    for _ in 1:3
        r = cis.(-2pi * f / fs .* n) .* s
        f += angle(sum(@view r[h+1:L]) * conj(sum(@view r[1:h]))) * fs / (2pi * h)
    end
    r = cis.(-2pi * f / fs .* n) .* s
    return f, angle(sum(r)), abs(sum(r)) / L, 10log10(p[k] / median(p[ks]))
end

function parse_868_cli(args::Vector{String})
    cfg = Config(gain1 = 40, gain2 = 40, save_raw = true); cc = CapCfg()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        obj = hasfield(CapCfg, sym) ? cc : hasfield(Config, sym) ? cfg : error("Unknown option '$k'")
        T = fieldtype(typeof(obj), sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(obj, sym, val)
    end
    abs(cc.offset) > cc.search + 10e3 || error("offset=$(cc.offset) Hz: the tone search (+-$(cc.search) Hz) would reach the DC spike")
    abs(cc.offset) < 0.4 * cfg.fs || error("offset=$(cc.offset) Hz is too close to the band edge for fs=$(cfg.fs)")
    cfg.f_lo = cc.f_tone - cc.offset
    return cfg, cc
end

function capture_868(args::Vector{String})
    cfg, cc = parse_868_cli(args)
    @printf("Generator at %.6f MHz, LO at %.6f MHz (tone expected at %+.1f kHz)\n\n",
            cc.f_tone / 1e6, cfg.f_lo / 1e6, cc.offset / 1e3)

    # ---- capture: whole blocks, stored first and processed afterwards so nothing waits on the refill ----
    r, rb = open_verified(cfg)
    fs = Float64(rb.fs1); lo = Float64(rb.lo)
    nsegb, bufsize = block_geometry(cfg)
    nblk = max(1, ceil(Int, cc.secs * fs / bufsize))
    blocks = Block[]
    t_start = now(UTC)
    try
        for _ in 1:cfg.skip_blocks
            read_block!(r, 0)
        end
        for k in 1:nblk
            push!(blocks, read_block!(r, (k - 1) * bufsize))
        end
    finally
        close_radio!(r)
    end
    nshort = count(b -> b.short, blocks)
    fsc = detect_fullscale(blocks[1])
    x1 = reduce(vcat, [to_complex(b.i1, b.q1, fsc) for b in blocks])
    x2 = reduce(vcat, [to_complex(b.i2, b.q2, fsc) for b in blocks])
    @printf("\nCaptured %d blocks = %.2f s (%d samples per channel), full scale %d, %d short block(s)\n",
            nblk, length(x1) / fs, length(x1), round(Int, fsc), nshort)

    # ---- per-channel level and clipping ----
    s1 = chan_stats(x1); s2 = chan_stats(x2)
    for (nm, s) in (("RX1", s1), ("RX2", s2))
        @printf("  %s: power %6.1f dBFS, peak %.3f of full scale, clipped %.4f %%\n",
                nm, 10log10(s.power), s.peak, 100s.clip)
    end
    (s1.clip > 1e-3 || s2.clip > 1e-3) && println("  WARNING: clipping. Lower gain1/gain2 or the generator level.")
    min(s1.power, s2.power) < 1e-5 &&
        println("  NOTE: below -50 dBFS the noise is only a few ADC counts. Raise gain1/gain2 (each +10 dB adds 10 dB).")

    # ---- cross spectrum over the whole capture ----
    fx = FX(cfg.nfft, nsegb, cfg.window)
    for b in blocks
        accumulate!(fx, to_complex(b.i1, b.q1, fsc), to_complex(b.i2, b.q2, fsc))
    end
    rec = finalize!(fx)
    sm = summarize(rec, cfg.dc_mask)
    f = bin_freqs(cfg.nfft, fs, lo)
    @printf("\nBand average (DC +-%d bins excluded) |rho| = %.4f; DC bin |rho| = %.4f; noise-only level 1/sqrt(nseg) = %.4f\n",
            cfg.dc_mask, abs(sm.rho), abs(sm.rho_dc), 1 / sqrt(fx.nseg))

    # ---- cross-correlation against lag: inverse FFT of the cross spectrum, normalised so identical signals give 1 ----
    # The DC spike and the outer 15 % of the band (rising ADC noise) are left out. conj() puts the peak at +tau when the
    # signal reaches RX2 later (V1 convention). One lag = 1/fs = 0.4 us, far more than the arc's largest geometric
    # delay (B/c = 0.83 ns for 25 cm), so the geometry shows up in the phase, not as a lag shift.
    nf = cfg.nfft; cdc = nf ÷ 2 + 1
    xkeep = abs.(f .- lo) .< 0.85 * fs / 2
    xkeep[max(1, cdc - cfg.dc_mask):min(nf, cdc + cfg.dc_mask)] .= false
    Vm = ifelse.(xkeep, ComplexF64.(rec.V), 0.0im)
    r_lag = fftshift(ifft(ifftshift(conj.(Vm)))) .* nf ./ sqrt(sum(rec.P1[xkeep]) * sum(rec.P2[xkeep]))
    lags = (-nf ÷ 2):(nf ÷ 2 - 1)
    kl = argmax(abs.(r_lag))
    @printf("Cross-correlation (DC and band edges excluded): zero lag |rho| = %.4f, phase %+.1f deg; peak |rho| = %.4f at lag %+d samples (%+.2f us)\n",
            abs(r_lag[nf ÷ 2 + 1]), rad2deg(angle(r_lag[nf ÷ 2 + 1])), abs(r_lag[kl]), lags[kl], 1e6 * lags[kl] / fs)

    # ---- tone: searched for only near the expected offset, so the DC spike or a spur cannot be mistaken for it ----
    fmin = cc.offset - cc.search; fmax = cc.offset + cc.search
    f1, ph1, a1, snr1 = tone_fit_in(x1, fs, fmin, fmax)
    f2, ph2, a2, snr2 = tone_fit_in(x2, fs, fmin, fmax)
    found = min(snr1, snr2) >= cc.min_snr_db
    @printf("Tone search %+.0f to %+.0f kHz: RX1 peak %+.3f kHz, %.1f dB above median; RX2 peak %+.3f kHz, %.1f dB\n",
            fmin / 1e3, fmax / 1e3, f1 / 1e3, snr1, f2 / 1e3, snr2)
    if !found
        println("  NO TONE FOUND in $(snr1 < cc.min_snr_db && snr2 < cc.min_snr_db ? "either channel" :
                snr1 < cc.min_snr_db ? "RX1" : "RX2") (needs >= $(cc.min_snr_db) dB). Check: generator RF output ON, " *
                "frequency $(cc.f_tone / 1e6) MHz, level, antenna connected; raise gain or bring the TX antenna closer.")
    end
    abs(f1 - f2) > 50 && found && @printf("  NOTE: RX1 and RX2 disagree on the tone frequency by %.0f Hz\n", f1 - f2)
    f_true = lo + f1
    k = argmin(abs.(f .- f_true))                    # FX bin holding the tone (fftshifted order)
    rho_t = rec.V[k] / sqrt(rec.P1[k] * rec.P2[k])
    @printf("  %s: true %.6f MHz (%+.0f Hz = %+.2f ppm from f_tone)\n", found ? "tone" : "strongest bin (not a tone)",
            f_true / 1e6, f_true - cc.f_tone, 1e6 * (f_true - cc.f_tone) / cc.f_tone)
    @printf("  RX1 %6.1f dBFS, RX2 %6.1f dBFS (RX2 - RX1 = %+.2f dB)\n", 20log10(a1), 20log10(a2), 20log10(a2 / a1))
    @printf("  coherence at %.6f MHz |rho| = %.4f, phase = %+.2f deg (tone fit: %+.2f deg)\n",
            f[k] / 1e6, abs(rho_t), rad2deg(angle(rho_t)), rad2deg(wrap_pi(ph1 - ph2)))

    # ---- phase at the tone bin, block by block (stability over the capture) ----
    fxb = FX(cfg.nfft, nsegb, cfg.window)
    kk = mod1(k - cfg.nfft ÷ 2, cfg.nfft)          # same bin in the unshifted FFT order
    ph_blk = Float64[]
    for b in blocks
        reset!(fxb)
        accumulate!(fxb, to_complex(b.i1, b.q1, fsc), to_complex(b.i2, b.q2, fsc))
        push!(ph_blk, angle(fxb.V[kk]))
    end
    ph_deg = rad2deg.(unwrap_simple(ph_blk))
    t_blk = (0:nblk-1) .* (bufsize / fs)
    @printf("  phase over %d blocks: rms %.2f deg, peak-to-peak %.2f deg\n",
            nblk, std(ph_deg), maximum(ph_deg) - minimum(ph_deg))

    # ---- outputs ----
    mkpath(cfg.out_dir)
    base = joinpath(cfg.out_dir, "cap868_" * Dates.format(t_start, "yyyymmdd_HHMMSS") *
                                 (isempty(cfg.tag) ? "" : "_" * cfg.tag))
    if cfg.save_raw
        write(base * "_rx1.cf32", x1); write(base * "_rx2.cf32", x2)
        open(base * ".txt", "w") do io
            println(io, "utc_start=", t_start, "\nf_tone_hz=", cc.f_tone, "\nf_lo_hz=", lo, "\nfs_hz=", fs,
                    "\ngain1_db=", rb.gain1, "\ngain2_db=", rb.gain2, "\nrx_port=", rb.port1,
                    "\nsamples=", length(x1), "\nformat=ComplexF32 interleaved I/Q, normalised to full scale (", fsc, ")")
        end
        println("\nRaw samples: $(base)_rx1.cf32, $(base)_rx2.cf32 (+ .txt)")
    end
    if cc.plot
        fm = f ./ 1e6
        p1 = plot(fm, 10log10.(rec.P1), label = "RX1", xlabel = "Frequency (MHz)", ylabel = "Power (dB, arb.)",
                  title = "Auto spectra")
        plot!(p1, fm, 10log10.(rec.P2), label = "RX2")
        plot!(p1, fm, 10log10.(abs.(rec.V)), label = "|cross| RX1xRX2", color = :black, alpha = 0.6)
        vline!(p1, [cc.f_tone / 1e6], label = "f_tone", ls = :dash, color = :gray)
        coh = abs.(rec.V) ./ sqrt.(rec.P1 .* rec.P2)
        p2 = plot(fm, coh, label = "", xlabel = "Frequency (MHz)",
                  ylabel = "|rho|", ylims = (0, 1.05), title = "Coherence per channel")
        good = coh .> 3 / sqrt(fx.nseg)                # bins clearly above the noise-only coherence
        p5 = scatter(fm[.!good], rad2deg.(angle.(rec.V[.!good])), ms = 1, msw = 0, color = :lightgray, label = "noise",
                     xlabel = "Frequency (MHz)", ylabel = "Phase (deg)", ylims = (-180, 180), yticks = -180:90:180,
                     title = "Cross-spectrum phase (RX1 x conj RX2)")
        scatter!(p5, fm[good], rad2deg.(angle.(rec.V[good])), ms = 2, msw = 0, color = :black, label = "|rho| > 3/sqrt(nseg)")
        lw = min(64, nf ÷ 2); sel = (nf ÷ 2 + 1 - lw):(nf ÷ 2 + lw)
        p6 = plot(lags[sel] ./ fs .* 1e6, abs.(r_lag[sel]), label = "", marker = :circle, ms = 2,
                  xlabel = "Lag (us, + = RX2 later)", ylabel = "|rho|", title = "Cross-correlation against lag")
        m = min(60, length(x1)); tus = (0:m-1) ./ fs .* 1e6
        p3 = plot(tus, real.(x1[1:m]), label = "RX1 I", xlabel = "Time (us)", ylabel = "Amplitude (full scale)",
                  title = "First samples", marker = :circle, ms = 2)
        plot!(p3, tus, real.(x2[1:m]), label = "RX2 I", marker = :circle, ms = 2)
        p4 = plot(t_blk, ph_deg, label = "", xlabel = "Time (s)", ylabel = "Phase (deg)", marker = :circle,
                  title = @sprintf("RX1-RX2 phase at %.4f MHz", f[k] / 1e6))
        savefig(plot(p1, p2, p5, p6, p3, p4, layout = (3, 2), size = (1400, 1300), margin = 5Plots.mm), base * ".png")
        println("Figure: $(base).png")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    capture_868(ARGS)
end
