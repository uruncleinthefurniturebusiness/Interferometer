# =====================================================================
# live_868.jl -- live two-channel spectrum of the 868-870 MHz ISM band, to find the Pluto+ tone
#
# Usage (from the Real folder):
#   julia live_868.jl [addr=usb:1.2.5 f_lo=869e6 gain1=20 gain2=20 avg=3 hold=true]
# Any pluto_interferometer.jl option (fs, nfft, ...) is also accepted. Ctrl+C stops it and closes the radio.
#
# The LO sits in the middle of the band (869.0 MHz), so 868-870 MHz is the central +-1 MHz (0.4 fs at 2.5 Msps,
# inside the filter roll-off). Shows RX1 and RX2 in dBFS against RF frequency with a peak hold, marks the strongest
# line away from DC (the DC spike is the receiver's own LO leakage, not a signal), and prints its frequency, level,
# height above the noise and the RX1-RX2 coherence and phase at it. A Pluto+ tone at 868.2 MHz (TX LO 868.1 MHz)
# shows as the tone plus a weaker carrier-leak line 100 kHz below it.
# =====================================================================

include(joinpath(@__DIR__, "pluto_interferometer.jl"))     # Config, open_verified, read_block!, detect_fullscale, to_complex, ...

using Plots
gr()

Base.@kwdef mutable struct LiveCfg
    band_lo::Float64 = 868.0e6     # band shown and searched (Hz)
    band_hi::Float64 = 870.0e6
    avg::Int         = 3           # blocks averaged per update (buffer_secs each)
    hold::Bool       = true        # show the RX1 peak hold
    dc_hz::Float64   = 2.5e3       # the peak search skips this close to the LO
    ymin::Float64    = -110.0      # bottom of the plot (dBFS)
end

function parse_live_cli(args::Vector{String})
    cfg = Config(f_lo = 869.0e6, gain1 = 20, gain2 = 20, buffer_secs = 0.1, kernel_buffers = 2); lc = LiveCfg()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        obj = hasfield(LiveCfg, sym) ? lc : hasfield(Config, sym) ? cfg : error("Unknown option '$k'")
        T = fieldtype(typeof(obj), sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(obj, sym, val)
    end
    lc.band_hi > lc.band_lo || error("band_hi must be above band_lo")
    (lc.band_lo < cfg.f_lo - 0.4cfg.fs || lc.band_hi > cfg.f_lo + 0.4cfg.fs) &&
        @warn "Part of $(lc.band_lo / 1e6)-$(lc.band_hi / 1e6) MHz is outside f_lo +- 0.4 fs (filter roll-off): levels there read low."
    return cfg, lc
end

function live_868(args::Vector{String})
    cfg, lc = parse_live_cli(args)
    fs = cfg.fs; nf = cfg.nfft
    nsegb, _ = block_geometry(cfg)
    w = Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:nf-1) ./ nf))
    g2 = sum(Float64.(w))^2                         # a full-scale tone reads 0 dBFS in its bin

    r, rb = open_verified(cfg)
    lo = Float64(rb.lo)
    f = lo .+ ((0:nf-1) .- nf ÷ 2) .* (fs / nf)     # RF frequency of each (fftshifted) bin
    fm = f ./ 1e6
    inb = (f .>= lc.band_lo) .& (f .<= lc.band_hi)
    srch = inb .& (abs.(f .- lo) .> lc.dc_hz)
    any(srch) || error("No bins between $(lc.band_lo / 1e6) and $(lc.band_hi / 1e6) MHz at this LO and fs")
    hold1 = fill(-Inf, nf)
    @printf("\nLive %.1f-%.1f MHz: LO %.4f MHz, %.2f Msps, bin width %.0f Hz, %d x %.2f s per update. Ctrl+C to stop.\n",
            lc.band_lo / 1e6, lc.band_hi / 1e6, lo / 1e6, fs / 1e6, fs / nf, lc.avg, cfg.buffer_secs)

    Base.exit_on_sigint(false)
    try
        for _ in 1:(cfg.kernel_buffers + cfg.skip_blocks); read_block!(r, 0); end     # stale buffers from start-up
        while true
            P1 = zeros(nf); P2 = zeros(nf); X = zeros(ComplexF64, nf); clip = 0.0
            for _ in 1:lc.avg
                b = read_block!(r, 0)
                fsc = detect_fullscale(b)
                x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
                clip = max(clip, chan_stats(x1).clip, chan_stats(x2).clip)
                F1 = fft(reshape(x1, nf, nsegb) .* w, 1)
                F2 = fft(reshape(x2, nf, nsegb) .* w, 1)
                P1 .+= vec(sum(abs2.(F1), dims = 2))
                P2 .+= vec(sum(abs2.(F2), dims = 2))
                X .+= vec(sum(F1 .* conj.(F2), dims = 2))
            end
            s = lc.avg * nsegb * g2
            P1 = fftshift(P1) ./ s; P2 = fftshift(P2) ./ s; X = fftshift(X) ./ s
            d1 = 10log10.(P1 .+ 1e-20); d2 = 10log10.(P2 .+ 1e-20)
            hold1 .= max.(hold1, d1)

            # Strongest line away from DC, its height above the band median, and the RX1-RX2 coherence over +-3 bins
            k = argmax(ifelse.(srch, P1 .+ P2, 0.0))
            win = max(1, k - 3):min(nf, k + 3)
            ρ = sum(X[win]) / sqrt(sum(P1[win]) * sum(P2[win]))
            above1 = 10log10(P1[k] / median(P1[srch])); above2 = 10log10(P2[k] / median(P2[srch]))

            ttl = @sprintf("Peak %.4f MHz: RX1 %.1f / RX2 %.1f dBFS, %.1f / %.1f dB above noise, |ρ| %.3f, phase %+.1f°%s",
                           fm[k], d1[k], d2[k], above1, above2, abs(ρ), rad2deg(angle(ρ)), clip > 1e-3 ? "   CLIPPING" : "")
            plt = plot(fm[inb], d1[inb], label = "RX1", lw = 1, xlims = (lc.band_lo / 1e6, lc.band_hi / 1e6),
                       ylims = (lc.ymin, 0), xlabel = "Frequency (MHz)", ylabel = "Power (dBFS per bin)",
                       title = ttl, titlefontsize = 10, size = (1200, 600), legend = :topright)
            plot!(plt, fm[inb], d2[inb], label = "RX2", lw = 1)
            lc.hold && plot!(plt, fm[inb], hold1[inb], label = "RX1 peak hold", color = :gray, alpha = 0.5)
            vline!(plt, [lo / 1e6], ls = :dot, color = :black, label = "LO (DC spike)")
            scatter!(plt, [fm[k]], [max(d1[k], d2[k])], label = "", ms = 5, color = :red)
            display(plt)
            @printf("\r%s   ", ttl)
        end
    catch e
        e isa InterruptException || rethrow()
        println("\nStopped.")
    finally
        close_radio!(r)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    live_868(ARGS)
end
