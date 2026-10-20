# =====================================================================
# scan_868.jl -- find where the signal generator really is
#
# Steps the Pluto LO across a frequency range twice: once with the generator's RF ON, then (after a pause for you to
# switch it off) with RF OFF. Anything that is much stronger with RF on is the generator, wherever it actually is.
#
# Usage (from the Interferometer folder):
#   julia scan_868.jl [f_start=325e6 f_stop=1000e6 gain1=40 gain2=40 fs=5e6 tag=...]
# Any pluto_interferometer.jl option is accepted. The stock Pluto tunes 325 MHz - 3.8 GHz.
#
# Each step keeps |baseband| < 0.4 fs (the band edges are filter roll-off) and skips the DC spike. Steps are 0.8 fs
# apart, so the kept bands join up; the LO grid is offset by lo_offset so a round generator frequency never sits on DC.
# Reports two things:
#   lines      narrow bins at least min_db above the same bin with RF off and above that step's median (a carrier)
#   steps      whole-band power with RF on minus RF off per step (a spread-out signal: sweep, modulation, unlocked PLL)
# Writes <out_dir>/scan_<time>[_tag]_lines.csv, _steps.csv and .png.
# =====================================================================

include(joinpath(@__DIR__, "pluto_interferometer.jl"))     # Config, open_verified, read_block!, detect_fullscale, to_complex, ...

using Plots

Base.@kwdef mutable struct ScanCfg
    f_start::Float64   = 325e6       # lowest frequency to cover (Hz)
    f_stop::Float64    = 1000e6      # highest frequency to cover (Hz)
    lo_offset::Float64 = 137e3       # moves the LO grid off round numbers (Hz)
    keep::Int          = 2           # buffers averaged per step
    min_db::Float64    = 15.0        # line threshold: RF-on bin this far above RF-off and above the step median (dB)
    step_db::Float64   = 1.0         # step threshold: whole-band power up by at least this much with RF on (dB)
    top::Int           = 15          # how many lines and steps to print
end

function parse_scan_cli(args::Vector{String})
    cfg = Config(gain1 = 40, gain2 = 40, fs = 5e6, buffer_secs = 0.02, kernel_buffers = 2); sc = ScanCfg()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        obj = hasfield(ScanCfg, sym) ? sc : hasfield(Config, sym) ? cfg : error("Unknown option '$k'")
        T = fieldtype(typeof(obj), sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(obj, sym, val)
    end
    sc.f_stop > sc.f_start || error("f_stop must be above f_start")
    cfg.kernel_buffers >= 1 || error("kernel_buffers must be >= 1 here (stale buffers are flushed after each retune)")
    cfg.f_lo = sc.f_start + 0.4 * cfg.fs + sc.lo_offset
    return cfg, sc
end

# One pass over all LO steps. Returns per-bin power (fftshifted, scaled so a tone reads its power in dBFS) for each
# channel as nfft x nsteps matrices, the LO frequencies the hardware reports, and the worst clipped fraction per step.
function scan_pass(r, cfg::Config, sc::ScanCfg, los::Vector{Float64}, label::String)
    nf = cfg.nfft
    nsegb, _ = block_geometry(cfg)
    w = Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:nf-1) ./ nf))
    g2 = sum(Float64.(w))^2
    n = length(los)
    P1 = zeros(nf, n); P2 = zeros(nf, n); lo_act = zeros(n); clip = zeros(n)
    t0 = time()
    for (j, lo) in enumerate(los)
        chk(AP.C_iio_channel_attr_write_longlong(r.lo, "frequency", Int64(round(lo))), "LO frequency")
        lo_act[j] = rd_ll(r.lo, "frequency")
        abs(lo_act[j] - lo) > 1e3 && @warn @sprintf("LO asked %.6f MHz, hardware reports %.6f MHz", lo / 1e6, lo_act[j] / 1e6)
        for _ in 1:(cfg.kernel_buffers + 1)          # buffers filled before or during the retune
            read_block!(r, 0)
        end
        for _ in 1:sc.keep
            b = read_block!(r, 0)
            fsc = detect_fullscale(b)
            x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
            clip[j] = max(clip[j], chan_stats(x1).clip, chan_stats(x2).clip)
            P1[:, j] .+= vec(sum(abs2.(fft(reshape(x1, nf, nsegb) .* w, 1)), dims = 2))
            P2[:, j] .+= vec(sum(abs2.(fft(reshape(x2, nf, nsegb) .* w, 1)), dims = 2))
        end
        P1[:, j] .= fftshift(P1[:, j]) ./ (sc.keep * nsegb * g2)
        P2[:, j] .= fftshift(P2[:, j]) ./ (sc.keep * nsegb * g2)
        (j % 25 == 0 || j == n) && @printf("  %s: step %d/%d, LO %.1f MHz (%.0f s)\n", label, j, n, lo_act[j] / 1e6, time() - t0)
    end
    return P1, P2, lo_act, clip
end

function scan_868(args::Vector{String})
    cfg, sc = parse_scan_cli(args)
    fs = cfg.fs; nf = cfg.nfft
    half = 0.4 * fs
    los = collect(cfg.f_lo:2half:(sc.f_stop + half))
    @printf("Scan %.1f - %.1f MHz: %d steps of %.1f MHz, fs %.1f MHz, gain %d/%d dB, bin width %.0f Hz\n\n",
            sc.f_start / 1e6, sc.f_stop / 1e6, length(los), 2half / 1e6, fs / 1e6, cfg.gain1, cfg.gain2, fs / nf)

    r, rb = open_verified(cfg)
    t_start = now(UTC)
    local on, off
    try
        println("\nGenerator: RF ON, CW (no sweep or modulation), level as you will use it. Press Enter to start.")
        readline()
        on = scan_pass(r, cfg, sc, los, "RF on")
        println("\nNow switch the generator RF OFF (change nothing else) and press Enter.")
        readline()
        off = scan_pass(r, cfg, sc, los, "RF off")
    finally
        close_radio!(r)
    end
    P1on, P2on, lo_act, clip_on = on
    P1off, P2off, _, clip_off = off
    maximum(clip_on) > 1e-3 && println("WARNING: clipping in some RF-on steps (max $(round(100maximum(clip_on), digits = 2)) %). Lower the gain.")

    bb = ((0:nf-1) .- nf ÷ 2) .* (fs / nf)
    use = (abs.(bb) .< half) .& (abs.(bb) .> (cfg.dc_mask + 0.5) * fs / nf)
    thr = 10^(sc.min_db / 10)

    # ---- narrow lines present only with RF on ----
    lines = NamedTuple[]
    for (ch, Pon, Poff) in ((1, P1on, P1off), (2, P2on, P2off)), j in eachindex(lo_act)
        med = median(Pon[use, j])
        for k in findall(use)
            p = Pon[k, j]
            (p > thr * med && p > thr * Poff[k, j]) || continue
            p == maximum(@view Pon[max(1, k - 3):min(nf, k + 3), j]) || continue      # one entry per line
            push!(lines, (f_hz = lo_act[j] + bb[k], ch = ch, on_dbfs = 10log10(p), off_dbfs = 10log10(Poff[k, j]),
                          on_minus_off_db = 10log10(p / Poff[k, j]), above_median_db = 10log10(p / med)))
        end
    end
    sort!(lines, by = l -> -l.on_dbfs)

    # ---- whole-band power per step, RF on minus RF off ----
    ex1 = [10log10(sum(P1on[use, j]) / sum(P1off[use, j])) for j in eachindex(lo_act)]
    ex2 = [10log10(sum(P2on[use, j]) / sum(P2off[use, j])) for j in eachindex(lo_act)]

    println("\n==== Narrow lines only with RF on (>= $(sc.min_db) dB above RF off and above the step median) ====")
    if isempty(lines)
        println("  none")
    else
        for l in lines[1:min(end, sc.top)]
            @printf("  %11.6f MHz  RX%d  on %6.1f dBFS  off %6.1f dBFS  (%+5.1f dB, %4.1f dB above median)\n",
                    l.f_hz / 1e6, l.ch, l.on_dbfs, l.off_dbfs, l.on_minus_off_db, l.above_median_db)
        end
        length(lines) > sc.top && println("  ... $(length(lines) - sc.top) more in the CSV")
    end
    println("\n==== Steps whose whole-band power rose >= $(sc.step_db) dB with RF on ====")
    big = sort(findall(j -> max(ex1[j], ex2[j]) >= sc.step_db, eachindex(lo_act)), by = j -> -max(ex1[j], ex2[j]))
    if isempty(big)
        println("  none")
    else
        for j in big[1:min(end, sc.top)]
            @printf("  %8.2f - %8.2f MHz  RX1 %+5.1f dB  RX2 %+5.1f dB\n", (lo_act[j] - half) / 1e6, (lo_act[j] + half) / 1e6, ex1[j], ex2[j])
        end
    end
    if isempty(lines) && isempty(big)
        println("\nNothing in $(sc.f_start / 1e6)-$(sc.f_stop / 1e6) MHz changes with the generator's RF. Its output is outside this range, " *
                "far too weak at this gain, or not reaching the antenna.")
    end
    println("Bursty outside signals (LoRa, Wi-Fi) can show up in one pass only: repeat the scan to confirm a candidate.")

    # ---- outputs ----
    mkpath(cfg.out_dir)
    base = joinpath(cfg.out_dir, "scan_" * Dates.format(t_start, "yyyymmdd_HHMMSS") * (isempty(cfg.tag) ? "" : "_" * cfg.tag))
    open(base * "_lines.csv", "w") do io
        println(io, "f_hz,channel,on_dbfs,off_dbfs,on_minus_off_db,above_median_db")
        for l in lines
            @printf(io, "%.1f,%d,%.2f,%.2f,%.2f,%.2f\n", l.f_hz, l.ch, l.on_dbfs, l.off_dbfs, l.on_minus_off_db, l.above_median_db)
        end
    end
    open(base * "_steps.csv", "w") do io
        println(io, "lo_hz,band_lo_hz,band_hi_hz,rx1_on_minus_off_db,rx2_on_minus_off_db,clip_on,clip_off")
        for j in eachindex(lo_act)
            @printf(io, "%.0f,%.0f,%.0f,%.3f,%.3f,%.5f,%.5f\n", lo_act[j], lo_act[j] - half, lo_act[j] + half, ex1[j], ex2[j], clip_on[j], clip_off[j])
        end
    end

    # overview: kept bins of every step joined up, decimated by taking the maximum of each group of bins
    ku = findall(use); g = 16
    function overview(P)
        fx = Float64[]; px = Float64[]
        for j in eachindex(lo_act), s in 1:g:length(ku)
            kk = ku[s:min(end, s + g - 1)]
            push!(fx, (lo_act[j] + bb[kk[1]]) / 1e6); push!(px, 10log10(maximum(P[kk, j])))
        end
        return fx, px
    end
    pl = Any[]
    for (ch, Pon, Poff) in ((1, P1on, P1off), (2, P2on, P2off))
        fo, po = overview(Pon); _, pf = overview(Poff)
        p = plot(fo, pf, label = "RF off", color = :gray, lw = 0.5, xlabel = "Frequency (MHz)", ylabel = "Peak per bin (dBFS)",
                 title = "RX$ch scan (max of $g bins)")
        plot!(p, fo, po, label = "RF on", color = ch == 1 ? 1 : 2, lw = 0.5, alpha = 0.8)
        push!(pl, p)
    end
    p3 = plot(lo_act ./ 1e6, ex1, label = "RX1", marker = :circle, ms = 2, xlabel = "Step centre (MHz)",
              ylabel = "On - off (dB)", title = "Whole-band power change with RF on")
    plot!(p3, lo_act ./ 1e6, ex2, label = "RX2", marker = :circle, ms = 2)
    hline!(p3, [sc.step_db], label = "threshold", ls = :dash, color = :gray)
    for (ch, p) in ((1, pl[1]), (2, pl[2]))
        mk = Float64[l.f_hz / 1e6 for l in lines[1:min(end, 10)] if l.ch == ch]
        isempty(mk) || vline!(p, mk, label = "RF-on lines", ls = :dot, color = :red)
    end
    savefig(plot(pl..., p3, layout = (3, 1), size = (1500, 1200), margin = 5Plots.mm), base * ".png")
    println("\nResults: $(base)_lines.csv, $(base)_steps.csv\nFigure:  $(base).png")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    scan_868(ARGS)
end
