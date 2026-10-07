# =====================================================================
# bt_fft.jl -- spectra of a raw capture from bt_capture.jl (average spectrum, max-hold, waterfall, burst close-up)
#
# Usage:
#     julia bt_fft.jl path=C:/PlutoCaptures/bt_XXXX_rx1.cs16                      # the whole file
#     julia bt_fft.jl path=... n=1000000                                          # the first 1 million samples (0.4 s)
#     julia bt_fft.jl path=... n=1000000 start=92000000                           # 1 million samples from sample 92,000,000
#     julia bt_fft.jl path=... start_s=36.9 dur_s=0.5                             # the same thing in seconds
# Options: path, n (number of samples to use; 0 = all), start (first sample, default 0), start_s / dur_s (alternative to
#          n and start), nfft (spectrum resolution, default 2048 = 1.22 kHz bins), wf_nfft (waterfall bins, default 256),
#          tbin (waterfall time step in s; default: 0.5 s for the whole file, otherwise the portion divided into about 400 steps),
#          fs (default 2.5e6), flo (centre Hz; read from the JSON if present), zoom_us (burst close-up width, default 1500),
#          packets (CSV from bt_process.jl, used to pick the strongest decoded packet inside the portion for the close-up)
#
# Output (next to the file; a portion gets the suffix _s<start>_n<samples> so it does not overwrite the full-file plots):
#     <stem>_psd.png, <stem>_waterfall.png, <stem>_burst.png, <stem>_psd.csv
#
# Scaling: spectra are in dBFS per bin, referenced so that a full-scale complex tone (|x| = 2048 counts) reads 0 dBFS in
# its bin. Noise reads lower the finer the bins are (noise per bin falls with the bin width), so compare tones with
# tones. "Max hold" is the largest value each bin reached in any single FFT frame; it shows short bursts that the
# average hides. A short portion has fewer averages, so the average spectrum is noisier.
# =====================================================================

using FFTW, Plots, Statistics, Printf
gr()

Base.@kwdef mutable struct FO
    path::String     = ""
    n::Int           = 0
    start::Int       = -1
    nfft::Int        = 2048
    wf_nfft::Int     = 256
    tbin::Float64    = NaN
    fs::Float64      = 2.5e6
    flo::Float64     = NaN
    start_s::Float64 = 0.0
    dur_s::Float64   = Inf
    zoom_us::Float64 = 1500.0
    packets::String  = ""
end

function parse_fo(args)
    o = FO()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        hasfield(FO, sym) || error("Unknown option '$k'")
        T = fieldtype(FO, sym)
        setfield!(o, sym, T == String ? String(v) : T <: Integer ? round(T, parse(Float64, v)) : parse(T, v))
    end
    return o
end

# Windows can refuse to overwrite a PNG that something has open; remove it first.
safe_save(plt, path) = (isfile(path) && rm(path; force = true); savefig(plt, path))

hann(n) = Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:n-1) ./ n))
dbfs(p) = 10 .* log10.(max.(p, 1e-20))

function read_samples(io, pos::Int, m::Int)
    seek(io, 4 * pos)
    buf = Vector{Int16}(undef, 2m); read!(io, buf)
    return ComplexF32.(buf[1:2:end], buf[2:2:end]) ./ 2048f0
end

function center_from_json(path)
    j = replace(path, r"_rx[12]\.cs16$" => ".json")
    isfile(j) || return NaN
    m = match(r"\"center_frequency_hz\":\s*([0-9.]+)", read(j, String))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

function masked_argmax(v, keep)
    m = copy(v); m[.!keep] .= -Inf
    return argmax(m)
end

# Accumulators for the average / max-hold spectrum and the waterfall
mutable struct Acc
    P::Vector{Float64}; H::Vector{Float64}; nfr::Int
    wf_max::Vector{Vector{Float32}}; wf_mean::Vector{Vector{Float32}}
    plans::Dict{Tuple{Int,Int},Any}
end

function accumulate!(a::Acc, x::AbstractVector{ComplexF32}, nf, wn, w1, w2, tb_frames)
    c1 = sum(w1); c2 = sum(w2)
    fr1 = length(x) ÷ nf
    if fr1 > 0
        p1 = get!(() -> plan_fft(zeros(ComplexF32, nf, fr1), 1), a.plans, (nf, fr1))
        A1 = abs2.(p1 * (reshape(@view(x[1:fr1*nf]), nf, :) .* w1)) ./ c1^2
        a.P .+= vec(sum(A1, dims = 2)); a.H .= max.(a.H, vec(maximum(A1, dims = 2))); a.nfr += fr1
    end
    nb = length(x) ÷ (tb_frames * wn)
    if nb > 0
        fr2 = nb * tb_frames
        p2 = get!(() -> plan_fft(zeros(ComplexF32, wn, fr2), 1), a.plans, (wn, fr2))
        A2 = reshape(abs2.(p2 * (reshape(@view(x[1:fr2*wn]), wn, :) .* w2)) ./ c2^2, wn, tb_frames, nb)
        for k in 1:nb
            push!(a.wf_max, Float32.(vec(maximum(view(A2, :, :, k), dims = 2))))
            push!(a.wf_mean, Float32.(vec(mean(view(A2, :, :, k), dims = 2))))
        end
    end
end

function main(args)
    o = parse_fo(args)
    isfile(o.path) || error("give path=<file.cs16> (file not found: '$(o.path)')")
    isfinite(o.flo) || (o.flo = center_from_json(o.path))
    isfinite(o.flo) || (o.flo = 2426.0e6; println("(centre frequency not found; assuming 2426 MHz)"))
    stem = replace(o.path, r"\.cs16$" => "")
    fs = o.fs
    n_total = filesize(o.path) ÷ 4
    s0 = o.start >= 0 ? o.start : round(Int, o.start_s * fs)
    s0 < n_total || error("start is beyond the end of the file ($n_total samples)")
    s1 = o.n > 0 ? min(n_total, s0 + o.n) : (isfinite(o.dur_s) ? min(n_total, s0 + round(Int, o.dur_s * fs)) : n_total)
    span = s1 - s0
    portion = span < n_total
    tag = portion ? @sprintf("_s%d_n%d", s0, span) : ""
    nf, wn = o.nfft, o.wf_nfft
    span >= 2nf || error("portion too short ($span samples); need at least $(2nf)")
    tbin = isfinite(o.tbin) ? o.tbin : (portion ? max(span / fs / 400, 2wn / fs) : 0.5)
    tb_frames = max(1, round(Int, tbin * fs / wn))
    tb_samples = tb_frames * wn
    w1 = hann(nf); w2 = hann(wn)
    acc = Acc(zeros(nf), zeros(nf), 0, Vector{Vector{Float32}}(), Vector{Vector{Float32}}(), Dict{Tuple{Int,Int},Any}())
    @printf("Reading %s: samples %d to %d (%.4f s from %.4f s)\n", o.path, s0, s1 - 1, span / fs, s0 / fs)
    if span <= 16_000_000                                    # a portion: read it all and process in one go
        x = open(io -> read_samples(io, s0, span), o.path, "r")
        accumulate!(acc, x, nf, wn, w1, w2, tb_frames)
    else                                                     # the whole file: three time bins per read
        chunk = 3 * tb_samples
        open(o.path, "r") do io
            pos = s0
            while pos + chunk <= s1
                accumulate!(acc, read_samples(io, pos, chunk), nf, wn, w1, w2, tb_frames)
                pos += chunk
                (pos - s0) % (20 * chunk) == 0 && @printf("  ... %.0f s\n", (pos - s0) / fs)
            end
        end
    end
    acc.nfr > 0 || error("no complete FFT frame in the portion")
    meanP = fftshift(acc.P ./ acc.nfr); maxP = fftshift(acc.H)
    fb = ((0:nf-1) .- nf ÷ 2) .* (fs / nf)
    fMHz = (o.flo .+ fb) ./ 1e6
    mean_db = dbfs(meanP); max_db = dbfs(maxP)
    short = span / fs < 2.0
    tscale, tunit = short ? (1e3, "ms") : (1.0, "s")

    # ---- numbers ----
    c = nf ÷ 2 + 1
    keep = trues(nf); keep[c-3:c+3] .= false
    floor_db = median(mean_db[keep])
    kpk = masked_argmax(mean_db, keep); kmx = masked_argmax(max_db, keep)
    @printf("\nAverage spectrum (%d bins of %.2f kHz, %d frames): noise floor (median) %.1f dBFS per bin\n", nf, fs / nf / 1e3, acc.nfr, floor_db)
    @printf("  DC spike (LO leakage): %.1f dB above the floor\n", mean_db[c] - floor_db)
    @printf("  strongest non-DC bin on average: %.3f MHz (%+.0f kHz from centre), %.1f dB above the floor (max hold there %.1f dB)\n",
            fMHz[kpk], fb[kpk] / 1e3, mean_db[kpk] - floor_db, max_db[kpk] - floor_db)
    @printf("  largest single-frame peak (DC excluded): %.1f dBFS at %+.0f kHz\n", max_db[kmx], fb[kmx] / 1e3)
    mid = abs.(fb) .< 1.0e6
    @printf("  average power: %.1f dBFS within +-1 MHz, %.1f dBFS over the whole 2.5 MHz\n", 10log10(sum(meanP[mid])), 10log10(sum(meanP)))

    # ---- plot 1: spectrum ----
    p1 = plot(fMHz, [mean_db max_db]; label = ["average" "max hold (single-frame peak)"], lw = [1.5 0.8], color = [:blue :red],
              xlabel = "frequency (MHz)", ylabel = "dBFS per bin", size = (1150, 520), legend = :topright,
              title = @sprintf("%s: %d samples (%.3f s), %d-point FFT (%.2f kHz bins)", basename(o.path), span, span / fs, nf, fs / nf / 1e3))
    vline!(p1, [o.flo / 1e6]; label = "LO", color = :gray, ls = :dash)
    safe_save(p1, stem * tag * "_psd.png")
    open(stem * tag * "_psd.csv", "w") do io
        println(io, "freq_hz,mean_dbfs,maxhold_dbfs")
        for i in 1:nf; @printf(io, "%.1f,%.3f,%.3f\n", o.flo + fb[i], mean_db[i], max_db[i]); end
    end

    # ---- plot 2: waterfall (max-hold per time bin) ----
    nt = length(acc.wf_max)
    nt > 0 || error("no waterfall bins; use a longer portion or a smaller tbin")
    Zmax = dbfs(reduce(hcat, acc.wf_max))[[wn÷2+1:wn; 1:wn÷2], :]                  # fftshift the frequency axis
    fw = ((0:wn-1) .- wn ÷ 2) .* (fs / wn)
    tt = (((0:nt-1) .+ 0.5) .* tb_samples ./ fs .+ s0 / fs) .* tscale
    lo, hi = quantile(vec(Zmax), 0.02), quantile(vec(Zmax), 0.9995)
    p2 = heatmap((o.flo .+ fw) ./ 1e6, tt, Zmax'; clims = (lo, hi), xlabel = "frequency (MHz)", ylabel = "time ($tunit)", color = :viridis,
                 title = @sprintf("max hold per %.3g ms, %d-point FFT (%.1f kHz bins), dBFS", tb_samples / fs * 1e3, wn, fs / wn / 1e3), size = (900, 800))
    safe_save(p2, stem * tag * "_waterfall.png")
    inb = abs.(fw) .< 1.0e6
    fl_t = median(vec(Zmax[inb, :]))
    hot = [maximum(Zmax[inb, k]) > fl_t + 20 for k in 1:nt]
    @printf("\nWaterfall (%d time bins of %.3g ms): median max-hold level %.1f dBFS per %.1f kHz bin; %.0f%% of time bins contain a burst more than 20 dB above it\n",
            nt, tb_samples / fs * 1e3, fl_t, fs / wn / 1e3, 100 * count(hot) / nt)

    # ---- plot 3: close-up of the strongest decoded packet in the portion (or its strongest 50 us window) ----
    pk = isempty(o.packets) ? stem * "_packets.csv" : o.packets
    centre = nothing
    if isfile(pk)
        best = (-Inf, 0, "", "")
        for (i, line) in enumerate(eachline(pk))
            i == 1 && continue
            cc = split(line, ',')
            (length(cc) >= 11 && cc[6] == "1") || continue
            smp = parse(Int, cc[1])
            (s0 <= smp < s1) || continue
            r = parse(Float64, cc[3])
            r > best[1] && (best = (r, smp, cc[7], cc[9]))
        end
        best[2] > 0 && (centre = best)
    end
    if centre === nothing
        x = open(io -> read_samples(io, s0, min(span, 4_000_000)), o.path, "r")
        e = [sum(abs2, @view x[(k-1)*125+1:k*125]) for k in 1:length(x)÷125]
        centre = (10log10(maximum(e) / 125), s0 + (argmax(e) - 1) * 125, "strongest 50 us window", "")
    end
    half = round(Int, o.zoom_us * 1e-6 * fs / 2)
    st = clamp(centre[2] - half ÷ 2, 0, max(0, n_total - 2half))
    seg = open(io -> read_samples(io, st, min(2half, n_total - st)), o.path, "r")
    zn, hop = 64, 8
    wz = hann(zn); cz = sum(wz)
    nz = (length(seg) - zn) ÷ hop + 1
    Z = reduce(hcat, [dbfs(abs2.(fftshift(fft(seg[(k-1)*hop+1:(k-1)*hop+zn] .* wz))) ./ cz^2) for k in 1:nz])
    tz = ((0:nz-1) .* hop .+ zn / 2) ./ fs .* 1e6
    fz = ((0:zn-1) .- zn ÷ 2) .* (fs / zn) ./ 1e3
    pw = dbfs([abs2(v) for v in seg])
    p3a = plot((0:length(seg)-1) ./ fs .* 1e6, pw; label = "", ylabel = "|x|^2 (dBFS)", lw = 1,
               title = @sprintf("%s %s: %.1f dBFS, from sample %d (%.4f s)", centre[3], centre[4], centre[1], st, st / fs))
    p3b = heatmap(tz, fz, Z; xlabel = "time (microseconds)", ylabel = "offset from centre (kHz)", color = :viridis, clims = (quantile(vec(Z), 0.05), maximum(Z)))
    safe_save(plot(p3a, p3b; layout = grid(2, 1, heights = [0.3, 0.7]), size = (1000, 700)), stem * tag * "_burst.png")

    println("\nWrote:\n  ", stem * tag, "_psd.png\n  ", stem * tag, "_waterfall.png\n  ", stem * tag, "_burst.png\n  ", stem * tag, "_psd.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
