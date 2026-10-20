# =====================================================================
# bt_capture.jl -- record the 2.4 GHz Bluetooth band from the ADALM-Pluto to disk, for post-processing
#
# Reuses the verified hardware layer of pluto_interferometer.jl (settings are written, then READ BACK and checked) and
# keeps the sample rate at 2.5 Msps per channel, as in V1.
#
# Usage (from this folder, with 3 Julia threads: capture, writer and main each need a thread):
#     julia -t 3 bt_capture.jl                       # 10 minutes of BLE advertising channel 38 (2426 MHz), RX1
#     julia -t 3 bt_capture.jl channel=37 minutes=2  # advertising channel 37 (2402 MHz), 2 minutes
#     julia -t 3 bt_capture.jl freq=2441e6 rx=both   # any centre frequency, both receive channels
#     julia bt_capture.jl look path=C:/PlutoCaptures/bt_XXXX_rx1.cs16     # quick look at a finished file
#
# Options (key=value):
#     channel=37|38|39   BLE advertising channel (2402, 2426, 2480 MHz)      default 38
#     freq=<Hz>          centre frequency, overrides channel
#     minutes=<n>        capture length                                       default 10
#     fs=<Hz>            sample rate per channel                              default 2.5e6
#     gain=<dB>          RX gain, both channels                               default 20 (see below)
#     rx=1|2|both        which receive channel(s) to store                    default 1
#     out_dir=<path>     output folder, keep it OUTSIDE OneDrive              default C:/PlutoCaptures
#     tag=<text>         added to the file names
#     preflight=true|false   short level check before the long capture        default true
#     burst_db=<dB>      threshold above the median for the burst counter     default 10
#
# What is written (stem = bt_<UTC date>_<time>_<tag>):
#     <stem>_rx1.cs16   raw samples, interleaved int16 I,Q, little-endian ("ci16_le"). 12-bit ADC values,
#                       right-aligned: full scale is +-2048 counts. 4 bytes per complex sample.
#     <stem>_rx2.cs16   same, only with rx=2 or rx=both
#     <stem>.json       all settings (from the hardware readback), start time, sizes, how to read the file
#     <stem>_blocks.csv one row per 0.1 s block: index, UTC arrival time, refill time, queue backlog, short-read flag
#     <stem>_log.csv    one row per second: levels, clipping, energy-burst count
#
# Size: 10 minutes at 2.5 Msps is 6.0 GB per stored channel (12 GB with rx=both).
#
# Reading it back: Julia  x = reinterpret(Int16, read(path)) ; z = complex.(x[1:2:end], x[2:2:end]) ./ 2048
#                  Python numpy.fromfile(path, dtype=numpy.int16).reshape(-1, 2) @ [1, 1j] / 2048
#                  GUI    inspectrum, URH or GNU Radio (complex int16 at 2.5 Msps)
#
# Notes on Bluetooth:
#   * BLE advertising packets are short bursts (about 0.1 to 0.4 ms) sent on channels 37, 38 and 39 one
#     after another; a 2.5 MHz capture covers one 2 MHz BLE channel. Capture all three by running three times.
#   * Classic Bluetooth (audio, file transfer) hops over 79 MHz at 1600 hops/s; a 2.5 MHz window sees only
#     about 3% of its packets. For a test, make the phone ADVERTISE (e.g. the nRF Connect app: Advertiser tab,
#     short interval) and hold it close to the antenna.
#   * Wi-Fi occupies the same band with much longer, stronger bursts; the burst counter does not tell them apart.
#   * Gain: a phone close to the antenna is strong. If the preflight reports clipping, lower the gain.
# =====================================================================

include(joinpath(@__DIR__, "pluto_interferometer.jl"))      # hardware layer, Block, chan_stats, detect_fullscale (does not run V1's main)

const BT_CHANNELS = Dict(37 => 2402.0e6, 38 => 2426.0e6, 39 => 2480.0e6)
const BT_BYTES_PER_SAMPLE = 4            # int16 I + int16 Q

Base.@kwdef mutable struct BTOpts
    channel::Int         = 38
    freq::Float64        = NaN
    minutes::Float64     = 10.0
    fs::Float64          = 2.5e6
    gain::Int            = 20
    rx::String           = "1"
    out_dir::String      = "C:/PlutoCaptures"
    tag::String          = ""
    preflight::Bool      = true
    burst_db::Float64    = 10.0
    queue_depth::Int     = 64
    path::String         = ""             # `look` mode
end

function parse_bt(args)
    o = BTOpts()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        hasfield(BTOpts, sym) || error("Unknown option '$k'")
        T = fieldtype(BTOpts, sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(o, sym, val)
    end
    o.rx in ("1", "2", "both") || error("rx must be 1, 2 or both")
    isfinite(o.freq) || haskey(BT_CHANNELS, o.channel) || error("channel must be 37, 38 or 39 (or give freq=)")
    return o
end

bt_freq(o::BTOpts) = isfinite(o.freq) ? o.freq : BT_CHANNELS[o.channel]
bt_nchan(o::BTOpts) = o.rx == "both" ? 2 : 1

# ---------------------------------------------------------------------
# Energy-burst counter. Short windows of power; a burst starts where the window power rises above
# (median + burst_db) and ends when it falls back. Counts any strong short signal (BLE, Wi-Fi, ...).
# ---------------------------------------------------------------------
const BURST_WIN = 500                      # samples per window (0.2 ms at 2.5 Msps)

function window_energies(i::Vector{Int16}, q::Vector{Int16})
    nwin = length(i) ÷ BURST_WIN
    e = zeros(Float64, nwin)
    @inbounds for w in 1:nwin
        s = 0.0
        for n in (w - 1) * BURST_WIN + 1 : w * BURST_WIN
            s += Float64(i[n])^2 + Float64(q[n])^2
        end
        e[w] = s / BURST_WIN
    end
    return e
end

function count_bursts(e::Vector{Float64}, burst_db::Float64)
    isempty(e) && return (0, 0.0, Int[])
    med = median(e)
    thr = med * 10^(burst_db / 10)
    n = 0; lens = Int[]; run = 0
    for x in e
        if x > thr
            run += 1
        else
            if run > 0; n += 1; push!(lens, run); end
            run = 0
        end
    end
    run > 0 && (n += 1; push!(lens, run))
    return (n, maximum(e) / max(med, 1e-9), lens)
end

# ---------------------------------------------------------------------
# Writer task: blocks in, raw files and logs out
# ---------------------------------------------------------------------
function bt_writer!(q::Channel{Block}, o::BTOpts, paths::Vector{String}, logpath::String, fs::Float64, bufsize::Int)
    ios = [open(p, "w") for p in paths]
    logio = open(logpath, "w")
    println(logio, "t_s,utc,rms_counts_rx1,peak_counts_rx1,clipped_fraction_rx1,bursts,max_burst_over_median_db")
    buf = Vector{Int16}(undef, 2 * bufsize)
    blocks_per_log = max(1, round(Int, fs / bufsize))
    nblk = 0; total_bursts = 0; clip_blocks = 0
    acc_bursts = 0; acc_peak = 0; acc_clip = 0.0; acc_pow = 0.0; acc_n = 0; acc_maxratio = 0.0
    all_lens = Int[]
    for b in q
        # raw samples to disk
        k = 0
        for (sel, (iv, qv)) in enumerate(((b.i1, b.q1), (b.i2, b.q2)))
            (o.rx == "both" || o.rx == string(sel)) || continue
            k += 1
            buf[1:2:end] .= iv; buf[2:2:end] .= qv
            write(ios[k], buf)
        end
        # statistics on the first stored channel
        iv, qv = o.rx == "2" ? (b.i2, b.q2) : (b.i1, b.q1)
        nb, ratio, lens = count_bursts(window_energies(iv, qv), o.burst_db)
        append!(all_lens, lens)
        total_bursts += nb; acc_bursts += nb
        pk = max(maximum(abs, iv), maximum(abs, qv))
        acc_peak = max(acc_peak, pk)
        nclip = count(v -> abs(v) >= 2007, iv) + count(v -> abs(v) >= 2007, qv)      # 0.98 of 2048
        acc_clip += nclip / (2 * length(iv)); nclip > 0 && (clip_blocks += 1)
        acc_pow += (sum(abs2, Float64.(iv)) + sum(abs2, Float64.(qv))) / length(iv)
        acc_n += 1; acc_maxratio = max(acc_maxratio, ratio)
        nblk += 1
        if nblk % blocks_per_log == 0
            @printf(logio, "%d,%s,%.2f,%d,%.3e,%d,%.1f\n", nblk ÷ blocks_per_log, string(b.utc), sqrt(acc_pow / acc_n / 2), acc_peak,
                    acc_clip / acc_n, acc_bursts, 10log10(max(acc_maxratio, 1e-9)))
            nblk % (blocks_per_log * 10) == 0 &&
                @printf("  %5.0f s  rms %.1f counts  peak %d  clipped %.1e  bursts in the last second %d\n", nblk * bufsize / fs, sqrt(acc_pow / acc_n / 2),
                        acc_peak, acc_clip / acc_n, acc_bursts)
            acc_bursts = 0; acc_peak = 0; acc_clip = 0.0; acc_pow = 0.0; acc_n = 0; acc_maxratio = 0.0
        end
    end
    foreach(close, ios); close(logio)
    return (blocks = nblk, bursts = total_bursts, clip_blocks = clip_blocks, burst_lens = all_lens)
end

function write_json(path::String; fields::Vector{Pair{String,Any}})
    open(path, "w") do io
        println(io, "{")
        for (i, (k, v)) in enumerate(fields)
            val = v isa AbstractString ? "\"" * replace(v, "\\" => "/", "\"" => "'") * "\"" :
                  v isa Bool ? string(v) : string(v)
            println(io, "  \"", k, "\": ", val, i < length(fields) ? "," : "")
        end
        println(io, "}")
    end
end

# ---------------------------------------------------------------------
# Level check before the long capture
# ---------------------------------------------------------------------
function bt_preflight(r::Radio, cfg::Config, fs::Float64, flo::Float64)
    println("\nPreflight: 1 s level check (hold the phone where it will be during the capture)")
    for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
    nblk = max(2, round(Int, fs / r.bufsize))
    fsc = 0.0f0; st = nothing; peak = 0.0; clipf = zeros(2); pw = zeros(2)
    P = zeros(Float64, 1024); nseg = 0
    for _ in 1:nblk
        b = read_block!(r, 0)
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        peak = max(peak, s1.peak, s2.peak)
        clipf[1] += s1.clip / nblk; clipf[2] += s2.clip / nblk
        pw[1] += s1.power / nblk; pw[2] += s2.power / nblk
        for s in 1:1024:length(x1)-1023
            P .+= abs2.(fftshift(fft(x1[s:s+1023])))
            nseg += 1
        end
    end
    @printf("  RX1 rms %.1f counts, RX2 rms %.1f counts (full scale +-%d), peak %.3f FS\n", sqrt(pw[1]) * fsc, sqrt(pw[2]) * fsc, Int(fsc), peak)
    @printf("  clipped fraction: RX1 %.2e, RX2 %.2e\n", clipf[1], clipf[2])
    P ./= nseg
    fb = ((0:1023) .- 512) .* (fs / 1024)
    k = argmax(P[[1:508; 517:1024]]); k = k > 508 ? k + 8 : k           # skip the DC bins
    @printf("  strongest line outside DC: %.3f MHz (%.0f kHz from the centre), %.1f dB above the median\n",
            (flo + fb[k]) / 1e6, fb[k] / 1e3, 10log10(P[k] / median(P)))
    ok = true
    if max(clipf...) > 1e-3
        println("  WARNING: clipping. Lower the gain (gain=10 or less) or move the phone away.")
        ok = false
    end
    if sqrt(max(pw...)) * fsc < 1.5
        println("  NOTE: the input is very quiet. Raise the gain or move the phone closer; check the antenna is on the RX port being stored.")
    end
    ok && println("  Levels look usable.")
    return ok
end

# ---------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------
function bt_capture(o::BTOpts)
    flo_req = bt_freq(o)
    cfg = Config()
    cfg.fs = o.fs; cfg.f_lo = flo_req; cfg.gain1 = o.gain; cfg.gain2 = o.gain; cfg.buffer_secs = 0.1
    cfg.skip_blocks = 10      # discard the first second: it includes stale data sitting in the Pluto's buffers (about 0.8 s)
    nchan = bt_nchan(o)
    secs = o.minutes * 60
    nsegb, bufsize = block_geometry(cfg)
    need = nchan * BT_BYTES_PER_SAMPLE * o.fs * secs
    isdir(o.out_dir) || mkpath(o.out_dir)
    free = try Float64(Base.diskstat(o.out_dir).available) catch; NaN end
    @printf("Plan: %.1f min at %.2f Msps, centre %.3f MHz, gain %d dB, storing RX%s -> about %.1f GB in %s\n",
            o.minutes, o.fs / 1e6, flo_req / 1e6, o.gain, o.rx, need / 1e9, o.out_dir)
    if isfinite(free)
        @printf("Free space: %.1f GB\n", free / 1e9)
        free > 1.2 * need || error("Not enough free disk space for this capture (need about $(round(need / 1e9, digits = 1)) GB plus margin)")
    end
    occursin("OneDrive", o.out_dir) && println("WARNING: out_dir is inside OneDrive; sync will contend with the writes and may drop samples. Use a local folder.")
    Threads.nthreads() < 3 && println("WARNING: fewer than 3 Julia threads; run `julia -t 3 bt_capture.jl` or capture may fall behind.")

    r, rb = open_verified(cfg)
    fs = Float64(rb.fs1); flo = Float64(rb.lo)
    if o.preflight
        bt_preflight(r, cfg, fs, flo)
    end

    stem = "bt_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * (isempty(o.tag) ? "" : "_" * o.tag)
    paths = String[]
    (o.rx == "1" || o.rx == "both") && push!(paths, joinpath(o.out_dir, stem * "_rx1.cs16"))
    (o.rx == "2" || o.rx == "both") && push!(paths, joinpath(o.out_dir, stem * "_rx2.cs16"))
    logpath = joinpath(o.out_dir, stem * "_log.csv")
    nblk = ceil(Int, secs * fs / bufsize)
    println("\nCapturing $(nblk) blocks (", round(nblk * bufsize / fs, digits = 1), " s). Ctrl-C stops early and keeps what was written.")

    q = Channel{Block}(o.queue_depth)
    writer = Threads.@spawn bt_writer!(q, o, paths, logpath, fs, bufsize)
    bind(q, writer)
    stop = Threads.Atomic{Bool}(false)
    rows = String[]; t_put = Float64[]; n_short = Ref(0)
    t_start_utc = now(UTC); t0wall = time()
    producer = Threads.@spawn begin
        err = nothing; t_first = time(); t_last = time()
        try
            for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
            t_first = time()
            for i in 1:nblk
                stop[] && break
                backlog = Base.n_avail(q)
                t0 = time(); b = read_block!(r, (i - 1) * bufsize); t1 = time()
                put!(q, b); t2 = time()
                push!(t_put, t2 - t1)
                b.short && (n_short[] += 1)
                push!(rows, @sprintf("%d,%s,%.1f,%d,%d", i - 1, string(b.utc), 1e3 * (t1 - t0), backlog, b.short ? 1 : 0))
            end
        catch e
            err = e
        end
        t_last = time()
        (err, t_first, t_last)
    end
    cap_err = nothing; cap_wall = NaN
    try
        cap_err, tf, tl = fetch(producer); cap_wall = tl - tf
    catch e
        if e isa InterruptException
            println("\nInterrupted; closing the files.")
            stop[] = true
            cap_err, tf, tl = fetch(producer); cap_wall = tl - tf
        else
            stop[] = true; cap_err = e
        end
    end
    close_radio!(r)
    close(q)
    res = fetch(writer)
    wall = time() - t0wall
    cap_err === nothing || println("CAPTURE ERROR: ", cap_err)

    open(joinpath(o.out_dir, stem * "_blocks.csv"), "w") do io
        println(io, "block,utc_arrival,refill_ms,queue_backlog,short_read")
        foreach(l -> println(io, l), rows)
    end
    stored = res.blocks * bufsize
    sizes = [filesize(p) for p in paths]
    expected_bytes = stored * BT_BYTES_PER_SAMPLE
    write_json(joinpath(o.out_dir, stem * ".json"), fields = Pair{String,Any}[
        "software" => "bt_capture.jl (uses pluto_interferometer.jl hardware layer)",
        "start_utc" => string(t_start_utc),
        "datatype" => "ci16_le (interleaved int16 I,Q, little-endian)",
        "full_scale_counts" => 2048,
        "sample_rate_hz" => rb.fs1,
        "center_frequency_hz" => rb.lo,
        "gain_rx1_db" => rb.gain1, "gain_rx2_db" => rb.gain2,
        "rf_bandwidth_hz" => rb.bw1,
        "tracking_quadrature_rfdc_bbdc" => "$(rb.quad) $(rb.rfdc) $(rb.bbdc)",
        "stored_channels" => o.rx,
        "files" => join(basename.(paths), " ; "),
        "samples_per_file" => stored,
        "duration_s" => stored / fs,
        "block_samples" => bufsize,
        "ble_channel_request" => isfinite(o.freq) ? "custom frequency" : string(o.channel),
        "capture_wall_seconds" => round(cap_wall, digits = 2),
        "total_wall_seconds" => round(wall, digits = 2),
        "short_reads" => n_short[],
        "max_enqueue_wait_ms" => isempty(t_put) ? 0.0 : round(1e3 * maximum(t_put), digits = 1),
        "note" => "RX1/RX2 share one LO. Sample loss cannot be detected from the data itself; see blocks.csv for timing."])

    println("\nDone.")
    @printf("  stored %.1f s (%d blocks); capture took %.1f s of wall time (%.1f s including the writer finishing)\n", stored / fs, res.blocks, cap_wall, wall)
    for (p, s) in zip(paths, sizes)
        @printf("  %s  %.2f GB%s\n", p, s / 1e9, s == expected_bytes ? "" : "  (SIZE MISMATCH: expected $(expected_bytes) bytes)")
    end
    n_short[] > 0 && println("  WARNING: $(n_short[]) short read(s).")
    if !isempty(t_put) && maximum(t_put) > 0.5
        @printf("  WARNING: the capture loop waited up to %.0f ms to hand a block to the writer (the Pluto buffers about 0.8 s); samples may have been lost.\n", 1e3 * maximum(t_put))
    end
    isfinite(cap_wall) && res.blocks == nblk && abs(cap_wall - nblk * bufsize / fs) / (nblk * bufsize / fs) > 0.02 &&
        println("  WARNING: capture wall time differs from the expected duration by more than 2%; check blocks.csv for stalls.")
    @printf("  energy bursts counted (>%g dB above the median, 0.2 ms windows): %d", o.burst_db, res.bursts)
    if !isempty(res.burst_lens)
        @printf("; median length %.1f windows (%.1f ms); %.0f%% of bursts last 2 windows or fewer (<= 0.4 ms, typical of BLE packets)",
                median(res.burst_lens), median(res.burst_lens) * BURST_WIN / fs * 1e3, 100 * count(<=(2), res.burst_lens) / length(res.burst_lens))
    end
    println()
    println("  clipped blocks: ", res.clip_blocks)
    println("Metadata: ", joinpath(o.out_dir, stem * ".json"))
    return nothing
end

# ---------------------------------------------------------------------
# `look`: quick summary of a finished file (average spectrum, bursts)
# ---------------------------------------------------------------------
function bt_look(o::BTOpts)
    isempty(o.path) && error("look needs path=<file.cs16>")
    fs = o.fs
    n = filesize(o.path) ÷ BT_BYTES_PER_SAMPLE
    @printf("%s: %d samples = %.1f s at %.2f Msps\n", o.path, n, n / fs, fs / 1e6)
    nfft = 1024
    plan = plan_fft(zeros(ComplexF32, nfft, 1))
    w = Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:nfft-1) ./ nfft))
    P = zeros(Float64, nfft); nseg = 0
    nbursts = 0; lens = Int[]; peakratio = 0.0
    open(o.path, "r") do io
        chunk = 1 << 21
        done = 0
        while done < n
            m = min(chunk, n - done)
            buf = Vector{Int16}(undef, 2m); read!(io, buf)
            iv = buf[1:2:end]; qv = buf[2:2:end]
            nb, ratio, l = count_bursts(window_energies(iv, qv), o.burst_db)
            nbursts += nb; append!(lens, l); peakratio = max(peakratio, ratio)
            x = ComplexF32.(iv, qv) ./ 2048f0
            nfull = length(x) ÷ nfft
            X = plan_fft(zeros(ComplexF32, nfft, nfull)) * (reshape(x[1:nfull*nfft], nfft, nfull) .* w)
            P .+= vec(sum(abs2.(X), dims = 2)); nseg += nfull
            done += m
        end
    end
    P ./= nseg
    Ps = fftshift(P)
    fb = ((0:nfft-1) .- nfft ÷ 2) .* (fs / nfft)
    med = median(Ps)
    println("Average spectrum (1024 bins, ", round(fs / nfft, digits = 0), " Hz each), dB relative to the median bin:")
    order = sortperm(Ps; rev = true)
    shown = Int[]
    for k in order
        any(abs(k - s) < 8 for s in shown) && continue
        push!(shown, k)
        @printf("  %+8.1f kHz from centre   %5.1f dB%s\n", fb[k] / 1e3, 10log10(Ps[k] / med), abs(fb[k]) < 3e3 ? "   (near DC: LO leakage)" : "")
        length(shown) >= 6 && break
    end
    @printf("Energy bursts (>%g dB above the median): %d in %.1f s = %.1f per second; strongest burst %.1f dB above the median\n",
            o.burst_db, nbursts, n / fs, nbursts / (n / fs), 10log10(max(peakratio, 1e-9)))
    isempty(lens) || @printf("  median length %.1f ms; %.0f%% last 0.4 ms or less\n", median(lens) * BURST_WIN / fs * 1e3, 100 * count(<=(2), lens) / length(lens))
end

# ---------------------------------------------------------------------
function bt_main(args)
    if !isempty(args) && args[1] == "look"
        return bt_look(parse_bt(args[2:end]))
    end
    return bt_capture(parse_bt(args))
end

if abspath(PROGRAM_FILE) == @__FILE__
    bt_main(ARGS)
end
