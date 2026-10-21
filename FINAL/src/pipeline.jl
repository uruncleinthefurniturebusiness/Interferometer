# Capture and processing. stream! runs a capture thread and a processing thread joined by a bounded queue, so a slow
# processing step never delays the next refill (which would overflow the Pluto's buffers). capture() is the simple
# synchronous version used for short measurements.

const DiagRow = NamedTuple{(:blk, :idx, :utc, :t_ns, :backlog, :t_read_ms, :short, :qfull),
                           Tuple{Int,Int,DateTime,UInt64,Int,Float64,Bool,Bool}}

function consume!(q::Channel{Block}, cfg::Config, fs::Float64, per_int::Int, on_block, on_record)
    nsegb, bufsize = block_geometry(cfg)
    fx = FX(cfg.nfft, nsegb, cfg.window)
    blk_ms = round(Int, 1000 * bufsize / fs)
    fsc = 0.0f0; nrec = 0; nblk = 0; nshort = 0; nqfull = 0
    proc = Float64[]
    for b in q
        t0 = time_ns()
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        add_block!(fx, b, x1, x2, s1, s2, blk_ms)
        on_block === nothing || on_block(b, x1, x2)
        nblk += 1; b.short && (nshort += 1); b.qfull && (nqfull += 1)
        if fx.blocks == per_int
            rec = finalize!(fx)
            on_record === nothing || on_record(rec)
            nrec += 1
            reset!(fx)
        end
        push!(proc, (time_ns() - t0) / 1e6)
    end
    return (records = nrec, blocks = nblk, short = nshort, qfull = nqfull, fullscale = fsc, proc_ms = proc)
end

# Capture nblk blocks (or until stop[] is set) through the correlator. on_block(b, x1, x2) and on_record(rec) run on
# the processing thread. Returns the processing summary, the per-block capture diagnostics and any capture error.
function stream!(r, cfg::Config, fs::Float64, nblk::Int; on_block = nothing, on_record = nothing,
                 stop = Threads.Atomic{Bool}(false), skip::Bool = true)
    _, bufsize = block_geometry(cfg)
    per_int = max(1, round(Int, cfg.t_int * fs / bufsize))
    q = Channel{Block}(cfg.queue_depth)
    diag = DiagRow[]
    consumer = Threads.@spawn consume!(q, cfg, fs, per_int, on_block, on_record)
    bind(q, consumer)
    producer = Threads.@spawn begin
        err = nothing
        try
            skip && flush!(r)
            for i in 1:nblk
                stop[] && break
                bl = Base.n_avail(q)
                t0 = time_ns()
                b = read_block!(r, (i - 1) * bufsize; qfull = bl >= cfg.queue_depth)
                push!(diag, (blk = i, idx = b.idx, utc = b.utc, t_ns = b.t_ns, backlog = bl,
                             t_read_ms = (time_ns() - t0) / 1e6, short = b.short, qfull = b.qfull))
                put!(q, b)
            end
        catch e
            err = e
        end
        err
    end
    cap_err = nothing
    try
        cap_err = fetch(producer)
    catch e
        if e isa InterruptException
            println("\nInterrupted; finishing the integration in progress...")
            stop[] = true
            cap_err = fetch(producer)
        else
            stop[] = true
            cap_err = e
        end
    end
    close(q)
    summ = fetch(consumer)
    return (summ..., diag = diag, err = cap_err)
end

# Synchronous capture of nblk blocks into one record (after flushing stale data). With tone=true it also fits the
# strongest tone in each block: (RX1 frequency, RX2 frequency, RX1-RX2 phase in deg).
function capture(r, cfg::Config, fs::Float64, nblk::Int; tone::Bool = false, flush::Bool = true)
    flush && flush!(r)
    nsegb, bufsize = block_geometry(cfg)
    fx = FX(cfg.nfft, nsegb, cfg.window)
    blk_ms = round(Int, 1000 * bufsize / fs)
    fsc = 0.0f0; pk1 = 0.0; pk2 = 0.0
    tones = NTuple{3,Float64}[]
    for i in 1:nblk
        b = read_block!(r, (i - 1) * bufsize)
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        pk1 = max(pk1, s1.peak); pk2 = max(pk2, s2.peak)
        add_block!(fx, b, x1, x2, s1, s2, blk_ms)
        if tone
            f1, p1 = tone_fit(x1, fs); f2, p2 = tone_fit(x2, fs)
            push!(tones, (f1, f2, rad2deg(wrap_pi(p1 - p2))))
        end
    end
    return (rec = finalize!(fx), fullscale = fsc, peak1 = pk1, peak2 = pk2, tones = tones)
end

blocks_for(cfg::Config, fs::Float64, secs::Real) = max(1, ceil(Int, secs * fs / block_geometry(cfg)[2]))
