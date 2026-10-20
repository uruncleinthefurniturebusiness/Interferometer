# =====================================================================
# observe.jl -- main program for the two-dish interferometer (ADALM-Pluto, both RX chains on one LO)
#
# Start it with the Pluto plugged in:
#     julia -t 4 observe.jl [key=value ...]
# It connects, writes and reads back every setting, and gives a prompt (type `help`). Or run one command and exit:
#     julia -t 4 observe.jl observe hours=6 baseline_m=19.2 baseline_az_deg=35 tag=sun1
#     julia observe.jl predict hours=12 baseline_m=19.2 baseline_az_deg=35
# Options: ObsCfg below, and Config in pluto_interferometer.jl (f_lo, fs, gain1, gain2, nfft, t_int, out_dir, save_raw, ...).
#
# `observe` captures both channels, correlates them with pluto_interferometer.jl's FX correlator and writes OBS_<time>_<tag>/ in out_dir:
#   obs.vis, obs.meta   visibility records in V1 format (`v1 show path=...` reads them)
#   obs.csv             one row per record: power, flags, band coherence, timing lag, Sun position, (u,v,w), predicted
#                       delay, and the fringe-stopped visibility: the measured visibility with the predicted geometric
#                       phase removed, so with the right geometry only the slowly varying instrument phase is left
#   obs.log             everything printed
#   rx1.cf32, rx2.cf32  raw samples, only with save_raw=true (about 40 MB/s)
# It runs until `hours` have passed or Ctrl-C, which finishes the current integration and closes the files.
#
# Convention as in pluto_interferometer.jl: V = x1*conj(x2); positive phase = the signal reaches RX2 later. The geometry is pluto_interferometer.jl's
# sim_geometry, so a measured fringe can be compared directly with `pluto_interferometer.jl sim`.
# =====================================================================

const V1_FILE = let a = joinpath(@__DIR__, "pluto_interferometer.jl"), b = joinpath(@__DIR__, "Previous", "pluto_interferometer.jl")
    isfile(a) ? a : b
end
include(V1_FILE)     # Config, open_verified, read_block!, consume!, summarize, write_record, sim_geometry, model_vis, ...

const OBS_VERSION = "observe 2.0"
const LATE_RMS_COUNTS = 2.4    # rms of the Pluto's own late-stage noise, flat from gain 0 to 40 dB (Experimental Results, 18 Oct, section 7)
const STEP_MS = 50.0           # timing step that suggests a lost block (one block is about 100 ms at the default settings)

Base.@kwdef mutable struct ObsCfg
    hours::Float64           = 1.0      # observing time
    target::String           = "sun"    # sun | none (none: no geometry, e.g. a blank-sky or internal-correlation run)
    lat_deg::Float64         = -33.9    # site latitude (approximate, as in pluto_interferometer.jl; set from a GPS reading)
    lon_deg::Float64         = 18.46    # site longitude, east positive (approximate; set from a GPS reading)
    baseline_m::Float64      = NaN      # dish separation, centre to centre (required for target=sun)
    baseline_az_deg::Float64 = NaN      # baseline azimuth, RX1 dish to RX2 dish, from north through east (required)
    cable_delay_ns::Float64  = 0.0      # extra delay of RX2 relative to RX1 from cables (measure with `v1 tone` and `v1 delayfit`)
    diam_arcmin::Float64     = 32.0     # Sun diameter for the predicted disk factor (placeholder, as in pluto_interferometer.jl)
    print_every::Int         = 10       # print one line every N records
    min_free_gb::Float64     = 5.0      # refuse to start unless this much disk space would be left at the end
    force::Bool              = false    # observe even if the preflight finds a fatal problem
end

# Changing any of these reconfigures the Pluto.
const RADIO_KEYS = (:addr, :f_lo, :fs, :gain1, :gain2, :rx_port, :bw_ratio, :buffer_secs, :kernel_buffers, :nfft,
                    :quad_track, :rf_dc_track, :bb_dc_track, :strict)

# Set key=value options on whichever of the two structs has the field. Returns the fields changed.
function apply_kv!(cfg::Config, oc::ObsCfg, kvs)
    changed = Symbol[]
    for a in kvs
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        obj = hasfield(ObsCfg, sym) ? oc : hasfield(Config, sym) ? cfg : error("Unknown option '$k'")
        T = fieldtype(typeof(obj), sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(obj, sym, val)
        push!(changed, sym)
    end
    oc.target in ("sun", "none") || error("target must be sun or none")
    return changed
end

need_baseline(oc::ObsCfg) = isfinite(oc.baseline_m) && isfinite(oc.baseline_az_deg) ||
    error("target=sun needs the surveyed baseline: baseline_m= and baseline_az_deg= (or set target=none)")

# ---------------------------------------------------------------------
# Sun position and baseline geometry
# ---------------------------------------------------------------------
# Low-precision solar coordinates (Astronomical Almanac formulas, about 0.01 deg from 1950 to 2050). On a 19 m baseline
# at 1420 MHz one fringe is about 0.63 deg, so 0.01 deg is about 6 deg of fringe phase, nearly constant over a day; it
# mostly joins the instrument phase. Refraction is ignored.
function sun_radec(t::DateTime)
    n = datetime2julian(t) - 2451545.0
    L = mod(280.460 + 0.9856474 * n, 360.0)
    g = deg2rad(mod(357.528 + 0.9856003 * n, 360.0))
    lam = deg2rad(L + 1.915 * sin(g) + 0.020 * sin(2g))
    eps = deg2rad(23.439 - 4.0e-7 * n)
    return mod(atan(cos(eps) * sin(lam), cos(lam)), 2pi), asin(sin(eps) * sin(lam))
end

# Greenwich mean sidereal time (rad) from UT, to about 0.1 s over this century.
gmst_rad(t::DateTime) = deg2rad(mod(280.46061837 + 360.98564736629 * (datetime2julian(t) - 2451545.0), 360.0))

# A copy of the configuration with V1's sim_* geometry fields set from the site and baseline, so that sim_geometry and
# model_vis are reused unchanged (same frame, same sign convention).
function geometry_cfg(cfg::Config, oc::ObsCfg, f_lo::Float64)
    g = deepcopy(cfg)
    g.f_lo = f_lo; g.sim_lat_deg = oc.lat_deg
    g.sim_baseline_m = isfinite(oc.baseline_m) ? oc.baseline_m : 0.0
    g.sim_az_deg = isfinite(oc.baseline_az_deg) ? oc.baseline_az_deg : 0.0
    g.sim_model = "disk"; g.sim_diam_arcmin = oc.diam_arcmin
    return g
end

# The Sun at time t: hour angle, declination, altitude, (u,v,w) in wavelengths, geometric delay of RX2 (s), fringe rate
# (Hz) and the predicted disk factor (negative past the first null).
function sun_geometry(gcfg::Config, oc::ObsCfg, t::DateTime)
    ra, dec = sun_radec(t)
    H = mod(gmst_rad(t) + deg2rad(oc.lon_deg) - ra + pi, 2pi) - pi
    gcfg.sim_dec_deg = rad2deg(dec); gcfg.sim_start_ha_h = rad2deg(H) / 15
    g0 = sim_geometry(gcfg, 0.0); g1 = sim_geometry(gcfg, 1.0)
    q = hypot(g0.u, g0.v)
    return (H = H, dec = dec, alt = g0.alt, u = g0.u, v = g0.v, w = g0.w, tau = g0.tau, fringe_hz = g1.w - g0.w,
            q = q, disk = model_vis(gcfg, q))
end

# ---------------------------------------------------------------------
# One observation: what happens to each record the correlator produces
# ---------------------------------------------------------------------
const CSV_HEADER = "rec,utc_start,utc_end,flags,pow1,pow2,clip1,clip2,rho_abs,rho_deg,rho_dc_abs,rho_dc_deg,lag_ms," *
                   "ha_deg,dec_deg,alt_deg,u,v,w,tau_ns,fringe_mhz,rho_fs_abs,rho_fs_deg,disk_pred"

mutable struct Obs
    cfg::Config
    oc::ObsCfg
    gcfg::Config
    f_lo::Float64
    fb::Vector{Float64}       # baseband frequency of each bin, fftshifted order (Hz)
    keep::BitVector           # bins used in band averages (DC bins excluded)
    rec_secs::Float64         # data seconds per record
    vis::IOStream
    csv::IOStream
    log::IOStream
    nrec::Int
    nflag::Int
    t0::Float64
    lags::Vector{Float64}
end

function Obs(cfg::Config, oc::ObsCfg, fs::Float64, f_lo::Float64, dir::String)
    _, bufsize = block_geometry(cfg)
    per_int = max(1, round(Int, cfg.t_int * fs / bufsize))
    n = cfg.nfft; c = n ÷ 2 + 1
    keep = trues(n); keep[max(1, c - cfg.dc_mask):min(n, c + cfg.dc_mask)] .= false
    csv = open(joinpath(dir, "obs.csv"), "w"); println(csv, CSV_HEADER)
    return Obs(cfg, oc, geometry_cfg(cfg, oc, f_lo), f_lo, collect(((0:n-1) .- n ÷ 2) .* (fs / n)), keep,
               per_int * bufsize / fs, open(joinpath(dir, "obs.vis"), "w"), csv, open(joinpath(dir, "obs.log"), "w"),
               0, 0, NaN, Float64[])
end

say(o::Obs, msg::AbstractString) = (println(msg); println(o.log, msg); flush(o.log))

# The same closure type in the warm-up and the real run, so consume! is compiled only once.
handler(o::Obs) = rec -> on_record!(o, rec)

function on_record!(o::Obs, rec::Record)
    o.nrec += 1
    write_record(o.vis, rec); flush(o.vis)
    sm = summarize(rec, o.cfg.dc_mask)
    # Timing: host time minus data time. A lost block adds a lasting step of one block length.
    o.nrec == 1 && (o.t0 = rec.t_end)
    lag = 1e3 * ((rec.t_end - o.t0) - (o.nrec - 1) * o.rec_secs)
    step = o.nrec > 1 && lag - o.lags[end] > STEP_MS
    push!(o.lags, lag)
    rec.flags != 0 && (o.nflag += 1)
    geo = ",,,,,,,,,,"; extra = ""
    if o.oc.target == "sun"
        g = sun_geometry(o.gcfg, o.oc, unix2datetime(0.5 * (rec.t_start + rec.t_end)))
        k = o.keep
        ph = 2pi .* (o.f_lo .+ o.fb[k]) .* (g.tau + 1e-9 * o.oc.cable_delay_ns)    # predicted phase per bin
        rho_fs = sum(ComplexF64.(rec.V[k]) .* cis.(-ph)) / sqrt(sum(Float64, rec.P1[k]) * sum(Float64, rec.P2[k]))
        geo = @sprintf("%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.6f,%.3f,%.5f", rad2deg(g.H), rad2deg(g.dec), rad2deg(g.alt),
                       g.u, g.v, g.w, 1e9 * g.tau, 1e3 * g.fringe_hz, abs(rho_fs), rad2deg(angle(rho_fs)), g.disk)
        extra = @sprintf("  | alt %4.1f  fringe %4.0f s  stopped |rho| %.4f ph %6.1f",
                         rad2deg(g.alt), 1 / max(abs(g.fringe_hz), 1e-9), abs(rho_fs), rad2deg(angle(rho_fs)))
    end
    @printf(o.csv, "%d,%s,%s,%d,%.4e,%.4e,%.3e,%.3e,%.6f,%.3f,%.6f,%.3f,%.2f,%s\n", o.nrec,
            Dates.format(unix2datetime(rec.t_start), "yyyy-mm-ddTHH:MM:SS.sss"), Dates.format(unix2datetime(rec.t_end), "yyyy-mm-ddTHH:MM:SS.sss"),
            rec.flags, rec.pow1, rec.pow2, rec.clip1, rec.clip2, abs(sm.rho), rad2deg(angle(sm.rho)),
            abs(sm.rho_dc), rad2deg(angle(sm.rho_dc)), lag, geo)
    flush(o.csv)
    step && say(o, @sprintf("  WARNING record %d: timing step %+.0f ms (a lost block adds about 100 ms; USB latency can also do this)",
                            o.nrec, lag - o.lags[end-1]))
    if o.oc.print_every > 0 && (o.nrec - 1) % o.oc.print_every == 0
        say(o, @sprintf("%s  rec %5d  |rho| %.4f ph %6.1f  P1 %.2e P2 %.2e  lag %+5.0f ms  %s%s",
                        Dates.format(unix2datetime(rec.t_start), "HH:MM:SS"), o.nrec, abs(sm.rho), rad2deg(angle(sm.rho)),
                        rec.pow1, rec.pow2, lag, flagstr(rec.flags), extra))
    end
    return nothing
end

# Persistent timing steps: the median lag of the next 5 records against the previous 5, so a single-record USB latency
# spike that recovers is ignored. Steps in the last few records are reported even though they cannot be shown to persist.
function lag_steps(lags::Vector{Float64})
    n = length(lags); steps = Tuple{Int,Float64}[]; i = 2
    while i <= n
        d = median(lags[i:min(i + 4, n)]) - median(lags[max(1, i - 5):i-1])
        abs(d) > STEP_MS ? (push!(steps, (i, d)); i += 5) : (i += 1)
    end
    return steps
end

# Run the whole record path once on synthetic data in a throwaway folder. Code compiled during an observation stalls the
# capture; this is the suspected cause of the queue-full events at record 4 in every run of 18 Oct (Experimental Results,
# section 9), so if they disappear, that confirms it.
function warmup!(cfg::Config, oc::ObsCfg)
    t = time(); tmp = mktempdir()
    o = Obs(cfg, oc, cfg.fs, cfg.f_lo, tmp)
    _, bufsize = block_geometry(cfg)
    x1, x2, _, _ = synth_stream(round(Int, 2 * o.rec_secs * cfg.fs / bufsize) * bufsize, cfg.fs, cfg.f_lo, 20.0, 10.0, 0.0)
    blocks = to_blocks(x1, x2, bufsize)
    q = Channel{Block}(length(blocks)); foreach(b -> put!(q, b), blocks); close(q)
    redirect_stdout(devnull) do
        consume!(q, cfg, cfg.fs, handler(o))
    end
    lag_steps(collect(1.0:10.0))
    close(o.vis); close(o.csv); close(o.log); rm(tmp; recursive = true, force = true)
    @printf("Warm-up: processing compiled in %.1f s\n", time() - t)
end

# ---------------------------------------------------------------------
# Preflight: levels, clipping and coherence from a few blocks, disk space, and the Sun now
# ---------------------------------------------------------------------
function preflight(r, rb, cfg::Config, oc::ObsCfg)
    nsegb, bufsize = block_geometry(cfg)
    for _ in 1:max(cfg.skip_blocks, cfg.kernel_buffers + 1); read_block!(r, 0); end     # discard stale kernel buffers
    fx = FX(cfg.nfft, nsegb, cfg.window); fsc = 0.0f0
    pw = zeros(2); clip = zeros(2)
    for i in 1:5
        b = read_block!(r, (i - 1) * bufsize)
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        pw .+= (s1.power, s2.power) ./ 5; clip .= max.(clip, (s1.clip, s2.clip))
        accumulate!(fx, x1, x2); fx.blocks += 1
    end
    sm = summarize(finalize!(fx), cfg.dc_mask)
    problems = String[]
    println("\nPreflight (5 blocks):")
    for ch in 1:2
        rms = sqrt(pw[ch]) * fsc
        @printf("  RX%d  rms %.1f counts, clipped fraction %.1e; the Pluto's own late-stage noise is about %.0f%% of this power\n",
                ch, rms, clip[ch], 100 * min((LATE_RMS_COUNTS / rms)^2, 1.0))
        pw[ch] < LOW_POWER && push!(problems, "RX$ch has almost no power: nothing connected, or the LNA is not powered")
        clip[ch] > 1e-3 && push!(problems, "RX$ch clips: lower gain$ch")
        (LATE_RMS_COUNTS / rms)^2 > 0.1 && println("  NOTE: on RX$ch the Pluto's own noise dilutes |rho| by more than 10%; raise gain$ch if the levels allow")
    end
    abs(10log10(pw[1] / pw[2])) > 10 && println("  NOTE: RX1 and RX2 power differ by more than 10 dB: check the LNA supplies, cables and gains")
    @printf("  band |rho| %.4f, phase %.1f deg (DC bins excluded)\n", abs(sm.rho), rad2deg(angle(sm.rho)))
    fs = Float64(rb.fs1)
    need_gb = oc.hours * 3600 * ((40 + 16 * cfg.nfft) / cfg.t_int + (cfg.save_raw ? 16 * fs : 0.0)) / 1e9
    d = cfg.out_dir; while !isdir(d) && dirname(d) != d; d = dirname(d); end
    free_gb = diskstat(d).available / 1e9
    @printf("  disk: %.2f h needs about %.1f GB; %.1f GB free\n", oc.hours, need_gb, free_gb)
    free_gb - need_gb < oc.min_free_gb && push!(problems, "not enough disk space")
    if oc.target == "sun" && isfinite(oc.baseline_m) && isfinite(oc.baseline_az_deg)
        g = sun_geometry(geometry_cfg(cfg, oc, Float64(rb.lo)), oc, now(UTC))
        @printf("  Sun now: altitude %.1f deg, hour angle %.1f deg, fringe period %.0f s, predicted disk factor %.3f\n",
                rad2deg(g.alt), rad2deg(g.H), 1 / max(abs(g.fringe_hz), 1e-9), g.disk)
    end
    foreach(p -> println("  PROBLEM: ", p), problems)
    isempty(problems) && println("  preflight OK")
    return isempty(problems)
end

# ---------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------
mutable struct Console
    cfg::Config
    oc::ObsCfg
    r::Any          # V1 Radio, or nothing when disconnected
    rb::Any
    warm::Bool
end

const HELP = """
Commands (options are key=value, the same names as on the command line):
  status                     connection, settings and free disk space
  set [key=value ...]        change settings (radio settings reconfigure the Pluto); with no arguments, list them
  check                      levels, clipping, coherence and disk space from a few blocks
  observe [key=value ...]    observe for `hours` (Ctrl-C stops early), e.g. observe hours=6 tag=sun1
  predict [key=value ...]    Sun track and fringe period from now, every 15 min (no capture)
  v1 <mode> [key=value ...]  run a pluto_interferometer.jl mode: check rate tone run delayfit show sim selftest
  connect, disconnect        open or release the Pluto (disconnect before unplugging it)
  quit"""

function connect!(c::Console)
    c.r === nothing || return true
    try
        c.r, c.rb = open_verified(c.cfg)
        println("Pluto ready.")
        return true
    catch e
        println("Could not open the Pluto: ", sprint(showerror, e), "\nType `connect` to try again.")
        return false
    end
end

disconnect!(c::Console) = (c.r === nothing || (close_radio!(c.r); c.r = nothing; c.rb = nothing); nothing)

need_radio(c::Console) = c.r !== nothing || connect!(c) || error("not connected to the Pluto")

function observe!(c::Console)
    cfg, oc = c.cfg, c.oc
    oc.target == "sun" && need_baseline(oc)
    c.warm || (warmup!(cfg, oc); c.warm = true)
    need_radio(c)
    r, rb = c.r, c.rb
    preflight(r, rb, cfg, oc) || oc.force || (println("Stopped by the preflight; fix the problem, or set force=true."); return)
    fs = Float64(rb.fs1); _, bufsize = block_geometry(cfg)
    dir = joinpath(cfg.out_dir, "OBS_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * (isempty(cfg.tag) ? "" : "_" * cfg.tag))
    mkpath(dir)
    o = Obs(cfg, oc, fs, Float64(rb.lo), dir)
    raw = cfg.save_raw ? (open(joinpath(dir, "rx1.cf32"), "w"), open(joinpath(dir, "rx2.cf32"), "w")) : nothing
    nblk = ceil(Int, oc.hours * 3600 * fs / bufsize)
    say(o, @sprintf("%s: observing %s for %.2f h, %.2f s per record, into %s. Ctrl-C stops.", OBS_VERSION,
                    oc.target == "sun" ? "the Sun" : "(no target)", oc.hours, o.rec_secs, dir))

    q = Channel{Block}(cfg.queue_depth)
    consumer = Threads.@spawn consume!(q, cfg, fs, handler(o); raw_ios = raw)
    bind(q, consumer)
    stop = Threads.Atomic{Bool}(false)
    producer = Threads.@spawn begin
        try
            # The kernel buffers filled while nothing was reading, possibly across a gap: discard them all.
            for _ in 1:max(cfg.skip_blocks, cfg.kernel_buffers + 1); read_block!(r, 0); end
            for i in 1:nblk
                stop[] && break
                put!(q, read_block!(r, (i - 1) * bufsize; qfull = Base.n_avail(q) >= cfg.queue_depth))
            end
            nothing
        catch e
            e
        end
    end
    t_wall = time()
    err = try
        fetch(producer)
    catch e
        e isa InterruptException || rethrow()
        say(o, "\nStopping: finishing the current integration...")
        stop[] = true
        fetch(producer)
    end
    close(q)
    summary = fetch(consumer)
    raw === nothing || foreach(close, raw)
    close(o.vis); close(o.csv)
    extra = Dict{String,Any}("software_obs" => OBS_VERSION, "wall_seconds" => round(time() - t_wall, digits = 2))
    foreach(f -> extra["obs_" * String(f)] = getfield(oc, f), fieldnames(ObsCfg))
    write_meta(joinpath(dir, "obs.meta"), cfg, rb, summary, extra)

    err === nothing || say(o, "CAPTURE ERROR: $(sprint(showerror, err)) (everything before it has been saved)")
    say(o, @sprintf("Done: %d records (%d flagged), %.1f s of data in %.1f s.", summary.records, o.nflag,
                    summary.blocks * bufsize / fs, time() - t_wall))
    summary.short > 0 && say(o, "WARNING: $(summary.short) short read(s).")
    summary.qfull > 0 && say(o, "WARNING: the processing queue was full $(summary.qfull) time(s); blocks may have been lost.")
    steps = lag_steps(o.lags)
    say(o, isempty(steps) ? "Timing: no lasting step (no evidence of a lost block)." :
           "Timing: lasting step(s) at " * join([@sprintf("record %d (%+.0f ms)", i, d) for (i, d) in steps], ", "))
    close(o.log)
end

function predict(c::Console)
    oc = c.oc; need_baseline(oc)
    g = geometry_cfg(c.cfg, oc, c.rb === nothing ? c.cfg.f_lo : Float64(c.rb.lo))
    t0 = now(UTC)
    @printf("Sun from %s UTC; baseline %.2f m at %.1f deg; site %.3f, %.3f; diameter %.1f arcmin\n",
            Dates.format(t0, "yyyy-mm-dd HH:MM"), oc.baseline_m, oc.baseline_az_deg, oc.lat_deg, oc.lon_deg, oc.diam_arcmin)
    println("  UTC    HA(deg)  dec(deg)  alt(deg)  |u,v|(wl)  fringe period(s)  disk factor")
    for m in 0:15:round(Int, 60 * oc.hours)
        t = t0 + Minute(m); s = sun_geometry(g, oc, t)
        @printf("  %s  %7.1f  %8.2f  %8.1f  %9.1f  %16.0f  %11.3f\n", Dates.format(t, "HH:MM"), rad2deg(s.H), rad2deg(s.dec),
                rad2deg(s.alt), s.q, 1 / max(abs(s.fringe_hz), 1e-9), s.disk)
    end
end

function status(c::Console)
    rb = c.rb
    println(rb === nothing ? "Pluto: not connected" :
            @sprintf("Pluto: %s; LO %d Hz, fs %d Hz, gains %.0f/%.0f dB", c.r.uri, rb.lo, rb.fs1, rb.gain1, rb.gain2))
    @printf("Target %s, baseline %s m at %s deg, site %.3f, %.3f, cable delay %.2f ns; t_int %.2f s, nfft %d\n",
            c.oc.target, c.oc.baseline_m, c.oc.baseline_az_deg, c.oc.lat_deg, c.oc.lon_deg, c.oc.cable_delay_ns, c.cfg.t_int, c.cfg.nfft)
    d = c.cfg.out_dir; while !isdir(d) && dirname(d) != d; d = dirname(d); end
    @printf("out_dir %s, %.1f GB free\n", c.cfg.out_dir, diskstat(d).available / 1e9)
end

# Returns false when the console should exit.
function run_command!(c::Console, line::AbstractString)
    w = String.(split(strip(line)))
    isempty(w) && return true
    cmd = lowercase(w[1]); args = w[2:end]
    if cmd in ("help", "?")
        println(HELP)
    elseif cmd == "status"
        status(c)
    elseif cmd == "set"
        if isempty(args)
            foreach(f -> println("  ", f, " = ", getfield(c.oc, f)), fieldnames(ObsCfg))
            foreach(f -> println("  ", f, " = ", getfield(c.cfg, f)), (:addr, :f_lo, :fs, :gain1, :gain2, :t_int, :nfft, :out_dir, :tag, :save_raw))
        elseif any(in(RADIO_KEYS), apply_kv!(c.cfg, c.oc, args)) && c.r !== nothing
            println("Reconfiguring the Pluto..."); disconnect!(c); connect!(c)
        end
    elseif cmd == "check"
        need_radio(c); preflight(c.r, c.rb, c.cfg, c.oc)
    elseif cmd == "observe"
        isempty(args) || run_command!(c, "set " * join(args, " "))
        observe!(c)
    elseif cmd == "predict"
        isempty(args) || run_command!(c, "set " * join(args, " "))
        predict(c)
    elseif cmd == "v1"
        isempty(args) && error("usage: v1 <mode> [key=value ...]")
        modes = Dict("check" => mode_check, "rate" => mode_rate, "tone" => mode_tone, "run" => mode_run,
                     "show" => mode_show, "sim" => mode_sim, "selftest" => mode_selftest, "delayfit" => mode_delayfit)
        haskey(modes, args[1]) || error("unknown V1 mode '$(args[1])'")
        cfg2 = deepcopy(c.cfg); apply_kv!(cfg2, ObsCfg(), args[2:end])
        hw = args[1] in ("check", "rate", "tone", "run")
        was = c.r !== nothing
        hw && disconnect!(c)             # V1's hardware modes open the Pluto themselves
        try
            modes[args[1]](cfg2)
        finally
            hw && was && connect!(c)
        end
    elseif cmd == "connect"
        connect!(c)
    elseif cmd == "disconnect"
        disconnect!(c); println("Pluto released.")
    elseif cmd in ("quit", "exit", "q")
        return false
    else
        println("Unknown command '$cmd'; type `help`.")
    end
    return true
end

function run_console(c::Console)
    println("$(OBS_VERSION) (with $(V1_VERSION)), $(Threads.nthreads()) threads. Type `help` for the commands.")
    Threads.nthreads() < 3 && println("NOTE: start with `julia -t 4 observe.jl` so capture and processing have threads of their own.")
    connect!(c)
    while true
        print("interferometer> ")
        line = try
            eof(stdin) ? "quit" : readline()
        catch e
            e isa InterruptException || rethrow()
            println(); continue
        end
        keep = try
            run_command!(c, line)
        catch e
            println(e isa InterruptException ? "\nInterrupted." : "Error: " * sprint(showerror, e))
            true
        end
        keep || break
    end
    disconnect!(c)
end

function obs_main(args::Vector{String})
    Base.exit_on_sigint(false)           # Ctrl-C stops an observation cleanly instead of killing the process mid-write
    c = Console(Config(), ObsCfg(), nothing, nothing, false)
    apply_kv!(c.cfg, c.oc, filter(a -> occursin('=', a), args))
    words = filter(a -> !occursin('=', a), args)
    if isempty(words)
        run_console(c)
    else
        try
            run_command!(c, join(words, " "))
        finally
            disconnect!(c)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    obs_main(ARGS)
end
