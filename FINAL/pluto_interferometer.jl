# =====================================================================
# pluto_interferometer.jl (FINAL) -- host processing for the two-channel ADALM-Pluto interferometer
#
#     julia -t 3 pluto_interferometer.jl <mode> [key=value ...]
#
# Modes
#   selftest   no hardware: correlator against a known delay, lost-sample detection, file round trip and follower,
#              fringe stopping, Sun ephemeris, delay fit, threaded pipeline
#   check      configure, read back, capture about 0.8 s, report levels, DC spike, spectral peak and coherence
#   run        capture -> FX correlator -> .vis records (+ .meta, + .csv with one row per record). With target=sun and
#              the baseline, each row also carries the Sun geometry and the fringe-stopped visibility
#   fake       the same as run, from a stand-in radio whose RX1-RX2 phase turns once per fake_period_s (for the viewer)
#   show       one line per record of a .vis file (path=..., default the newest in out_dir)
#   sim        no hardware: a simulated observation (Sun, disk, Gaussian or point source on the baseline) through the
#              same correlator, checked against its own expectation; writes .vis, .meta and .truth.csv
#   (none)     a menu: the same modes from a prompt, with the current settings on screen; v opens viewer.jl in its own
#              process (its window survives Ctrl-C at the menu)
#
# Examples
#   julia -t 3 pluto_interferometer.jl                       (menu)
#   julia -t 3 pluto_interferometer.jl check addr=usb:1.2.5
#   julia -t 3 pluto_interferometer.jl run duration_s=3600 tag=sun1 target=sun baseline_m=20 baseline_az_deg=35
#   julia -t 3 pluto_interferometer.jl fake duration_s=600 fake_period_s=60
#   julia viewer.jl                       (in a second terminal: live fringes and spectra of the newest run)
#   julia -t 3 pluto_interferometer.jl sim baseline_m=20 baseline_az_deg=35 sim_dec_deg=-10 sim_snr_db=-20
#
# Convention: V = x1 * conj(x2); a positive phase means the signal reaches RX2 later.
# Settings: Config and GeoCfg in src/config.jl, SimCfg (sim_...) in src/sim.jl. Every run writes all radio settings and reads them back.
# =====================================================================

include(joinpath(@__DIR__, "src", "PlutoIF.jl"))
const P = PlutoIF
using Dates, Printf, Statistics, Random, FFTW

const STEP_MS = 50.0      # a persistent timing step this large suggests a lost block (one block is about 100 ms)

const USAGE = """
pluto_interferometer.jl modes: selftest | check | run | fake | show | sim   (no mode: menu)
  julia -t 3 pluto_interferometer.jl
  julia pluto_interferometer.jl selftest
  julia -t 3 pluto_interferometer.jl check [addr=usb:X.Y.Z f_lo=1420e6 fs=2.5e6 gain1=40 gain2=40]
  julia -t 3 pluto_interferometer.jl run   [duration_s=600 tag=... target=none|sun baseline_m=.. baseline_az_deg=.. cable_delay_ns=0]
  julia -t 3 pluto_interferometer.jl fake  [duration_s=600 fake_period_s=60 fake_snr_db=-3]
  julia pluto_interferometer.jl show       [path=<file>.vis]
  julia -t 3 pluto_interferometer.jl sim   baseline_m=.. baseline_az_deg=.. [sim_model=sun sim_dec_deg=0 sim_hours=6 sim_snr_db=-20 ...]
Options: Config and GeoCfg in src/config.jl, SimCfg in src/sim.jl."""

# ---------------------------------------------------------------------
# run / fake
# ---------------------------------------------------------------------
const CSV_HEAD = "rec,utc_start,utc_end,flags,pow1,pow2,clip1,clip2,rho_abs,rho_deg,rho_dc_abs,rho_dc_deg,lag_ms"
const CSV_GEO  = ",ha_deg,dec_deg,alt_deg,u,v,w,tau_ns,fringe_mhz,rho_fs_abs,rho_fs_deg,disk_pred"
const RECORD_LAYOUT = "Float64 t_start, Float64 t_end, Int32 nseg, UInt32 flags, Float32 pow1, pow2, clip1, clip2, then ComplexF32 V[nfft], Float32 P1[nfft], Float32 P2[nfft]; fftshifted (DC at nfft/2+1)"
const FLAGS_DOC = "1 clip, 2 low power, 4 short read, 8 queue full"

# Output files are <out_dir>/PI_<utc>[_tag][suffix].*
function out_base(cfg::P.Config, suffix::String)
    isdir(cfg.out_dir) || mkpath(cfg.out_dir)
    return joinpath(cfg.out_dir, "PI_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * (isempty(cfg.tag) ? "" : "_" * cfg.tag) * suffix)
end

# The first lines of every .meta file.
meta_head(cfg::P.Config, mode) = ["software" => P.SOFTWARE, "mode" => mode, "start_utc" => string(now(UTC)),
    "nfft" => cfg.nfft, "window" => cfg.window, "t_int_requested_s" => cfg.t_int, "dc_mask" => cfg.dc_mask]

function start_meta(base, cfg, geo, rb, mode)
    P.write_meta(base * ".meta", [meta_head(cfg, mode);
        "fs_hz" => rb.fs1; "f_lo_hz" => rb.lo; "rf_bandwidth_hz" => rb.bw1; "gain1_db" => rb.gain1; "gain2_db" => rb.gain2;
        "tracking_rx1_rx2" => "quadrature $(rb.quad), rf_dc $(rb.rfdc), bb_dc $(rb.bbdc)";
        "target" => geo.target; "lat_deg" => geo.lat_deg; "lon_deg" => geo.lon_deg; "baseline_m" => geo.baseline_m;
        "baseline_az_deg" => geo.baseline_az_deg; "cable_delay_ns" => geo.cable_delay_ns; "diam_arcmin" => geo.diam_arcmin;
        "convention" => P.CONVENTION; "record_layout" => RECORD_LAYOUT; "flags" => FLAGS_DOC])
end

# Everything after the radio is open: files, the per-record handler, the threaded capture, the summary.
function run_core(cfg::P.Config, geo::P.GeoCfg, r, rb; mode = "run", quiet = false)
    fs = Float64(rb.fs1); f_lo = Float64(rb.lo)
    base = out_base(cfg, mode == "fake" ? "_FAKE" : "")
    start_meta(base, cfg, geo, rb, mode)
    vis = open(base * ".vis", "w"); csv = open(base * ".csv", "w")
    sun = geo.target == "sun"
    println(csv, CSV_HEAD, sun ? CSV_GEO : "")
    keep = P.band_keep(cfg.nfft, cfg.dc_mask); fbb = P.bb_freqs(cfg.nfft, fs)
    _, bufsize = P.block_geometry(cfg)
    per_int = max(1, round(Int, cfg.t_int * fs / bufsize)); rec_secs = per_int * bufsize / fs
    nrec = Ref(0); t0 = Ref(NaN); lags = Float64[]; nflag = Ref(0)

    function on_record(rec::P.Record)
        nrec[] += 1
        P.write_record(vis, rec); flush(vis)
        sm = P.summarize(rec, cfg.dc_mask)
        nrec[] == 1 && (t0[] = rec.t_end)
        lag = 1e3 * ((rec.t_end - t0[]) - (nrec[] - 1) * rec_secs)     # host time minus data time
        push!(lags, lag)
        rec.flags != 0 && (nflag[] += 1)
        tstr(t) = Dates.format(unix2datetime(t), "yyyy-mm-ddTHH:MM:SS.sss")
        @printf(csv, "%d,%s,%s,%d,%.4e,%.4e,%.3e,%.3e,%.6f,%.3f,%.6f,%.3f,%.2f", nrec[], tstr(rec.t_start), tstr(rec.t_end),
                rec.flags, rec.pow1, rec.pow2, rec.clip1, rec.clip2, abs(sm.rho), rad2deg(angle(sm.rho)),
                abs(sm.rho_dc), rad2deg(angle(sm.rho_dc)), lag)
        extra = ""
        if sun
            g = P.sun_geometry(geo, f_lo, unix2datetime(0.5 * (rec.t_start + rec.t_end)))
            rfs = P.fringe_stop(rec, fbb, keep, f_lo, g.tau + 1e-9 * geo.cable_delay_ns)
            @printf(csv, ",%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.6f,%.3f,%.5f", rad2deg(g.H), rad2deg(g.dec), rad2deg(g.alt),
                    g.u, g.v, g.w, 1e9 * g.tau, 1e3 * g.fringe_hz, abs(rfs), rad2deg(angle(rfs)), g.disk)
            extra = @sprintf("  | alt %4.1f fringe %4.0f s stopped |rho| %.4f ph %6.1f", rad2deg(g.alt),
                             1 / max(abs(g.fringe_hz), 1e-9), abs(rfs), rad2deg(angle(rfs)))
        end
        println(csv); flush(csv)
        if !quiet && cfg.print_every > 0 && (nrec[] - 1) % cfg.print_every == 0
            @printf("%s rec %5d  |rho| %.4f ph %7.2f  P1 %.2e P2 %.2e  lag %+5.0f ms  %s%s\n",
                    Dates.format(unix2datetime(rec.t_start), "HH:MM:SS"), nrec[], abs(sm.rho), rad2deg(angle(sm.rho)),
                    rec.pow1, rec.pow2, lag, P.flagstr(rec.flags), extra)
        end
    end

    nblk = ceil(Int, cfg.duration_s * fs / bufsize)
    quiet || println("\n$mode: $nblk blocks of $(bufsize) samples, $(per_int) blocks per record ($(round(rec_secs, digits = 3)) s), ",
                     "$(cfg.nfft)-point FX ($(round(fs / cfg.nfft, digits = 1)) Hz bins)\nOutput: $base.{vis,meta,csv}\n",
                     "Live display: julia viewer.jl path=\"$(base).vis\"\n")
    t_wall = time()
    res = try
        P.stream!(r, cfg, fs, nblk; on_record)
    finally
        P.close_radio!(r); close(vis); close(csv)
    end
    wall = time() - t_wall
    steps = P.lag_steps(lags, STEP_MS)
    P.write_meta(base * ".meta", ["end_utc" => string(now(UTC)), "records" => res.records, "blocks" => res.blocks,
        "short_reads" => res.short, "queue_full_events" => res.qfull, "fullscale_counts" => res.fullscale,
        "wall_seconds" => round(wall, digits = 2), "expected_seconds" => round(res.blocks * bufsize / fs, digits = 2),
        "timing_steps" => length(steps), "flagged_records" => nflag[],
        "capture_error" => res.err === nothing ? "none" : replace(string(res.err), '\n' => ' ')]; append = true)
    if !quiet
        @printf("\nDone: %d records, %d blocks, wall %.1f s (data %.1f s). Flagged records %d, short reads %d, queue full %d, timing steps %d.\n",
                res.records, res.blocks, wall, res.blocks * bufsize / fs, nflag[], res.short, res.qfull, length(steps))
        isempty(res.proc_ms) || @printf("Processing per block: median %.1f ms, max %.1f ms (block %.0f ms); queue backlog max %d of %d.\n",
                                        median(res.proc_ms), maximum(res.proc_ms), 1e3 * bufsize / fs,
                                        isempty(res.diag) ? 0 : maximum(d.backlog for d in res.diag), cfg.queue_depth)
        res.err === nothing || println("CAPTURE ERROR: ", res.err, " (the files cover the data captured before it)")
    end
    return (base = base, records = res.records, res = res)
end

# Compile the whole record path on synthetic data first: code compiled during a capture stalls the capture threads.
function warmup(cfg::P.Config, geo::P.GeoCfg)
    w = deepcopy(cfg); w.duration_s = 2.2 * cfg.t_int; w.out_dir = mktempdir(); w.tag = "warmup"
    r, rb = P.open_fake(w; realtime = false, verbose = false)
    run_core(w, geo, r, rb; mode = "fake", quiet = true)
    return nothing
end

function mode_run(cfg::P.Config, geo::P.GeoCfg; fake::Bool = false)
    geo.target in ("none", "sun") || error("target must be none or sun")
    geo.target == "sun" && P.need_baseline(geo)
    print("Warming up (compiling the record path on synthetic data)... "); warmup(cfg, geo); println("done")
    r, rb = fake ? P.open_fake(cfg) : P.open_verified(cfg)
    return run_core(cfg, geo, r, rb; mode = fake ? "fake" : "run")
end

# ---------------------------------------------------------------------
# check
# ---------------------------------------------------------------------
function mode_check(cfg::P.Config)
    r, rb = P.open_verified(cfg)
    fs = Float64(rb.fs1)
    c = try P.capture(r, cfg, fs, 8) finally P.close_radio!(r) end
    rec = c.rec; fsc = c.fullscale; n = cfg.nfft; dc = P.dc_index(n)
    sm = P.summarize(rec, cfg.dc_mask)
    f = P.bin_freqs(n, fs, Float64(rb.lo))
    println("\nFull scale detected: +-$(Int(fsc)) counts ($(fsc == 2048.0f0 ? "12-bit, right-aligned" : "left-aligned in 16 bits"))")
    for (nm, pw, pk, cl) in (("RX1", rec.pow1, c.peak1, rec.clip1), ("RX2", rec.pow2, c.peak2, rec.clip2))
        @printf("  %s  power %.3e FS^2 (%.1f dBFS, rms %.1f counts)  peak %.3f FS  clipped %.2e\n", nm, pw, P.dbfs(pw), sqrt(pw) * fsc, pk, cl)
    end
    for (nm, Pk) in (("RX1", rec.P1), ("RX2", rec.P2))
        Pm = Float64.(Pk); Pm[.!P.band_keep(n, cfg.dc_mask)] .= -Inf
        k = argmax(Pm)
        @printf("  %s DC spike %.1f dB above the median bin; strongest other bin %.6f MHz (%+.2f kHz), %.1f dB above the median\n",
                nm, 10log10(Pk[dc] / median(Pk)), f[k] / 1e6, (f[k] - rb.lo) / 1e3, 10log10(Pk[k] / median(Pk)))
    end
    @printf("Band |rho| (DC +-%d bins out) %.4f at %.1f deg; strongest bin %.4f at %.1f deg; DC bin %.4f at %.1f deg\n",
            cfg.dc_mask, abs(sm.rho), rad2deg(angle(sm.rho)), abs(sm.rho_pk), rad2deg(angle(sm.rho_pk)), abs(sm.rho_dc), rad2deg(angle(sm.rho_dc)))
    @printf("Noise level of |rho|: about %.4f per bin, %.5f for the band (%d segments)\n", 1 / sqrt(rec.nseg), 1 / sqrt(rec.nseg * sum(P.band_keep(n, cfg.dc_mask))), rec.nseg)
    println("Flags: ", P.flagstr(rec.flags))
end

# ---------------------------------------------------------------------
# show
# ---------------------------------------------------------------------
function mode_show(cfg::P.Config)
    path = isempty(cfg.path) ? P.newest_vis(cfg.out_dir) : cfg.path
    isempty(path) && error("No .vis files in $(cfg.out_dir)")
    v = P.load_vis(path); meta = v.meta
    println("$path: $(length(v.recs)) records, nfft $(v.nfft), fs $(meta["fs_hz"]) Hz, LO $(meta["f_lo_hz"]) Hz, target $(get(meta, "target", "?"))")
    for (i, rec) in enumerate(v.recs)
        sm = P.summarize(rec, v.dc_mask)
        @printf("%5d %s  |rho| %.4f ph %7.2f  P1 %.2e P2 %.2e  %s\n", i, Dates.format(unix2datetime(rec.t_start), "HH:MM:SS"),
                abs(sm.rho), rad2deg(angle(sm.rho)), rec.pow1, rec.pow2, P.flagstr(rec.flags))
    end
end

# ---------------------------------------------------------------------
# sim (no hardware): the physics is in src/sim.jl; this writes the files and prints the records
# ---------------------------------------------------------------------
function mode_sim(cfg::P.Config, geo::P.GeoCfg, sc::P.SimCfg)
    base = out_base(cfg, "_SIM")
    P.write_meta(base * ".meta", [meta_head(cfg, "sim");
        "fs_hz" => cfg.fs; "f_lo_hz" => cfg.f_lo; "target" => "none";
        "lat_deg" => geo.lat_deg; "baseline_m" => geo.baseline_m; "baseline_az_deg" => geo.baseline_az_deg; "diam_arcmin" => geo.diam_arcmin;
        [string(f) => getfield(sc, f) for f in fieldnames(P.SimCfg)];
        "convention" => P.CONVENTION; "record_layout" => "as for run; record times are simulated, sim_step_s apart"; "flags" => FLAGS_DOC])
    vis = open(base * ".vis", "w"); csv = open(base * ".truth.csv", "w")
    println(csv, "rec,t_start_unix,hour_angle_deg,alt_deg,u,v,w,tau_ns,vnorm,beam,rho_exp_abs,rho_exp_deg,rho_abs,rho_deg,phase_err_deg")
    n = Ref(0)
    function on_result(rec, tr, rho, err, sig_ph)
        n[] += 1
        P.write_record(vis, rec); flush(vis)
        @printf(csv, "%d,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.5f,%.5f,%.6g,%.3f,%.6g,%.3f,%.3f\n", n[], rec.t_start, rad2deg(tr.H), rad2deg(tr.alt),
                tr.u, tr.v, tr.w, 1e9 * tr.tau, tr.vnorm, tr.beam, abs(tr.rho_exp), rad2deg(angle(tr.rho_exp)), abs(rho), rad2deg(angle(rho)), rad2deg(err))
        if cfg.print_every > 0 && (n[] - 1) % cfg.print_every == 0
            @printf("H %6.1f deg alt %4.1f  q %6.1f  tau %6.2f ns  |V| %.3f | exp |rho| %.4f ph %7.2f | meas |rho| %.4f ph %7.2f | err %6.2f deg (sigma %.1f)\n",
                    rad2deg(tr.H), rad2deg(tr.alt), hypot(tr.u, tr.v), 1e9 * tr.tau, tr.vnorm * tr.beam, abs(tr.rho_exp),
                    rad2deg(angle(tr.rho_exp)), abs(rho), rad2deg(angle(rho)), rad2deg(err), rad2deg(sig_ph))
        end
    end
    res = try
        P.simulate_observation(cfg, geo, sc; on_result)
    finally
        close(vis); close(csv)
    end
    P.write_meta(base * ".meta", ["end_utc" => string(now(UTC)), "records" => res.records, "sim_ta_over_ts" => res.a2,
        "sim_med_phase_err_sigma" => res.med_err_sigma, "sim_med_rho_ratio" => res.med_ratio, "sim_pass" => res.pass]; append = true)
    println("\nSim summary: $(res.records) records ($(res.skipped) record times skipped below $(sc.sim_min_alt_deg) deg altitude)")
    if isnan(res.med_err_sigma)
        println("  no record had an expected correlation above the noise (resolved out, outside the beam, or too faint)")
    else
        @printf("  median |phase error| %.2f deg = %.2f x the expected noise (0.67 expected if the noise model is right; pass below 1.2)\n",
                res.med_err_deg, res.med_err_sigma)
        println("  median |rho| ratio, measured/expected, records with SNR > 6: ",
                isnan(res.med_ratio) ? "n/a (all records below SNR 6)" : @sprintf("%.3f (pass within 0.9 to 1.1)", res.med_ratio))
    end
    println("  ", res.pass ? "PASS" : "FAIL: the correlator output disagrees with the simulation's own expectation")
    println("Output: $base.{vis,meta,truth.csv}   (julia viewer.jl path=\"$base.vis\")")
    return res.pass
end

# ---------------------------------------------------------------------
# selftest (no hardware)
# ---------------------------------------------------------------------
# Common complex noise with channel 2 delayed by tau (envelope by a phase ramp, carrier by the LO term), plus
# independent noise at snr_db below the common signal.
function synth_pair(n, fs, f_lo, tau, phi_deg, snr_db; seed = 1)
    rng = MersenneTwister(seed)
    s = randn(rng, ComplexF64, n)
    s2 = ifft(fft(s) .* cis.(-2pi .* (f_lo .+ fftfreq(n, fs)) .* tau)) .* cis(-deg2rad(phi_deg))
    npow = 10^(-snr_db / 10)
    return s .+ sqrt(npow) .* randn(rng, ComplexF64, n), s2 .+ sqrt(npow) .* randn(rng, ComplexF64, n), npow
end

function to_blocks(x1, x2, bufsize; counts_rms = 250.0)
    sc = counts_rms / std(real.(x1))
    cl(v) = Int16.(clamp.(round.(v .* sc), -2047, 2047))
    return [P.Block((i - 1) * bufsize, now(UTC), time_ns(), cl(real.(x1[r])), cl(imag.(x1[r])), cl(real.(x2[r])), cl(imag.(x2[r])), false, false)
            for (i, r) in enumerate((k * bufsize + 1):((k + 1) * bufsize) for k in 0:(length(x1) ÷ bufsize - 1))]
end

function records_from(cfg, blocks)
    recs = P.Record[]
    q = Channel{P.Block}(length(blocks)); foreach(b -> put!(q, b), blocks); close(q)
    _, bufsize = P.block_geometry(cfg)
    per_int = max(1, round(Int, cfg.t_int * cfg.fs / bufsize))
    s = P.consume!(q, cfg, cfg.fs, per_int, nothing, rec -> push!(recs, rec))
    return recs, s
end

function mode_selftest()
    cfg = P.Config(); cfg.t_int = 0.5
    _, bufsize = P.block_geometry(cfg); fs = cfg.fs; n = cfg.nfft
    ok = true
    check(name, pass) = (println("  ", pass ? "PASS" : "FAIL", "  ", name); ok &= pass)

    println("== 1. FX correlator against a known delay (20 m baseline, 15 deg off broadside, SNR 10 dB)")
    tau = 20 * sind(15) / P.C_LIGHT
    x1, x2, npow = synth_pair(10bufsize, fs, cfg.f_lo, tau, 0.0, 10.0)
    recs, s = records_from(cfg, to_blocks(x1, x2, bufsize))
    f = P.bin_freqs(n, fs, cfg.f_lo)
    errs = Float64[]; mags = Float64[]
    for rec in recs, k in (n ÷ 8):(n - n ÷ 8)
        rho = rec.V[k] / sqrt(rec.P1[k] * rec.P2[k])
        push!(errs, abs(P.wrap_pi(angle(rho) - 2pi * f[k] * tau))); push!(mags, abs(rho))
    end
    @printf("  %d records; median phase error %.3f deg (limit 2); median |rho| %.4f, expected %.4f (limit 5%%)\n",
            length(recs), rad2deg(median(errs)), median(mags), 1 / (1 + npow))
    check("phase and coherence", rad2deg(median(errs)) < 2 && abs(median(mags) * (1 + npow) - 1) < 0.05 && length(recs) == 2)

    println("== 2. Tone monitor finds a lost-sample gap, and only that one")
    fr = P.FakeRadio(cfg; realtime = false); P.start_stream!(fr, cfg)
    fr.src = :tone; fr.tone_dbfs0 = -60.0; fr.g1 = fr.g2 = 40; fr.lost_every = 6
    tm = P.ToneMon(fs, bufsize); bad = Int[]
    for i in 1:12
        b = P.read_block!(fr, 0)
        u = P.tone_update!(tm, P.to_complex(b.i1, b.q1, 2048.0f0), P.to_complex(b.i2, b.q2, 2048.0f0))
        isfinite(u.res1) && abs(u.res1) > 0.25 && abs(u.res2) > 0.25 && push!(bad, i)
    end
    println("  gaps flagged before blocks ", bad, " (7 samples skipped before blocks 6 and 12)")
    check("lost samples detected", bad == [6, 12])

    println("== 3. File round trip, and the follower reading a file that is still being written")
    path = joinpath(mktempdir(), "t.vis")
    io = open(path, "w"); P.write_record(io, recs[1]); flush(io)
    fol = P.VisFollower(path, n)
    a = length(P.poll!(fol))
    buf = IOBuffer(); P.write_record(buf, recs[2]); bytes = take!(buf)
    write(io, bytes[1:1000]); flush(io); b = length(P.poll!(fol))
    write(io, bytes[1001:end]); close(io); c2 = P.poll!(fol)
    back = P.read_records(path, n)
    check("round trip exact, follower read 1, 0, 1", a == 1 && b == 0 && length(c2) == 1 && c2[1].V == recs[2].V &&
          length(back) == 2 && back[1].V == recs[1].V && back[2].P2 == recs[2].P2)

    println("== 4. Fringe stopping: Sun on a 20 m baseline at 35 deg, instrument phase 40 deg, SNR 0 dB")
    geo = P.GeoCfg(target = "sun", baseline_m = 20.0, baseline_az_deg = 35.0)
    keep = P.band_keep(n, cfg.dc_mask); fbb = P.bb_freqs(n, fs)
    raw = Float64[]; stopped = ComplexF64[]
    for (i, m) in enumerate(0:2:10)
        t = DateTime(2026, 10, 20, 10, 30) + Minute(m)
        g = P.sun_geometry(geo, cfg.f_lo, t)
        y1, y2, _ = synth_pair(5bufsize, fs, cfg.f_lo, g.tau, 40.0, 0.0; seed = i)
        rr, _ = records_from(cfg, to_blocks(y1, y2, bufsize))
        push!(raw, rad2deg(angle(P.band_rho(rr[1], keep))))
        push!(stopped, P.fringe_stop(rr[1], fbb, keep, cfg.f_lo, g.tau))
    end
    cs = P.circ_stats(rad2deg.(angle.(stopped)))
    @printf("  raw band phase spans %.0f deg over 10 min; fringe-stopped %.2f +- %.2f deg (injected 40); |rho| %.3f (expected %.3f)\n",
            maximum(P.unwrap_deg(raw)) - minimum(P.unwrap_deg(raw)), cs.mean, cs.std, mean(abs.(stopped)), 0.5)
    check("stopped phase is the instrument phase", abs(cs.mean - 40) < 2 && cs.std < 2 && abs(mean(abs.(stopped)) - 0.5) < 0.03)

    println("== 5. Sun ephemeris at the 2024 equinox and June solstice")
    d1 = rad2deg(P.sun_radec(DateTime(2024, 3, 20, 3, 6))[2]); d2 = rad2deg(P.sun_radec(DateTime(2024, 6, 20, 20, 51))[2])
    @printf("  declination %.3f deg (0) and %.3f deg (23.44)\n", d1, d2)
    check("ephemeris within 0.05 deg", abs(d1) < 0.05 && abs(d2 - 23.44) < 0.05)

    println("== 6. Delay and phase from a cross spectrum (37.3 ns, -50 deg, phase noise 5 deg per bin)")
    rng = MersenneTwister(3)
    fk = collect(-0.7e6:610.35:0.7e6)
    rho = 0.8 .* cis.(deg2rad(-50) .+ 2pi .* fk .* 37.3e-9 .+ deg2rad(5) .* randn(rng, length(fk)))
    d = P.delay_search(fk, rho, ones(length(fk)))
    @printf("  fitted %.3f +- %.3f ns (expected scatter about 0.72 ns), %.2f deg\n", d.tau_ns, d.tau_err_ns, d.phi0_deg)
    check("delay within 3 sigma, sigma about right, phase within 1 deg",
          abs(d.tau_ns - 37.3) < 3d.tau_err_ns && 0.5 < d.tau_err_ns < 1.0 && abs(P.wrap180(d.phi0_deg + 50)) < 1)

    println("== 7. Threaded pipeline (capture thread, queue, processing thread) on the stand-in radio")
    fr = P.FakeRadio(cfg; realtime = false); P.start_stream!(fr, cfg)
    fr.sky_snr_db = 0.0; fr.sky_period_s = 1e9; fr.sky_phase_deg = -20.0
    got = P.Record[]
    res = P.stream!(fr, cfg, fs, 25; on_record = rec -> push!(got, rec))
    sm = P.summarize(got[end], cfg.dc_mask)
    @printf("  %d blocks, %d records, short %d, queue full %d; band |rho| %.3f at %.1f deg\n",
            res.blocks, res.records, res.short, res.qfull, abs(sm.rho), rad2deg(angle(sm.rho)))
    check("records, flags and phase (injected -20 deg)", res.records == 5 && res.err === nothing && abs(P.wrap180(rad2deg(angle(sm.rho)) + 20)) < 3)

    println("\n", ok ? "SELFTEST PASSED" : "SELFTEST FAILED")
    return ok
end

# ---------------------------------------------------------------------
# menu (no mode given)
# ---------------------------------------------------------------------
const MODES = ("selftest", "check", "run", "fake", "show", "sim")

const MENU_ITEMS = (
    ("1", "selftest", "check the software against known answers; no radio needed"),
    ("2", "check",    "write and read back the settings, short capture: levels, DC spike, coherence"),
    ("3", "run",      "record visibilities (.vis, .meta, .csv) for duration_s"),
    ("4", "fake",     "the same as run from the stand-in radio, to try the viewer without hardware"),
    ("5", "show",     "one line per record of the newest .vis file (or path=...)"),
    ("6", "sim",      "simulated observation through the same correlator; needs baseline_m, baseline_az_deg"))

function print_menu(cfg::P.Config, geo::P.GeoCfg)
    println("\n", "="^90)
    println("Pluto interferometer ($(P.SOFTWARE))   radio $(cfg.addr)   files in $(cfg.out_dir)")
    @printf("Requested: LO %.4f MHz, %.3f Msps, gain %d / %d dB, %d-point FX, %.1f s records, run %.0f s, target %s\n",
            cfg.f_lo / 1e6, cfg.fs / 1e6, cfg.gain1, cfg.gain2, cfg.nfft, cfg.t_int, cfg.duration_s, geo.target)
    println("(check and run write these to the Pluto and read them back before measuring)")
    println("-"^90)
    for (k, name, what) in MENU_ITEMS
        @printf("  %s  %-9s %s\n", k, name, what)
    end
    println("  v  viewer    live display of the visibility in its own window, e.g. v avg=1 smooth=4")
    println("  s  set k=v   change settings, e.g. set gain1=30 gain2=30 (any command also takes k=v)")
    println("  ?  status    every setting")
    println("  h  help ")
    println("  q  quit      exit this menu")
    println("  Enter  this menu      ")
    println("Open the viewer (v) before or during a run; it follows the newest run. Ctrl-C stops a command and returns here.")
end

# The live display in its own process (viewer.jl), so plotting never stalls the capture threads. detach() gives it its
# own process group, so Ctrl-C at the menu does not close it; its messages go to viewer.log in out_dir.
function launch_viewer(cfg::P.Config, args::Vector{String}, viewer::Ref{Any})
    if viewer[] !== nothing && process_running(viewer[])
        println("The viewer is already open. Close its window first to start it with other settings.")
        return nothing
    end
    isdir(cfg.out_dir) || mkpath(cfg.out_dir)
    log = joinpath(cfg.out_dir, "viewer.log")
    cmd = `$(Base.julia_cmd()) $(joinpath(@__DIR__, "viewer.jl")) dir=$(cfg.out_dir) $args`
    io = open(log, "w")
    viewer[] = try
        run(pipeline(detach(cmd); stdout = io, stderr = io); wait = false)
    finally
        close(io)
    end
    println("Viewer starting in its own window (the first start takes a while to compile). Messages: $log")
    return nothing
end

function print_status(cfg::P.Config, geo::P.GeoCfg, sc::P.SimCfg)
    for o in (cfg, geo, sc), f in fieldnames(typeof(o))
        @printf("  %-16s %s\n", f, getfield(o, f))
    end
end

# One command at a time: a run blocks the prompt, so no other code is compiled while the capture threads run.
function menu(cfg::P.Config, geo::P.GeoCfg, sc::P.SimCfg)
    bykey = Dict(k => name for (k, name, _) in MENU_ITEMS)
    viewer = Ref{Any}(nothing)
    print_menu(cfg, geo)
    while true
        print("\n[1-6, v, s, ?, h, q] > ")
        line = try
            eof(stdin) ? "q" : readline()
        catch e
            e isa InterruptException || rethrow()
            ""
        end
        words = String.(split(strip(line)))
        isempty(words) && (print_menu(cfg, geo); continue)
        c = lowercase(words[1])
        c in ("q", "quit", "exit") && break
        if c in ("v", "view", "viewer")              # its key=value options are the viewer's, not ours
            try
                launch_viewer(cfg, words[2:end], viewer)
            catch e
                println("ERROR: ", sprint(showerror, e))
            end
            continue
        end
        try
            P.apply_kv!(words[2:end], cfg, geo, sc)
            if c in ("s", "set")
                println(length(words) == 1 ? "Usage: set key=value ...   (? lists the keys)" : "Set " * join(words[2:end], ", ") * ". Enter shows the menu.")
            elseif c in ("?", "status")
                print_status(cfg, geo, sc)
            elseif c in ("h", "help")
                println(USAGE)
            elseif get(bykey, c, c) in MODES
                dispatch(get(bykey, c, c), cfg, geo, sc)
            else
                println("Unknown command '$c'. Enter shows the menu.")
            end
        catch e
            e isa InterruptException ? println("\nStopped; back to the menu.") : println("ERROR: ", sprint(showerror, e))
        end
    end
end

# ---------------------------------------------------------------------
function dispatch(mode::AbstractString, cfg::P.Config, geo::P.GeoCfg, sc::P.SimCfg)
    mode == "selftest" && return mode_selftest()
    mode == "check"    && return mode_check(cfg)
    mode == "run"      && return mode_run(cfg, geo)
    mode == "fake"     && return mode_run(cfg, geo; fake = true)
    mode == "show"     && return mode_show(cfg)
    mode == "sim"      && return mode_sim(cfg, geo, sc)
    error("Unknown mode '$mode'")
end

function main(args::Vector{String})
    Base.exit_on_sigint(false)
    cfg = P.Config(); geo = P.GeoCfg(); sc = P.SimCfg()
    isempty(args) && (menu(cfg, geo, sc); return)
    mode = args[1]
    P.apply_kv!(args[2:end], cfg, geo, sc)
    mode in MODES || (println(USAGE); error("Unknown mode '$mode'"))
    ok = dispatch(mode, cfg, geo, sc)
    mode in ("selftest", "sim") && ok === false && exit(1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
