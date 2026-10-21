# =====================================================================
# stress_test.jl (FINAL) -- characterises the receiving ADALM-Pluto (Rev D, two RX channels), so every receiver
# number in the report comes from a recorded measurement.
#
#     julia -t 3 stress_test.jl <test> [key=value ...]
#
# Tests (each writes CSV files and rows of summary.tsv in the results folder; `report` turns them into a LaTeX
# table and PDF figures). The Pluto+ is only the signal source, run with tx_source.jl on its own laptop; this
# script tells you what to set there and waits for Enter.
#   readback   LO, sample rate, RF bandwidth, gain and tracking settings written and read back          ~1 min, no source
#   rate       sustained sample rate delivered to the host, 2.1 to 6 Msps, no processing                ~2 min, no source
#   floor      1420 MHz, receive only: noise and internal correlation against gain, DC spike, spurs,
#              antennas removed and then connected                                                     ~3 min, no source
#   levels     869 MHz tone: gain-step accuracy, linear range, clipping point, SNR, image rejection,
#              spurious-free range, RX1/RX2 match, LO offset to the Pluto+                             ~3 min, Pluto+ tone
#   coherence  869 MHz noise: coherence of a common noise signal against the SNR prediction, inter-channel
#              delay and phase, bandpass phase ripple                                                  ~3 min, Pluto+ noise
#   soak       869 MHz tone, full threaded pipeline for soak_min: lost samples, timing steps, sample clock
#              against the host clock, inter-channel phase drift and stability, tone coherence          soak_min + 2 min
#   report     results_table.tex and fig_*.pdf from what has been measured so far
#   all        every test in the order above, then report
#   plan       print the session plan and the Pluto+ commands
#
# Examples
#   julia -t 3 stress_test.jl all addr=usb:1.2.5
#   julia -t 3 stress_test.jl levels addr=usb:1.2.5 out=C:/PlutoCaptures/STRESS_20261021
#   julia stress_test.jl report out=C:/PlutoCaptures/STRESS_20261021
#   julia -t 3 stress_test.jl all fake=true soak_min=0.5        (no hardware: exercises every code path)
#
# Over-the-air set-up for the 869 MHz tests: the two receive antennas side by side on the bar, the Pluto+ antenna at
# broadside about 2-3 m away, nobody moving near them. Over the air the measured delay and phase include the
# geometry (about zero at broadside) and reflections; there is no splitter.
# =====================================================================

include(joinpath(@__DIR__, "src", "PlutoIF.jl"))
const P = PlutoIF
using Dates, Printf, Statistics, Random, FFTW

Base.@kwdef mutable struct StressCfg
    out::String           = ""          # results folder (default <out_dir>/STRESS_<utc>)
    fake::Bool            = false       # use the stand-in radio (no hardware)
    f_rx::Float64         = 1420.0e6    # receive-only tests
    f_tx_lo::Float64      = 869.0e6     # receiver LO for the Pluto+ tests
    tone_hz::Float64      = 869.2e6     # Pluto+ tone (TX LO 300 kHz below: its leakage and image land away from the measured bins)
    tone_offset::Float64  = 300e3       # tx_source.jl offset= for the tone
    tone_atten::Float64   = -50.0       # suggested Pluto+ attenuation for the tone (dB)
    noise_atten::Float64  = -40.0       # suggested Pluto+ attenuation for the noise (dB)
    noise_bw_frac::Float64 = 0.6        # tx_source.jl bw_frac= for the noise
    rates::String         = "2.1,2.5,3.0,3.5,4.0,5.0,6.0"   # Msps
    rate_secs::Float64    = 10.0
    floor_gains::String   = "0:10:70"
    floor_secs::Float64   = 2.0
    level_gains::String   = "0:2:70"
    level_secs::Float64   = 0.5
    coh_gains::String     = "20,40"
    coh_secs::Float64     = 5.0
    soak_min::Float64     = 15.0
    soak_gain::Int        = 20
    lo_random::Int        = 20          # extra random LO requests in 1400-1430 MHz
end

mutable struct Ctx
    cfg::P.Config
    sc::StressCfg
    out::String
    run_id::String
    src::Symbol        # what the Pluto+ is doing: :none, :tone, :noise (drives the stand-in radio in fake mode)
    r::Any             # the open radio, if any
end

pf(b::Bool) = b ? "PASS" : "FAIL"
fmt(x, d = 3) = isfinite(x) ? Printf.format(Printf.Format("%.$(d)f"), x) : "n/a"

# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------
function csv_write(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join((x isa AbstractFloat ? @sprintf("%.7g", x) : string(x) for x in r), ","))
        end
    end
end

const SUMMARY_HEAD = "run_id\ttest\tlabel\tvalue\tconditions\tspec\tresult"

function summ!(ctx::Ctx, test, label, value, cond; spec = "", result = "")
    path = joinpath(ctx.out, "summary.tsv")
    isfile(path) || open(io -> println(io, SUMMARY_HEAD), path, "w")
    open(path, "a") do io
        println(io, join((ctx.run_id, test, label, value, cond, spec, result), '\t'))
    end
    @printf("  %-46s %-30s %s\n", label, value, isempty(result) ? "" : "[$result]")
end

# ---------------------------------------------------------------------
# Radio and prompts
# ---------------------------------------------------------------------
function open_rx(ctx::Ctx, f_lo, gain; realtime = false, fs = ctx.cfg.fs)
    c = deepcopy(ctx.cfg); c.f_lo = f_lo; c.gain1 = c.gain2 = round(Int, gain); c.fs = fs
    if ctx.sc.fake
        r, rb = P.open_fake(c; realtime, verbose = false)
        r.src = ctx.src; r.tone_dbfs0 = -68.0; r.noise_dbfs0 = -72.0
        r.tone_bb = ctx.sc.tone_hz - f_lo; r.noise_bw_frac = ctx.sc.noise_bw_frac
    else
        r, rb = P.open_verified(c; verbose = false)
    end
    ctx.r = r
    return r, rb, c
end

close_rx!(ctx::Ctx) = (ctx.r === nothing || P.close_radio!(ctx.r); ctx.r = nothing)

# Returns the answer: "" = go on, "s" = skip, "r" = repeat (where offered).
function ask(ctx::Ctx, msg; opts = "Enter = done, s = skip this test")
    println("\n>>> ", msg)
    ctx.sc.fake && (println("    (fake: carrying on)"); return "")
    print("    [", opts, "]: ")
    return lowercase(strip(readline()))
end

function pluto_plus!(ctx::Ctx, state::Symbol)
    sc = ctx.sc
    cmd = state == :none  ? "stop the source: Enter in the running tx_source.jl, or `julia tx_source.jl off`" :
          state == :tone  ? "julia tx_source.jl tone freq=$(sc.tone_hz) offset=$(sc.tone_offset) atten=$(sc.tone_atten)" :
                            "julia tx_source.jl noise freq=$(sc.f_tx_lo) bw_frac=$(sc.noise_bw_frac) atten=$(sc.noise_atten)"
    a = ask(ctx, "On the Pluto+ laptop: $cmd")
    a == "s" && return false
    ctx.src = state
    ctx.r isa P.FakeRadio && (ctx.r.src = state)
    return true
end

nblocks(ctx, fs, secs) = P.blocks_for(ctx.cfg, fs, secs)

# ---------------------------------------------------------------------
# readback
# ---------------------------------------------------------------------
function t_readback(ctx::Ctx)
    println("\n=== readback: settings written and read back")
    sc = ctx.sc; cfg = deepcopy(ctx.cfg); cfg.f_lo = sc.f_rx
    r = sc.fake ? P.FakeRadio(cfg; realtime = false) : P.open_radio(cfg; verbose = false)
    rows = Any[]
    try
        los = [433.92e6, 869.0e6, 1420.0e6, P.HI_REST, 2450.0e6, 3800.0e6]
        append!(los, 1400e6 .+ 30e6 .* rand(MersenneTwister(1), sc.lo_random))
        for f in los
            act = try P.set_lo!(r, f) catch; NaN end
            push!(rows, ("lo_hz", f, act, act - f))
        end
        P.set_lo!(r, sc.f_rx)
        for fs in (2.084e6, 2.1e6, 2.5e6, 3.0e6, 4.0e6, 6.0e6, 10e6, 20e6, 30.72e6)
            c = deepcopy(cfg); c.fs = fs
            rb = try P.configure!(r, c); P.read_back(r) catch; nothing end
            if rb === nothing
                push!(rows, ("fs_hz", fs, NaN, NaN))
            else
                push!(rows, ("fs_hz", fs, rb.fs1, rb.fs1 - fs)); push!(rows, ("fs_rx2_hz", fs, rb.fs2, rb.fs2 - fs))
                push!(rows, ("rf_bw_hz", fs, rb.bw1, rb.bw1 - fs))
            end
        end
        P.configure!(r, cfg)
        for g in 0:73
            g1, g2 = try P.set_gain!(r, g, g) catch; (NaN, NaN) end
            push!(rows, ("gain1_db", g, g1, g1 - g)); push!(rows, ("gain2_db", g, g2, g2 - g))
        end
        for v in (0, 1)
            c = deepcopy(cfg); c.quad_track = c.rf_dc_track = c.bb_dc_track = v
            P.configure!(r, c); rb = P.read_back(r)
            ok = all(==(v), (rb.quad..., rb.rfdc..., rb.bbdc...))
            push!(rows, ("tracking_all", v, ok ? v : -1, ok ? 0 : 1))
        end
        P.configure!(r, cfg)
    finally
        P.close_radio!(r)
    end
    csv_write(joinpath(ctx.out, "readback.csv"), ["setting", "requested", "readback", "difference"], rows)
    sel(k) = [x for x in rows if x[1] == k]
    lo = sel("lo_hz"); loerr = [abs(x[4]) for x in lo if isfinite(x[4])]
    summ!(ctx, "readback", "LO read-back error (max)", "$(fmt(maximum(loerr), 1)) Hz",
          "$(length(loerr)) requests, 433.92-3800 MHz incl. $(sc.lo_random) random in 1400-1430 MHz",
          spec = "<= 100 Hz (SP-02)", result = pf(maximum(loerr) <= 100))
    summ!(ctx, "readback", "LO requests returned exactly", "$(count(==(0), loerr)) of $(length(loerr))", "integer-Hz requests")
    fsr = [x for x in sel("fs_hz") if isfinite(x[4])]
    fe = maximum(abs(x[4]) / x[2] for x in fsr)
    summ!(ctx, "readback", "Sample-rate read-back error (max, relative)", @sprintf("%.1e", fe), "$(length(fsr)) rates, 2.084-30.72 Msps",
          spec = "<= 1e-4 (SP-02)", result = pf(fe <= 1e-4))
    rx2 = all(x[4] == 0 for x in sel("fs_rx2_hz") if isfinite(x[4]))
    summ!(ctx, "readback", "RX2 sample rate equals RX1", rx2 ? "yes" : "no", "every rate", spec = "equal (SP-02)", result = pf(rx2))
    gs = [x for x in vcat(sel("gain1_db"), sel("gain2_db")) if isfinite(x[3])]
    gmax = maximum(x[3] for x in gs); gerr = maximum(abs(x[4]) for x in gs if x[2] <= gmax)
    summ!(ctx, "readback", "Gain read-back error (max)", "$(fmt(gerr, 2)) dB", "0 to $(Int(gmax)) dB in 1 dB steps, both channels",
          spec = "<= 1 dB (SP-02)", result = pf(gerr <= 1))
    summ!(ctx, "readback", "Highest gain accepted at $(sc.f_rx / 1e6) MHz", "$(fmt(gmax, 0)) dB", "requests up to 73 dB")
    tr = sel("tracking_all")
    summ!(ctx, "readback", "Tracking loops switch on and off", all(x[4] == 0 for x in tr) ? "yes" : "no",
          "quadrature, RF DC, BB DC on both channels", result = pf(all(x[4] == 0 for x in tr)))
end

# ---------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------
function t_rate(ctx::Ctx)
    println("\n=== rate: sustained sample rate delivered to the host (no processing)")
    sc = ctx.sc; rows = Any[]
    for fsm in P.parse_list(sc.rates)
        fs = fsm * 1e6
        local r, rb, c
        try
            r, rb, c = open_rx(ctx, sc.f_rx, 40; realtime = true, fs = fs)
        catch e
            println("  $fsm Msps: not accepted ($(sprint(showerror, e)[1:min(end, 120)]))")
            push!(rows, (fs, NaN, 0, 0, NaN, NaN, NaN, 0, NaN)); continue
        end
        _, bufsize = P.block_geometry(c)
        nb = 0; nshort = 0; tmax = 0.0; t_first = 0.0; t_last = 0.0
        try
            P.flush!(r)
            t0 = time()
            while time() - t0 < sc.rate_secs
                ta = time(); b = P.read_block!(r, 0); tb = time()
                nb += 1; b.short && (nshort += 1); tmax = max(tmax, tb - ta)
                nb == 1 && (t_first = tb); t_last = tb
            end
        finally
            close_rx!(ctx)
        end
        deliv = (nb - 1) * bufsize / (t_last - t_first)
        @printf("  requested %.3f Msps (read back %.3f): delivered %.3f Msps (%.1f%%), %d blocks, %d short, longest refill %.0f ms\n",
                fsm, rb.fs1 / 1e6, deliv / 1e6, 100deliv / fs, nb, nshort, 1e3tmax)
        push!(rows, (fs, rb.fs1, bufsize, nb, t_last - t_first, deliv, deliv / fs, nshort, 1e3tmax))
    end
    csv_write(joinpath(ctx.out, "rate.csv"), ["fs_req_hz", "fs_readback_hz", "block_samples", "blocks", "seconds", "delivered_sps", "ratio", "short_reads", "max_refill_ms"], rows)
    ok = [x for x in rows if isfinite(x[7])]
    r25 = [x for x in ok if abs(x[1] - 2.5e6) < 1]
    if !isempty(r25)
        x = r25[1]
        summ!(ctx, "rate", "Delivered rate at 2.5 Msps", @sprintf("%.4f Msps (%.2f%%)", x[6] / 1e6, 100x[7]),
              "$(fmt(x[5], 1)) s, both channels, no processing", spec = ">= 98% (SP-04)", result = pf(x[7] >= 0.98 && x[8] == 0))
    end
    keep = [x for x in ok if x[7] >= 0.98 && x[8] == 0]
    summ!(ctx, "rate", "Highest sustained rate (>= 98%)", isempty(keep) ? "none" : @sprintf("%.2f Msps", maximum(x[1] for x in keep) / 1e6),
          "requested $(sc.rates) Msps")
    summ!(ctx, "rate", "Highest delivered rate", @sprintf("%.3f Msps", maximum(x[6] for x in ok) / 1e6), "any request (USB 2.0 ceiling)")
end

# ---------------------------------------------------------------------
# floor (receive only, 1420 MHz)
# ---------------------------------------------------------------------
function spur_rows(rec, fbb, keep, f_lo; thresh_db = 10.0, top = 20)
    P1 = Float64.(rec.P1); P2 = Float64.(rec.P2)
    m1 = median(P1[keep]); m2 = median(P2[keep])
    cand = [k for k in findall(keep) if 10log10(P1[k] / m1) > thresh_db || 10log10(P2[k] / m2) > thresh_db]
    sort!(cand, by = k -> -(P1[k] / m1 + P2[k] / m2))
    rho = P.rho_bins(rec)
    return [(fbb[k], f_lo + fbb[k], 10log10(P1[k] / m1), 10log10(P2[k] / m2), abs(rho[k])) for k in cand[1:min(top, end)]]
end

function t_floor(ctx::Ctx)
    println("\n=== floor: noise and internal correlation at $(ctx.sc.f_rx / 1e6) MHz, receive only")
    sc = ctx.sc; rows = Any[]; spurs = Any[]
    for state in ("off", "on")
        a = ask(ctx, state == "off" ? "Remove BOTH receive antennas (leave the RX ports open). The Pluto+ can stay off." :
                                      "Connect both receive antennas again.")
        a == "s" && continue
        r, rb, c = open_rx(ctx, sc.f_rx, 0)
        fs = Float64(rb.fs1); n = c.nfft; keep = P.band_keep(n, c.dc_mask); fbb = P.bb_freqs(n, fs); dc = P.dc_index(n)
        try
            for g in P.parse_list(sc.floor_gains)
                P.set_gain!(r, g, g)
                cp = P.capture(r, c, fs, nblocks(ctx, fs, sc.floor_secs))
                rec = cp.rec; rb_ = P.band_rho(rec, keep)
                lvl = 1 / sqrt(rec.nseg * sum(keep))
                sp = spur_rows(rec, fbb, keep, rb.lo)
                foreach(s -> push!(spurs, (state, g, s...)), sp)
                row = (state, g, rec.pow1, rec.pow2, sqrt(rec.pow1) * cp.fullscale, sqrt(rec.pow2) * cp.fullscale,
                       abs(rb_), rad2deg(angle(rb_)), lvl, 10log10(rec.P1[dc] / median(rec.P1[keep])), 10log10(rec.P2[dc] / median(rec.P2[keep])),
                       abs(rec.V[dc]) / sqrt(rec.P1[dc] * rec.P2[dc]), length(sp), 10log10(rec.pow1 / rec.pow2), rec.clip1, rec.clip2)
                push!(rows, row)
                @printf("  antennas %-3s gain %2d: rms %.2f / %.2f counts, band |rho| %.4f (noise level %.4f), DC %.1f dB, spurs %d\n",
                        state, g, row[5], row[6], row[7], lvl, row[10], row[13])
            end
        finally
            close_rx!(ctx)
        end
    end
    isempty(rows) && return
    csv_write(joinpath(ctx.out, "floor.csv"), ["antennas", "gain_db", "pow1_fs2", "pow2_fs2", "rms1_counts", "rms2_counts", "band_rho",
              "band_rho_deg", "rho_noise_level", "dc1_db", "dc2_db", "rho_dc", "n_spurs", "match_db", "clip1", "clip2"], rows)
    csv_write(joinpath(ctx.out, "floor_spurs.csv"), ["antennas", "gain_db", "f_bb_hz", "f_hz", "db1_above_median", "db2_above_median", "rho"], spurs)
    st = any(x -> x[1] == "off", rows) ? "off" : "on"
    R = [x for x in rows if x[1] == st]
    cond = st == "off" ? "antennas removed, ports open, $(sc.f_rx / 1e6) MHz" : "antennas connected (no off run), $(sc.f_rx / 1e6) MHz"
    late = median(vcat([x[5] for x in R if x[2] <= 20], [x[6] for x in R if x[2] <= 20]))
    summ!(ctx, "floor", "Late-stage noise (gain <= 20 dB)", "$(fmt(late, 2)) counts rms", cond)
    platep = median(vcat([x[3] for x in R if x[2] <= 20], [x[4] for x in R if x[2] <= 20]))
    for g in (40, 60, 70)
        x = findfirst(y -> y[2] == g, R); x === nothing && continue
        p = 0.5 * (R[x][3] + R[x][4])
        summ!(ctx, "floor", "Noise from before the gain stages at $g dB", "$(fmt(100 * max(0, 1 - platep / p), 0))% of recorded",
              "1 - P(gain <= 20)/P($g); $cond")
    end
    x40 = findfirst(y -> y[2] == 40, R)
    if x40 !== nothing
        x = R[x40]
        summ!(ctx, "floor", "Internal band |rho| at 40 dB", "$(fmt(x[7], 4)) (noise level $(fmt(x[9], 4)))", cond,
              spec = "<= 0.01 (SP-07)", result = st == "off" ? pf(x[7] <= 0.01) : "")
        summ!(ctx, "floor", "DC spike at 40 dB", "$(fmt(x[10], 1)) / $(fmt(x[11], 1)) dB above median", "RX1 / RX2, tracking on; $cond")
        summ!(ctx, "floor", "Spurs > 10 dB above median at 40 dB", string(x[13]), "list in floor_spurs.csv; $cond")
        summ!(ctx, "floor", "Noise power RX1/RX2 at 40 dB", "$(fmt(x[14], 2)) dB", cond)
    end
    summ!(ctx, "floor", "Largest internal band |rho| over gain", fmt(maximum(x[7] for x in R), 4), "gains $(sc.floor_gains); $cond")
end

# ---------------------------------------------------------------------
# levels (869 MHz tone, RX gain swept)
# ---------------------------------------------------------------------
# Bins belonging to the source: the tone, its image, the Pluto+ LO leakage and the Pluto+ image.
function source_lines(sc::StressCfg, tone_bb)
    txlo = sc.tone_hz - sc.tone_offset - sc.f_tx_lo
    return [tone_bb, -tone_bb, txlo, 2txlo - tone_bb]
end

function tone_metrics(rec, tones, fs, sc, keep)
    n = length(rec.V)
    fb = median(t[1] for t in tones)
    win = P.tone_bins(n, fs, fb); img = P.tone_bins(n, fs, -fb)
    other = copy(keep)
    for f in source_lines(sc, fb); other[P.tone_bins(n, fs, f; half = 5)] .= false; end
    nb1 = median(rec.P1[other]); nb2 = median(rec.P2[other])
    t1 = P.bins_power(rec.P1, win, nb1); t2 = P.bins_power(rec.P2, win, nb2)
    i1 = P.bins_power(rec.P1, img, nb1)
    snr1 = 10log10(max(sum(rec.P1[win]) - 7nb1, 1e-30) / (7nb1)); snr2 = 10log10(max(sum(rec.P2[win]) - 7nb2, 1e-30) / (7nb2))
    rho = abs(sum(rec.V[win])) / sqrt(sum(rec.P1[win]) * sum(rec.P2[win]))
    sfdr = 10log10(maximum(rec.P1[other]) / maximum(rec.P1[win]))
    cs = P.circ_stats([t[3] for t in tones])
    return (fb = fb, t1 = P.dbfs(t1), t2 = P.dbfs(t2), nb1 = P.dbfs(nb1 / n), nb2 = P.dbfs(nb2 / n), snr1 = snr1, snr2 = snr2,
            img_dbc = P.dbfs(i1) - P.dbfs(t1), img_limited = sum(rec.P1[img]) < 2 * 7nb1, sfdr = sfdr, rho = rho,
            dphi = cs.mean, dphi_sd = cs.std, clip1 = rec.clip1, clip2 = rec.clip2)
end

# Measure at one gain until the user accepts the level (real hardware), or once (fake).
function level_check(ctx, r, c, fs, gain; what = "tone", snr_min = 30.0)
    while true
        P.set_gain!(r, gain, gain)
        cp = P.capture(r, c, fs, 5; tone = true)
        m = tone_metrics(cp.rec, cp.tones, fs, ctx.sc, P.band_keep(c.nfft, c.dc_mask))
        @printf("  level check at gain %d: %s at %+.1f kHz, %.1f / %.1f dBFS, SNR %.1f / %.1f dB, clipped %.1e / %.1e\n",
                gain, what, m.fb / 1e3, m.t1, m.t2, m.snr1, m.snr2, m.clip1, m.clip2)
        good = min(m.snr1, m.snr2) >= snr_min && max(m.clip1, m.clip2) < 1e-3
        good || println("  wanted: SNR >= $snr_min dB on both, clipping < 1e-3. Change the Pluto+ atten= and measure again.")
        a = ask(ctx, "Level OK?"; opts = "Enter = accept, r = measure again, s = skip this test")
        a == "s" && return false
        a == "r" || return true
    end
end

function t_levels(ctx::Ctx)
    println("\n=== levels: tone at $(ctx.sc.tone_hz / 1e6) MHz, receiver gain swept")
    sc = ctx.sc
    r, rb, c = open_rx(ctx, sc.f_tx_lo, 20)
    rows = Any[]
    try
        pluto_plus!(ctx, :tone) || return
        fs = Float64(rb.fs1); keep = P.band_keep(c.nfft, c.dc_mask)
        level_check(ctx, r, c, fs, 20) || return
        for g in P.parse_list(sc.level_gains)
            P.set_gain!(r, g, g)
            cp = P.capture(r, c, fs, nblocks(ctx, fs, sc.level_secs); tone = true)
            m = tone_metrics(cp.rec, cp.tones, fs, sc, keep)
            push!(rows, (g, m.fb, m.t1, m.t2, m.nb1, m.nb2, m.snr1, m.snr2, m.img_dbc, m.img_limited ? 1 : 0, m.sfdr, m.rho, m.dphi, m.dphi_sd, m.clip1, m.clip2))
            @printf("  gain %2d: tone %6.1f / %6.1f dBFS, SNR %5.1f / %5.1f dB, image %6.1f dBc%s, |rho| %.4f, phase %7.2f, clip %.1e\n",
                    g, m.t1, m.t2, m.snr1, m.snr2, m.img_dbc, m.img_limited ? " (noise)" : "", m.rho, m.dphi, max(m.clip1, m.clip2))
        end
    finally
        close_rx!(ctx)
    end
    csv_write(joinpath(ctx.out, "levels.csv"), ["gain_db", "f_bb_hz", "tone1_dbfs", "tone2_dbfs", "noise1_dbfs_bin", "noise2_dbfs_bin", "snr1_db", "snr2_db",
              "image_dbc", "image_noise_limited", "sfdr_dbc", "rho_tone", "dphi_deg", "dphi_sd_deg", "clip1", "clip2"], rows)
    analyse_levels(ctx, rows)
end

function analyse_levels(ctx::Ctx, rows)
    sc = ctx.sc
    lin = [x for x in rows if min(x[7], x[8]) >= 20 && max(x[15], x[16]) < 1e-4]
    cond = "tone $(sc.tone_hz / 1e6) MHz over the air, LO $(sc.f_tx_lo / 1e6) MHz"
    if length(lin) < 3
        summ!(ctx, "levels", "Linear range", "too few points", "need SNR >= 20 dB and no clipping at 3+ gains; change the Pluto+ level")
        return
    end
    g = [x[1] for x in lin]
    for (ch, col) in (("RX1", 3), ("RX2", 4))
        fl = P.fit_line(g, [x[col] for x in lin])
        summ!(ctx, "levels", "Gain slope $ch", "$(fmt(fl.b, 3)) +- $(fmt(fl.se_b, 3)) dB/dB", "gains $(Int(minimum(g)))-$(Int(maximum(g))) dB; $cond", spec = "1.000")
        summ!(ctx, "levels", "Gain-step error $ch (max deviation from line)", "$(fmt(maximum(abs.(fl.res)), 2)) dB", "same range")
        if col == 3
            above = [x for x in rows if x[1] > maximum(g) && max(x[15], x[16]) <= 1e-3]     # compression below clipping only
            comp = findfirst(x -> x[3] < fl.a + fl.b * x[1] - 1, above)
            summ!(ctx, "levels", "1 dB compression (RX1)", comp === nothing ? "not reached before clipping" :
                  "gain $(Int(above[comp][1])) dB, tone $(fmt(above[comp][3], 1)) dBFS", "against the straight line; $cond")
        end
    end
    clip = findfirst(x -> max(x[15], x[16]) > 1e-3, rows)
    summ!(ctx, "levels", "Clipping onset", clip === nothing ? "not reached" :
          "gain $(Int(rows[clip][1])) dB; last clean $(fmt(rows[max(clip - 1, 1)][3], 1)) dBFS", "clip fraction > 1e-3")
    best = argmax(x -> min(x[7], x[8]), rows)
    summ!(ctx, "levels", "Largest tone SNR (7 bins, 1.8 kHz)", "$(fmt(min(best[7], best[8]), 1)) dB at gain $(Int(best[1])) dB", cond)
    nl = [x[5] for x in rows if x[1] <= 20]
    isempty(nl) || summ!(ctx, "levels", "Dynamic range (clean tone to noise per bin)",
                         "$(fmt(maximum(x[3] for x in lin) - median(x[5] for x in lin), 1)) dB", "per 610 Hz bin, linear range")
    im = [x[9] for x in lin if x[10] == 0]
    summ!(ctx, "levels", "Image rejection (RX1)", isempty(im) ? "> $(fmt(-maximum(x[9] for x in lin), 1)) dB (image below noise)" :
          "$(fmt(-median(im), 1)) dB (median)", "tone at +$(fmt(median(x[2] for x in lin) / 1e3, 1)) kHz, image at the mirror bin; quadrature tracking on")
    summ!(ctx, "levels", "Spurious-free range (RX1)", "$(fmt(-median(x[11] for x in lin), 1)) dBc (median)", "largest other bin, source lines excluded")
    summ!(ctx, "levels", "Tone power RX1 - RX2", "$(fmt(median(x[3] - x[4] for x in lin), 2)) dB", "over the air: includes the two antennas; $cond")
    summ!(ctx, "levels", "Tone |rho| (minimum, linear range)", fmt(minimum(x[12] for x in lin), 4), cond, spec = ">= 0.98 (SP-06)",
          result = pf(minimum(x[12] for x in lin) >= 0.98))
    ph = P.unwrap_deg([x[13] for x in lin])
    summ!(ctx, "levels", "RX1-RX2 phase change with gain", "$(fmt(maximum(ph) - minimum(ph), 2)) deg peak-to-peak", "gains $(Int(minimum(g)))-$(Int(maximum(g))) dB")
    off = median(x[2] for x in lin) - (sc.tone_hz - sc.f_tx_lo)
    summ!(ctx, "levels", "Tone offset (Pluto against Pluto+ reference)", "$(fmt(off, 1)) Hz = $(fmt(off / sc.tone_hz * 1e6, 2)) ppm",
          "difference of the two 40 MHz references, not an absolute LO error")
end

# ---------------------------------------------------------------------
# coherence (869 MHz noise, source off then on)
# ---------------------------------------------------------------------
function t_coherence(ctx::Ctx)
    println("\n=== coherence: common noise at $(ctx.sc.f_tx_lo / 1e6) MHz +- $(ctx.sc.noise_bw_frac * ctx.cfg.fs / 2e6) MHz")
    sc = ctx.sc; gains = P.parse_list(sc.coh_gains)
    r, rb, c = open_rx(ctx, sc.f_tx_lo, gains[1])
    fs = Float64(rb.fs1); n = c.nfft; keep = P.band_keep(n, c.dc_mask); fbb = P.bb_freqs(n, fs)
    off = Dict{Float64,P.Record}(); on = Dict{Float64,P.Record}()
    try
        pluto_plus!(ctx, :none) || return
        for g in gains
            P.set_gain!(r, g, g); off[g] = P.capture(r, c, fs, nblocks(ctx, fs, sc.coh_secs)).rec
            @printf("  source off, gain %2d: P1 %.1f dBFS, P2 %.1f dBFS\n", g, P.dbfs(off[g].pow1), P.dbfs(off[g].pow2))
        end
        pluto_plus!(ctx, :noise) || return
        while true
            P.set_gain!(r, gains[end], gains[end])
            cp = P.capture(r, c, fs, 5)
            rise1 = P.dbfs(cp.rec.pow1) - P.dbfs(off[gains[end]].pow1); rise2 = P.dbfs(cp.rec.pow2) - P.dbfs(off[gains[end]].pow2)
            @printf("  level check at gain %d: power rise %.1f / %.1f dB over source off, clipped %.1e / %.1e\n",
                    Int(gains[end]), rise1, rise2, cp.rec.clip1, cp.rec.clip2)
            println("  wanted: a rise of 6-20 dB on both channels (band SNR about 0-20 dB), and no clipping")
            a = ask(ctx, "Level OK?"; opts = "Enter = accept, r = measure again, s = skip this test")
            a == "s" && return
            a == "r" || break
        end
        for g in gains
            P.set_gain!(r, g, g); on[g] = P.capture(r, c, fs, nblocks(ctx, fs, sc.coh_secs)).rec
        end
    finally
        close_rx!(ctx)
    end
    spec_rows = Any[]
    edge = 0.9 * sc.noise_bw_frac * fs / 2
    for g in gains
        a = on[g]; b = off[g]
        S1 = Float64.(a.P1) .- Float64.(b.P1); S2 = Float64.(a.P2) .- Float64.(b.P2)
        Vs = ComplexF64.(a.V) .- ComplexF64.(b.V)
        den = sqrt.(Float64.(a.P1) .* Float64.(a.P2))
        rm = Vs ./ den; re = sqrt.(max.(S1, 0) .* max.(S2, 0)) ./ den
        band = [k for k in 1:n if keep[k] && abs(fbb[k]) <= edge && S1[k] > 0 && S2[k] > 0]
        d = P.delay_search(fbb[band], rm[band], abs2.(rm[band]))
        model = cis.(deg2rad(d.phi0_deg) .+ 2pi .* fbb .* d.tau_ns .* 1e-9)
        eta_band = abs(sum(Vs[band] .* conj.(model[band]))) / sum(sqrt.(S1[band] .* S2[band]))
        eta = median(abs.(rm[band]) ./ re[band])
        grp = 64; ng = length(band) ÷ grp
        gph = [rad2deg(angle(sum(rm[band[(j-1)*grp+1:j*grp]] .* conj.(model[band[(j-1)*grp+1:j*grp]])))) for j in 1:ng]
        ripple = sqrt(mean(abs2, gph))
        snr = median(10log10.(S1[band] ./ Float64.(b.P1[band])))
        for k in 1:4:n
            push!(spec_rows, (g, fbb[k], a.P1[k], a.P2[k], b.P1[k], b.P2[k], abs(rm[k]), rad2deg(angle(rm[k])), re[k], k in band ? 1 : 0))
        end
        cond = "gain $(Int(g)) dB, band SNR $(fmt(snr, 1)) dB per bin, $(length(band)) bins over +-$(fmt(edge / 1e3, 0)) kHz, over the air at broadside"
        summ!(ctx, "coherence", "Coherence efficiency at $(Int(g)) dB (median per bin)", fmt(eta, 3), "measured |rho| / predicted from autos; $cond", spec = "1.000")
        summ!(ctx, "coherence", "Coherence efficiency at $(Int(g)) dB (band, delay removed)", fmt(eta_band, 3), cond)
        summ!(ctx, "coherence", "Inter-channel delay at $(Int(g)) dB", "$(fmt(d.tau_ns, 2)) +- $(fmt(d.tau_err_ns, 2)) ns ($(fmt(d.tau_ns * 1e-9 * fs, 4)) samples)",
              "includes the geometry (about 0 at broadside); $cond", spec = "< 200 ns (SP-01)", result = pf(abs(d.tau_ns) < 0.5e9 / fs))
        summ!(ctx, "coherence", "Inter-channel phase at $(Int(g)) dB", "$(fmt(d.phi0_deg, 1)) deg at the LO", cond)
        summ!(ctx, "coherence", "Bandpass phase ripple at $(Int(g)) dB", "$(fmt(ripple, 2)) deg rms", "$(grp)-bin groups after the delay fit; $cond")
    end
    csv_write(joinpath(ctx.out, "coherence_spectrum.csv"), ["gain_db", "f_bb_hz", "P1_on", "P2_on", "P1_off", "P2_off", "rho_abs", "rho_deg", "rho_pred", "in_band"], spec_rows)
end

# ---------------------------------------------------------------------
# soak (869 MHz tone, threaded pipeline)
# ---------------------------------------------------------------------
function t_soak(ctx::Ctx)
    sc = ctx.sc
    println("\n=== soak: $(sc.soak_min) min through the full pipeline, tone at $(sc.tone_hz / 1e6) MHz, gain $(sc.soak_gain)")
    r, rb, c = open_rx(ctx, sc.f_tx_lo, sc.soak_gain; realtime = true)
    fs = Float64(rb.fs1); _, bufsize = P.block_geometry(c); n = c.nfft
    keep = P.band_keep(n, c.dc_mask)
    sc.fake && (r.lost_every = 100)
    tone_rows = Any[]; rec_rows = Any[]
    tm = P.ToneMon(fs, bufsize)
    local res
    try
        pluto_plus!(ctx, :tone) || return
        level_check(ctx, r, c, fs, sc.soak_gain) || return
        a = ask(ctx, "Soak for $(sc.soak_min) min: keep people and metal still near the antennas.")
        a == "s" && return
        on_block = (b, x1, x2) -> (u = P.tone_update!(tm, x1, x2); push!(tone_rows, (length(tone_rows) + 1, u.f, u.res1, u.res2, u.dphi)))
        on_record = function (rec)
            win = P.tone_bins(n, fs, tm.prev_f[1])
            rt = sum(rec.V[win]) / sqrt(sum(rec.P1[win]) * sum(rec.P2[win]))
            rbnd = P.band_rho(rec, keep)
            push!(rec_rows, (length(rec_rows) + 1, rec.t_start, rec.flags, rec.pow1, rec.pow2, abs(rt), rad2deg(angle(rt)), abs(rbnd), rec.clip1, rec.clip2))
            length(rec_rows) % 60 == 0 && @printf("  %5.1f min: tone |rho| %.4f, phase %7.2f deg, flags %s\n", length(rec_rows) / 60, abs(rt), rad2deg(angle(rt)), P.flagstr(rec.flags))
        end
        nblk = ceil(Int, sc.soak_min * 60 * fs / bufsize)
        res = P.stream!(r, c, fs, nblk; on_block, on_record)
    finally
        close_rx!(ctx)
    end
    d = res.diag; nb = min(length(d), length(tone_rows))
    nb < 20 && (println("  too few blocks for the analysis"); return)
    blk_s = bufsize / fs
    host = [(d[i].t_ns - d[1].t_ns) / 1e9 for i in 1:nb]; data = [(i - 1) * blk_s for i in 1:nb]
    s0 = min(10, nb ÷ 4) + 1
    fl = P.fit_line(data[s0:end], host[s0:end])
    lag = 1e3 .* (host .- (fl.a .+ fl.b .* data))
    steps = P.lag_steps(lag[s0:end], 50.0)
    step1 = 2pi * median(t[2] for t in tone_rows[1:nb]) / fs
    thr = 0.5 * abs(step1)
    lost = [t[1] for t in tone_rows[2:nb] if abs(t[3]) > thr && abs(t[4]) > thr]
    one = [t[1] for t in tone_rows[2:nb] if xor(abs(t[3]) > thr, abs(t[4]) > thr)]
    quiet = [t[3] for t in tone_rows[2:nb] if abs(t[3]) <= thr]
    ph = P.unwrap_deg([t[5] for t in tone_rows[1:nb]])
    tmin = data ./ 60
    pfit = P.fit_line(tmin, ph)
    fr = [t[2] for t in tone_rows[1:nb]]
    csv_write(joinpath(ctx.out, "soak_blocks.csv"), ["block", "t_data_s", "t_host_s", "lag_detrended_ms", "backlog", "refill_ms", "short", "qfull",
              "proc_ms", "tone_bb_hz", "res1_rad", "res2_rad", "dphi_deg", "dphi_unwrapped_deg"],
              [(i, data[i], host[i], lag[i], d[i].backlog, d[i].t_read_ms, Int(d[i].short), Int(d[i].qfull), i <= length(res.proc_ms) ? res.proc_ms[i] : NaN,
                fr[i], tone_rows[i][3], tone_rows[i][4], tone_rows[i][5], ph[i]) for i in 1:nb])
    csv_write(joinpath(ctx.out, "soak_records.csv"), ["record", "t_start_unix", "flags", "pow1", "pow2", "rho_tone", "rho_tone_deg", "rho_band", "clip1", "clip2"], rec_rows)
    dur = nb * blk_s
    cond = "$(fmt(dur / 60, 1)) min, $(fs / 1e6) Msps, gain $(sc.soak_gain) dB, threaded pipeline with FX"
    summ!(ctx, "soak", "Lost-sample events (both channels)", "$(length(lost)) in $(nb - 1) block joins", "one sample = $(fmt(rad2deg(step1), 1)) deg of tone; $cond",
          spec = "0", result = pf(isempty(lost) && !ctx.sc.fake))
    summ!(ctx, "soak", "Phase jumps in one channel only", string(length(one)), "would mean the channels lost alignment")
    summ!(ctx, "soak", "Tone phase noise at block joins", "$(fmt(rad2deg(std(quiet)), 3)) deg rms", "detection limit for a lost sample")
    summ!(ctx, "soak", "Short reads / queue-full events", "$(res.short) / $(res.qfull)", cond, spec = "0 (SP-04)", result = pf(res.short == 0 && res.qfull == 0))
    summ!(ctx, "soak", "Wall time against data time", @sprintf("%+.3f%%", 100 * (fl.b - 1)), cond, spec = "within 2% (SP-04)", result = pf(abs(fl.b - 1) < 0.02))
    summ!(ctx, "soak", "Sample clock against host clock", "$(fmt((fl.b - 1) * 1e6, 1)) ppm", "slope of host time against sample count")
    summ!(ctx, "soak", "Timing jitter (host - data, detrended)", "$(fmt(std(lag[s0:end]), 1)) ms rms, $(length(steps)) steps > 50 ms", "USB and scheduling")
    pm = isempty(res.proc_ms) ? NaN : median(res.proc_ms); px = isempty(res.proc_ms) ? NaN : maximum(res.proc_ms)
    summ!(ctx, "soak", "Processing time per block", "$(fmt(pm, 1)) ms median, $(fmt(px, 0)) ms max", "block = $(fmt(1e3blk_s, 0)) ms; queue backlog max $(maximum(x.backlog for x in d[1:nb]))")
    summ!(ctx, "soak", "RX1-RX2 phase drift", "$(fmt(pfit.b, 3)) deg/min", "tone over the air; $cond")
    summ!(ctx, "soak", "RX1-RX2 phase peak-to-peak", "$(fmt(maximum(ph) - minimum(ph), 2)) deg", cond)
    summ!(ctx, "soak", "RX1-RX2 phase scatter after drift removed", "$(fmt(std(pfit.res), 3)) deg rms", "per 0.1 s block")
    for lag_s in (0.1, 1.0, 10.0, 60.0, 300.0)
        k = round(Int, lag_s / blk_s)
        v = P.phase_change_rms(ph, k)
        isfinite(v) && summ!(ctx, "soak", "Phase change over $(lag_s < 1 ? lag_s : Int(lag_s)) s", "$(fmt(v, 3)) deg rms", "structure function / sqrt 2")
    end
    rt = [x[6] for x in rec_rows]
    isempty(rt) || summ!(ctx, "soak", "Tone |rho| (median / minimum)", "$(fmt(median(rt), 4)) / $(fmt(minimum(rt), 4))", cond, spec = ">= 0.98 (SP-06)", result = pf(minimum(rt) >= 0.98))
    summ!(ctx, "soak", "Tone frequency drift (Pluto against Pluto+)", "$(fmt(maximum(fr) - minimum(fr), 2)) Hz peak-to-peak", "relative drift of the two references")
end

# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------
function read_csv(path)
    lines = readlines(path)
    head = split(lines[1], ',')
    cols = Dict(String(h) => String[] for h in head)
    for l in lines[2:end]
        isempty(l) && continue
        v = split(l, ',')
        for (h, x) in zip(head, v); push!(cols[String(h)], String(x)); end
    end
    return cols
end
num(c, k) = parse.(Float64, c[k])

function latest_summary(out)
    path = joinpath(out, "summary.tsv")
    isfile(path) || return Vector{Vector{String}}()
    rows = [String.(split(l, '\t')) for l in readlines(path)[2:end] if !isempty(l)]
    last_run = Dict{String,String}()
    for r in rows; last_run[r[2]] = r[1]; end
    return [r for r in rows if last_run[r[2]] == r[1]]
end

function tex(s::AbstractString)
    s = replace(s, "\\" => "\\textbackslash{}", "%" => "\\%", "&" => "\\&", "#" => "\\#", "_" => "\\_", "\$" => "\\\$")
    return replace(s, "+-" => "\$\\pm\$", ">=" => "\$\\ge\$", "<=" => "\$\\le\$", " deg" => "\$^\\circ\$", "|rho|" => "\$|\\rho|\$",
                   "rho" => "\$\\rho\$", "/ sqrt 2" => "/ \$\\sqrt{2}\$", "> " => "\$>\$ ", "< " => "\$<\$ ")
end

const GROUPS = ("readback" => "Settings and read-back", "rate" => "Sample rate", "floor" => "Noise floor and internal correlation (1420 MHz)",
                "levels" => "Levels and spectral purity (869 MHz tone)", "coherence" => "Coherence, delay and phase (869 MHz noise)",
                "soak" => "Continuity and stability (soak)")

function write_table(ctx::Ctx, rows)
    path = joinpath(ctx.out, "results_table.tex")
    open(path, "w") do io
        println(io, "% Generated by stress_test.jl report, ", Dates.format(now(UTC), "yyyy-mm-dd HH:MM"), " UTC, from ", ctx.out)
        println(io, "% Needs \\usepackage{longtable,array}. Every value is in summary.tsv with the run it came from.")
        col(w) = ">{\\raggedright\\arraybackslash}p{$(w)\\textwidth}"
        println(io, "\\begin{longtable}{|", join(col.((0.25, 0.2, 0.27, 0.11, 0.07)), "|"), "|}")
        println(io, "    \\caption{Measured characteristics of the receiving ADALM-Pluto (Rev D).}\n    \\label{tab:pluto-char} \\\\")
        hdr = "    \\hline \\textbf{Characteristic} & \\textbf{Measured} & \\textbf{Conditions} & \\textbf{Spec} & \\textbf{Result} \\\\ \\hline"
        println(io, hdr, "\n    \\endfirsthead\n", hdr, "\n    \\endhead\n    \\hline\n    \\endfoot")
        for (t, title) in GROUPS
            R = [r for r in rows if r[2] == t]
            isempty(R) && continue
            println(io, "    \\multicolumn{5}{|l|}{\\textbf{", tex(title), "}} \\\\ \\hline")
            for r in R
                println(io, "    ", join(tex.(r[3:7]), " & "), " \\\\ \\hline")
            end
        end
        println(io, "\\end{longtable}")
    end
    println("Wrote ", path)
end

function make_figures(ctx::Ctx)
    out = ctx.out
    files = filter(f -> isfile(joinpath(out, f)), ["rate.csv", "floor.csv", "levels.csv", "coherence_spectrum.csv", "soak_blocks.csv"])
    isempty(files) && return
    println("Loading Plots for the figures...")
    @eval Main using Plots
    Base.invokelatest(draw_figures, out, files)     # Plots was loaded after this function was compiled
end

function draw_figures(out, files)
    Pl = Main.Plots
    Pl.gr(); Pl.default(fontfamily = "Computer Modern", framestyle = :box, grid = true, legend = :best, titlefontsize = 10, guidefontsize = 9,
                        left_margin = 6Pl.mm, bottom_margin = 6Pl.mm)
    sv(p, name) = (Pl.savefig(p, joinpath(out, name)); println("Wrote ", joinpath(out, name)))
    if "rate.csv" in files
        c = read_csv(joinpath(out, "rate.csv")); x = num(c, "fs_req_hz") ./ 1e6; y = num(c, "delivered_sps") ./ 1e6
        p = Pl.plot(x, x, ls = :dash, color = :gray, label = "requested", xlabel = "Requested rate (Msps)", ylabel = "Delivered rate (Msps)", size = (500, 350))
        Pl.plot!(p, x, y, marker = :circle, label = "delivered (both channels)")
        sv(p, "fig_rate.pdf")
    end
    if "floor.csv" in files
        c = read_csv(joinpath(out, "floor.csv")); st = c["antennas"]; g = num(c, "gain_db")
        p1 = Pl.plot(xlabel = "Gain (dB)", ylabel = "Noise (counts rms)", yscale = :log10, title = "Noise against gain")
        p2 = Pl.plot(xlabel = "Gain (dB)", ylabel = "Band |ρ|", title = "Internal correlation", yscale = :log10)
        for s in unique(st)
            i = findall(==(s), st)
            Pl.plot!(p1, g[i], num(c, "rms1_counts")[i], marker = :circle, label = "RX1, antennas $s")
            Pl.plot!(p1, g[i], num(c, "rms2_counts")[i], marker = :utriangle, label = "RX2, antennas $s")
            Pl.plot!(p2, g[i], max.(num(c, "band_rho")[i], 1e-5), marker = :circle, label = "antennas $s")
            Pl.plot!(p2, g[i], num(c, "rho_noise_level")[i], ls = :dash, label = "noise level ($s)")
        end
        Pl.hline!(p2, [0.01], color = :red, ls = :dot, label = "SP-07 limit")
        sv(Pl.plot(p1, p2, layout = (1, 2), size = (900, 350)), "fig_floor.pdf")
    end
    if "levels.csv" in files
        c = read_csv(joinpath(out, "levels.csv")); g = num(c, "gain_db")
        p1 = Pl.plot(g, num(c, "tone1_dbfs"), marker = :circle, label = "tone RX1", xlabel = "Gain (dB)", ylabel = "dBFS", title = "Tone and noise against gain")
        Pl.plot!(p1, g, num(c, "tone2_dbfs"), marker = :utriangle, label = "tone RX2")
        Pl.plot!(p1, g, num(c, "noise1_dbfs_bin"), label = "noise per bin RX1")
        cl = findall(x -> x > 1e-3, max.(num(c, "clip1"), num(c, "clip2")))
        isempty(cl) || Pl.vline!(p1, [g[cl[1]]], color = :red, ls = :dot, label = "clipping")
        p2 = Pl.plot(g, -num(c, "image_dbc"), marker = :circle, label = "image rejection", xlabel = "Gain (dB)", ylabel = "dB", title = "Spectral purity (RX1)")
        Pl.plot!(p2, g, -num(c, "sfdr_dbc"), marker = :diamond, label = "spurious-free range")
        Pl.plot!(p2, g, num(c, "snr1_db"), label = "tone SNR (7 bins)")
        sv(Pl.plot(p1, p2, layout = (1, 2), size = (900, 350)), "fig_levels.pdf")
    end
    if "coherence_spectrum.csv" in files
        c = read_csv(joinpath(out, "coherence_spectrum.csv")); g = num(c, "gain_db"); f = num(c, "f_bb_hz") ./ 1e3; inb = num(c, "in_band") .== 1
        p1 = Pl.plot(xlabel = "Baseband frequency (kHz)", ylabel = "|ρ|", title = "Coherence of a common noise signal")
        p2 = Pl.plot(xlabel = "Baseband frequency (kHz)", ylabel = "Phase (deg)", title = "Cross-spectrum phase")
        for gg in unique(g)
            i = findall(k -> g[k] == gg && inb[k], eachindex(g))
            Pl.scatter!(p1, f[i], num(c, "rho_abs")[i], ms = 1, msw = 0, label = "measured, $(Int(gg)) dB")
            Pl.plot!(p1, f[i], num(c, "rho_pred")[i], lw = 1.5, label = "predicted from autos, $(Int(gg)) dB")
            Pl.scatter!(p2, f[i], num(c, "rho_deg")[i], ms = 1, msw = 0, label = "$(Int(gg)) dB")
        end
        sv(Pl.plot(p1, p2, layout = (1, 2), size = (900, 350)), "fig_coherence.pdf")
    end
    if "soak_blocks.csv" in files
        c = read_csv(joinpath(out, "soak_blocks.csv")); t = num(c, "t_data_s") ./ 60
        p1 = Pl.plot(t, num(c, "dphi_unwrapped_deg"), label = "", xlabel = "Time (min)", ylabel = "RX1-RX2 phase (deg)", title = "Phase of the common tone")
        p2 = Pl.plot(t, num(c, "lag_detrended_ms"), label = "", xlabel = "Time (min)", ylabel = "Host - data time (ms)", title = "Timing (clock rate removed)")
        sv(Pl.plot(p1, p2, layout = (1, 2), size = (900, 350)), "fig_soak.pdf")
    end
end

function t_report(ctx::Ctx)
    println("\n=== report: ", ctx.out)
    rows = latest_summary(ctx.out)
    isempty(rows) && (println("  no summary.tsv yet"); return)
    write_table(ctx, rows)
    make_figures(ctx)
end

# ---------------------------------------------------------------------
const PLAN = """
Session plan (about 1 hour). Receiver: the original Pluto on this laptop. Source: the Pluto+ on its own laptop,
driven with Real/tx_source.jl. This script prints each Pluto+ command and waits.

  1 readback   ~1 min    nothing to set
  2 rate       ~2 min    nothing to set
  3 floor      ~3 min    antennas removed, then connected; Pluto+ off
  4 levels     ~3 min    Pluto+ tone  869.2 MHz (TX LO 868.9), start atten -50; the script asks for SNR >= 30 dB at gain 20
  5 coherence  ~3 min    Pluto+ off, then noise 869.0 MHz +- 0.75 MHz, start atten -40; 6-20 dB power rise wanted
  6 soak       soak_min  Pluto+ tone again, same level as step 4
  7 report               results_table.tex, fig_*.pdf

Set-up for 4-6: both receive antennas on the bar, the Pluto+ antenna at broadside 2-3 m away, at the same height.
Everything transmitted stays inside 868-870 MHz. Run the whole session with:  julia -t 3 stress_test.jl all addr=usb:X.Y.Z
"""

function main(args)
    Base.exit_on_sigint(false)
    isempty(args) && (println(PLAN); return)
    test = args[1]
    cfg = P.Config(); sc = StressCfg()
    P.apply_kv!(args[2:end], cfg, sc)
    test == "plan" && (println(PLAN); return)
    out = isempty(sc.out) ? joinpath(cfg.out_dir, "STRESS_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * (sc.fake ? "_FAKE" : "")) : sc.out
    mkpath(out)
    ctx = Ctx(cfg, sc, out, Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS"), :none, nothing)
    open(joinpath(out, "settings_$(replace(ctx.run_id, ':' => '-'))_$test.txt"), "w") do io
        println(io, "test = $test\nsoftware = $(P.SOFTWARE)\nrun_id = $(ctx.run_id)\nhost = $(gethostname())\njulia = $(VERSION)")
        for o in (cfg, sc), f in fieldnames(typeof(o)); println(io, f, " = ", getfield(o, f)); end
    end
    println("Results in ", out, sc.fake ? "   (FAKE radio: numbers describe the stand-in, not the Pluto)" : "")
    tests = Dict("readback" => t_readback, "rate" => t_rate, "floor" => t_floor, "levels" => t_levels,
                 "coherence" => t_coherence, "soak" => t_soak, "report" => t_report)
    order = test == "all" ? ["readback", "rate", "floor", "levels", "coherence", "soak", "report"] : [test]
    for t in order
        haskey(tests, t) || error("Unknown test '$t'. Tests: readback rate floor levels coherence soak report all plan")
        try
            tests[t](ctx)
        catch e
            e isa InterruptException && (println("\nStopped."); close_rx!(ctx); break)
            println("\n  ERROR in $t: ", sprint(showerror, e))
            close_rx!(ctx)
            test == "all" || rethrow()
        end
    end
    println("\nDone. Results in ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
