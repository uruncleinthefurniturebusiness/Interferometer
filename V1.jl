# =====================================================================
# V1.jl -- ADALM-Pluto as a two-channel interferometer (single file)
#
# Plan and reasoning: V1.md in the vault.
#
# Usage (from this folder):
#     julia -t 3 V1.jl <mode> [key=value ...]
#
# Modes
#     selftest   no hardware: simulator -> same correlator path, drop-monitor test, file round trip
#     check      connect, configure, READ BACK settings, short capture, per-channel health + spectrum peak
#     rate       sustained capture, no processing: real sample rate vs host clock (add tone=true to
#                also watch a tone for dropped samples)
#     run        full pipeline: capture -> FX correlator -> timestamped visibility records on disk
#     show       summarise a visibility file written by `run`  (path=... or newest in out_dir)
#     sim        no hardware: simulated drift-scan observation of a chosen source (Sun disk, point, Gaussian,
#                optional HI line, shared-LO offset, instrument phase), run through the same correlator;
#                compares measured visibilities with the simulation's own expectation
#
# Options (key=value, defaults in the Config struct below), e.g.
#     julia -t 3 V1.jl check f_lo=1420e6 fs=2.5e6 gain1=30 gain2=30
#     julia -t 3 V1.jl run duration_s=600 t_int=2 nfft=4096
#
# Signal convention: visibility V = x1 * conj(x2). A positive phase means the signal reaches
# channel 2 LATER than channel 1 (delay tau gives phase +2*pi*f_true*tau).
#
# NOTE ON STATE: the Pluto keeps its settings between runs. Every run therefore writes ALL settings
# and then reads them back; a mismatch is fatal unless strict=false.
# ============================
using AdalmPluto
using FFTW
using Statistics
using Dates
using Printf
using Random

const AP = AdalmPluto

const V1_VERSION    = "V1.0"
const C_LIGHT       = 2.99792458e8
const MIN_FS_HZ     = 2_083_333.0     # AD9361 minimum rate with the default filter chain (verify in datasheet)
const CLIP_LEVEL    = 0.98f0          # |I| or |Q| above this fraction of full scale counts as clipped
const LOW_POWER     = 1.0e-7          # mean power (full-scale^2) below this means "nothing connected"
const FLAG_CLIP     = 0x01
const FLAG_LOWPOW   = 0x02
const FLAG_SHORT    = 0x04            # buffer refill returned fewer bytes than expected
const FLAG_QFULL    = 0x08            # processing queue was full when this block arrived (host falling behind)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
Base.@kwdef mutable struct Config
    addr::String          = "auto"          # "auto" tries usb: then ip:192.168.2.1; or e.g. "usb:1.2.5"
    f_lo::Float64         = 1420.0e6        # shared RX LO (Hz); HI line then sits at +405.75 kHz baseband
    fs::Float64           = 2.5e6           # complex sample rate per channel (Hz); must be >= ~2.083e6
    gain1::Int            = 40              # RX1 manual gain (dB)
    gain2::Int            = 40              # RX2 manual gain (dB)
    rx_port::String       = "A_BALANCED"
    bw_ratio::Float64     = 1.0             # rf_bandwidth = bw_ratio * fs (check the readback!)
    buffer_secs::Float64  = 0.1             # one refill = about this many seconds (rounded to whole FFT segments)
    kernel_buffers::Int   = 8               # Pluto-side DMA buffers (0 = leave the libiio default)
    skip_blocks::Int      = 2               # discard this many start-up blocks
    nfft::Int             = 4096            # FX channels (bin width = fs/nfft)
    t_int::Float64        = 1.0             # integration time per record (s)
    window::String        = "hann"          # "hann" or "rect"
    duration_s::Float64   = 60.0
    out_dir::String       = "C:/PlutoCaptures"   # keep OUTSIDE the OneDrive-synced folder
    tag::String           = ""
    save_raw::Bool        = false           # also write raw cf32 per channel (about 20 MB/s!)
    queue_depth::Int      = 32
    strict::Bool          = true            # abort if the hardware readback differs from the request
    tone::Bool            = false           # `rate` mode: monitor a dominant tone for dropped samples
    path::String          = ""              # `show` mode: visibility file
    dc_mask::Int          = 4               # bins either side of DC excluded from the band average and peak search
    quad_track::Int       = -1              # AD9361 quadrature (I/Q) tracking: -1 leave as is, 0 off, 1 on
    rf_dc_track::Int      = -1              # AD9361 RF DC-offset tracking:     -1 leave as is, 0 off, 1 on
    bb_dc_track::Int      = -1              # AD9361 baseband DC-offset tracking: -1 leave as is, 0 off, 1 on
    # ---- `sim` mode (no hardware): simulated observation, see "Simulated observation" below ----
    sim_model::String       = "sun"         # point | disk | gaussian | sun (disk with optional limb brightening)
    sim_diam_arcmin::Float64 = 32.0         # disk diameter / Gaussian FWHM, arcmin (placeholder: verify the radio Sun)
    sim_limb::Float64       = 0.0           # sun model: I(r) = 1 + limb*(r/a)^8
    sim_snr_db::Float64     = -20.0         # T_A/T_S per channel for the TOTAL source flux, dB
    sim_flux_jy::Float64    = NaN           # physical override: with sim_area_m2 and sim_tsys_k, T_A/T_S = S*A/2761.3/T_S
    sim_area_m2::Float64    = NaN
    sim_tsys_k::Float64     = NaN
    sim_baseline_m::Float64 = 20.0
    sim_az_deg::Float64     = 90.0          # baseline azimuth from north through east (90 = east-west; ASSUMED, not yet surveyed)
    sim_lat_deg::Float64    = -33.9         # site latitude (Cape Town, approximate)
    sim_dec_deg::Float64    = 0.0           # source declination
    sim_start_ha_h::Float64 = -3.0          # starting hour angle (hours)
    sim_hours::Float64      = 6.0           # simulated duration
    sim_step_s::Float64     = 120.0         # simulated time between records (each record uses t_int seconds of data)
    sim_min_alt_deg::Float64 = 10.0         # skip records with the source below this altitude
    sim_hpbw_deg::Float64   = 0.0           # >0 = fixed dishes at the meridian with this primary-beam FWHM (overrides sim_dish_m for the beam)
    sim_dish_m::Float64     = NaN           # dish diameter (m): gives effective area (with sim_eff) and, for fixed dishes, the beamwidth
    sim_eff::Float64        = 0.55          # aperture efficiency (placeholder: get the real value)
    sim_beam_k::Float64     = 1.15          # HPBW = k * lambda / D (about 1.02 uniform aperture, 1.1 to 1.2 tapered)
    sim_track::Bool         = true          # true: dishes track the source (no beam loss); false: fixed at the meridian
    sim_phase_off_deg::Float64 = 0.0        # instrument phase added to the measured phase (calibration target)
    sim_gain2_db::Float64   = 0.0           # RX2 gain relative to RX1
    sim_dc::Float64         = 0.0           # shared DC offset amplitude, in units of the noise rms (LO leakage)
    sim_dc_phase_deg::Float64 = 0.0
    sim_line::Bool          = false         # add an HI-like spectral line
    sim_line_v_kms::Float64 = 0.0           # line velocity (radio convention, relative to rest)
    sim_line_fwhm_kms::Float64 = 10.0
    sim_line_amp::Float64   = 0.05          # peak line power relative to the receiver-noise PSD
    sim_line_vis::Float64   = 0.0           # normalized visibility of the line (0 = resolved out: autos only)
    sim_seed::Int           = 1
    # ---- `tone` / `delayfit` modes: generator tone through a splitter, delay from the phase-versus-frequency slope ----
    tn_sweep::String        = ""            # e.g. "-1000:200:1000": baseband offsets in kHz from the LO; prompts you to set the generator at each
    tn_cable_m::Float64     = 0.0           # extra cable on RX2 relative to RX1 for this run (m); stored in the CSV, used by delayfit
    tn_vf::Float64          = 0.66          # cable velocity factor (measure it, or use the datasheet value)
    tn_file::String         = ""            # results CSV (default: <out_dir>/tone_results.csv)
    tn_tag2::String         = ""            # delayfit: second configuration tag to compare against `tag`
end

function parse_cli(args::Vector{String})
    isempty(args) && (println(usage()); exit(0))
    mode = args[1]
    cfg = Config()
    for a in args[2:end]
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        hasfield(Config, sym) || error("Unknown option '$k'")
        T = fieldtype(Config, sym)
        val = T == String  ? String(v) :
              T == Bool    ? (lowercase(v) in ("1", "true", "yes", "on")) :
              T <: Integer ? round(T, parse(Float64, v)) :
                             parse(T, v)
        setfield!(cfg, sym, val)
    end
    return mode, cfg
end

usage() = """
V1.jl modes: selftest | check | rate | run | show | sim | tone | delayfit
  julia -t 3 V1.jl check    [f_lo=1420e6 fs=2.5e6 gain1=40 gain2=40 addr=auto]
  julia -t 3 V1.jl rate     [duration_s=60 tone=true]
  julia -t 3 V1.jl run      [duration_s=600 t_int=1 nfft=4096 window=hann]
  julia V1.jl show          [path=<file>.vis]
  julia V1.jl selftest
  julia -t 3 V1.jl tone     [duration_s=10 tag=A tn_cable_m=0 tn_sweep="-1000:200:1000"]   (generator tone through a splitter)
  julia V1.jl delayfit      [tag=A tn_tag2=B]                                              (delay from the phase slope)
  julia -t 3 V1.jl sim      [sim_model=sun sim_hours=6 sim_snr_db=-20 sim_az_deg=90 sim_dec_deg=0 ...]   (no hardware)
See the header of V1.jl for all options."""

function validate(cfg::Config)
    cfg.fs >= MIN_FS_HZ || error("fs=$(cfg.fs) Hz is below the AD9361 minimum of about $(MIN_FS_HZ) Hz; " *
                                 "the Pluto would silently keep its previous rate. Use fs >= 2.1e6.")
    cfg.nfft >= 64 || error("nfft must be >= 64")
    cfg.window in ("hann", "rect") || error("window must be hann or rect")
    cfg.t_int > 0 && cfg.buffer_secs > 0 || error("t_int and buffer_secs must be positive")
    (0 <= cfg.gain1 <= 73 && 0 <= cfg.gain2 <= 73) || @warn "gain outside the usual 0..73 dB range"
    (325e6 <= cfg.f_lo <= 3.8e9) || @warn "f_lo outside the stock Pluto range (325 MHz - 3.8 GHz)"
    return nothing
end

# Samples per refill: a whole number of FFT segments, so no segment ever straddles two buffers.
function block_geometry(cfg::Config)
    nsegb = max(1, round(Int, cfg.fs * cfg.buffer_secs / cfg.nfft))
    return nsegb, nsegb * cfg.nfft
end

# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
struct Block
    idx::Int                 # index of the block's first sample (0-based, after start-up skip)
    utc::DateTime            # host UTC when the refill returned (= end of the block, plus USB latency)
    t_ns::UInt64
    i1::Vector{Int16}
    q1::Vector{Int16}
    i2::Vector{Int16}
    q2::Vector{Int16}
    short::Bool
    qfull::Bool
end

struct Record
    t_start::Float64         # unix seconds, UTC
    t_end::Float64
    nseg::Int32
    flags::UInt32
    pow1::Float32            # mean power (full-scale^2)
    pow2::Float32
    clip1::Float32           # fraction of samples at/near full scale
    clip2::Float32
    V::Vector{ComplexF32}    # cross spectrum  <X1 conj(X2)>, fftshifted (DC bin at nfft/2+1)
    P1::Vector{Float32}      # auto spectra, same ordering
    P2::Vector{Float32}
end

# ---------------------------------------------------------------------
# Hardware layer
# ---------------------------------------------------------------------
mutable struct Radio
    ctx::Ptr{AP.iio_context}
    rxdev::Ptr{AP.iio_device}
    buf::Ptr{AP.iio_buffer}
    bufsize::Int
    chs::Vector{Ptr{AP.iio_channel}}     # RX1 I, RX1 Q, RX2 I, RX2 Q
    phy::Vector{Ptr{AP.iio_channel}}     # RX1, RX2 control channels
    lo::Ptr{AP.iio_channel}
    raw::Vector{Vector{UInt8}}
    uri::String
end

chk(ret, what) = (ret < 0 ? error("Writing '$what' failed (libiio returned $ret)") : ret)

function rd_ll(ch, attr)
    ret, v = AP.C_iio_channel_attr_read_longlong(ch, attr)
    ret < 0 && error("Reading '$attr' failed (libiio returned $ret)")
    return Int(v)
end

function rd_str(ch, attr)
    ret, s = AP.C_iio_channel_attr_read(ch, attr)
    ret < 0 && error("Reading '$attr' failed (libiio returned $ret)")
    return strip(replace(String(s), '\0' => ""))
end

rd_gain(ch) = parse(Float64, split(rd_str(ch, "hardwaregain"))[1])

function connect(addr::String)
    candidates = addr == "auto" ? ["usb:", "ip:192.168.2.1"] : [addr]
    for uri in candidates
        try
            ctx = AP.C_iio_create_context_from_uri(uri)
            if ctx != C_NULL && AP.C_iio_context_get_devices_count(ctx) > 0
                return ctx, uri
            end
        catch
            # try the next candidate
        end
    end
    error("No Pluto found via $(join(candidates, ", ")). Run `iio_info -s` and pass addr=usb:X.Y.Z.")
end

function open_radio(cfg::Config)
    validate(cfg)
    ctx, uri = connect(cfg.addr)
    println("Connected via $uri\n", AP.C_iio_context_get_description(ctx), "\n")
    local radio
    try
        _, rxdev = AP.findTRXDevices(ctx)
        rxdev == C_NULL && error("RX streaming device (cf-ad9361-lpc) not found")

        i1, q1 = AP.findIQChannels(rxdev, "voltage0", "voltage1", false)
        i2, q2 = AP.findIQChannels(rxdev, "voltage2", "voltage3", false)
        if any(==(C_NULL), (i1, q1, i2, q2))
            error("RX2 (voltage2/voltage3) not exposed. The Pluto must be in 2R2T mode; check `iio_info` " *
                  "shows four RX data channels and see Hardware Setup.md.")
        end
        chs = [i1, q1, i2, q2]
        foreach(AP.C_iio_channel_enable, chs)

        _, phy1 = AP.findTRXChannels(ctx, 0, 0)
        _, phy2 = AP.findTRXChannels(ctx, 0, 1)
        _, lo   = AP.findLoChannels(ctx)

        # --- write EVERY setting and check each return code (the old code ignored them) ---
        fs  = Int64(round(cfg.fs))
        bw  = Int64(round(cfg.bw_ratio * cfg.fs))
        for (phy, g) in ((phy1, cfg.gain1), (phy2, cfg.gain2))
            chk(AP.C_iio_channel_attr_write(phy, "rf_port_select", cfg.rx_port), "rf_port_select")
            chk(AP.C_iio_channel_attr_write_longlong(phy, "sampling_frequency", fs), "sampling_frequency")
            chk(AP.C_iio_channel_attr_write_longlong(phy, "rf_bandwidth", bw), "rf_bandwidth")
            chk(AP.C_iio_channel_attr_write(phy, "gain_control_mode", "manual"), "gain_control_mode")
            chk(AP.C_iio_channel_attr_write_longlong(phy, "hardwaregain", Int64(g)), "hardwaregain")
            for (attr, v) in (("quadrature_tracking_en", cfg.quad_track), ("rf_dc_offset_tracking_en", cfg.rf_dc_track),
                              ("bb_dc_offset_tracking_en", cfg.bb_dc_track))
                v >= 0 && chk(AP.C_iio_channel_attr_write_longlong(phy, attr, Int64(v)), attr)
            end
        end
        chk(AP.C_iio_channel_attr_write_longlong(lo, "frequency", Int64(round(cfg.f_lo))), "LO frequency")
        sleep(0.3)   # let the AD9361 retune / recalibrate

        _, bufsize = block_geometry(cfg)
        if cfg.kernel_buffers > 0
            r = AP.C_iio_device_set_kernel_buffers_count(rxdev, UInt32(cfg.kernel_buffers))
            r < 0 && @warn "Could not set kernel buffer count (libiio returned $r); using the default"
        end
        buf = try
            AP.C_iio_device_create_buffer(rxdev, UInt(bufsize), false)
        catch e
            error("Could not create the capture buffer ($bufsize samples). Try a smaller buffer_secs or " *
                  "kernel_buffers. ($e)")
        end

        raw = [zeros(UInt8, bufsize * 2) for _ in 1:4]
        radio = Radio(ctx, rxdev, buf, bufsize, chs, [phy1, phy2], lo, raw, uri)
    catch
        AP.C_iio_context_destroy(ctx)
        rethrow()
    end
    return radio
end

close_radio!(r::Radio) = (AP.C_iio_buffer_destroy(r.buf); AP.C_iio_context_destroy(r.ctx); nothing)

function read_back(r::Radio)
    return (fs1   = rd_ll(r.phy[1], "sampling_frequency"), fs2 = rd_ll(r.phy[2], "sampling_frequency"),
            bw1   = rd_ll(r.phy[1], "rf_bandwidth"),       bw2 = rd_ll(r.phy[2], "rf_bandwidth"),
            lo    = rd_ll(r.lo, "frequency"),
            gain1 = rd_gain(r.phy[1]),                      gain2 = rd_gain(r.phy[2]),
            mode1 = rd_str(r.phy[1], "gain_control_mode"),  mode2 = rd_str(r.phy[2], "gain_control_mode"),
            port1 = rd_str(r.phy[1], "rf_port_select"),     port2 = rd_str(r.phy[2], "rf_port_select"),
            quad = (rd_ll(r.phy[1], "quadrature_tracking_en"),   rd_ll(r.phy[2], "quadrature_tracking_en")),
            rfdc = (rd_ll(r.phy[1], "rf_dc_offset_tracking_en"), rd_ll(r.phy[2], "rf_dc_offset_tracking_en")),
            bbdc = (rd_ll(r.phy[1], "bb_dc_offset_tracking_en"), rd_ll(r.phy[2], "bb_dc_offset_tracking_en")))
end

# Print requested vs actual and return the list of (message, fatal) problems.
function verify_readback(rb, cfg::Config)
    probs = Tuple{String,Bool}[]
    function row(name, req, act, bad::Bool, fatal::Bool)
        @printf("  %-22s requested %-16s actual %-16s %s\n", name, req, act, bad ? "<-- MISMATCH" : "ok")
        bad && push!(probs, ("$name: requested $req, hardware reports $act", fatal))
    end
    println("Hardware readback:")
    row("sampling_frequency", @sprintf("%.0f", cfg.fs), string(rb.fs1),
        abs(rb.fs1 - cfg.fs) > 1e-4 * cfg.fs, true)
    row("sampling_frequency RX2", string(rb.fs1), string(rb.fs2), rb.fs1 != rb.fs2, true)
    row("rf_bandwidth", @sprintf("%.0f", cfg.bw_ratio * cfg.fs), string(rb.bw1),
        abs(rb.bw1 - cfg.bw_ratio * cfg.fs) > 0.2 * cfg.bw_ratio * cfg.fs, false)
    row("LO frequency", @sprintf("%.0f", cfg.f_lo), string(rb.lo), abs(rb.lo - cfg.f_lo) > 100, true)
    row("gain RX1 (dB)", string(cfg.gain1), @sprintf("%.2f", rb.gain1), abs(rb.gain1 - cfg.gain1) > 1.0, true)
    row("gain RX2 (dB)", string(cfg.gain2), @sprintf("%.2f", rb.gain2), abs(rb.gain2 - cfg.gain2) > 1.0, true)
    row("gain mode", "manual", "$(rb.mode1)/$(rb.mode2)", !(rb.mode1 == "manual" && rb.mode2 == "manual"), true)
    row("rf port", cfg.rx_port, "$(rb.port1)/$(rb.port2)", !(rb.port1 == cfg.rx_port && rb.port2 == cfg.rx_port), false)
    @printf("  %-22s quadrature %s, rf-dc %s, bb-dc %s   (RX1, RX2; 1 = on)\n", "AD9361 tracking", rb.quad, rb.rfdc, rb.bbdc)
    for (nm, req, act) in (("quadrature_tracking", cfg.quad_track, rb.quad), ("rf_dc_tracking", cfg.rf_dc_track, rb.rfdc),
                           ("bb_dc_tracking", cfg.bb_dc_track, rb.bbdc))
        req >= 0 && any(!=(req), act) && push!(probs, ("$nm: requested $req, hardware reports $act", true))
    end
    return probs
end

function open_verified(cfg::Config)
    r = open_radio(cfg)
    rb = read_back(r)
    probs = verify_readback(rb, cfg)
    fatal = [m for (m, f) in probs if f]
    if !isempty(fatal) && cfg.strict
        close_radio!(r)
        error("Hardware did not accept the requested configuration:\n  " * join(fatal, "\n  ") *
              "\nFix the request, or pass strict=false to continue with what the hardware reports.")
    end
    isempty(probs) || println("WARNING: $(length(probs)) readback difference(s); using the hardware's values.")
    return r, rb
end

# One refill -> four Int16 vectors (copied, so the raw buffers can be reused immediately).
function read_block!(r::Radio, idx::Int; qfull::Bool = false)
    nbytes = AP.C_iio_buffer_refill(r.buf)
    utc = now(UTC); t_ns = time_ns()
    nbytes < 0 && error("iio_buffer_refill failed with code $nbytes")
    for k in 1:4
        AP.C_iio_channel_read!(r.chs[k], r.buf, r.raw[k])
    end
    a = ntuple(k -> copy(reinterpret(Int16, r.raw[k])), 4)
    return Block(idx, utc, t_ns, a[1], a[2], a[3], a[4], nbytes != r.bufsize * 8, qfull)
end

# ---------------------------------------------------------------------
# Sample conversion and per-channel health
# ---------------------------------------------------------------------
# The container is 16-bit but the ADC is 12-bit; whether the value is right-aligned (+-2048) or
# left-aligned (+-32768) is checked from the data rather than assumed.
function detect_fullscale(b::Block)
    m = 0
    low_zero = true
    for v in (b.i1, b.q1, b.i2, b.q2), s in v
        a = abs(Int(s)); a > m && (m = a)
        low_zero &= (s & Int16(15)) == 0
    end
    return (m > 2048 || (low_zero && m > 0)) ? 32768.0f0 : 2048.0f0
end

to_complex(i::Vector{Int16}, q::Vector{Int16}, fsc::Float32) = ComplexF32.(i ./ fsc, q ./ fsc)

function chan_stats(x::Vector{ComplexF32})
    n = length(x); p = 0.0; pk = 0.0f0; nclip = 0; sr = 0.0; si = 0.0
    @inbounds for v in x
        re = real(v); im = imag(v)
        p += re * re + im * im; sr += re; si += im
        a = max(abs(re), abs(im)); a > pk && (pk = a); a >= CLIP_LEVEL && (nclip += 1)
    end
    return (power = p / n, peak = Float64(pk), clip = nclip / n, dc = complex(sr / n, si / n))
end

# ---------------------------------------------------------------------
# FX correlator
# ---------------------------------------------------------------------
mutable struct FX
    nfft::Int
    nsegb::Int
    win::Vector{Float32}
    wpow::Float64
    plan::Any
    V::Vector{ComplexF64}
    P1::Vector{Float64}
    P2::Vector{Float64}
    nseg::Int
    blocks::Int
    t_first::DateTime
    t_last::DateTime
    pw1::Float64; pw2::Float64
    clip1::Float64; clip2::Float64
    flags::UInt32
end

function FX(nfft::Int, nsegb::Int, window::String)
    w = window == "hann" ? Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:nfft-1) ./ nfft)) : ones(Float32, nfft)
    plan = plan_fft(Matrix{ComplexF32}(undef, nfft, nsegb), 1)
    fx = FX(nfft, nsegb, w, sum(abs2, Float64.(w)), plan, zeros(ComplexF64, nfft), zeros(nfft), zeros(nfft),
            0, 0, DateTime(0), DateTime(0), 0.0, 0.0, 0.0, 0.0, 0x00)
    return fx
end

function reset!(fx::FX)
    fill!(fx.V, 0); fill!(fx.P1, 0); fill!(fx.P2, 0)
    fx.nseg = 0; fx.blocks = 0
    fx.pw1 = fx.pw2 = fx.clip1 = fx.clip2 = 0.0
    fx.flags = 0x00
    return fx
end

function accumulate!(fx::FX, x1::Vector{ComplexF32}, x2::Vector{ComplexF32})
    X1 = fx.plan * (reshape(x1, fx.nfft, fx.nsegb) .* fx.win)
    X2 = fx.plan * (reshape(x2, fx.nfft, fx.nsegb) .* fx.win)
    fx.V  .+= vec(sum(X1 .* conj.(X2), dims = 2))
    fx.P1 .+= vec(sum(abs2.(X1), dims = 2))
    fx.P2 .+= vec(sum(abs2.(X2), dims = 2))
    fx.nseg += fx.nsegb
    return fx
end

function finalize!(fx::FX)::Record
    s = fx.nseg * fx.wpow
    nb = max(fx.blocks, 1)
    flags = fx.flags
    p1 = fx.pw1 / nb; p2 = fx.pw2 / nb
    c1 = fx.clip1 / nb; c2 = fx.clip2 / nb
    c1 > 1e-3 || c2 > 1e-3 ? (flags |= FLAG_CLIP) : nothing
    p1 < LOW_POWER || p2 < LOW_POWER ? (flags |= FLAG_LOWPOW) : nothing
    return Record(datetime2unix(fx.t_first), datetime2unix(fx.t_last), Int32(fx.nseg), flags,
                  Float32(p1), Float32(p2), Float32(c1), Float32(c2),
                  ComplexF32.(fftshift(fx.V ./ s)), Float32.(fftshift(fx.P1 ./ s)), Float32.(fftshift(fx.P2 ./ s)))
end

bin_freqs(nfft, fs, f_lo) = f_lo .+ ((0:nfft-1) .- nfft ÷ 2) .* (fs / nfft)

# Band-average, DC-bin and peak-bin coherence of a record. The direct-conversion DC spike (LO leakage)
# is common to both channels, so it is excluded (+-dc_mask bins) from the band average and the peak
# search and reported separately.
function summarize(rec::Record, dc_mask::Int = 4)
    n = length(rec.V); c = n ÷ 2 + 1
    keep = trues(n); keep[max(1, c - dc_mask):min(n, c + dc_mask)] .= false
    V = sum(rec.V[keep]); P1 = sum(rec.P1[keep]); P2 = sum(rec.P2[keep])
    rho = V / sqrt(P1 * P2)
    rho_dc = rec.V[c] / sqrt(rec.P1[c] * rec.P2[c])
    pw = Float64.(rec.P1 .+ rec.P2); pw[.!keep] .= -Inf
    k = argmax(pw)
    rho_pk = rec.V[k] / sqrt(rec.P1[k] * rec.P2[k])
    return (rho = rho, rho_dc = rho_dc, k = k, rho_pk = rho_pk)
end

flagstr(f::UInt32) = isempty(strip((f & FLAG_CLIP != 0 ? "CLIP " : "") * (f & FLAG_LOWPOW != 0 ? "LOWPOW " : "") *
                                   (f & FLAG_SHORT != 0 ? "SHORT " : "") * (f & FLAG_QFULL != 0 ? "QFULL" : ""))) ? "-" :
                   strip((f & FLAG_CLIP != 0 ? "CLIP " : "") * (f & FLAG_LOWPOW != 0 ? "LOWPOW " : "") *
                         (f & FLAG_SHORT != 0 ? "SHORT " : "") * (f & FLAG_QFULL != 0 ? "QFULL" : ""))

# ---------------------------------------------------------------------
# Consumer: blocks -> integrations -> records
# ---------------------------------------------------------------------
function consume!(q::Channel{Block}, cfg::Config, fs_eff::Float64, on_record::Function; raw_ios = nothing)
    nsegb, bufsize = block_geometry(cfg)
    fx = FX(cfg.nfft, nsegb, cfg.window)
    per_int = max(1, round(Int, cfg.t_int * fs_eff / bufsize))
    fsc = 0.0f0
    nrec = 0; nblk = 0; nshort = 0; nqfull = 0
    blk_secs = bufsize / fs_eff
    proc = Float64[]                       # per-block processing time (ms), for the queue diagnostics
    for b in q
        tproc = time_ns()
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        fx.blocks == 0 && (fx.t_first = b.utc - Millisecond(round(Int, 1000 * blk_secs)))
        fx.t_last = b.utc
        accumulate!(fx, x1, x2)
        fx.blocks += 1
        fx.pw1 += s1.power; fx.pw2 += s2.power
        fx.clip1 += s1.clip; fx.clip2 += s2.clip
        b.short && (fx.flags |= FLAG_SHORT; nshort += 1)
        b.qfull && (fx.flags |= FLAG_QFULL; nqfull += 1)
        if raw_ios !== nothing
            write(raw_ios[1], x1); write(raw_ios[2], x2)
        end
        nblk += 1
        if fx.blocks == per_int
            rec = finalize!(fx)
            on_record(rec)
            nrec += 1
            reset!(fx)
        end
        push!(proc, (time_ns() - tproc) / 1e6)
    end
    return (records = nrec, blocks = nblk, short = nshort, qfull = nqfull, fullscale = fsc,
            proc_med_ms = isempty(proc) ? NaN : median(proc), proc_max_ms = isempty(proc) ? NaN : maximum(proc))
end

# ---------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------
function write_record(io::IO, rec::Record)
    write(io, rec.t_start, rec.t_end, rec.nseg, rec.flags, rec.pow1, rec.pow2, rec.clip1, rec.clip2)
    write(io, rec.V); write(io, rec.P1); write(io, rec.P2)
end

function read_records(path::String, nfft::Int)
    recs = Record[]
    open(path, "r") do io
        while !eof(io)
            t0 = read(io, Float64); t1 = read(io, Float64); ns = read(io, Int32); fl = read(io, UInt32)
            p1 = read(io, Float32); p2 = read(io, Float32); c1 = read(io, Float32); c2 = read(io, Float32)
            V = Vector{ComplexF32}(undef, nfft); read!(io, V)
            P1 = Vector{Float32}(undef, nfft); read!(io, P1)
            P2 = Vector{Float32}(undef, nfft); read!(io, P2)
            push!(recs, Record(t0, t1, ns, fl, p1, p2, c1, c2, V, P1, P2))
        end
    end
    return recs
end

function write_meta(path::String, cfg::Config, rb, summary, extra::Dict)
    open(path, "w") do io
        println(io, "software = $V1_VERSION")
        println(io, "written_utc = $(now(UTC))")
        println(io, "nfft = $(cfg.nfft)")
        println(io, "window = $(cfg.window)")
        println(io, "t_int_requested_s = $(cfg.t_int)")
        println(io, "fs_hz = $(rb === nothing ? cfg.fs : rb.fs1)")
        println(io, "f_lo_hz = $(rb === nothing ? cfg.f_lo : rb.lo)")
        println(io, "gain1_db = $(rb === nothing ? cfg.gain1 : rb.gain1)")
        println(io, "gain2_db = $(rb === nothing ? cfg.gain2 : rb.gain2)")
        rb === nothing || println(io, "rf_bandwidth_hz = $(rb.bw1)")
        rb === nothing || println(io, "tracking_rx1_rx2 = quadrature $(rb.quad), rf_dc $(rb.rfdc), bb_dc $(rb.bbdc)")
        println(io, "fullscale_counts = $(summary.fullscale)")
        println(io, "records = $(summary.records)")
        println(io, "blocks = $(summary.blocks)")
        println(io, "short_reads = $(summary.short)")
        println(io, "queue_full_events = $(summary.qfull)")
        for (k, v) in extra
            println(io, "$k = $v")
        end
        println(io, "convention = V = x1 * conj(x2); positive phase = signal reaches channel 2 later")
        println(io, "record_layout = Float64 t_start, Float64 t_end, Int32 nseg, UInt32 flags, " *
                    "Float32 pow1, pow2, clip1, clip2, then ComplexF32 V[nfft], Float32 P1[nfft], Float32 P2[nfft]; " *
                    "spectra fftshifted (DC at index nfft/2+1)")
        println(io, "flags = 1 clip, 2 low power, 4 short read, 8 queue full")
    end
end

function read_meta(path::String)
    d = Dict{String,String}()
    for line in eachline(path)
        occursin(" = ", line) || continue
        k, v = split(line, " = "; limit = 2)
        d[strip(k)] = strip(v)
    end
    return d
end

# ---------------------------------------------------------------------
# Tone monitor: finds a dominant tone and checks its phase continuity across block boundaries.
# A dropped sample shifts the tone phase by 2*pi*f/fs per sample, far above the noise.
# ---------------------------------------------------------------------
wrap_pi(x) = mod(x + pi, 2pi) - pi

function tone_fit(x::AbstractVector{ComplexF32}, fs::Float64; L::Int = 65536)
    L = min(L, length(x))
    s = ComplexF64.(@view x[1:L])
    S = fft(s)
    p = abs2.(S)
    k = argmax(p)
    a = log(p[mod1(k - 1, L)] + 1e-300); b = log(p[k] + 1e-300); c = log(p[mod1(k + 1, L)] + 1e-300)
    d = 0.5 * (a - c) / (a - 2b + c)
    f = ((k - 1) + (isfinite(d) ? d : 0.0)) * fs / L
    f > fs / 2 && (f -= fs)
    n = collect(0.0:L-1)
    h = L ÷ 2
    for _ in 1:3
        r = cis.(-2pi * f / fs .* n) .* s
        dphi = angle(sum(@view r[h+1:L]) * conj(sum(@view r[1:h])))
        f += dphi * fs / (2pi * h)
    end
    r = cis.(-2pi * f / fs .* n) .* s
    return f, angle(sum(r))
end

mutable struct ToneMon
    fs::Float64
    bufsize::Int
    prev_f::Vector{Float64}
    prev_ph::Vector{Float64}
    have::Bool
    freqs::Vector{Float64}          # RX1 tone baseband frequency per block
    res1::Vector{Float64}
    res2::Vector{Float64}
    xphase::Vector{Float64}         # RX2 - RX1 tone phase per block
end
ToneMon(fs, bufsize) = ToneMon(fs, bufsize, zeros(2), zeros(2), false, Float64[], Float64[], Float64[], Float64[])

function tone_update!(tm::ToneMon, x1, x2)
    f1, p1 = tone_fit(x1, tm.fs); f2, p2 = tone_fit(x2, tm.fs)
    if tm.have
        r = Float64[]
        for (k, (f, p)) in enumerate(((f1, p1), (f2, p2)))
            favg = 0.5 * (f + tm.prev_f[k])
            push!(r, wrap_pi(p - (tm.prev_ph[k] + 2pi * favg * tm.bufsize / tm.fs)))
        end
        push!(tm.res1, r[1]); push!(tm.res2, r[2])
    end
    push!(tm.freqs, f1)
    push!(tm.xphase, wrap_pi(p2 - p1))
    tm.prev_f .= (f1, f2); tm.prev_ph .= (p1, p2); tm.have = true
    return nothing
end

function tone_report(tm::ToneMon, f_lo::Float64; thresh = 0.25)
    println("Tone monitor ($(length(tm.freqs)) blocks):")
    isempty(tm.freqs) && return nothing
    @printf("  tone at f_lo %+.3f kHz (true %.6f MHz), spread %.3f Hz over the run\n",
            median(tm.freqs) / 1e3, (f_lo + median(tm.freqs)) / 1e6, maximum(tm.freqs) - minimum(tm.freqs))
    bad = [i for i in eachindex(tm.res1) if abs(tm.res1[i]) > thresh || abs(tm.res2[i]) > thresh]
    @printf("  boundary phase residual: RX1 rms %.4f rad, RX2 rms %.4f rad (threshold %.2f)\n",
            sqrt(mean(abs2, tm.res1)), sqrt(mean(abs2, tm.res2)), thresh)
    if isempty(bad)
        println("  no phase discontinuity found at block boundaries (no evidence of dropped samples)")
    else
        println("  DISCONTINUITY at block boundaries $(bad[1:min(end, 20)])$(length(bad) > 20 ? " ..." : "") -> samples were probably dropped")
    end
    xp = unwrap_simple(tm.xphase)
    @printf("  RX2-RX1 tone phase: mean %.2f deg, peak-to-peak drift %.2f deg\n",
            rad2deg(mean(xp)), rad2deg(maximum(xp) - minimum(xp)))
    return bad
end

function unwrap_simple(v)
    out = copy(v)
    for i in 2:length(out)
        out[i] = out[i-1] + wrap_pi(v[i] - out[i-1])
    end
    return out
end

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------
function mode_check(cfg::Config)
    r, rb = open_verified(cfg)
    nsegb, bufsize = block_geometry(cfg)
    fs = Float64(rb.fs1)
    try
        for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
        fx = FX(cfg.nfft, nsegb, cfg.window)
        fsc = 0.0f0
        s1 = nothing; s2 = nothing
        nblk = 8
        for i in 1:nblk
            b = read_block!(r, (i - 1) * bufsize)
            fsc == 0.0f0 && (fsc = detect_fullscale(b))
            x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
            s1 = chan_stats(x1); s2 = chan_stats(x2)
            accumulate!(fx, x1, x2); fx.blocks += 1
            fx.pw1 += s1.power; fx.pw2 += s2.power; fx.clip1 += s1.clip; fx.clip2 += s2.clip
            fx.t_first = b.utc; fx.t_last = b.utc
        end
        rec = finalize!(fx)
        sm = summarize(rec, cfg.dc_mask)
        println("\nFull scale detected: +-$(Int(fsc)) counts ($(fsc == 2048.0f0 ? "12-bit, right-aligned" : "left-aligned in 16 bits"))")
        println("Per-channel health (last block):")
        for (n, s) in (("RX1", s1), ("RX2", s2))
            @printf("  %s  power %.3e (rms %.1f counts)  peak %.3f FS  clipped %.2e  DC %+.2e%+.2ej\n",
                    n, s.power, sqrt(s.power) * fsc, s.peak, s.clip, real(s.dc), imag(s.dc))
        end
        f = bin_freqs(cfg.nfft, fs, Float64(rb.lo))
        c = cfg.nfft ÷ 2 + 1
        for (n, P) in (("RX1", rec.P1), ("RX2", rec.P2))
            Pm = Float64.(P); Pm[max(1, c - cfg.dc_mask):min(end, c + cfg.dc_mask)] .= -Inf
            k = argmax(Pm)
            @printf("  %s DC spike %.1f dB above median;  spectral peak (DC excluded) at %.6f MHz (baseband %+.2f kHz), %.1f dB above median\n",
                    n, 10log10(P[c] / median(P)), f[k] / 1e6, (f[k] - rb.lo) / 1e3, 10log10(P[k] / median(P)))
        end
        @printf("Cross-correlation, DC bins (+-%d) excluded: band |rho| = %.4f, phase %.1f deg;  at peak bin |rho| = %.4f, phase %.1f deg\n",
                cfg.dc_mask, abs(sm.rho), rad2deg(angle(sm.rho)), abs(sm.rho_pk), rad2deg(angle(sm.rho_pk)))
        @printf("Cross-correlation at the DC bin alone: |rho| = %.4f, phase %.1f deg  (a high value here is shared LO leakage / offset, not sky)\n",
                abs(sm.rho_dc), rad2deg(angle(sm.rho_dc)))
        println("|rho| by distance from DC (bin width $(round(fs / cfg.nfft, digits = 1)) Hz); a floor well above the noise level means correlated hardware pickup:")
        for (lo_b, hi_b) in ((1, 4), (5, 20), (21, 100), (101, 500), (501, cfg.nfft ÷ 2 - 1))
            idx = filter(i -> 1 <= i <= cfg.nfft, vcat((c - hi_b):(c - lo_b), (c + lo_b):(c + hi_b)))
            isempty(idx) && continue
            rho_b = sum(rec.V[idx]) / sqrt(sum(rec.P1[idx]) * sum(rec.P2[idx]))
            @printf("  %4d..%-4d bins (%7.1f..%-7.1f kHz): |rho| = %.4f  (noise floor ~ %.4f)\n", lo_b, hi_b,
                    lo_b * fs / cfg.nfft / 1e3, hi_b * fs / cfg.nfft / 1e3, abs(rho_b), 1 / sqrt(fx.nseg * length(idx)))
        end
        @printf("Noise floor for |rho| with this many segments: about %.4f per bin, %.5f for the whole band\n",
                1 / sqrt(fx.nseg), 1 / sqrt(fx.nseg * cfg.nfft))
        println("Flags: ", flagstr(rec.flags))
        println("\nExpected with independent antennas on noise: |rho| near 0. With one source split to both: near 1 at the tone bin.")
    finally
        close_radio!(r)
    end
end

function mode_rate(cfg::Config)
    r, rb = open_verified(cfg)
    _, bufsize = block_geometry(cfg)
    fs = Float64(rb.fs1)
    nblk = ceil(Int, cfg.duration_s * fs / bufsize)
    tm = cfg.tone ? ToneMon(fs, bufsize) : nothing
    t = Float64[]; nshort = 0; fsc = 0.0f0
    println("\nCapturing $(nblk) blocks of $(bufsize) samples ($(round(nblk * bufsize / fs, digits = 1)) s at $(fs / 1e6) Msps)...")
    try
        for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
        t0 = time_ns()
        for i in 1:nblk
            b = read_block!(r, (i - 1) * bufsize)
            push!(t, (time_ns() - t0) / 1e9)
            b.short && (nshort += 1)
            if tm !== nothing
                fsc == 0.0f0 && (fsc = detect_fullscale(b))
                tone_update!(tm, to_complex(b.i1, b.q1, fsc), to_complex(b.i2, b.q2, fsc))
            end
            i % max(1, nblk ÷ 10) == 0 && @printf("  %3d%%  %.1f s elapsed\n", round(Int, 100i / nblk), t[end])
        end
    catch e
        e isa InterruptException || (close_radio!(r); rethrow())
        println("Interrupted.")
    finally
        close_radio!(r)
    end
    n = length(t)
    n < 2 && return
    wall = t[end]
    expected = n * bufsize / fs
    ratio = wall / expected
    half = n ÷ 2
    fs_est = half >= 2 ? (n - half) * bufsize / (t[end] - t[half]) : NaN
    dts = diff(t)
    println("\nResult:")
    @printf("  blocks %d, wall time %.2f s, expected %.2f s, wall/expected = %.4f\n", n, wall, expected, ratio)
    @printf("  sample rate from host clock (2nd half): %.1f Hz  (hardware reports %d Hz)\n", fs_est, rb.fs1)
    @printf("  per-block time: min %.1f ms, median %.1f ms, max %.1f ms (nominal %.1f ms)\n",
            1e3minimum(dts), 1e3median(dts), 1e3maximum(dts), 1e3bufsize / fs)
    println("  short reads: $nshort")
    if ratio > 1.02
        println("  VERDICT: FALLING BEHIND -- the host/USB cannot sustain $(fs / 1e6) Msps; lower fs. Samples may be lost.")
    elseif ratio < 0.98
        println("  VERDICT: faster than real time -- the hardware is NOT running at the reported rate (buffering artefact or wrong rate).")
    else
        println("  VERDICT: keeping pace with real time. (This alone does not prove nothing was dropped; use tone=true with a tone source.)")
    end
    tm === nothing || tone_report(tm, Float64(rb.lo))
    return nothing
end

function mode_run(cfg::Config)
    r, rb = open_verified(cfg)
    nsegb, bufsize = block_geometry(cfg)
    fs = Float64(rb.fs1)
    Threads.nthreads() < 2 && println("NOTE: only 1 Julia thread; run `julia -t 3 V1.jl ...` for headroom.")
    isdir(cfg.out_dir) || mkpath(cfg.out_dir)
    base = joinpath(cfg.out_dir, "V1_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * (isempty(cfg.tag) ? "" : "_" * cfg.tag))
    vis_io = open(base * ".vis", "w")
    raw_ios = cfg.save_raw ? (open(base * "_rx1.cf32", "w"), open(base * "_rx2.cf32", "w")) : nothing
    nblk = ceil(Int, cfg.duration_s * fs / bufsize)
    println("\nRun: $(nblk) blocks, integrating $(cfg.t_int) s, nfft $(cfg.nfft) ($(round(fs / cfg.nfft, digits = 1)) Hz bins)")
    println("Output: $(base).vis (+ .meta)\n")

    function on_record(rec::Record)
        write_record(vis_io, rec); flush(vis_io)
        sm = summarize(rec, cfg.dc_mask)
        f = Float64(rb.lo) + ((sm.k - 1) - cfg.nfft ÷ 2) * fs / cfg.nfft
        @printf("%s  |rho|=%.4f ph=%7.2f deg | peak %.4f MHz |rho|=%.3f | P1=%.1e P2=%.1e | %s\n",
                Dates.format(unix2datetime(rec.t_start), "HH:MM:SS"), abs(sm.rho), rad2deg(angle(sm.rho)),
                f / 1e6, abs(sm.rho_pk), rec.pow1, rec.pow2, flagstr(rec.flags))
    end

    q = Channel{Block}(cfg.queue_depth)
    task = Threads.@spawn consume!(q, cfg, fs, on_record; raw_ios = raw_ios)
    bind(q, task)
    t_wall = time()
    summary = nothing
    backlog = Int[]; t_read = Float64[]; t_put = Float64[]      # diagnostics: queue depth, time in refill, time waiting to enqueue
    # The capture loop runs in its own task. Its blocking libiio read would otherwise tie up the main thread, and the
    # processing task could not run there (it was observed to leave the queue about 90% full); with capture on a
    # worker thread and the main thread only waiting, the other threads stay free for processing.
    stop = Threads.Atomic{Bool}(false)
    producer = Threads.@spawn begin
        err = nothing
        try
            for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
            for i in 1:nblk
                stop[] && break
                push!(backlog, Base.n_avail(q))
                full = backlog[end] >= cfg.queue_depth
                t0 = time(); blk = read_block!(r, (i - 1) * bufsize; qfull = full); t1 = time()
                put!(q, blk); t2 = time()
                push!(t_read, t1 - t0); push!(t_put, t2 - t1)
                Threads.nthreads() == 1 && yield()
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
    close_radio!(r)
    close(q)
    summary = fetch(task)
    cap_err === nothing || println("CAPTURE ERROR: ", cap_err, "  (the output below covers only the data captured before it)")
    wall = time() - t_wall
    close(vis_io)
    raw_ios === nothing || foreach(close, raw_ios)
    write_meta(base * ".meta", cfg, rb, summary, Dict("wall_seconds" => round(wall, digits = 2),
                                                      "expected_seconds" => round(nblk * bufsize / fs, digits = 2)))
    println("\nDone. $(summary.records) records, $(summary.blocks) blocks, wall $(round(wall, digits = 1)) s " *
            "(expected $(round(nblk * bufsize / fs, digits = 1)) s).")
    summary.short > 0 && println("WARNING: $(summary.short) short read(s).")
    summary.qfull > 0 && println("WARNING: processing queue was full $(summary.qfull) time(s); the host fell behind.")
    if !isempty(backlog)
        @printf("Diagnostics: queue backlog max %d (of %d), mean %.1f; refill time median %.0f ms, max %.0f ms (block = %.0f ms); wait to enqueue max %.0f ms\n",
                maximum(backlog), cfg.queue_depth, mean(backlog), 1e3median(t_read), 1e3maximum(t_read), 1e3bufsize / fs, 1e3maximum(t_put))
        @printf("             processing time per block: median %.1f ms, max %.1f ms (thread count %d)\n", summary.proc_med_ms, summary.proc_max_ms, Threads.nthreads())
    end
end

function mode_show(cfg::Config)
    path = cfg.path
    if isempty(path)
        files = filter(f -> endswith(f, ".vis"), readdir(cfg.out_dir; join = true))
        isempty(files) && error("No .vis files in $(cfg.out_dir)")
        path = last(sort(files, by = mtime))
    end
    meta = read_meta(replace(path, r"\.vis$" => ".meta"))
    nfft = parse(Int, meta["nfft"]); fs = parse(Float64, meta["fs_hz"]); lo = parse(Float64, meta["f_lo_hz"])
    recs = read_records(path, nfft)
    println("$(path): $(length(recs)) records, nfft $nfft, fs $(fs / 1e6) Msps, LO $(lo / 1e6) MHz")
    f = bin_freqs(nfft, fs, lo)
    for (i, rec) in enumerate(recs)
        sm = summarize(rec, cfg.dc_mask)
        @printf("%4d %s  |rho|=%.4f ph=%7.2f deg  peak %.4f MHz |rho|=%.3f  %s\n", i,
                Dates.format(unix2datetime(rec.t_start), "HH:MM:SS"), abs(sm.rho), rad2deg(angle(sm.rho)),
                f[sm.k] / 1e6, abs(sm.rho_pk), flagstr(rec.flags))
    end
end

# ---------------------------------------------------------------------
# Self-test (no hardware)
# ---------------------------------------------------------------------
function synth_stream(n::Int, fs::Float64, f_lo::Float64, baseline::Float64, theta_deg::Float64, snr_db::Float64; seed = 1)
    rng = MersenneTwister(seed)
    tau = baseline * sind(theta_deg) / C_LIGHT
    s = randn(rng, ComplexF64, n)
    sd = ifft(fft(s) .* cis.(-2pi .* fftfreq(n, fs) .* tau))        # envelope delay
    s2 = sd .* cis(-2pi * f_lo * tau)                                # carrier phase
    npow = 10^(-snr_db / 10)
    n1 = sqrt(npow) .* randn(rng, ComplexF64, n); n2 = sqrt(npow) .* randn(rng, ComplexF64, n)
    return s .+ n1, s2 .+ n2, tau, npow
end

function to_blocks(x1, x2, bufsize; counts_rms = 250.0, drop_at = Int[], drop_n = 0)
    # scale so each of I and Q has about `counts_rms` counts, clip to the 12-bit range
    sc = counts_rms / std(real.(x1))
    cl(v) = Int16.(clamp.(round.(v .* sc), -2047, 2047))
    blocks = Block[]
    pos = 1; idx = 0
    nb = length(x1) ÷ bufsize
    for i in 1:nb
        i in drop_at && (pos += drop_n)
        rng = pos:pos+bufsize-1
        push!(blocks, Block(idx, now(UTC), time_ns(), cl(real.(x1[rng])), cl(imag.(x1[rng])),
                            cl(real.(x2[rng])), cl(imag.(x2[rng])), false, false))
        pos += bufsize; idx += bufsize
    end
    return blocks
end

function mode_selftest(cfg::Config)
    cfg.nfft = 4096; cfg.fs = 2.5e6; cfg.f_lo = 1420.0e6; cfg.window = "hann"; cfg.t_int = 0.5
    nsegb, bufsize = block_geometry(cfg)
    ok = true
    println("== Test 1: FX correlator vs known delay (noise-like source, 20 m baseline, 15 deg off broadside)")
    snr_db = 10.0
    nblk = 10
    x1, x2, tau, npow = synth_stream(nblk * bufsize, cfg.fs, cfg.f_lo, 20.0, 15.0, snr_db)
    blocks = to_blocks(x1, x2, bufsize)
    recs = Record[]
    q = Channel{Block}(length(blocks)); foreach(b -> put!(q, b), blocks); close(q)
    summ = consume!(q, cfg, cfg.fs, rec -> push!(recs, rec))
    f = bin_freqs(cfg.nfft, cfg.fs, cfg.f_lo)
    expected_mag = 1 / (1 + npow)
    errs = Float64[]; mags = Float64[]
    for rec in recs
        for k in cfg.nfft ÷ 8 : cfg.nfft - cfg.nfft ÷ 8
            rho = rec.V[k] / sqrt(rec.P1[k] * rec.P2[k])
            push!(errs, abs(wrap_pi(angle(rho) - 2pi * f[k] * tau))); push!(mags, abs(rho))
        end
    end
    med_err = rad2deg(median(errs)); med_mag = median(mags)
    @printf("  records %d, injected delay %.2f ns\n", length(recs), tau * 1e9)
    @printf("  median |phase error| per bin %.3f deg (pass < 2)\n", med_err)
    @printf("  median |rho| %.4f, expected %.4f (pass within 5%%)\n", med_mag, expected_mag)
    pass1 = med_err < 2.0 && abs(med_mag / expected_mag - 1) < 0.05 && summ.records == length(recs) && length(recs) >= 1
    println("  ", pass1 ? "PASS" : "FAIL"); ok &= pass1

    println("== Test 2: tone monitor detects a dropped-sample gap")
    n = 8 * bufsize
    f_tone = 405.75e3
    tone = cis.(2pi * f_tone .* (0:n+100) ./ cfg.fs)
    noise() = 0.03 .* randn(ComplexF64, n + 101)
    t1 = tone .+ noise(); t2 = tone .* cis(0.7) .+ noise()
    tb = to_blocks(t1, t2, bufsize; drop_at = [5], drop_n = 5)
    tm = ToneMon(cfg.fs, bufsize)
    fsc = detect_fullscale(tb[1])
    for b in tb
        tone_update!(tm, to_complex(b.i1, b.q1, fsc), to_complex(b.i2, b.q2, fsc))
    end
    bad = tone_report(tm, cfg.f_lo)
    pass2 = bad == [4]       # residual i compares block i+1 with block i; the gap is before block 5
    println("  flagged boundaries: ", bad, " (expected [4])  ", pass2 ? "PASS" : "FAIL"); ok &= pass2

    println("== Test 3: record file round trip")
    path = joinpath(mktempdir(), "t.vis")
    open(path, "w") do io; foreach(rec -> write_record(io, rec), recs); end
    back = read_records(path, cfg.nfft)
    pass3 = length(back) == length(recs) && all(back[i].V == recs[i].V && back[i].P1 == recs[i].P1 &&
                                                back[i].t_start == recs[i].t_start for i in eachindex(recs))
    println("  ", pass3 ? "PASS" : "FAIL"); ok &= pass3

    println("== Test 4: Bessel J1 and a short simulated observation (point source, east-west baseline)")
    jok = abs(besselj1(1.0) - 0.4400505857) < 1e-6 && abs(besselj1(5.0) + 0.3275791376) < 1e-6 &&
          abs(besselj1(10.0) - 0.0434727462) < 1e-6 && abs(besselj1(3.8317059702)) < 1e-6
    println("  J1 table values: ", jok ? "PASS" : "FAIL"); ok &= jok
    scfg = Config(); scfg.nfft = 4096; scfg.fs = 2.5e6; scfg.f_lo = 1420.0e6; scfg.window = "hann"; scfg.t_int = 0.5
    scfg.sim_model = "point"; scfg.sim_snr_db = -17.0; scfg.sim_hours = 0.12; scfg.sim_step_s = 120.0
    scfg.sim_phase_off_deg = 30.0
    sres = simulate_observation(scfg; write_files = false, verbose = false)
    @printf("  %d records, median |phase error| %.2f deg = %.2f x expected noise, |rho| ratio %.3f\n",
            sres.records, sres.med_err_deg, sres.med_err_sigma, sres.med_ratio)
    pass4 = sres.pass && sres.records >= 3
    println("  ", pass4 ? "PASS" : "FAIL"); ok &= pass4

    println("== Test 5: antenna inputs (dish diameter to effective area and beamwidth)")
    acfg = Config(); acfg.sim_dish_m = 10.0; acfg.sim_eff = 0.55; acfg.sim_track = false
    ant = sim_antenna(acfg)
    aok = abs(ant.area - 0.55 * pi * 25) < 1e-9 && abs(ant.hpbw_deg - rad2deg(1.15 * C_LIGHT / 1420e6 / 10)) < 1e-9 && ant.fixed
    @printf("  D = 10 m, eff 0.55: A_e = %.1f m^2, HPBW = %.2f deg, fixed = %s\n", ant.area, ant.hpbw_deg, ant.fixed)
    println("  ", aok ? "PASS" : "FAIL"); ok &= aok

    println("== Test 6: delay from the phase-versus-frequency slope (synthetic tone sweep with phase wrapping)")
    rngt = MersenneTwister(7)
    fb6 = collect(-1.0e6:2.0e5:1.0e6)
    for tau_true in (13.0, 50.5, 101.0)
        ph6 = [rad2deg(wrap_pi(deg2rad(30.0 + 360 * tau_true * 1e-9 * f + 0.6 * randn(rngt)))) for f in fb6]
        fd = fit_delay(fb6, ph6, fill(0.6, length(fb6)))
        dok = abs(fd.tau_ns - tau_true) < max(0.5, 3 * fd.tau_err_ns) && abs(fd.phi0 - 30.0) < 2.0
        @printf("  true %.1f ns: fitted %.2f +- %.2f ns, phase at LO %.1f deg  %s\n", tau_true, fd.tau_ns, fd.tau_err_ns, fd.phi0, dok ? "PASS" : "FAIL")
        ok &= dok
    end

    println("\n", ok ? "SELFTEST PASSED" : "SELFTEST FAILED")
    return ok
end

# ---------------------------------------------------------------------
# Simulated observation (`sim` mode). No hardware needed.
#
# Generates the two channels a pair of dishes would deliver for a chosen source as the Earth rotates, and runs
# them through the SAME consume!/FX code as real captures, writing the same .vis/.meta files.
# Physics (see the vault note "Simulating the source" in the Thompson note):
#   * the source is complex Gaussian noise (Thompson Ch. 1 s1.2.1), common to both channels with weight equal to the
#     source's normalized visibility; the rest of the source power and all receiver noise are independent per channel
#   * source-to-noise power ratio per channel is T_A/T_S (Thompson Eq. 1.6, 6.45-6.51); expected correlation is
#     rho = a2*v/(1+a2) before the delay phase, a2 = T_A/T_S, v = |V_norm| * beam factor
#   * the delay of channel 2 is the geometric delay -b.s/c (carrier phase + fractional-sample envelope delay)
#   * convention: V = x1*conj(x2); phase = +2*pi*f*tau; for a point source at the phase centre this is -2*pi*w
#     (Thompson Eq. 4.1 for u,v,w)
# ---------------------------------------------------------------------
const OMEGA_E = 7.2921159e-5      # Earth rotation rate (rad/s)
const HI_REST = 1420.405751e6     # Hz

# Bessel J1 (Numerical-Recipes rational approximation, abs error ~1e-8); avoids a SpecialFunctions dependency.
function besselj1(x::Float64)
    ax = abs(x)
    if ax < 8.0
        y = x * x
        n = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        d = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))))
        return n / d
    end
    z = 8.0 / ax; y = z * z; xx = ax - 2.356194491
    a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
    a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
    ans = sqrt(0.636619772 / ax) * (cos(xx) * a1 - z * sin(xx) * a2)
    return x < 0 ? -ans : ans
end

# Visibility (un-normalized) of a uniform disk of angular radius a (rad) at projected baseline q (wavelengths): Thompson Eq. 10.80
disk_vis(a, q) = (x = 2pi * a * q; x < 1e-9 ? pi * a^2 : pi * a^2 * 2 * besselj1(x) / x)

# Normalized visibility of the chosen source model at projected baseline q (wavelengths). Real, can be negative.
function model_vis(cfg::Config, q::Float64)
    th = cfg.sim_diam_arcmin * pi / (180 * 60)        # diameter (disk, sun) or FWHM (gaussian), rad
    m = cfg.sim_model
    if m == "point"
        return 1.0
    elseif m == "disk"
        x = pi * th * q
        return x < 1e-9 ? 1.0 : 2 * besselj1(x) / x
    elseif m == "gaussian"
        return exp(-(pi * th * q)^2 / (4 * log(2)))   # Thompson Table 10.2 with FWHM = sqrt(8 ln2) a
    elseif m == "sun"                                 # stacked concentric disks, Thompson Eq. 10.82-10.84
        a = th / 2; N = 64; num = 0.0; den = 0.0
        for i in 1:N
            a_in = a * (i - 1) / N; a_out = a * i / N
            I = 1.0 + cfg.sim_limb * (((a_in + a_out) / 2) / a)^8
            num += I * (disk_vis(a_out, q) - disk_vis(a_in, q))
            den += I * pi * (a_out^2 - a_in^2)
        end
        return num / den
    end
    error("sim_model must be point, disk, gaussian or sun")
end

# Baseline geometry at simulated time tsec: (u,v,w) in wavelengths of the LO, geometric delay of channel 2 (s), altitude.
function sim_geometry(cfg::Config, tsec::Float64)
    lat = deg2rad(cfg.sim_lat_deg); dec = deg2rad(cfg.sim_dec_deg)
    H = deg2rad(15 * cfg.sim_start_ha_h) + OMEGA_E * tsec
    az = deg2rad(cfg.sim_az_deg)
    east = cfg.sim_baseline_m * sin(az); north = cfg.sim_baseline_m * cos(az)
    X = -north * sin(lat); Y = east; Z = north * cos(lat)           # equatorial frame, metres
    bs = X * cos(dec) * cos(H) - Y * cos(dec) * sin(H) + Z * sin(dec)    # baseline . source direction
    lam = C_LIGHT / cfg.f_lo
    u = (sin(H) * X + cos(H) * Y) / lam
    v = (-sin(dec) * cos(H) * X + sin(dec) * sin(H) * Y + cos(dec) * Z) / lam
    alt = asin(clamp(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(H), -1, 1))
    return (H = H, u = u, v = v, w = bs / lam, tau = -bs / C_LIGHT, alt = alt)
end

# One block of synthetic samples for a fixed geometry. All sources are expressed in units of the receiver-noise
# variance (unit complex Gaussian noise per channel).
function sim_block(rng, n::Int, sp, idx::Int, utc::DateTime)::Block
    cw() = randn(rng, ComplexF32, n)
    C  = (sp.a * sp.sv) .* cw()                  # source component common to both channels
    I1 = (sp.a * sp.si) .* cw()                  # source power that is NOT common (resolved structure)
    I2 = (sp.a * sp.si) .* cw()
    if sp.gline !== nothing                      # spectral line: coloured noise, common part weighted by the line's visibility
        colored(w) = ifft(fft(w) .* sp.gline)
        C  .+= sp.sqvl .* colored(cw())
        I1 .+= sp.sqnl .* colored(cw())
        I2 .+= sp.sqnl .* colored(cw())
    end
    x1 = C .+ I1 .+ cw()
    Cd = ifft(fft(C) .* sp.ramp) .* sp.carrier   # channel 2 sees the common component delayed by tau_g
    x2 = (Cd .+ I2 .+ cw()) .* sp.gain2
    if sp.dc != 0
        x1 .+= sp.dc1; x2 .+= sp.dc2
    end
    cl(v) = Int16.(clamp.(round.(v .* sp.scale), -2047, 2047))
    return Block(idx, utc, time_ns(), cl(real.(x1)), cl(imag.(x1)), cl(real.(x2)), cl(imag.(x2)), false, false)
end

# Expected band-average rho for the same DC mask the measurement uses (white continuum plus optional line).
function expected_band_rho(cfg::Config, fs::Float64, a2::Float64, v::Float64, tau::Float64, theta::Float64, neg::Bool)
    nfft = cfg.nfft
    fb = ((0:nfft-1) .- nfft ÷ 2) .* (fs / nfft)
    ell = cfg.sim_line ? cfg.sim_line_amp .* exp.(-4 * log(2) .* ((fb .- sim_line_center(cfg)) ./ sim_line_width(cfg)) .^ 2) : zeros(nfft)
    c = nfft ÷ 2 + 1
    keep = trues(nfft); keep[max(1, c - cfg.dc_mask):min(nfft, c + cfg.dc_mask)] .= false
    ph = 2pi .* (cfg.f_lo .+ fb) .* tau .+ theta .+ (neg ? pi : 0.0)
    cross = (a2 * v .+ ell .* cfg.sim_line_vis) .* cis.(ph)
    return sum(cross[keep]) / sum((1 + a2) .+ ell[keep])
end
sim_line_center(cfg::Config) = HI_REST * (1 - cfg.sim_line_v_kms / (C_LIGHT / 1e3)) - cfg.f_lo
sim_line_width(cfg::Config) = cfg.sim_line_fwhm_kms / (C_LIGHT / 1e3) * HI_REST

# Antenna inputs derived from the dish diameter: effective area A_e = eff*pi*D^2/4, primary-beam FWHM = k*lambda/D, and
# whether the dishes are fixed (beam loss as the source drifts) or tracking (none).
function sim_antenna(cfg::Config)
    lam = C_LIGHT / cfg.f_lo
    area = isfinite(cfg.sim_area_m2) ? cfg.sim_area_m2 :
           (isfinite(cfg.sim_dish_m) ? cfg.sim_eff * pi * cfg.sim_dish_m^2 / 4 : NaN)
    hpbw = cfg.sim_hpbw_deg > 0 ? cfg.sim_hpbw_deg :
           (isfinite(cfg.sim_dish_m) ? rad2deg(cfg.sim_beam_k * lam / cfg.sim_dish_m) : 0.0)
    fixed = !cfg.sim_track || cfg.sim_hpbw_deg > 0
    return (area = area, hpbw_deg = hpbw, fixed = fixed)
end

function simulate_observation(cfg::Config; write_files::Bool = true, verbose::Bool = true)
    validate(cfg)
    nsegb, bufsize = block_geometry(cfg)
    fs = cfg.fs
    per_int = max(1, round(Int, cfg.t_int * fs / bufsize))
    blk_secs = bufsize / fs
    rng = MersenneTwister(cfg.sim_seed)
    ant = sim_antenna(cfg)
    ant.fixed && ant.hpbw_deg <= 0 && error("Fixed dishes need a beamwidth: set sim_dish_m or sim_hpbw_deg")
    physical = all(isfinite, (cfg.sim_flux_jy, ant.area, cfg.sim_tsys_k))
    a2 = physical ? cfg.sim_flux_jy * ant.area / (2 * 1.380649e-23 / 1e-26) / cfg.sim_tsys_k : 10^(cfg.sim_snr_db / 10)
    theta = deg2rad(cfg.sim_phase_off_deg)
    nrec = floor(Int, cfg.sim_hours * 3600 / cfg.sim_step_s) + 1
    utc0 = now(UTC)

    w = cfg.window == "hann" ? 0.5 .- 0.5 .* cos.(2pi .* (0:cfg.nfft-1) ./ cfg.nfft) : ones(cfg.nfft)
    sigma_comp = sqrt(sum(w .^ 4)) / (sum(w .^ 2) * sqrt(2 * per_int * nsegb))     # noise per component of band-average rho

    verbose && @printf("Simulated observation: model %s, %.1f arcmin, T_A/T_S = %.4g (%.1f dB), baseline %.1f m az %.0f deg, dec %.1f, lat %.1f\n",
                       cfg.sim_model, cfg.sim_diam_arcmin, a2, 10log10(a2), cfg.sim_baseline_m, cfg.sim_az_deg, cfg.sim_dec_deg, cfg.sim_lat_deg)
    verbose && @printf("  %d record times over %.1f h (step %.0f s), %.2f s of data each, band-average rho noise per component about %.2e\n",
                       nrec, cfg.sim_hours, cfg.sim_step_s, per_int * blk_secs, sigma_comp)
    verbose && println("  antenna: ", isfinite(ant.area) ? @sprintf("A_e = %.1f m^2 (%.0f Jy/K); ", ant.area, 2761.3 / ant.area) : "A_e unknown; ",
                       ant.fixed ? @sprintf("fixed dishes, HPBW %.2f deg", ant.hpbw_deg) : "tracking dishes (no beam loss)")
    g0 = sim_geometry(cfg, 0.0); g1 = sim_geometry(cfg, 1.0)
    verbose && @printf("  start: |u,v| = %.1f wavelengths, fringe rate about %.2f mHz (period %.0f s)\n",
                       hypot(g0.u, g0.v), 1e3 * abs(g1.w - g0.w), 1 / max(abs(g1.w - g0.w), 1e-12))

    # per-run constants
    f = fftfreq(bufsize, fs)
    gline = cfg.sim_line ? Float32.(sqrt(cfg.sim_line_amp) .* exp.(-2 * log(2) .* ((f .- sim_line_center(cfg)) ./ sim_line_width(cfg)) .^ 2)) : nothing
    dcamp = ComplexF32(cfg.sim_dc)
    truths = NamedTuple[]
    errs = Float64[]; errs_sig = Float64[]; ratios = Float64[]; snr_hi = Bool[]
    idx_ref = Ref(0)
    tlock = ReentrantLock()

    base = joinpath(cfg.out_dir, "V1_" * Dates.format(now(UTC), "yyyymmdd_HHMMSS") * "_sim" * (isempty(cfg.tag) ? "" : "_" * cfg.tag))
    vis_io = nothing; csv_io = nothing
    if write_files
        isdir(cfg.out_dir) || mkpath(cfg.out_dir)
        vis_io = open(base * ".vis", "w"); csv_io = open(base * ".truth.csv", "w")
        println(csv_io, "t_start_unix,hour_angle_deg,alt_deg,u,v,w,tau_ns,vnorm,beam,rho_exp_abs,rho_exp_phase_deg,rho_meas_abs,rho_meas_phase_deg,phase_err_deg")
    end

    function on_record(rec::Record)
        idx_ref[] += 1
        tr = lock(() -> truths[idx_ref[]], tlock)
        sm = summarize(rec, cfg.dc_mask)
        err = wrap_pi(angle(sm.rho) - angle(tr.rho_exp))
        σφ = sigma_comp / max(abs(tr.rho_exp), 1e-12)
        if abs(tr.rho_exp) > sigma_comp        # phase is only meaningful when the expected signal exceeds the noise
            push!(errs, abs(err)); push!(errs_sig, abs(err) / σφ); push!(ratios, abs(sm.rho) / abs(tr.rho_exp))
            push!(snr_hi, abs(tr.rho_exp) / sigma_comp > 6)
        end
        write_files && (write_record(vis_io, rec); flush(vis_io))
        write_files && @printf(csv_io, "%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.5f,%.5f,%.6g,%.3f,%.6g,%.3f,%.3f\n", rec.t_start, rad2deg(tr.H), rad2deg(tr.alt),
                               tr.u, tr.v, tr.w, tr.tau * 1e9, tr.vnorm, tr.beam, abs(tr.rho_exp), rad2deg(angle(tr.rho_exp)),
                               abs(sm.rho), rad2deg(angle(sm.rho)), rad2deg(err))
        verbose && @printf("H %6.1f deg alt %4.1f  q %6.1f  tau %6.2f ns  |V| %.3f | exp |rho| %.4f ph %7.2f | meas |rho| %.4f ph %7.2f | err %6.2f deg (sigma %.1f)\n",
                           rad2deg(tr.H), rad2deg(tr.alt), hypot(tr.u, tr.v), tr.tau * 1e9, tr.vnorm * tr.beam, abs(tr.rho_exp), rad2deg(angle(tr.rho_exp)),
                           abs(sm.rho), rad2deg(angle(sm.rho)), rad2deg(err), rad2deg(σφ))
    end

    q = Channel{Block}(cfg.queue_depth)
    task = Threads.@spawn consume!(q, cfg, fs, on_record)
    bind(q, task)
    nskipped = 0
    try
        for k in 0:nrec-1
            tsec = k * cfg.sim_step_s
            g = sim_geometry(cfg, tsec)
            if g.alt < deg2rad(cfg.sim_min_alt_deg)
                nskipped += 1; continue
            end
            qb = hypot(g.u, g.v)
            vn = model_vis(cfg, qb)
            beam = ant.fixed ? exp(-4 * log(2) * (g.H * cos(deg2rad(cfg.sim_dec_deg)) / deg2rad(ant.hpbw_deg))^2) : 1.0
            v = abs(vn) * beam
            neg = vn < 0
            rho_exp = expected_band_rho(cfg, fs, a2, v, g.tau, theta, neg)
            lock(tlock) do; push!(truths, (H = g.H, alt = g.alt, u = g.u, v = g.v, w = g.w, tau = g.tau, vnorm = abs(vn), beam = beam, rho_exp = rho_exp)); end
            ramp = ComplexF32.(cis.(-2pi .* f .* g.tau))
            carrier = ComplexF32(cis(-(2pi * cfg.f_lo * g.tau + theta + (neg ? pi : 0.0))))
            sigma_t = sqrt(1 + a2 + cfg.sim_dc^2)
            sp = (a = Float32(sqrt(a2)), sv = Float32(sqrt(v)), si = Float32(sqrt(1 - v)), ramp = ramp, carrier = carrier,
                  gline = gline, sqvl = Float32(sqrt(cfg.sim_line_vis)), sqnl = Float32(sqrt(1 - cfg.sim_line_vis)),
                  gain2 = Float32(10^(cfg.sim_gain2_db / 20)), dc = cfg.sim_dc,
                  dc1 = dcamp, dc2 = ComplexF32(cfg.sim_dc * cis(deg2rad(cfg.sim_dc_phase_deg))),
                  scale = Float32(250.0 / sqrt(sigma_t^2 / 2)))
            for j in 1:per_int
                put!(q, sim_block(rng, bufsize, sp, (j - 1) * bufsize, utc0 + Millisecond(round(Int, 1000 * (tsec + j * blk_secs)))))
                Threads.nthreads() == 1 && yield()
            end
        end
    finally
        close(q)
    end
    summary = fetch(task)
    if write_files
        close(vis_io); close(csv_io)
        write_meta(base * ".meta", cfg, nothing, summary, Dict("mode" => "sim", "sim_model" => cfg.sim_model, "sim_diam_arcmin" => cfg.sim_diam_arcmin,
                   "sim_ta_over_ts" => a2, "sim_baseline_m" => cfg.sim_baseline_m, "sim_az_deg" => cfg.sim_az_deg, "sim_dec_deg" => cfg.sim_dec_deg,
                   "sim_lat_deg" => cfg.sim_lat_deg, "sim_hours" => cfg.sim_hours, "sim_step_s" => cfg.sim_step_s, "sim_phase_off_deg" => cfg.sim_phase_off_deg,
                   "sim_line" => cfg.sim_line, "sim_dc" => cfg.sim_dc, "sim_seed" => cfg.sim_seed))
    end
    if isempty(errs)
        verbose && println("\nSim summary: $(summary.records) records, but none had an expected correlation above the noise (source resolved out, outside the beam, or too faint); no phase statistics.")
        return (records = summary.records, pass = false, med_err_deg = NaN, med_err_sigma = NaN, med_ratio = NaN)
    end
    med_err_deg = rad2deg(median(errs)); med_sig = median(errs_sig)
    hi = findall(snr_hi)
    med_ratio = isempty(hi) ? NaN : median(ratios[hi])
    pass = med_sig < 1.2 && (isnan(med_ratio) || abs(med_ratio - 1) < 0.1)
    if verbose
        println("\nSim summary: $(summary.records) records ($(nskipped) record times skipped below $(cfg.sim_min_alt_deg) deg altitude)")
        @printf("  median |phase error| %.2f deg = %.2f x the expected noise (0.67 expected if the noise model is right)\n", med_err_deg, med_sig)
        println("  median |rho| ratio (measured/expected, records with SNR > 6): ", isnan(med_ratio) ? "n/a (all records below SNR 6)" : @sprintf("%.3f", med_ratio))
        println("  ", pass ? "PASS" : "CHECK: measured values disagree with the simulation's own expectation")
        write_files && println("Output: $(base).vis, .meta, .truth.csv")
    end
    return (records = summary.records, pass = pass, med_err_deg = med_err_deg, med_err_sigma = med_sig, med_ratio = med_ratio)
end

mode_sim(cfg::Config) = (simulate_observation(cfg).pass || println("Sim did not pass its own consistency check."); nothing)

# ---------------------------------------------------------------------
# Tone through a splitter (`tone`) and delay from the phase slope (`delayfit`). The generator signal is split to both
# receive inputs through cables of known length difference. A completely coherent tone gives |rho| close to 1 (limited
# only by the SNR), and its RX1-RX2 phase is 2*pi*f*tau + a fixed instrument phase. The ABSOLUTE phase cannot be
# predicted from a cable length (at 1.42 GHz 1 mm of cable is about 2.5 degrees and the instrument phase is unknown),
# so the checks are (1) the phase STABILITY, (2) the SLOPE of phase against frequency, which gives the total delay
# tau, and (3) the DIFFERENCE in delay between two cable configurations, which must equal the extra length / (vf*c).
# Convention as everywhere: V = x1*conj(x2); positive phase = signal reaches RX2 later.
# ---------------------------------------------------------------------
function circ_stats(deg::Vector{Float64})
    z = mean(cis.(deg2rad.(deg)))
    return (mean = rad2deg(angle(z)), std = rad2deg(sqrt(max(-2 * log(max(abs(z), 1e-12)), 0.0))), R = abs(z))
end

function measure_tone(r::Radio, cfg::Config, rb, secs::Float64)
    nsegb, bufsize = block_geometry(cfg)
    fs = Float64(rb.fs1); flo = Float64(rb.lo)
    nblk = max(2, ceil(Int, secs * fs / bufsize))
    for _ in 1:cfg.skip_blocks; read_block!(r, 0); end
    fx = FX(cfg.nfft, nsegb, cfg.window)
    ph = Float64[]; fts = Float64[]; fsc = 0.0f0; clip = 0.0
    for i in 1:nblk
        b = read_block!(r, 0)
        fsc == 0.0f0 && (fsc = detect_fullscale(b))
        x1 = to_complex(b.i1, b.q1, fsc); x2 = to_complex(b.i2, b.q2, fsc)
        s1 = chan_stats(x1); s2 = chan_stats(x2)
        clip = max(clip, s1.clip, s2.clip)
        f1, p1 = tone_fit(x1, fs); f2, p2 = tone_fit(x2, fs)
        push!(fts, f1); push!(ph, rad2deg(wrap_pi(p1 - p2)))      # both phases are at the same sample, so the frequency error cancels
        accumulate!(fx, x1, x2); fx.blocks += 1
        fx.pw1 += s1.power; fx.pw2 += s2.power
    end
    fx.t_first = fx.t_last = now(UTC)
    rec = finalize!(fx)
    fbb = median(fts)
    binw = fs / cfg.nfft
    kt = clamp(round(Int, fbb / binw) + cfg.nfft ÷ 2 + 1, 5, cfg.nfft - 4)
    win = (kt - 3):(kt + 3)
    c = cfg.nfft ÷ 2 + 1
    keep = trues(cfg.nfft); keep[max(1, c - cfg.dc_mask):min(cfg.nfft, c + cfg.dc_mask)] .= false; keep[win] .= false
    fl1 = median(Float64.(rec.P1[keep])); fl2 = median(Float64.(rec.P2[keep]))
    n1 = 7fl1; n2 = 7fl2
    t1 = max(sum(Float64.(rec.P1[win])) - n1, 1e-30); t2 = max(sum(Float64.(rec.P2[win])) - n2, 1e-30)
    rho = sum(rec.V[win]) / sqrt(sum(Float64.(rec.P1[win])) * sum(Float64.(rec.P2[win])))
    rho_exp = sqrt(t1 * t2) / sqrt((t1 + n1) * (t2 + n2))
    cs = circ_stats(ph)
    P1k = Float64.(rec.P1); P1k[.!keep] .= -Inf
    ks = argmax(P1k)
    spur_db = 10log10(max(P1k[ks] - fl1, 1e-30) / max(maximum(Float64.(rec.P1[win])) - fl1, 1e-30))
    spur_khz = (ks - cfg.nfft ÷ 2 - 1) * binw / 1e3
    return (spur_db = spur_db, spur_khz = spur_khz, f_bb = fbb, f_true = flo + fbb, f_lo = flo, phase = cs.mean, phase_std = cs.std, rho = abs(rho), rho_phase = rad2deg(angle(rho)),
            rho_exp = rho_exp, snr1_db = 10log10(t1 / n1), snr2_db = 10log10(t2 / n2), nblk = nblk, clip = clip, drift = rad2deg(wrap_pi(deg2rad(ph[end] - ph[1]))))
end

parse_sweep(s::String) = (p = parse.(Float64, split(s, ':')); length(p) == 3 ? collect(p[1]:p[2]:p[3]) : error("tn_sweep must be start:step:stop in kHz"))

const TONE_HEADER = "utc,tag,f_lo_hz,f_bb_hz,f_true_hz,phase_deg,phase_std_deg,rho,rho_expected,snr1_db,snr2_db,gain1_db,gain2_db,cable_m,vf,blocks"

# Fit phase(f) = phi0 + 360*tau*f over baseband frequency f (Hz), unwrapping in frequency order. Returns delay in ns.
function fit_delay(fbb::Vector{Float64}, phase_deg::Vector{Float64}, std_deg::Vector{Float64})
    o = sortperm(fbb); f = fbb[o]; ph = copy(phase_deg[o]); sd = max.(std_deg[o], 0.2)
    for i in 2:length(ph)
        ph[i] = ph[i-1] + rad2deg(wrap_pi(deg2rad(ph[i] - ph[i-1])))
    end
    w = 1 ./ sd .^ 2
    fm = sum(w .* f) / sum(w); pm = sum(w .* ph) / sum(w)
    slope = sum(w .* (f .- fm) .* (ph .- pm)) / sum(w .* (f .- fm) .^ 2)         # degrees per Hz
    phi0 = pm - slope * fm
    res = ph .- (phi0 .+ slope .* f)
    se = sqrt(1 / sum(w .* (f .- fm) .^ 2))                                       # degrees per Hz, from the quoted per-point scatter
    return (tau_ns = slope / 360 * 1e9, tau_err_ns = se / 360 * 1e9, phi0 = phi0, rms = sqrt(mean(res .^ 2)), n = length(f))
end

function read_tone_rows(path::String)
    rows = NamedTuple[]
    for line in eachline(path)
        (startswith(line, "utc,") || isempty(strip(line))) && continue
        c = split(line, ',')
        length(c) >= 16 || continue
        push!(rows, (tag = String(c[2]), f_bb = parse(Float64, c[4]), phase = parse(Float64, c[6]), std = parse(Float64, c[7]),
                     cable = parse(Float64, c[14]), vf = parse(Float64, c[15])))
    end
    return rows
end

function report_delay(rows, label)
    isempty(rows) && (println("  no rows for ", label); return nothing)
    length(rows) >= 3 || (println("  only $(length(rows)) point(s) for $label; need at least 3 frequencies"); return nothing)
    fd = fit_delay([r.f_bb for r in rows], [r.phase for r in rows], [r.std for r in rows])
    @printf("  %-12s %2d points, baseband %.0f to %.0f kHz: delay %.2f +- %.2f ns, phase at LO %.1f deg, residual rms %.2f deg, extra cable %.2f m\n",
            label, fd.n, minimum(r.f_bb for r in rows) / 1e3, maximum(r.f_bb for r in rows) / 1e3, fd.tau_ns, fd.tau_err_ns, fd.phi0, fd.rms, mean(r.cable for r in rows))
    return (fd = fd, cable = mean(r.cable for r in rows), vf = mean(r.vf for r in rows))
end

function mode_delayfit(cfg::Config)
    file = isempty(cfg.tn_file) ? joinpath(cfg.out_dir, "tone_results.csv") : cfg.tn_file
    rows = read_tone_rows(file)
    println("Delay from the phase-versus-frequency slope, ", file)
    A = report_delay(filter(r -> isempty(cfg.tag) || r.tag == cfg.tag, rows), isempty(cfg.tag) ? "(all rows)" : cfg.tag)
    if !isempty(cfg.tn_tag2)
        B = report_delay(filter(r -> r.tag == cfg.tn_tag2, rows), cfg.tn_tag2)
        if A !== nothing && B !== nothing
            dt = B.fd.tau_ns - A.fd.tau_ns
            exp_ns = (B.cable - A.cable) / (B.vf * C_LIGHT) * 1e9
            err = sqrt(A.fd.tau_err_ns^2 + B.fd.tau_err_ns^2)
            @printf("\n  measured delay difference %.2f +- %.2f ns; expected from the cable change %.2f ns (extra %.2f m at velocity factor %.2f)\n",
                    dt, err, exp_ns, B.cable - A.cable, B.vf)
            @printf("  difference %.2f ns (%.1f%% of the expected value). %s\n", dt - exp_ns, 100 * abs(dt - exp_ns) / max(abs(exp_ns), 1e-9),
                    abs(dt - exp_ns) < max(3err, 0.05 * abs(exp_ns)) ? "AGREES within the fit error / 5% velocity-factor uncertainty" : "DISAGREES: check the cable length, the velocity factor and the generator settings")
        end
    end
    return nothing
end

function mode_tone(cfg::Config)
    r, rb = open_verified(cfg)
    file = isempty(cfg.tn_file) ? joinpath(cfg.out_dir, "tone_results.csv") : cfg.tn_file
    isdir(dirname(file)) || mkpath(dirname(file))
    isfile(file) || open(io -> println(io, TONE_HEADER), file, "w")
    offs = isempty(cfg.tn_sweep) ? Union{Nothing,Float64}[nothing] : Union{Nothing,Float64}[parse_sweep(cfg.tn_sweep)...]
    println("\nTone measurement: $(cfg.duration_s) s per point, results appended to $file (tag '$(cfg.tag)', extra cable $(cfg.tn_cable_m) m)")
    cnt = 0
    try
        for off in offs
            if off !== nothing
                @printf("\nSet the generator to %.6f MHz (LO %+.0f kHz), then press Enter   [s = skip this point, q = stop]: ", (Float64(rb.lo) + off * 1e3) / 1e6, off)
                ans = strip(readline())
                ans == "q" && break
                ans == "s" && continue
            end
            m = measure_tone(r, cfg, rb, cfg.duration_s)
            @printf("  tone at %+.3f kHz (true %.6f MHz): %.1f dB (RX1), %.1f dB (RX2) above the noise\n", m.f_bb / 1e3, m.f_true / 1e6, m.snr1_db, m.snr2_db)
            @printf("  coherence at the tone |rho| = %.4f (expected from the SNRs %.4f)\n", m.rho, m.rho_exp)
            @printf("  strongest other line (DC excluded): %+.1f kHz, %.1f dB below the tone%s\n", m.spur_khz, -m.spur_db,
                    m.spur_db > -30 ? "   <-- SPUR: a harmonic or alias, or a second signal. Filter the generator output or move the tone." : "")
            @printf("  phase RX1-RX2 = %.2f deg, standard deviation %.2f deg over %d blocks, drift first to last %.2f deg\n", m.phase, m.phase_std, m.nblk, m.drift)
            min(m.snr1_db, m.snr2_db) < 10 && println("  WARNING: no clear tone on both channels (below 10 dB). Check the generator output, the splitter and the attenuators.")
            m.clip > 1e-3 && println("  WARNING: clipping. Add attenuation.")
            abs(m.rho - m.rho_exp) > 0.1 && println("  NOTE: measured |rho| differs from the SNR-based expectation by more than 0.1: the channels may not see the same signal, or the tone is not alone in the band.")
            open(file, "a") do io
                @printf(io, "%s,%s,%.0f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.2f,%.2f,%.1f,%.1f,%.3f,%.3f,%d\n", string(now(UTC)), cfg.tag, m.f_lo, m.f_bb, m.f_true, m.phase, m.phase_std,
                        m.rho, m.rho_exp, m.snr1_db, m.snr2_db, rb.gain1, rb.gain2, cfg.tn_cable_m, cfg.tn_vf, m.nblk)
            end
            cnt += 1
        end
    finally
        close_radio!(r)
    end
    if cnt >= 3
        println("\nDelay from this run's points:")
        rows = filter(rw -> rw.tag == cfg.tag, read_tone_rows(file))
        report_delay(rows[max(1, end - cnt + 1):end], isempty(cfg.tag) ? "(this run)" : cfg.tag)
    end
    println("\nNext: change the cable (set tn_cable_m= and a new tag=), repeat the sweep, then: julia V1.jl delayfit tag=<A> tn_tag2=<B>")
    return nothing
end

# ---------------------------------------------------------------------
function main(args::Vector{String})
    mode, cfg = parse_cli(args)
    mode == "selftest" ? (mode_selftest(cfg) || exit(1)) :
    mode == "check"    ? mode_check(cfg) :
    mode == "rate"     ? mode_rate(cfg) :
    mode == "run"      ? mode_run(cfg) :
    mode == "show"     ? mode_show(cfg) :
    mode == "sim"      ? mode_sim(cfg) :
    mode == "tone"     ? mode_tone(cfg) :
    mode == "delayfit" ? mode_delayfit(cfg) :
    error("Unknown mode '$mode'.\n" * usage())
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
