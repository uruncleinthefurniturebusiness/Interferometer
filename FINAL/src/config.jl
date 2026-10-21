# Constants, settings, data types and command-line parsing shared by every FINAL program.

const SOFTWARE    = "FINAL 1.0"
const C_LIGHT     = 2.99792458e8
const HI_REST     = 1420.405751e6
const OMEGA_E     = 7.2921159e-5     # Earth rotation rate (rad/s)
const MIN_FS_HZ   = 2_083_334.0      # lowest rate the AD9363 accepts with the default filter chain (below it a decimating FIR must be loaded)
const CLIP_LEVEL  = 0.98f0           # |I| or |Q| at or above this fraction of full scale counts as clipped
const LOW_POWER   = 1.0e-7           # mean power (full scale^2) below this means nothing is connected
const FLAG_CLIP   = 0x01
const FLAG_LOWPOW = 0x02
const FLAG_SHORT  = 0x04             # the refill returned fewer bytes than a full block
const FLAG_QFULL  = 0x08             # the processing queue was full when the block arrived (host falling behind)
const CONVENTION  = "V = x1 * conj(x2); positive phase = signal reaches channel 2 later"

Base.@kwdef mutable struct Config
    # radio
    addr::String           = "auto"       # "auto" tries usb: then ip:192.168.2.1; with two Plutos on one laptop give usb:X.Y.Z
    f_lo::Float64          = 1420.0e6     # shared RX LO (Hz); the HI line then sits at +405.75 kHz
    fs::Float64            = 2.5e6        # complex sample rate per channel (Hz)
    gain1::Int             = 40           # manual gain (dB)
    gain2::Int             = 40
    rx_port::String        = "A_BALANCED"
    bw_ratio::Float64      = 1.0          # rf_bandwidth = bw_ratio * fs
    quad_track::Int        = 1            # AD9363 tracking loops: 1 on, 0 off, -1 leave as found
    rf_dc_track::Int       = 1
    bb_dc_track::Int       = 1
    buffer_secs::Float64   = 0.1          # one refill is about this long (rounded to whole FFT segments)
    kernel_buffers::Int    = 8            # Pluto-side DMA buffers
    strict::Bool           = true         # stop if the read-back differs from the request
    # processing
    nfft::Int              = 4096         # FX channels (bin width fs/nfft)
    t_int::Float64         = 1.0          # integration per record (s)
    window::String         = "hann"       # hann | rect
    dc_mask::Int           = 4            # bins either side of DC left out of band averages
    queue_depth::Int       = 32           # blocks between the capture and processing threads
    # run
    duration_s::Float64    = 60.0
    out_dir::String        = "C:/PlutoCaptures"   # keep outside the OneDrive folder (the sync client disturbs sustained writes)
    tag::String            = ""
    print_every::Int       = 1            # print one line every N records
    path::String           = ""           # show: the .vis file (default: newest in out_dir)
    # fake radio (no hardware): a common noise source whose RX1-RX2 phase turns once per fake_period_s
    fake_period_s::Float64 = 60.0
    fake_snr_db::Float64   = -3.0         # common source power relative to the receiver noise, per channel
    fake_phase_deg::Float64 = 30.0        # instrument phase added on RX2
end

Base.@kwdef mutable struct GeoCfg
    target::String           = "none"     # none | sun
    lat_deg::Float64         = -33.9      # site latitude (approximate; set from a GPS reading)
    lon_deg::Float64         = 18.46      # site longitude, east positive
    baseline_m::Float64      = NaN        # dish separation, centre to centre
    baseline_az_deg::Float64 = NaN        # azimuth from the RX1 dish to the RX2 dish, from north through east
    cable_delay_ns::Float64  = 0.0        # extra delay of RX2 relative to RX1 from the cables
    diam_arcmin::Float64     = 32.0       # Sun diameter for the predicted disk factor (placeholder: cite a 1.4 GHz value)
end

# One refill from the Pluto: four Int16 streams and when they arrived.
struct Block
    idx::Int            # index of the first sample (0-based, counted by the host)
    utc::DateTime       # host UTC when the refill returned (end of the block plus USB latency)
    t_ns::UInt64        # host monotonic clock at the same moment
    i1::Vector{Int16}
    q1::Vector{Int16}
    i2::Vector{Int16}
    q2::Vector{Int16}
    short::Bool
    qfull::Bool
end

# One integration, as written to the .vis file.
struct Record
    t_start::Float64         # unix seconds, UTC
    t_end::Float64
    nseg::Int32
    flags::UInt32
    pow1::Float32            # mean power (full scale^2)
    pow2::Float32
    clip1::Float32           # fraction of samples at or near full scale
    clip2::Float32
    V::Vector{ComplexF32}    # cross spectrum <X1 conj(X2)>, fftshifted (DC at index nfft/2+1)
    P1::Vector{Float32}      # auto spectra, same ordering
    P2::Vector{Float32}
end

function parse_value(T, v::AbstractString)
    T == String  && return String(v)
    T == Bool    && return lowercase(v) in ("1", "true", "yes", "on")
    T <: Integer && return round(T, parse(Float64, v))
    return parse(T, v)
end

# key=value arguments, each set on the first object that has the field.
function apply_kv!(args, objs...)
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        i = findfirst(o -> hasfield(typeof(o), sym), objs)
        i === nothing && error("Unknown option '$k'")
        o = objs[i]
        setfield!(o, sym, parse_value(fieldtype(typeof(o), sym), v))
    end
    return nothing
end

# "0:10:70" or "2.1,2.5,3" -> Vector{Float64}
function parse_list(s::AbstractString)
    occursin(':', s) || return parse.(Float64, split(s, ','))
    p = parse.(Float64, split(s, ':'))
    length(p) == 3 || error("Ranges are start:step:stop, got '$s'")
    return collect(p[1]:p[2]:p[3])
end

# Samples per refill: a whole number of FFT segments, so no segment straddles two refills.
function block_geometry(cfg::Config)
    nsegb = max(1, round(Int, cfg.fs * cfg.buffer_secs / cfg.nfft))
    return nsegb, nsegb * cfg.nfft
end

function validate(cfg::Config)
    cfg.fs >= MIN_FS_HZ || error(@sprintf("fs = %.0f Hz is below the AD9363 minimum of about %.0f Hz with the default filters; the Pluto would keep its previous rate.", cfg.fs, MIN_FS_HZ))
    cfg.nfft >= 64 || error("nfft must be at least 64")
    cfg.window in ("hann", "rect") || error("window must be hann or rect")
    cfg.t_int > 0 && cfg.buffer_secs > 0 || error("t_int and buffer_secs must be positive")
    (0 <= cfg.gain1 <= 73 && 0 <= cfg.gain2 <= 73) || @warn "gain outside 0..73 dB"
    (325e6 <= cfg.f_lo <= 3.8e9) || @warn "f_lo outside the AD9363 range (325 MHz to 3.8 GHz)"
    return nothing
end
