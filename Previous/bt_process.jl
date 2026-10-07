# =====================================================================
# bt_process.jl -- post-process raw 2.4 GHz captures from bt_capture.jl: find bursts and decode BLE advertising packets
#
# Input : <stem>_rx1.cs16 (or _rx2), interleaved int16 I,Q, 12-bit values (full scale +-2048), 2.5 Msps, centred on a
#         BLE advertising channel (37 = 2402 MHz, 38 = 2426 MHz, 39 = 2480 MHz).
# Output: a summary on screen and <stem>_packets.csv (one row per burst that synchronised).
#
# Usage:
#     julia bt_process.jl path=C:/PlutoCaptures/bt_XXXX_rx1.cs16 channel=38
#     julia bt_process.jl path=... channel=38 start_s=60 dur_s=120      # only part of the file
#     julia bt_process.jl selftest                                      # synthetic packets, no file needed
# Options: path, channel (37|38|39), fs (default 2.5e6), thr_db (burst threshold above the median, default 8),
#          start_s, dur_s, out (CSV path), top (advertisers listed, default 12).
#
# Method (all steps are done per burst):
#   1. Energy in 50 us windows; a burst is a run of windows more than thr_db above the chunk median.
#   2. Remove the DC (LO leakage) estimate, low-pass at 750 kHz, take the unwrapped phase.
#   3. GFSK demodulation: BLE 1M has 1 Msym/s and +-250 kHz deviation, so the phase moves by about +-90 degrees per
#      symbol. At 2.5 Msps that is 2.5 samples per symbol, so the phase is interpolated and sampled at 10 trial
#      timing offsets; the symbol is 1 when the phase change over the symbol is above the carrier-offset level.
#   4. Sync on the 40-bit preamble + access address (0xAA, 0x8E89BED6), re-estimate the carrier offset from it.
#   5. De-whiten, read the header and length, check the CRC-24. The whitening start-up and CRC conventions are tried
#      in all plausible variants and the one that actually validates real packets is reported (see "Conventions").
#   6. Parse the PDU (type, advertiser address, name, manufacturer) of the packets whose CRC is valid.
#
# Only packets whose CRC is valid are trusted. A burst that syncs but fails the CRC is counted separately.
# Limits: 2.5 samples per symbol is coarse, so weak packets fail; Wi-Fi and classic Bluetooth bursts are not decoded.
# =====================================================================

using DSP, Statistics, Printf, Dates, Random

const SAMPLE_FS = 2.5e6
const FULL_SCALE = 2048.0f0
const BURST_W = 125                      # samples per detection window (50 us at 2.5 Msps)

# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------
Base.@kwdef mutable struct PO
    path::String      = ""
    channel::Int      = 38
    fs::Float64       = SAMPLE_FS
    thr_db::Float64   = 8.0
    start_s::Float64  = 0.0
    dur_s::Float64    = Inf
    out::String       = ""
    top::Int          = 12
end

function parse_po(args)
    o = PO()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        hasfield(PO, sym) || error("Unknown option '$k'")
        T = fieldtype(PO, sym)
        setfield!(o, sym, T == String ? String(v) : T <: Integer ? round(T, parse(Float64, v)) : parse(T, v))
    end
    o.channel in (37, 38, 39) || error("channel must be 37, 38 or 39")
    return o
end

# ---------------------------------------------------------------------
# BLE basics: bit order, CRC, whitening
# ---------------------------------------------------------------------
bits_lsb(bytes) = UInt8[(b >> k) & 1 for b in bytes for k in 0:7]
bytes_from_bits(bits) = UInt8[sum(UInt8(bits[8i + k + 1]) << k for k in 0:7) for i in 0:length(bits) ÷ 8 - 1]

const SYNC_BITS = vcat(bits_lsb([0xAA]), bits_lsb([0xD6, 0xBE, 0x89, 0x8E]))     # preamble, access address 0x8E89BED6, as sent
const SYNC_BOOL = SYNC_BITS .== 1

# CRC-24, polynomial x^24+x^10+x^9+x^6+x^4+x^3+x+1 (0x00065B), bits fed in air order
function crc24_bits(bits::AbstractVector{UInt8}, init::UInt32)
    r = init & 0x00ffffff
    for b in bits
        fb = ((r >> 23) & 0x1) ⊻ UInt32(b)
        r = (r << 1) & 0x00ffffff
        fb != 0 && (r ⊻= 0x0000065b)
    end
    return r
end

fold_msb(bits) = foldl((a, b) -> (a << 1) | UInt32(b), bits; init = UInt32(0))
fold_lsb(bits) = foldl((a, ib) -> a | (UInt32(ib[2]) << (ib[1] - 1)), enumerate(bits); init = UInt32(0))

# Whitening LFSR x^7+x^4+1. Variants differ in how the channel index is loaded into positions 1..6.
function whitening_stream(chan::Int, n::Int, variant::Symbol)
    p = zeros(UInt8, 7)                                   # p[k+1] is position k
    p[1] = 1
    cb = UInt8[(chan >> k) & 1 for k in 5:-1:0]            # bit 5 down to bit 0
    p[2:7] .= variant == :A ? cb : reverse(cb)
    out = Vector{UInt8}(undef, n)
    for i in 1:n
        o = p[7]
        out[i] = o
        q = similar(p)
        q[1] = o
        for k in 2:7; q[k] = p[k-1]; end
        q[5] = p[4] ⊻ o                                    # position 4 receives position 3 xor the output
        p = q
    end
    return out
end

struct Combo
    wv::Symbol          # whitening variant
    init::UInt32        # CRC initial value
    order::Symbol       # :msb or :lsb: how the 24 received CRC bits map onto the register value
end
combo_name(c::Combo) = @sprintf("whiten-%s init-0x%06X crc-%s-first", c.wv, c.init, c.order)
const ALL_COMBOS = [Combo(w, i, o) for w in (:A, :B) for i in (UInt32(0x555555), UInt32(0xAAAAAA)) for o in (:msb, :lsb)]

# rb: bits received after the sync pattern (still whitened). Returns (combo, pdu bytes) or nothing.
function try_decode(rb::Vector{UInt8}, ks::Dict{Symbol,Vector{UInt8}}, combos::Vector{Combo})
    length(rb) >= 16 + 24 || return nothing
    for c in combos
        k = ks[c.wv]
        hdr = bytes_from_bits(rb[1:16] .⊻ k[1:16])
        for len in unique((Int(hdr[2] & 0x3f), Int(hdr[2])))
            need = 8 * (2 + len + 3)
            (len >= 6 || len == 0) || continue
            (need <= length(rb) && need <= length(k)) || continue
            d = rb[1:need] .⊻ k[1:need]
            npdu = 8 * (2 + len)
            crc = crc24_bits(d[1:npdu], c.init)
            rx = c.order == :msb ? fold_msb(d[npdu+1:need]) : fold_lsb(d[npdu+1:need])
            rx == crc && return (combo = c, pdu = bytes_from_bits(d[1:npdu]))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------
# PDU parsing
# ---------------------------------------------------------------------
const PDU_NAMES = ["ADV_IND", "ADV_DIRECT_IND", "ADV_NONCONN_IND", "SCAN_REQ", "SCAN_RSP", "CONNECT_IND", "ADV_SCAN_IND", "ADV_EXT_IND"]
const COMPANIES = Dict(0x004C => "Apple", 0x0006 => "Microsoft", 0x0075 => "Samsung", 0x00E0 => "Google", 0x0059 => "Nordic",
                       0x02E5 => "Espressif", 0x0087 => "Garmin", 0x0157 => "Huami", 0x000F => "Broadcom", 0x00D2 => "Dialog",
                       0x0499 => "Ruuvi", 0x0131 => "Cypress", 0x038F => "Xiaomi", 0x01D7 => "Qualcomm", 0x0171 => "Amazon")
mac_str(b) = join([uppercase(string(x, base = 16, pad = 2)) for x in reverse(b)], ':')

function parse_pdu(pdu::Vector{UInt8})
    t = Int(pdu[1] & 0x0f); txadd = Int((pdu[1] >> 6) & 1)
    payload = pdu[3:end]
    addr = ""; addr2 = ""; name = ""; company = ""; ad = UInt8[]
    if t in (0, 2, 4, 6) && length(payload) >= 6
        addr = mac_str(payload[1:6]); ad = payload[7:end]             # AdvA: the transmitter
    elseif t == 1 && length(payload) >= 12
        addr = mac_str(payload[1:6]); addr2 = mac_str(payload[7:12])   # AdvA (transmitter), target InitA
    elseif t in (3, 5) && length(payload) >= 12
        addr = mac_str(payload[1:6]); addr2 = mac_str(payload[7:12])   # ScanA / InitA (the TRANSMITTER, a scanner), target AdvA
    end
    i = 1
    while i < length(ad)
        l = Int(ad[i]); l == 0 && break
        i + l > length(ad) && break
        ty = ad[i+1]; data = ad[i+2:i+l]
        if ty in (0x08, 0x09)
            name = String(UInt8[c for c in data if 0x20 <= c < 0x7f])
        elseif ty == 0xff && length(data) >= 2
            cid = UInt16(data[1]) | (UInt16(data[2]) << 8)
            company = get(COMPANIES, cid, @sprintf("0x%04X", cid))
        end
        i += l + 1
    end
    return (type = t, type_name = t < length(PDU_NAMES) ? PDU_NAMES[t+1] : "type$t", txadd = txadd, len = length(payload),
            addr = addr, addr2 = addr2, name = name, company = company)
end

# ---------------------------------------------------------------------
# GFSK demodulation of one burst
# ---------------------------------------------------------------------
function unwrap_phase(x::Vector{ComplexF32})
    ph = Vector{Float64}(undef, length(x))
    ph[1] = angle(x[1])
    for i in 2:length(x)
        d = angle(x[i]) - angle(x[i-1])
        d > pi && (d -= 2pi); d < -pi && (d += 2pi)
        ph[i] = ph[i-1] + d
    end
    return ph
end

@inline function phase_at(ph::Vector{Float64}, t::Float64)
    i = clamp(floor(Int, t), 1, length(ph) - 1)
    return ph[i] + (t - i) * (ph[i+1] - ph[i])
end

const LPF_TAPS = Dict{Float64,Vector{Float64}}()
function lowpass(fs)
    get!(LPF_TAPS, fs) do
        digitalfilter(Lowpass(0.75e6 / (fs / 2)), FIRWindow(hanning(41)))      # cutoff relative to Nyquist
    end
end

# Returns (bits after the sync, offset index, sync Hamming distance, carrier offset in kHz) or nothing.
function demod_burst(x::Vector{ComplexF32}, fs::Float64; osteps = 10, maxdist = 4)
    length(x) > 400 || return nothing
    y = filt(lowpass(fs), 1.0, x)[21:end]                  # drop the filter's start-up
    ph = unwrap_phase(ComplexF32.(y))
    T = fs / 1e6
    best = nothing
    for oi in 0:osteps-1
        t0 = 1.0 + oi * T / osteps
        nsym = floor(Int, (length(ph) - 1 - (t0 - 1)) / T) - 1
        nsym > 60 || continue
        dph = [phase_at(ph, t0 + (k + 1) * T) - phase_at(ph, t0 + k * T) for k in 0:nsym-1]
        bits = dph .> median(dph)
        for pos in 1:(length(bits) - 39)
            d = count(bits[pos:pos+39] .!= SYNC_BOOL)
            if best === nothing || d < best.dist
                best = (dist = d, pos = pos, oi = oi, t0 = t0, dph = dph)
            end
        end
    end
    (best === nothing || best.dist > maxdist) && return nothing
    seg = best.dph[best.pos:best.pos+39]
    ones_m = mean(seg[SYNC_BOOL]); zeros_m = mean(seg[.!SYNC_BOOL])
    c2 = (ones_m + zeros_m) / 2
    bits2 = best.dph .> c2
    d2 = count(bits2[best.pos:best.pos+39] .!= SYNC_BOOL)
    rb = UInt8.(bits2[best.pos+40:end])
    return (rb = rb, oi = best.oi, dist = d2, cfo_khz = c2 * 1e6 / (2pi) / 1e3, swing = ones_m - zeros_m)
end

# ---------------------------------------------------------------------
# Burst finding
# ---------------------------------------------------------------------
function find_bursts(x::Vector{ComplexF32}, thr_db::Float64; W = BURST_W, merge = 3, minw = 2, maxw = 70)
    nwin = length(x) ÷ W
    e = Vector{Float32}(undef, nwin)
    for w in 1:nwin
        s = 0.0f0
        @inbounds for n in (w-1)*W+1:w*W
            s += abs2(x[n])
        end
        e[w] = s / W
    end
    thr = median(e) * 10^(thr_db / 10)
    above = e .> thr
    runs = Tuple{Int,Int}[]
    i = 1
    while i <= nwin
        if !above[i]; i += 1; continue; end
        last = i; k = i + 1
        while k <= nwin && k - last <= merge + 1
            above[k] && (last = k)
            k += 1
        end
        minw <= last - i + 1 <= maxw && push!(runs, (i, last))
        i = last + 1
    end
    return runs, W, median(e)
end

# ---------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------
mutable struct Tally
    bursts::Int; synced::Int; crc_ok::Int; chunks::Int
    combos::Dict{String,Int}
end

function process_file(o::PO; combos = ALL_COMBOS, quiet = false)
    fs = o.fs
    n_total = filesize(o.path) ÷ 4
    first_sample = round(Int, o.start_s * fs)
    last_sample = isfinite(o.dur_s) ? min(n_total, first_sample + round(Int, o.dur_s * fs)) : n_total
    ks = Dict(:A => whitening_stream(o.channel, 8 * (2 + 255 + 3), :A), :B => whitening_stream(o.channel, 8 * (2 + 255 + 3), :B))
    chunk = 1 << 22; ov = 6000
    tally = Tally(0, 0, 0, 0, Dict{String,Int}())
    rows = NamedTuple[]
    open(o.path, "r") do io
        pos = first_sample
        while pos < last_sample
            m = min(chunk + ov, last_sample - pos)
            seek(io, 4 * pos)
            buf = Vector{Int16}(undef, 2m); read!(io, buf)
            x = ComplexF32.(buf[1:2:end], buf[2:2:end])
            dc = mean(@view x[1:min(end, 1 << 18)]); x .-= dc
            runs, W, med = find_bursts(x, o.thr_db)
            owned = pos + chunk < last_sample ? chunk : m                    # bursts starting past `owned` belong to the next chunk
            for (w0, w1) in runs
                s0 = (w0 - 1) * W + 1
                (s0 > owned || (pos > first_sample && w0 == 1)) && continue
                a = max(1, s0 - 4W); b = min(length(x), w1 * W + 4W)
                seg = x[a:b]
                tally.bursts += 1
                dm = demod_burst(seg, fs)
                dm === nothing && continue
                tally.synced += 1
                dec = try_decode(dm.rb, ks, combos)
                core = x[s0:min(length(x), w1 * W)]
                rssi = 10log10(max(mean(abs2, core), 1e-9) / FULL_SCALE^2)
                if dec === nothing
                    push!(rows, (sample = pos + a - 1, t = (pos + a - 1) / fs, rssi = rssi, cfo = dm.cfo_khz, dist = dm.dist, crc = false,
                                 combo = "", type = "", len = 0, addr = "", addr2 = "", name = "", company = ""))
                else
                    tally.crc_ok += 1
                    tally.combos[combo_name(dec.combo)] = get(tally.combos, combo_name(dec.combo), 0) + 1
                    p = parse_pdu(dec.pdu)
                    push!(rows, (sample = pos + a - 1, t = (pos + a - 1) / fs, rssi = rssi, cfo = dm.cfo_khz, dist = dm.dist, crc = true,
                                 combo = combo_name(dec.combo), type = p.type_name, len = p.len, addr = p.addr, addr2 = p.addr2, name = p.name, company = p.company))
                end
            end
            tally.chunks += 1
            quiet || tally.chunks % 20 == 0 && @printf("  ... %.0f s processed, %d bursts, %d synced, %d CRC ok\n", (pos + m) / fs, tally.bursts, tally.synced, tally.crc_ok)
            pos += chunk
        end
    end
    return rows, tally, (last_sample - first_sample) / fs
end

function report(o::PO, rows, tally, secs)
    println("\n== Results for $(o.path)")
    @printf("  processed %.1f s on channel %d\n", secs, o.channel)
    @printf("  energy bursts %d (%.1f per second); synchronised on the BLE access address: %d; CRC valid: %d\n", tally.bursts, tally.bursts / secs, tally.synced, tally.crc_ok)
    tally.synced > 0 && @printf("  %.0f%% of the synchronised bursts passed the CRC\n", 100 * tally.crc_ok / tally.synced)
    println("  Conventions that produced CRC-valid packets: ", isempty(tally.combos) ? "none" : "")
    for (k, v) in sort(collect(tally.combos); by = last, rev = true)
        println("    ", v, " x  ", k)
    end
    good = filter(r -> r.crc, rows)
    isempty(good) && return nothing
    println("\n  Packet types:")
    types = Dict{String,Int}(); for r in good; types[r.type] = get(types, r.type, 0) + 1; end
    for (k, v) in sort(collect(types); by = last, rev = true); @printf("    %-16s %d\n", k, v); end
    println("\n  Transmitters (by sender address). A SCAN_REQ or CONNECT_IND is sent by a *scanner*, so a strong one is a nearby scanning device.")
    println("  Random private addresses change about every 15 minutes.")
    byaddr = Dict{String,Vector{NamedTuple}}()
    for r in good; isempty(r.addr) || push!(get!(byaddr, r.addr, NamedTuple[]), r); end
    order = sort(collect(keys(byaddr)); by = a -> length(byaddr[a]), rev = true)
    @printf("    %-19s %6s %10s %9s  %-26s %s\n", "address", "count", "interval", "RSSI dBFS", "packet types", "name / manufacturer")
    for a in order[1:min(o.top, length(order))]
        rr = sort(byaddr[a]; by = r -> r.t)
        iv = length(rr) > 1 ? median(diff([r.t for r in rr])) * 1e3 : NaN
        nm = firstnonempty([r.name for r in rr])
        co = firstnonempty([r.company for r in rr])
        tc = Dict{String,Int}(); for r in rr; tc[r.type] = get(tc, r.type, 0) + 1; end
        tstr = join(["$(k) x$(v)" for (k, v) in sort(collect(tc); by = last, rev = true)[1:min(2, length(tc))]], ", ")
        @printf("    %-19s %6d %8.1f ms %9.1f  %-26s %s %s\n", a, length(rr), iv, mean(r.rssi for r in rr), tstr, nm, isempty(co) ? "" : "[$co]")
    end
    @printf("\n  %d distinct addresses; strongest packet %.1f dBFS, weakest decoded %.1f dBFS\n", length(order), maximum(r.rssi for r in good), minimum(r.rssi for r in good))
    return nothing
end

firstnonempty(v) = (i = findfirst(!isempty, v); i === nothing ? "" : v[i])

function write_csv(path::String, rows)
    open(path, "w") do io
        println(io, "sample,time_s,rssi_dbfs,cfo_khz,sync_errors,crc_ok,pdu_type,payload_len,transmitter_address,target_address,name,manufacturer,convention")
        for r in rows
            @printf(io, "%d,%.6f,%.1f,%.1f,%d,%d,%s,%d,%s,%s,\"%s\",%s,%s\n", r.sample, r.t, r.rssi, r.cfo, r.dist, r.crc ? 1 : 0, r.type, r.len,
                    r.addr, r.addr2, replace(r.name, "\"" => "'"), r.company, r.combo)
        end
    end
end

# ---------------------------------------------------------------------
# Synthetic packets (self-test): GFSK, BT = 0.5, h = 0.5, with a carrier offset, timing offset and noise
# ---------------------------------------------------------------------
function air_bits(chan::Int, pdu::Vector{UInt8}, combo::Combo)
    pb = bits_lsb(pdu)
    crc = crc24_bits(pb, combo.init)
    cb = combo.order == :msb ? UInt8[(crc >> k) & 1 for k in 23:-1:0] : UInt8[(crc >> k) & 1 for k in 0:23]
    body = vcat(pb, cb)
    ks = whitening_stream(chan, length(body), combo.wv)
    return vcat(SYNC_BITS, body .⊻ ks)
end

function gfsk_waveform(bits::Vector{UInt8}; fs_out = 2.5e6, cfo = 0.0, start_frac = 0.37, amp = 400.0)
    U = 40                                              # samples per symbol at 40 Msps
    nrz = repeat(Float64.(2 .* bits .- 1), inner = U)
    tt = (-2U:2U) ./ U
    g = exp.(-2pi^2 * 0.5^2 .* tt .^ 2 ./ log(2)); g ./= sum(g)
    f = conv(nrz, g)[2U+1:end-2U] .* 250e3
    ph = 2pi .* cumsum(f) ./ (U * 1e6)
    hi = cis.(ph .+ 2pi .* cfo .* (0:length(ph)-1) ./ (U * 1e6))
    step = U * 1e6 / fs_out                             # 16 high-rate samples per output sample
    idx = collect(1.0 + start_frac * step:step:length(hi) - 1)
    out = [hi[floor(Int, t)] * (1 - (t - floor(t))) + hi[floor(Int, t) + 1] * (t - floor(t)) for t in idx]
    return ComplexF32.(amp .* out)
end

function mode_selftest()
    println("== bt_process self-test: synthetic BLE advertising packets through the full chain")
    rng = MersenneTwister(5)
    combo = Combo(:A, UInt32(0x555555), :msb)
    chan = 38
    adv = UInt8[0x02, 0x01, 0x06, 0x0a, 0x09, 0x50, 0x6c, 0x75, 0x74, 0x6f, 0x54, 0x65, 0x73, 0x74, 0x03, 0xff, 0x4c, 0x00]
    addr = UInt8[0x11, 0x22, 0x33, 0x44, 0x55, 0xC6]
    payload = vcat(addr, adv)
    pdu = vcat(UInt8[0x40, length(payload)], payload)         # ADV_IND, TxAdd = 1 (random address)
    nbur = 40
    gap = 3000
    pieces = ComplexF32[]
    truth = Int[]
    snr_list = vcat(fill(30.0, 20), fill(18.0, 10), fill(12.0, 10))
    for i in 1:nbur
        w = gfsk_waveform(air_bits(chan, pdu, combo); cfo = 120e3 * (2rand(rng) - 1), start_frac = rand(rng), amp = 400.0)
        sigma = 400.0 / sqrt(10^(snr_list[i] / 10) * 2)
        append!(pieces, ComplexF32.(sigma .* randn(rng, ComplexF64, gap + rand(rng, 0:999))))
        push!(truth, length(pieces) + 1)
        append!(pieces, w .+ ComplexF32.(sigma .* randn(rng, ComplexF64, length(w))))
    end
    append!(pieces, ComplexF32.(8.0 .* randn(rng, ComplexF64, 4000)))
    # weak background noise level equal to the 30 dB case
    noise = ComplexF32.(2.0 .* randn(rng, ComplexF64, length(pieces))); pieces .+= noise
    path = joinpath(mktempdir(), "synthetic_ch38.cs16")
    buf = Vector{Int16}(undef, 2length(pieces))
    buf[1:2:end] .= Int16.(clamp.(round.(real.(pieces)), -2047, 2047)); buf[2:2:end] .= Int16.(clamp.(round.(imag.(pieces)), -2047, 2047))
    write(path, buf)
    o = PO(path = path, channel = chan)
    rows, tally, secs = process_file(o; quiet = true)
    good = filter(r -> r.crc, rows)
    ok_name = all(r.name == "PlutoTest" for r in good) && all(r.addr == "C6:55:44:33:22:11" for r in good) && all(r.type == "ADV_IND" for r in good) && all(r.company == "Apple" for r in good)
    @printf("  sent %d packets (20 at 30 dB, 10 at 18 dB, 10 at 12 dB SNR); synced %d; CRC valid %d\n", nbur, tally.synced, tally.crc_ok)
    @printf("  decoded fields correct on all valid packets: %s (name PlutoTest, address C6:55:44:33:22:11, company Apple)\n", ok_name)
    pass = tally.crc_ok >= 28 && ok_name && length(tally.combos) == 1 && occursin("whiten-A", first(keys(tally.combos)))
    println("  ", pass ? "PASS" : "FAIL")
    return pass
end

# ---------------------------------------------------------------------
function main(args)
    if !isempty(args) && args[1] == "selftest"
        mode_selftest() || exit(1)
        return
    end
    o = parse_po(args)
    isempty(o.path) && error("give path=<file.cs16>")
    isfile(o.path) || error("File not found: $(o.path)")
    println("Processing ", o.path, " (", round(filesize(o.path) / 1e9, digits = 2), " GB)")
    rows, tally, secs = process_file(o)
    report(o, rows, tally, secs)
    out = isempty(o.out) ? replace(o.path, r"\.cs16$" => "") * "_packets.csv" : o.out
    write_csv(out, rows)
    println("\nWrote ", length(rows), " rows to ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
