# The ADALM-Pluto (two RX channels on one AD9363): connect, write every setting, read it back, stream blocks.
# The Pluto keeps its settings between runs, so every run writes all of them and reads them back.

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
    kernel_buffers::Int
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
        end
    end
    error("No Pluto found via $(join(candidates, ", ")). Run `iio_info -s` and pass addr=usb:X.Y.Z.")
end

# Connect, find both RX chains, and write the configuration. No stream yet.
function open_radio(cfg::Config; verbose::Bool = true)
    validate(cfg)
    ctx, uri = connect(cfg.addr)
    verbose && println("Connected via $uri\n", AP.C_iio_context_get_description(ctx), "\n")
    local r
    try
        _, rxdev = AP.findTRXDevices(ctx)
        rxdev == C_NULL && error("RX streaming device (cf-ad9361-lpc) not found")
        i1, q1 = AP.findIQChannels(rxdev, "voltage0", "voltage1", false)
        i2, q2 = AP.findIQChannels(rxdev, "voltage2", "voltage3", false)
        any(==(C_NULL), (i1, q1, i2, q2)) &&
            error("RX2 (voltage2/voltage3) not exposed. The Pluto must be in two-channel mode (iio_info shows four RX data channels).")
        chs = [i1, q1, i2, q2]
        foreach(AP.C_iio_channel_enable, chs)
        _, phy1 = AP.findTRXChannels(ctx, 0, 0)
        _, phy2 = AP.findTRXChannels(ctx, 0, 1)
        _, lo   = AP.findLoChannels(ctx)
        r = Radio(ctx, rxdev, Ptr{AP.iio_buffer}(C_NULL), 0, chs, [phy1, phy2], lo, Vector{UInt8}[], uri, cfg.kernel_buffers)
        configure!(r, cfg)
    catch
        AP.C_iio_context_destroy(ctx)
        rethrow()
    end
    return r
end

# Write every setting, checking each return code.
function configure!(r::Radio, cfg::Config)
    validate(cfg)
    fs = Int64(round(cfg.fs)); bw = Int64(round(cfg.bw_ratio * cfg.fs))
    for (phy, g) in ((r.phy[1], cfg.gain1), (r.phy[2], cfg.gain2))
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
    set_lo!(r, cfg.f_lo)
    return r
end

function set_lo!(r::Radio, f::Float64)
    chk(AP.C_iio_channel_attr_write_longlong(r.lo, "frequency", Int64(round(f))), "LO frequency")
    sleep(0.3)   # let the AD9363 retune and recalibrate
    return rd_ll(r.lo, "frequency")
end

# Both gains at once; returns the read-back values.
function set_gain!(r::Radio, g1::Real, g2::Real)
    chk(AP.C_iio_channel_attr_write_longlong(r.phy[1], "hardwaregain", Int64(round(g1))), "hardwaregain")
    chk(AP.C_iio_channel_attr_write_longlong(r.phy[2], "hardwaregain", Int64(round(g2))), "hardwaregain")
    return rd_gain(r.phy[1]), rd_gain(r.phy[2])
end

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

# Requested against read back. Returns (message, fatal) for each difference.
function verify_readback(rb, cfg::Config; verbose::Bool = true)
    probs = Tuple{String,Bool}[]
    function row(name, req, act, bad::Bool, fatal::Bool)
        verbose && @printf("  %-22s requested %-16s actual %-16s %s\n", name, req, act, bad ? "<-- MISMATCH" : "ok")
        bad && push!(probs, ("$name: requested $req, hardware reports $act", fatal))
    end
    verbose && println("Hardware read-back:")
    row("sampling_frequency", @sprintf("%.0f", cfg.fs), string(rb.fs1), abs(rb.fs1 - cfg.fs) > 1e-4 * cfg.fs, true)
    row("sampling_frequency RX2", string(rb.fs1), string(rb.fs2), rb.fs1 != rb.fs2, true)
    row("rf_bandwidth", @sprintf("%.0f", cfg.bw_ratio * cfg.fs), string(rb.bw1), abs(rb.bw1 - cfg.bw_ratio * cfg.fs) > 0.2 * cfg.bw_ratio * cfg.fs, false)
    row("LO frequency", @sprintf("%.0f", cfg.f_lo), string(rb.lo), abs(rb.lo - cfg.f_lo) > 100, true)
    row("gain RX1 (dB)", string(cfg.gain1), @sprintf("%.2f", rb.gain1), abs(rb.gain1 - cfg.gain1) > 1.0, true)
    row("gain RX2 (dB)", string(cfg.gain2), @sprintf("%.2f", rb.gain2), abs(rb.gain2 - cfg.gain2) > 1.0, true)
    row("gain mode", "manual", "$(rb.mode1)/$(rb.mode2)", !(rb.mode1 == "manual" && rb.mode2 == "manual"), true)
    row("rf port", cfg.rx_port, "$(rb.port1)/$(rb.port2)", !(rb.port1 == cfg.rx_port && rb.port2 == cfg.rx_port), false)
    for (nm, req, act) in (("quadrature tracking", cfg.quad_track, rb.quad), ("RF DC tracking", cfg.rf_dc_track, rb.rfdc),
                           ("BB DC tracking", cfg.bb_dc_track, rb.bbdc))
        req >= 0 && row(nm, string(req), "$(act[1])/$(act[2])", any(!=(req), act), true)
    end
    return probs
end

function start_stream!(r::Radio, cfg::Config)
    stop_stream!(r)
    _, bufsize = block_geometry(cfg)
    if r.kernel_buffers > 0
        ret = AP.C_iio_device_set_kernel_buffers_count(r.rxdev, UInt32(r.kernel_buffers))
        ret < 0 && @warn "Could not set the kernel buffer count (libiio returned $ret); using the default"
    end
    r.buf = AP.C_iio_device_create_buffer(r.rxdev, UInt(bufsize), false)
    r.buf == C_NULL && error("Could not create the capture buffer ($bufsize samples)")
    r.bufsize = bufsize
    r.raw = [zeros(UInt8, bufsize * 2) for _ in 1:4]
    return r
end

function stop_stream!(r::Radio)
    r.buf != C_NULL && AP.C_iio_buffer_destroy(r.buf)
    r.buf = Ptr{AP.iio_buffer}(C_NULL)
    return r
end

close_radio!(r::Radio) = (stop_stream!(r); AP.C_iio_context_destroy(r.ctx); nothing)

# One refill -> four Int16 vectors (copied, so the raw buffers can be reused at once).
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

# While the host is not reading, the Pluto's DMA buffers fill with samples from that time and then stop. After any
# pause or change (gain, source moved, prompt), read them all out before measuring, or up to kernel_buffers x 0.1 s
# of stale data ends up in the measurement.
flush!(r::Radio) = (for _ in 1:(max(r.kernel_buffers, 4) + 2); read_block!(r, 0); end; r)

# Configure, read back, stop on a fatal difference (unless strict=false), start streaming.
function open_verified(cfg::Config; verbose::Bool = true)
    r = open_radio(cfg; verbose)
    rb = read_back(r)
    probs = verify_readback(rb, cfg; verbose)
    fatal = [m for (m, f) in probs if f]
    if !isempty(fatal) && cfg.strict
        close_radio!(r)
        error("The hardware did not accept the configuration:\n  " * join(fatal, "\n  ") * "\nFix the request, or pass strict=false.")
    end
    isempty(probs) || println("WARNING: $(length(probs)) read-back difference(s); using the hardware's values.")
    start_stream!(r, cfg)
    return r, rb
end
