using AdalmPluto
const AP = AdalmPluto   # shorthand to access AdalmPluto's internal (non-exported) low-level functions

# =================================================================
# Dual-channel (RX1 + RX2) capture for ADALM-Pluto
#
# AdalmPluto.jl's public API (openPluto/recv!) only supports a single
# RX channel pair. This module goes one level below it, using the
# package's own internal libiio bindings, to enable and read BOTH
# RX1 (voltage0/voltage1) and RX2 (voltage2/voltage3) on the
# cf-ad9361-lpc buffer device simultaneously.
#
# Because RX1 and RX2 share the same physical local oscillator on the
# AD9361 (confirmed via `iio_info`: both are driven by the single
# RX_LO / altvoltage0 channel), samples from the two channels are
# phase-coherent by hardware design — exactly what's needed for
# interferometry between two antennas.
# =================================================================

mutable struct DualRXRadio
    context::Ptr{AP.iio_context}
    rxDevice::Ptr{AP.iio_device}
    bufPtr::Ptr{AP.iio_buffer}
    bufferSize::Int
    rx0_i::Ptr{AP.iio_channel}   # RX1 - I
    rx0_q::Ptr{AP.iio_channel}   # RX1 - Q
    rx1_i::Ptr{AP.iio_channel}   # RX2 - I
    rx1_q::Ptr{AP.iio_channel}   # RX2 - Q
    raw_i1::Array{UInt8}
    raw_q1::Array{UInt8}
    raw_i2::Array{UInt8}
    raw_q2::Array{UInt8}
end

"""
    openDualRX(carrierFreq, samplingRate, gain, bufferSize=1_000_000; addr="auto", rxPort="A_BALANCED")

Opens Pluto and configures BOTH RX1 and RX2 channels for simultaneous, phase-coherent capture.

# Arguments
- `carrierFreq::Int`  : centre frequency in Hz (shared by both RX channels — they use the same LO).
- `samplingRate::Int` : sample rate in Hz (applied to both channels).
- `gain::Int`         : manual RX gain in dB (applied to both channels).
- `bufferSize::Int`   : samples per channel per buffer refill.

# Keywords
- `addr::String`  : e.g. "usb:1.2.5" or "ip:192.168.2.1". Required in practice — auto-detect
                     via `openPluto`'s "auto" logic is not replicated here for simplicity.
- `rxPort::String`: RF port for both RX channels (default matches Pluto's default "A_BALANCED").
"""
function openDualRX(carrierFreq::Int, samplingRate::Int, gain::Int, bufferSize::Int=1_000_000;
                     addr::String, rxPort::String = "A_BALANCED")

    context = AP.createContext(addr)
    println("Description:\n", AP.C_iio_context_get_description(context), "\n")

    txDevice, rxDevice = AP.findTRXDevices(context)

    # --- Buffer-side channels (the actual I/Q data streams) ---
    rx0_i, rx0_q = AP.findIQChannels(rxDevice, "voltage0", "voltage1", false)  # RX1
    rx1_i, rx1_q = AP.findIQChannels(rxDevice, "voltage2", "voltage3", false)  # RX2

    AP.C_iio_channel_enable(rx0_i)
    AP.C_iio_channel_enable(rx0_q)
    AP.C_iio_channel_enable(rx1_i)
    AP.C_iio_channel_enable(rx1_q)

    # --- Physical (phy device) control channels for RX1 and RX2 ---
    # findTRXChannels(context, txID, rxID) -> (txChannel, rxChannel) on the ad9361-phy device.
    # rxID=0 -> RX1 physical channel, rxID=1 -> RX2 physical channel.
    _, rxPhy1 = AP.findTRXChannels(context, 0, 0)
    _, rxPhy2 = AP.findTRXChannels(context, 0, 1)

    # LO is shared between RX1 and RX2 on the AD9361 -- this is what makes them phase-coherent.
    _, rxLo = AP.findLoChannels(context)

    for rxPhy in (rxPhy1, rxPhy2)
        AP.C_iio_channel_attr_write(rxPhy, "rf_port_select", rxPort)
        AP.C_iio_channel_attr_write_longlong(rxPhy, "rf_bandwidth", samplingRate)
        AP.C_iio_channel_attr_write_longlong(rxPhy, "sampling_frequency", samplingRate)
        AP.C_iio_channel_attr_write(rxPhy, "gain_control_mode", "manual")
        AP.C_iio_channel_attr_write_longlong(rxPhy, "hardwaregain", Int64(gain))
    end
    AP.C_iio_channel_attr_write_longlong(rxLo, "frequency", Int64(carrierFreq))

    # --- Create the buffer AFTER all 4 channels are enabled ---
    # (buffer layout / sample size depends on which channels are enabled at creation time)
    bufPtr = AP.C_iio_device_create_buffer(rxDevice, UInt(bufferSize), false)

    # Each of the 4 channels is a 12-bit sample packed into 16 bits (2 bytes), per `iio_info`
    # (format "le:S12/16>>0"), regardless of how many channels are enabled overall.
    bytesPerChannelSample = 2
    raw_i1 = zeros(UInt8, bufferSize * bytesPerChannelSample)
    raw_q1 = zeros(UInt8, bufferSize * bytesPerChannelSample)
    raw_i2 = zeros(UInt8, bufferSize * bytesPerChannelSample)
    raw_q2 = zeros(UInt8, bufferSize * bytesPerChannelSample)

    return DualRXRadio(context, rxDevice, bufPtr, bufferSize,
                        rx0_i, rx0_q, rx1_i, rx1_q,
                        raw_i1, raw_q1, raw_i2, raw_q2)
end

"""
    captureDualRX!(radio)

Refills the buffer once and returns `(rx1::Vector{ComplexF32}, rx2::Vector{ComplexF32})`,
one buffer's worth of samples (`radio.bufferSize` each) from RX1 and RX2 respectively.
Because both channels come from the same buffer refill, the two returned vectors are
time-aligned sample-for-sample.
"""
function captureDualRX!(radio::DualRXRadio)
    nbytes = AP.C_iio_buffer_refill(radio.bufPtr)
    if nbytes < 0
        error("iio_buffer_refill failed with code $nbytes")
    end

    AP.C_iio_channel_read!(radio.rx0_i, radio.bufPtr, radio.raw_i1)
    AP.C_iio_channel_read!(radio.rx0_q, radio.bufPtr, radio.raw_q1)
    AP.C_iio_channel_read!(radio.rx1_i, radio.bufPtr, radio.raw_i2)
    AP.C_iio_channel_read!(radio.rx1_q, radio.bufPtr, radio.raw_q2)

    norm = Float32(2^11)  # 12-bit right-justified samples (le:S12/16>>0, range -2048..2047) -> [-1, 1]

    rx1 = (reinterpret(Int16, radio.raw_i1) .+ 1im .* reinterpret(Int16, radio.raw_q1)) ./ norm
    rx2 = (reinterpret(Int16, radio.raw_i2) .+ 1im .* reinterpret(Int16, radio.raw_q2)) ./ norm

    return ComplexF32.(rx1), ComplexF32.(rx2)
end

"""
    closeDualRX!(radio)

Destroys the buffer and context, releasing the USB/network handle.
"""
function closeDualRX!(radio::DualRXRadio)
    AP.C_iio_buffer_destroy(radio.bufPtr)
    AP.C_iio_context_destroy(radio.context)
    return nothing
end


# =================================================================
# Example usage / sanity check
# =================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    pluto_addr   = "usb:1.2.5"   # <-- check with `iio_info -s` first
    center_freq  = Int(1420e6)
    sample_rate  = Int(5e6)
    gain         = 40
    n_samples    = 100_000

    radio = openDualRX(center_freq, sample_rate, gain, n_samples; addr = pluto_addr)

    rx1, rx2 = captureDualRX!(radio)

    closeDualRX!(radio)

    println("RX1: ", length(rx1), " samples | mean power: ", sum(abs2, rx1) / length(rx1))
    println("RX2: ", length(rx2), " samples | mean power: ", sum(abs2, rx2) / length(rx2))

    # --- Basic coherence sanity check: normalized cross-correlation at zero lag ---
    # With both channels sharing an LO, even receiving the same noise/signal on two
    # antennas fed from a common source (e.g. a signal splitter) should show strong
    # correlation. Two independent antennas on real astronomical noise will show much
    # weaker correlation unless looking at a common bright source -- that's expected physics,
    # not a bug.
    xcorr0 = sum(rx1 .* conj.(rx2)) / sqrt(sum(abs2, rx1) * sum(abs2, rx2))
    println("Normalized zero-lag cross-correlation |xcorr0| = ", abs(xcorr0))
end