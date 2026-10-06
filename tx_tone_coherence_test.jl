using AdalmPluto
const AP = AdalmPluto

# =================================================================
# TX test-tone + dual-RX coherence check
#
# Uses Pluto's OWN transmitter to generate a continuous low-power carrier
# on a safe ISM frequency, then captures that same signal on BOTH RX
# channels simultaneously. Since RX1 and RX2 share a local oscillator
# (see dual_channel_capture.jl), a genuinely common source received by
# both channels should show a much higher cross-correlation than the
# near-zero baseline measured with independent noise on each channel.
#
# ANTENNA SETUP FOR THIS TEST:
#   - Move one antenna to the TX port.
#   - Leave the other antenna on RX1.
#   - Leave RX2's port bare (near-field coupling picks up some signal).
#   - Keep everything a few cm apart on the same desk.
#
# IMPORTANT: test_freq is deliberately NOT 1420 MHz (protected radio
# astronomy band) -- using a license-exempt ISM frequency instead.
# =================================================================

pluto_addr   = "usb:1.2.5"     # <-- check with `iio_info -s` first
test_freq    = Int(433.5e6)    # safe ISM frequency for this bench test
sample_rate  = Int(2e6)
rx_gain      = 40
tx_atten_dB  = -30             # heavy attenuation: this is a cm-range bench test, not real transmission
n_rx_samples = 100_000
tx_buf_size  = 1024            # small cyclic buffer, repeated by hardware

context = AP.createContext(pluto_addr)
println("Description:\n", AP.C_iio_context_get_description(context), "\n")

txDevice, rxDevice = AP.findTRXDevices(context)

# ---------------- RX1 + RX2 setup (same as dual_channel_capture.jl) ----------------
rx0_i, rx0_q = AP.findIQChannels(rxDevice, "voltage0", "voltage1", false)
rx1_i, rx1_q = AP.findIQChannels(rxDevice, "voltage2", "voltage3", false)
AP.C_iio_channel_enable(rx0_i); AP.C_iio_channel_enable(rx0_q)
AP.C_iio_channel_enable(rx1_i); AP.C_iio_channel_enable(rx1_q)

_, rxPhy1 = AP.findTRXChannels(context, 0, 0)
_, rxPhy2 = AP.findTRXChannels(context, 0, 1)
_, rxLo   = AP.findLoChannels(context)

for rxPhy in (rxPhy1, rxPhy2)
    AP.C_iio_channel_attr_write(rxPhy, "rf_port_select", "A_BALANCED")
    AP.C_iio_channel_attr_write_longlong(rxPhy, "rf_bandwidth", sample_rate)
    AP.C_iio_channel_attr_write_longlong(rxPhy, "sampling_frequency", sample_rate)
    AP.C_iio_channel_attr_write(rxPhy, "gain_control_mode", "manual")
    AP.C_iio_channel_attr_write_longlong(rxPhy, "hardwaregain", Int64(rx_gain))
end
AP.C_iio_channel_attr_write_longlong(rxLo, "frequency", Int64(test_freq))

rxBufPtr = AP.C_iio_device_create_buffer(rxDevice, UInt(n_rx_samples), false)

# ---------------- TX setup: continuous low-power carrier ----------------
tx0_i, tx0_q = AP.findIQChannels(txDevice, "voltage0", "voltage1", true)
AP.C_iio_channel_enable(tx0_i); AP.C_iio_channel_enable(tx0_q)

txPhy, _ = AP.findTRXChannels(context, 0, 0)
txLo, _  = AP.findLoChannels(context)

AP.C_iio_channel_attr_write(txPhy, "rf_port_select", "A")
AP.C_iio_channel_attr_write_longlong(txPhy, "rf_bandwidth", sample_rate)
AP.C_iio_channel_attr_write_longlong(txPhy, "sampling_frequency", sample_rate)
AP.C_iio_channel_attr_write_longlong(txPhy, "hardwaregain", Int64(tx_atten_dB))
AP.C_iio_channel_attr_write_longlong(txLo, "frequency", Int64(test_freq))

# Cyclic buffer: the hardware repeats this buffer's content automatically after one push.
txBufPtr = AP.C_iio_device_create_buffer(txDevice, UInt(tx_buf_size), true)
txFirst = AP.C_iio_buffer_start(txBufPtr)
txStep  = AP.C_iio_buffer_step(txBufPtr)

# Fill with a constant unmodulated carrier: I = fixed amplitude, Q = 0.
# (Matches the Q14-alignment technique used by AdalmPluto.jl's own send()/populateBuffer!.)
amplitude = 0.3
tmpI = Int16(round(amplitude * (1 << 15))) >> 2
tmpQ = Int16(0)
packed = Int32(tmpQ) << 16 + tmpI

cPos = txFirst
for _ in 1:tx_buf_size
    unsafe_store!(Ptr{Int32}(cPos), packed)
    global cPos += txStep
end

AP.C_iio_buffer_push(txBufPtr)
println("TX tone started at $(test_freq/1e6) MHz, $(tx_atten_dB) dB attenuation.")

# ---------------- Capture on both RX channels ----------------
sleep(0.2)  # let the TX settle
nbytes = AP.C_iio_buffer_refill(rxBufPtr)
if nbytes < 0
    error("iio_buffer_refill failed with code $nbytes")
end

raw_i1 = zeros(UInt8, n_rx_samples * 2)
raw_q1 = zeros(UInt8, n_rx_samples * 2)
raw_i2 = zeros(UInt8, n_rx_samples * 2)
raw_q2 = zeros(UInt8, n_rx_samples * 2)

AP.C_iio_channel_read!(rx0_i, rxBufPtr, raw_i1)
AP.C_iio_channel_read!(rx0_q, rxBufPtr, raw_q1)
AP.C_iio_channel_read!(rx1_i, rxBufPtr, raw_i2)
AP.C_iio_channel_read!(rx1_q, rxBufPtr, raw_q2)

norm = Float32(2^15)
rx1 = ComplexF32.((reinterpret(Int16, raw_i1) .+ 1im .* reinterpret(Int16, raw_q1)) ./ norm)
rx2 = ComplexF32.((reinterpret(Int16, raw_i2) .+ 1im .* reinterpret(Int16, raw_q2)) ./ norm)

println("RX1: ", length(rx1), " samples | mean power: ", sum(abs2, rx1) / length(rx1))
println("RX2: ", length(rx2), " samples | mean power: ", sum(abs2, rx2) / length(rx2))

xcorr0 = sum(rx1 .* conj.(rx2)) / sqrt(sum(abs2, rx1) * sum(abs2, rx2))
println("Normalized zero-lag cross-correlation |xcorr0| = ", abs(xcorr0))
println("(Compare this to your earlier no-common-source baseline of ~0.0076)")

# ---------------- Cleanup ----------------
AP.C_iio_buffer_destroy(rxBufPtr)
AP.C_iio_buffer_destroy(txBufPtr)
AP.C_iio_context_destroy(context)
println("Done — radio closed.")
