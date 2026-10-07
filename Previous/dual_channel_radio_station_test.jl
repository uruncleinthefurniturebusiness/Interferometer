include("dual_channel_capture.jl")
using DSP
using WAV

# =================================================================
# Dual-channel test capture: two separate antennas on RX1/RX2,
# both pointed at the same known strong FM broadcast station.
#
# This is a sanity/coherence test before going to the actual dishes:
# with two INDEPENDENT antennas (not a split single feed), the two
# channels see the same broadcast signal but via different paths
# (cable length, antenna placement, local multipath), so we expect
# high but not perfect correlation -- unlike a split-single-antenna
# test, which would show near-1.0 coherence trivially.
# =================================================================

station_freq = Int(104.9e6)      # Heart 104.9 FM (Cape Town)
sample_rate  = Int(2_400_000)     # 2.4 MHz -- plenty of margin around a ~200 kHz FM channel,
                                   # and close to Feng et al.'s measured ~2.7 Msps/stream USB2
                                   # ceiling for THIS project's dual-channel mode, so this run
                                   # doubles as a real throughput check, not just a coherence one.
gain         = 70                 # FM broadcast is strong -- start both channels equal;
                                   # if one antenna is weaker/further from the transmitter,
                                   # revisit per-channel gain (openDualRX currently applies the
                                   # same gain to both RX1 and RX2).
capture_secs = 2                  # Kept short for a first test -- increase once this runs clean.
pluto_addr   = "usb:1.2.5"        # confirmed via `iio_info -s`

n_samples = sample_rate * capture_secs

println("Opening Pluto dual-RX at $(station_freq/1e6) MHz, $(sample_rate/1e6) Msps/channel, gain $(gain) dB...")
radio = openDualRX(station_freq, sample_rate, gain, n_samples; addr = pluto_addr)

rx1, rx2 = captureDualRX!(radio)
closeDualRX!(radio)

# -----------------------------------------------------------------
# Dropped-sample check -- captureDualRX! doesn't currently verify the
# refill returned a FULL buffer, so do it here explicitly. This is
# exactly the "confirm it holds without dropped samples" check flagged
# as unverified for this project's hardware.
# -----------------------------------------------------------------
if length(rx1) != n_samples || length(rx2) != n_samples
    println("WARNING: expected $(n_samples) samples/channel, got RX1=$(length(rx1)), RX2=$(length(rx2)) -- possible dropped samples.")
else
    println("Buffer complete: $(n_samples) samples/channel as expected, no evidence of dropped samples.")
end

println("RX1 mean power: ", sum(abs2, rx1) / length(rx1))
println("RX2 mean power: ", sum(abs2, rx2) / length(rx2))

# -----------------------------------------------------------------
# Cross-correlation vs. lag (not just zero-lag) -- with two separate
# antennas there may be a real relative delay (differing cable length,
# antenna placement) worth seeing, not just a single coherence number.
# Search a modest window since two indoor antennas a few metres apart
# imply a delay of nanoseconds -- far under one sample period at
# 2.4 Msps (~417 ns/sample) -- so the peak should land within a few
# samples of zero unless something in the setup is unexpectedly long.
# -----------------------------------------------------------------
lag_window = 200
xc = DSP.xcorr(rx1, rx2; padmode = :none)
center = length(rx1)  # index of zero lag in DSP.xcorr's output
lags = -lag_window:lag_window
mags = abs.(xc[center .+ lags]) ./ sqrt(sum(abs2, rx1) * sum(abs2, rx2))

peak_idx = argmax(mags)
peak_lag = lags[peak_idx]
peak_mag = mags[peak_idx]

println("Peak normalized cross-correlation: $(round(peak_mag, digits=4)) at lag $(peak_lag) samples ($(round(peak_lag/sample_rate*1e9, digits=1)) ns)")
println("Zero-lag normalized cross-correlation: $(round(mags[lag_window+1], digits=4))")

# -----------------------------------------------------------------
# FM demodulate each channel SEPARATELY and save as WAV -- a real
# intelligibility check (do both antennas actually hear the station?)
# beyond just the numeric correlation above.
# -----------------------------------------------------------------
function demod_and_save(sig::Vector{ComplexF32}, sample_rate::Int, outfile::String)
    demod = angle.(sig[2:end] .* conj.(sig[1:end-1]))

    audio_rate = 48_000
    decim = sample_rate ÷ audio_rate

    nyquist = sample_rate / 2
    cutoff_normalized = 15_000 / nyquist
    filt = digitalfilter(Lowpass(cutoff_normalized), Butterworth(4))
    demod_filtered = filtfilt(filt, demod)

    audio = demod_filtered[1:decim:end]
    peak = maximum(abs.(audio))
    if peak > 0
        audio = audio ./ peak
    end

    wavwrite(audio, outfile; Fs = audio_rate)
    println("Saved $(outfile) -- $(round(length(audio)/audio_rate, digits=1))s of audio")
end

demod_and_save(rx1, sample_rate, "heart_fm_rx1.wav")
demod_and_save(rx2, sample_rate, "heart_fm_rx2.wav")

println("\nDone. Listen to both WAV files -- if both are intelligible Heart FM audio, both antennas/channels are working.")
println("A low cross-correlation despite both being intelligible just means independent local multipath/noise -- expected with separate antennas, not a fault.")
