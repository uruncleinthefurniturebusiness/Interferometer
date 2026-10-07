using AdalmPluto
using DSP
using WAV

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
station_freq  = Int(94.5e6)      # Heart FM
sample_rate   = Int(2_400_000)    # 2.4 MHz — plenty of margin around a ~200 kHz FM channel
gain          = 70                # FM broadcast is strong; lower gain than the 1420 MHz work
capture_secs  = 5
pluto_addr    = "usb:1.2.5"       # <-- CHECK THIS with `iio_info -s` first, it can shift between sessions

# ---------------------------------------------------------------
# Connect and capture
# ---------------------------------------------------------------
radio = openPluto(station_freq, sample_rate, gain; addr = pluto_addr, bandwidth = sample_rate)

n_samples = sample_rate * capture_secs
sig = zeros(ComplexF32, n_samples)
recv!(sig, radio)
close(radio)

println("Captured $(length(sig)) samples (~$(capture_secs)s) at $(station_freq/1e6) MHz")
println("Mean power: ", sum(abs2, sig) / length(sig))

# ---------------------------------------------------------------
# FM demodulation (simple phase-difference / quadrature discriminator)
# ---------------------------------------------------------------
demod = angle.(sig[2:end] .* conj.(sig[1:end-1]))

# ---------------------------------------------------------------
# Low-pass filter (audio bandwidth) then decimate to a standard audio rate
# ---------------------------------------------------------------
audio_rate = 48_000
decim = sample_rate ÷ audio_rate   # 2_400_000 / 48_000 = 50

nyquist = sample_rate / 2
cutoff_normalized = 15_000 / nyquist
responsetype = Lowpass(cutoff_normalized)
designmethod = Butterworth(4)
filt = digitalfilter(responsetype, designmethod)
demod_filtered = filtfilt(filt, demod)

audio = demod_filtered[1:decim:end]

# Normalize to [-1, 1] before writing (avoids clipping / silent files)
peak = maximum(abs.(audio))
if peak > 0
    audio = audio ./ peak
end

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
outfile = "heart_fm_104_9.wav"
wavwrite(audio, outfile; Fs = audio_rate)
println("Saved $(outfile) — $(length(audio)/audio_rate) seconds of audio")