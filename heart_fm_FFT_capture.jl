using AdalmPluto
using GLMakie
using FFTW
using Statistics

# =================================================================
# Interactive Pluto Spectrum Viewer
# Sliders control center frequency, gain, and sample rate.
# Press "Capture" to grab a fresh buffer and update the plot.
# =================================================================

pluto_addr = "usb:1.2.5"   # <-- update with current `iio_info -s` output if needed

fig = Figure(size = (900, 600))

# --- Plot area ---
ax = Axis(fig[1, 1],
    title = "Live Spectrum",
    xlabel = "Frequency offset (Hz)",
    ylabel = "Power (dB)")

# Observable data the plot will track
freqs_obs = Observable(Float64[])
mag_db_obs = Observable(Float64[])

lines!(ax, freqs_obs, mag_db_obs)

# --- Controls ---
controls = fig[2, 1] = GridLayout()

Label(controls[1, 1], "Center Freq (MHz)")
freq_slider = Slider(controls[1, 2], range = 70:0.1:1500, startvalue = 104.9)
freq_label = Label(controls[1, 3], @lift(string(round($(freq_slider.value), digits=1), " MHz")))

Label(controls[2, 1], "Gain (dB)")
gain_slider = Slider(controls[2, 2], range = 0:1:71, startvalue = 40)
gain_label = Label(controls[2, 3], @lift(string($(gain_slider.value), " dB")))

Label(controls[3, 1], "Sample Rate (MHz)")
rate_slider = Slider(controls[3, 2], range = 0.5:0.1:10, startvalue = 2.4)
rate_label = Label(controls[3, 3], @lift(string(round($(rate_slider.value), digits=1), " MHz")))

capture_btn = Button(controls[4, 1:2], label = "Capture & Update")
status_label = Label(controls[4, 3], "Ready")

# --- Capture + FFT logic ---
function do_capture!()
    status_label.text[] = "Capturing..."

    center_freq  = Int(round(freq_slider.value[] * 1e6))
    gain         = Int(round(gain_slider.value[]))
    sample_rate  = Int(round(rate_slider.value[] * 1e6))
    n_samples    = sample_rate  # ~1 second worth

    try
        radio = openPluto(center_freq, sample_rate, gain; addr = pluto_addr, bandwidth = sample_rate)
        sig = zeros(ComplexF32, n_samples)
        recv!(sig, radio)
        close(radio)

        spectrum = fftshift(fft(sig))
        mag = abs.(spectrum)
        mag_db = 20 .* log10.(mag .+ eps(Float32))
        freqs = collect(range(-sample_rate/2, sample_rate/2, length = length(spectrum)))

        freqs_obs[] = freqs
        mag_db_obs[] = mag_db

        # Auto-scale axis to the new data
        autolimits!(ax)

        peak_idx = argmax(mag)
        peak_offset = freqs[peak_idx]
        noise_floor = median(mag)
        ratio = mag[peak_idx] / noise_floor

        status_label.text[] = "Peak at $(round(peak_offset)) Hz offset | Peak/noise ratio: $(round(ratio, digits=1))"
    catch e
        status_label.text[] = "Error: $(e)"
        rethrow(e)
    end
end

on(capture_btn.clicks) do _
    do_capture!()
end

# Take one capture immediately so the plot isn't empty on startup
do_capture!()

display(fig)

# Keep the window open (needed when running as a script rather than in the REPL)
println("Window open — press Enter in this terminal to exit.")
readline()