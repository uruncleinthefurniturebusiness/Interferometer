using AdalmPluto
using GLMakie
using FFTW
using Statistics

GLMakie.activate!()  # force GLMakie as the active backend, in case another backend (e.g. CairoMakie) was set as default

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

        # --- Averaged periodogram (Welch's method, simplified) ---
        # Rather than one huge FFT (which renders as an unreadable "wall" of
        # points), split into short chunks, FFT each, and average the power.
        # This gives a smooth, low-noise spectrum with a manageable point count.
        nfft = 4096
        nchunks = length(sig) ÷ nfft
        avg_power = zeros(Float64, nfft)
        for i in 1:nchunks
            chunk = @view sig[(i-1)*nfft+1 : i*nfft]
            avg_power .+= abs2.(fftshift(fft(chunk)))
        end
        avg_power ./= nchunks

        mag_db = 10 .* log10.(avg_power .+ eps(Float32))
        freqs = collect(range(-sample_rate/2, sample_rate/2, length = nfft))

        freqs_obs[] = freqs
        mag_db_obs[] = mag_db

        # Auto-scale axis to the new data
        autolimits!(ax)

        peak_idx = argmax(avg_power)
        peak_offset = freqs[peak_idx]
        noise_floor = median(avg_power)
        ratio = avg_power[peak_idx] / noise_floor

        status_label.text[] = "Peak at $(round(peak_offset)) Hz offset | Peak/noise ratio: $(round(ratio, digits=1))"
    catch e
        status_label.text[] = "Error: $(e)"
        @warn "Capture failed" exception=e
    end
end

on(capture_btn.clicks) do _
    do_capture!()
end

# Show the window FIRST, so it appears even if the initial capture fails
screen = display(fig)
println("Screen object: ", screen)

# Take one capture immediately so the plot isn't empty on startup.
# Wrapped so a failure here (e.g. stale pluto_addr) doesn't prevent the window from showing.
try
    do_capture!()
catch e
    status_label.text[] = "Initial capture failed: $(e)"
    @warn "Initial capture failed — window is still open, fix pluto_addr and click Capture" exception=e
end

# Keep the window open (recommended pattern for GLMakie when running as a script)
println("Window should be open now. Waiting for it to close...")
wait(screen)