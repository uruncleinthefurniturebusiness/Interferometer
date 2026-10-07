using AdalmPluto
using Plots
using FFTW
#plotly() # to switch 
#gr()
pythonplot()



# =================================================================
# Configurable IQ capture from ADALM-Pluto
# =================================================================

"""
    capture_iq(target_freq, bandwidth; gain=40, duration_secs=1, addr="usb:1.2.5")

Captures IQ samples from PlutoSDR at a given target frequency and bandwidth,
returning them as a `Vector{ComplexF32}`.

# Arguments
- `target_freq::Real` : centre (carrier) frequency in Hz, e.g. `104.9e6`
- `bandwidth::Real`   : sample rate / bandwidth in Hz, e.g. `5e6`

# Keywords
- `gain::Int`            : RX gain in dB (default 40)
- `duration_secs::Real`  : how many seconds worth of samples to capture (default 1)
- `addr::String`         : Pluto's context address, e.g. "usb:1.2.5" or "ip:192.168.2.1"
                            — check current value with `iio_info -s` before running.

# Returns
- `Vector{ComplexF32}` of length `round(Int, bandwidth * duration_secs)`

# Example
```julia
iq = capture_iq(1420e6, 5e6; gain=64, duration_secs=1, addr="usb:1.2.5")
println(length(iq), " samples captured")
println("Mean power: ", sum(abs2, iq) / length(iq))
```
"""


#plotly();    # Specify Plotly backend which allows zooming with a mouse inside Jupyter Notebook.

default(size=(800,300)); #Plot canvas size

default(label=""); # Turn off legends

default(ticks=:native);



function capture_iq(target_freq::Real, bandwidth::Real;
                     gain::Int = 40,
                     duration_secs::Real = 1,
                     addr::String = "usb:1.2.5")


    target_freq = Int(round(target_freq))
    sample_rate = Int(round(bandwidth))
    n_samples   = Int(round(sample_rate * duration_secs))

    radio = openPluto(target_freq, sample_rate, gain; addr = addr, bandwidth = sample_rate)

    iq = zeros(ComplexF32, n_samples)
    recv!(iq, radio)

    close(radio)

    return iq
end


# =================================================================
# Example usage (only runs when this file is executed directly)
# =================================================================
#if abspath(PROGRAM_FILE) == @__FILE__
    # --- Edit these to your target ---
    target_freq   = 96e6   # Hz
    bandwidth     = 2e6      # Hz
    gain          = 64
    duration_secs = 1
    pluto_addr    = "usb:1.2.5"   # <-- check with `iio_info -s` first

    global iq = capture_iq(target_freq, bandwidth;
                        gain = gain, duration_secs = duration_secs, addr = pluto_addr)

    n = length(iq)
    f_axis = (((0:n-1) .* (bandwidth/n)) .- (bandwidth/2) .+ target_freq)/1e6

    println("Captured ", length(iq), " IQ samples")
    println("Mean power: ", sum(abs2, iq) / length(iq))
    println("First 5 samples: ", iq[1:5])
#end

p1 = plot(real(iq), label = "I (real)", xlabel = "Sample index", ylabel = "Amplitude")
#display(p1)

IQ = fft(iq)
p2 = plot(f_axis, abs.(fftshift(IQ)), label = "Magnitude", xlabel = "Frequency bin", ylabel = "Magnitude")
#isplay(p2)

display(plot(p1, p2, layout = (2, 1), size = (800, 600)))