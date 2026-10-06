# Simulation of the data capture and processing of the sun for the interferometry project
# Joshua Smith 
# 2026 - EEE4022S

using Plots, Random, FFTW
include("signal_shift_function 1.jl")  # Include the signal shifting function

# Var declarations
sample_rate = 2.5e6 # MSPS, set as it would be on the Pluto
fc = 1.42e9         # Hz, the frequency of the hydrogen line
baseline = 20       # m, the distance between the two antennas
c = 3e8             # m/s, the speed of light
N_samples = 1000    # Number of samples per capture

# Angular sweep range (-3° to +3°)
theta_range = range(deg2rad(-3.0), deg2rad(3.0), length=1000)

# Array to store complex visibility peaks
C_max = Vector{ComplexF64}(undef, length(theta_range))

# Simulation Loop across sky positions
for (i, theta) in enumerate(theta_range)
    # Calculate geometric delay between antennas
    delta_tau = (baseline * sin(theta)) / c
    
    # Simulate IQ1 and IQ2 with geometric RF phase shift
    base_signal = randn(ComplexF64, N_samples)
    
    IQ1 = base_signal
    shifted_base = shift_sampled_signal(base_signal, delta_tau, 1/sample_rate, "noncircular")

    IQ2 = shifted_base.*exp.(-2π * im* fc * delta_tau)  # Apply phase shift due to geometric delay
    
    # 1. Frequency domain conversion
    FFT1 = fft(IQ1)
    FFT2 = fft(IQ2)
    
    # 2. Conjugate multiplication & Inverse FFT (Cross-correlation)
    corr_time = ifft(FFT1 .* conj.(FFT2))
    
    # 3. Peak visibility extraction
    val, idx = findmax(abs.(corr_time))
    C_max[i] = corr_time[idx]
end

# Plot 1: Interference Fringes (Phase Angle)
p1 = plot(rad2deg.(theta_range), angle.(C_max), 
    xlabel="Angle (degrees)", 
    ylabel="Phase (radians)", 
    title="Interferometer Fringe Pattern", 
    legend=false)

# Plot 2: Reconstructed Source (FFT of C_max)
p2 = plot(abs.(fftshift(fft(C_max))), 
    ylabel="Magnitude", 
    title="Reconstructed Visibility Spectrum", 
    legend=false)

# Combine plots and explicitly force display
plt = plot(p1, p2, layout=(2, 1))
display(plt)

# Save image locally as a fallback
savefig("interferometer_sim.png")

# Prevents terminal scripts from closing the plot window immediately
println("Press [ENTER] to exit...")
readline()