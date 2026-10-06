using AdalmPluto

# Opening the radio with 100MHz carrier frequency, 3MHz sampling rate, and 64dB gain.
radio = openPluto(Int(100e6), Int(3e6), 64; bandwidth=Int(20e6));

# Receive the samples
sig = zeros(ComplexF32, 1024*1024) # 1 MiS buffer
recv!(sig, radio);

plot(real(sig), title="Received Signal", xlabel="Sample Index", ylabel="Amplitude")

# Close the radio
close(radio);