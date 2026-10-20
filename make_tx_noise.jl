# make_tx_noise.jl -- noise waveform for the Pluto+ transmitter (IIO Oscilloscope, DDS mode "DAC Buffer Output")
#
# Writes a text waveform in IIO Oscilloscope's format: a first line "TEXT", then one sample per line, "I, Q" (TX1 only)
# or "I1, Q1, I2, Q2" (independent noise on TX1 and TX2). IIO Oscilloscope scales the file so its largest value reaches
# full scale (then applies its Scale (dBFS) box), so the absolute numbers here do not matter: set the radiated power with
# the TX attenuation. The buffer is transmitted cyclically.
#
# Usage: julia make_tx_noise.jl [channels=1|2] [n=262144] [bw_frac=1.0] [clip=4.0] [seed=1] [path=pluto_noise.txt]
#   channels  1: TX1 only (arc-test run C). 2: independent noise on TX1 and TX2 (two incoherent sources, run D)
#   n         samples in the cyclic buffer, which repeats every n/fs_tx seconds. 262144 at 4 Msps is 66 ms: spectral lines
#             every 15 Hz (far finer than V1's 610 Hz FX bins), and a TX1-TX2 correlation floor of about 1/sqrt(n) = 0.002
#   bw_frac   occupied bandwidth as a fraction of the TX sample rate (1.0 = white across the whole TX band)
#   clip      clip each component at this many standard deviations, so one rare sample does not set the full scale

using FFTW, Random, Printf, Statistics

function parse_noise_args(args::Vector{String})
    o = Dict("channels" => "1", "n" => "262144", "bw_frac" => "1.0", "clip" => "4.0", "seed" => "1", "path" => "pluto_noise.txt")
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        haskey(o, k) || error("Unknown option '$k' (options: $(join(sort(collect(keys(o))), ", ")))")
        o[String(k)] = String(v)
    end
    return o
end

# Complex Gaussian noise, band-limited by zeroing FFT bins outside +-bw_frac/2 cycles per sample. Filtering in the
# frequency domain keeps the buffer exactly periodic, so the cyclic repeat has no join.
function band_noise(rng, n::Int, bw_frac::Float64)
    x = randn(rng, ComplexF64, n)
    bw_frac >= 1 && return x
    X = fft(x)
    X[abs.(fftfreq(n)) .> bw_frac / 2] .= 0
    return ifft(X)
end

function make_noise(args::Vector{String})
    o = parse_noise_args(args)
    ch = parse(Int, o["channels"]); n = parse(Int, o["n"]); bw = parse(Float64, o["bw_frac"]); cl = parse(Float64, o["clip"])
    ch in (1, 2) || error("channels must be 1 or 2")
    0 < bw <= 1 || error("bw_frac must be in (0, 1]")
    n >= 4096 || error("n must be at least 4096")
    rng = MersenneTwister(parse(Int, o["seed"]))

    sig = [band_noise(rng, n, bw) for _ in 1:ch]
    nclip = 0
    for s in sig
        s ./= sqrt(mean(abs2, s) / 2)                                   # unit variance per component
        nclip += count(v -> abs(real(v)) > cl, s) + count(v -> abs(imag(v)) > cl, s)
        s .= complex.(clamp.(real.(s), -cl, cl), clamp.(imag.(s), -cl, cl))
    end
    pk = maximum(max(maximum(abs.(real.(s))), maximum(abs.(imag.(s)))) for s in sig)
    for s in sig
        s ./= pk                                                        # peak = 1, as IIO Oscilloscope will scale it
    end

    path = o["path"]
    open(path, "w") do io
        println(io, "TEXT")
        if ch == 1
            for v in sig[1]
                @printf(io, "%.5f, %.5f\n", real(v), imag(v))
            end
        else
            for (a, b) in zip(sig[1], sig[2])
                @printf(io, "%.5f, %.5f, %.5f, %.5f\n", real(a), imag(a), real(b), imag(b))
            end
        end
    end

    rms = sqrt(mean(abs2, sig[1]) / 2)                                  # per component, relative to full scale
    @printf("Wrote %s: %d samples, %s, occupied bandwidth %.0f%% of the TX sample rate\n", path, n,
            ch == 1 ? "TX1 only (2 columns)" : "independent noise on TX1 and TX2 (4 columns)", 100bw)
    @printf("Level: %.1f dB below a full-scale tone (per component rms %.3f of full scale); %.4f%% of values clipped at %.1f sigma\n",
            -20log10(sqrt(2) * rms), rms, 100nclip / (2n * ch), cl)
    if ch == 2
        rho = abs(sum(sig[1] .* conj.(sig[2]))) / sqrt(sum(abs2, sig[1]) * sum(abs2, sig[2]))
        @printf("TX1-TX2 correlation over the buffer %.4f (expected about %.4f for independent noise)\n", rho, 1 / sqrt(n))
    end
    @printf("At a TX sample rate fs the noise covers the TX LO +- %.2f*fs (e.g. fs = 4 Msps: +- %.1f MHz); keep that inside the ISM band.\n",
            bw / 2, 2bw)
    @printf("The buffer repeats every n/fs = %.1f ms at 4 Msps.\n", 1e3n / 4e6)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_noise(ARGS)
end
