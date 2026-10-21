# =====================================================================
# tx_source.jl -- the Pluto+ as the arc-test source: a CW tone or band-limited noise, transmitted cyclically
#
# Usage (on the transmitting laptop, from this folder):
#   julia tx_source.jl tone  freq=868.3e6 [offset=100e3 atten=-40 amp=0.5 fs=2.5e6 addr=usb:]
#   julia tx_source.jl noise freq=868.8e6 [bw_frac=0.6 channels=1 atten=-40 amp=0.5 fs=2.5e6 n=262144 seed=1]
#   julia tx_source.jl off   [addr=usb:]       set maximum attenuation on TX1 (and TX2) and close
# Add dry=true to build the waveform and run the checks without touching the radio.
# Runs until Enter (or for secs=N), then sets maximum attenuation and closes the radio.
#
# tone:  TX LO = freq - offset and the baseband tone at +offset, so the RF tone sits at freq and the TX LO leakage
#        (carrier feedthrough) is a separate weak line at freq - offset. The tone frequency is rounded to a whole
#        number of cycles in the buffer (steps of fs/n), so the cyclic repeat has no phase jump.
# noise: TX LO = freq; complex Gaussian noise occupying freq +- bw_frac*fs/2. channels=2 sends independent noise on
#        TX1 and TX2 (two incoherent sources, arc-test run D; needs the Pluto+ in two-channel mode).
# Every setting is written, read back and logged to `log` (default C:/PlutoCaptures/tx_log.csv) for the methodology.
# Refuses 1400-1427 MHz (protected passive band); warns outside 433.05-434.79, 868-870 and 2400-2483.5 MHz.
# Based on Previous/tx_tone_coherence_test.jl (cyclic buffer; samples scaled as in AdalmPluto.jl's populateBuffer!).
# =====================================================================

using AdalmPluto, FFTW, Random, Printf, Statistics, Dates
const AP = AdalmPluto

const TX_FULL   = 16384.0     # amp = 1 maps to 2^14 (-6 dB below the 16-bit container, as ADI's examples use)
const MAX_ATTEN = -89.75      # dB, the AD9361's largest TX attenuation
const ISM_BANDS = ((433.05e6, 434.79e6), (868.0e6, 870.0e6), (2400e6, 2483.5e6))

Base.@kwdef mutable struct TxCfg
    freq::Float64    = NaN             # RF frequency of the tone, or the centre of the noise (Hz)
    offset::Float64  = 100e3           # tone: baseband offset; TX LO = freq - offset (Hz)
    fs::Float64      = 2.5e6           # TX sample rate (Hz); at least about 2.083 MHz with the default filters
    atten::Float64   = -40.0           # TX attenuation (dB, -89.75 to 0); sets the radiated power
    amp::Float64     = 0.5             # waveform peak as a fraction of 2^14
    channels::Int    = 1               # noise: 1 = TX1, 2 = independent noise on TX1 and TX2
    bw_frac::Float64 = 0.6             # noise: occupied bandwidth as a fraction of fs
    n::Int           = 262144          # samples in the cyclic buffer
    seed::Int        = 1               # noise: random seed
    clip::Float64    = 4.0             # noise: clip each component at this many standard deviations
    secs::Float64    = 0.0             # > 0: transmit for this long, then stop (0 = until Enter)
    addr::String     = "usb:"          # e.g. usb:1.2.5 or ip:192.168.2.1 (from iio_info -s)
    log::String      = "C:/PlutoCaptures/tx_log.csv"
    dry::Bool        = false           # build and check the waveform without the radio
end

const TX_USAGE = """
tx_source.jl modes: tone | noise | off
  julia tx_source.jl tone  freq=868.3e6 [offset=100e3 atten=-40 amp=0.5 fs=2.5e6 addr=usb:]
  julia tx_source.jl noise freq=868.8e6 [bw_frac=0.6 channels=1 atten=-40 fs=2.5e6 n=262144 seed=1]
  julia tx_source.jl off   [addr=usb:]
Add dry=true to check the plan without the radio. Stops on Enter, or after secs=N."""

function parse_tx(args)
    c = TxCfg()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2)
        sym = Symbol(k)
        hasfield(TxCfg, sym) || error("Unknown option '$k'")
        T = fieldtype(TxCfg, sym)
        setfield!(c, sym, T == String ? String(v) : T == Bool ? (lowercase(v) in ("1", "true", "yes", "on")) :
                          T <: Integer ? round(T, parse(Float64, v)) : parse(T, v))
    end
    return c
end

function check_emission(lo::Float64, hi::Float64)
    (lo < 1427e6 && hi > 1400e6) && error(@sprintf("The emission %.3f-%.3f MHz overlaps 1400-1427 MHz, a protected passive band (ITU RR 5.340). Choose an ISM frequency.", lo / 1e6, hi / 1e6))
    any(b -> b[1] <= lo && hi <= b[2], ISM_BANDS) ||
        @warn @sprintf("The emission %.3f-%.3f MHz is not inside 433.05-434.79, 868-870 or 2400-2483.5 MHz. Check the ICASA short-range-device limits before transmitting.", lo / 1e6, hi / 1e6)
    return nothing
end

# Complex Gaussian noise, band-limited by zeroing FFT bins outside +-bw_frac/2 cycles per sample (keeps the buffer
# exactly periodic), clipped at `clip` sigma, then scaled so its largest component equals `amp`.
function noise_wave(rng, n::Int, bw_frac::Float64, clip::Float64, amp::Float64)
    x = randn(rng, ComplexF64, n)
    if bw_frac < 1
        X = fft(x); X[abs.(fftfreq(n)) .> bw_frac / 2] .= 0; x = ifft(X)
    end
    x ./= sqrt(mean(abs2, x) / 2)
    x .= complex.(clamp.(real.(x), -clip, clip), clamp.(imag.(x), -clip, clip))
    return x .* (amp / maximum(max(abs(real(v)), abs(imag(v))) for v in x))
end

function write_attr(ch, attr::String, v::String)
    ch == C_NULL && error("Channel missing for '$attr'")
    ret = AP.C_iio_channel_attr_write(ch, attr, v)
    ret < 0 && error("Writing $attr = $v failed (code $ret)")
    return nothing
end
read_ll(ch, attr) = AP.C_iio_channel_attr_read_longlong(ch, attr)[2]
read_str(ch, attr) = strip(AP.C_iio_channel_attr_read(ch, attr)[2])

function quiet!(phys)
    for p in phys
        p == C_NULL && continue
        AP.C_iio_channel_attr_write(p, "hardwaregain", @sprintf("%.2f", MAX_ATTEN))
    end
end

function log_line(c::TxCfg, mode, t0, t1, txlo, fbb, rb, desc)
    isdir(dirname(c.log)) || mkpath(dirname(c.log))
    isfile(c.log) || open(io -> println(io, "utc_start,utc_stop,mode,freq_hz,tx_lo_hz,f_bb_hz,fs_hz,rf_bw_hz,atten,amp,channels,bw_frac,n,seed,addr,device"), c.log, "w")
    open(c.log, "a") do io
        @printf(io, "%s,%s,%s,%.1f,%d,%.3f,%d,%d,%s,%.3f,%d,%.3f,%d,%d,%s,\"%s\"\n", string(t0), string(t1), mode, c.freq, rb.lo, fbb,
                rb.fs, rb.bw, rb.gain, c.amp, mode == "noise" ? c.channels : 1, c.bw_frac, c.n, c.seed, c.addr, replace(desc, '"' => '\''))
    end
end

function tx_main(args::Vector{String})
    Base.exit_on_sigint(false)                 # Ctrl-C raises an exception, so the radio is still quietened and closed
    isempty(args) && (println(TX_USAGE); return nothing)
    mode = args[1]; c = parse_tx(args[2:end])
    mode in ("tone", "noise", "off") || error("Unknown mode '$mode'.\n" * TX_USAGE)
    -89.75 <= c.atten <= 0 || error("atten must be between -89.75 and 0 dB")
    0 < c.amp <= 1 || error("amp must be in (0, 1]")
    c.channels in (1, 2) || error("channels must be 1 or 2")
    mode == "tone" && c.channels == 2 && error("channels=2 is for noise (two independent sources); a tone uses TX1")

    waves = Vector{Vector{ComplexF64}}(); txlo = NaN; fbb = 0.0
    if mode != "off"
        isfinite(c.freq) || error("Give freq= (Hz), e.g. freq=868.3e6")
        c.fs >= 2.084e6 || error("fs below about 2.083 MHz is rejected by the AD9361 default filters (the Pluto would keep its old rate)")
        c.n >= 4096 || error("n must be at least 4096")
        if mode == "tone"
            k = round(Int, c.offset * c.n / c.fs); k == 0 && error("offset too small for this fs and n")
            fbb = k * c.fs / c.n; txlo = c.freq - fbb
            check_emission(min(txlo, c.freq), max(txlo, c.freq))
            push!(waves, c.amp .* cis.(2pi * k .* (0:c.n-1) ./ c.n))
            @printf("Tone at %.6f MHz: TX LO %.6f MHz, baseband %+.3f kHz (%d whole cycles in %d samples)\n", c.freq / 1e6, txlo / 1e6, fbb / 1e3, k, c.n)
        else
            txlo = c.freq; half = c.bw_frac * c.fs / 2
            check_emission(c.freq - half, c.freq + half)
            rng = MersenneTwister(c.seed)
            for _ in 1:c.channels; push!(waves, noise_wave(rng, c.n, c.bw_frac, c.clip, c.amp)); end
            rms = sqrt(mean(abs2, waves[1]) / 2)
            @printf("Noise %.3f-%.3f MHz on %s; buffer repeats every %.1f ms; level %.1f dB below a tone of the same amp\n",
                    (c.freq - half) / 1e6, (c.freq + half) / 1e6, c.channels == 1 ? "TX1" : "TX1 and TX2 (independent)",
                    1e3 * c.n / c.fs, -20log10(sqrt(2) * rms / c.amp))
            if c.channels == 2
                r = abs(sum(waves[1] .* conj.(waves[2]))) / sqrt(sum(abs2, waves[1]) * sum(abs2, waves[2]))
                @printf("TX1-TX2 correlation %.4f (about %.4f expected for independent noise)\n", r, 1 / sqrt(c.n))
            end
        end
        @printf("Peak %.0f of 32767 in the DAC word; attenuation %.2f dB; sample rate %.3f Msps\n", c.amp * TX_FULL, c.atten, c.fs / 1e6)
    end
    c.dry && (println("dry=true: radio not touched."); return nothing)

    ctx = AP.createContext(c.addr)
    desc = string(AP.C_iio_context_get_description(ctx))
    println("Connected to ", c.addr, ": ", desc)
    txDev, _ = AP.findTRXDevices(ctx)
    phy1, _ = AP.findTRXChannels(ctx, 0, 0)
    phy2, _ = AP.findTRXChannels(ctx, 1, 0)          # C_NULL unless the Pluto+ runs in two-channel mode
    txLoCh, _ = AP.findLoChannels(ctx)
    buf = C_NULL; t0 = nothing
    try
        if mode == "off"
            quiet!((phy1, phy2))
            println("TX1", phy2 == C_NULL ? "" : " and TX2", " set to ", read_str(phy1, "hardwaregain"), ". Nothing transmitted on purpose.")
            return nothing
        end
        chans = [AP.findIQChannels(txDev, "voltage0", "voltage1", true)]
        if c.channels == 2
            (phy2 == C_NULL || AP.findIQChannels(txDev, "voltage2", "voltage3", true)[1] == C_NULL) &&
                error("TX2 not found: put the Pluto+ in two-channel (2r2t) mode, or use channels=1")
            push!(chans, AP.findIQChannels(txDev, "voltage2", "voltage3", true))
        end
        phys = c.channels == 2 ? (phy1, phy2) : (phy1,)
        quiet!((phy1, phy2))                         # start quiet; the level is set last
        for p in phys; write_attr(p, "rf_port_select", "A"); end
        write_attr(phy1, "sampling_frequency", string(round(Int, c.fs)))
        write_attr(phy1, "rf_bandwidth", string(round(Int, c.fs)))
        write_attr(txLoCh, "frequency", string(round(Int, txlo)))
        rb = (fs = read_ll(phy1, "sampling_frequency"), bw = read_ll(phy1, "rf_bandwidth"), lo = read_ll(txLoCh, "frequency"), gain = "")
        abs(rb.fs - c.fs) > 1 && error("Sample rate read back as $(rb.fs), requested $(c.fs)")
        abs(rb.lo - txlo) > 10 && error("TX LO read back as $(rb.lo) Hz, requested $(round(Int, txlo)) Hz")
        abs(rb.bw - c.fs) > 0.05c.fs && @warn "TX RF bandwidth read back as $(rb.bw) Hz (requested $(round(Int, c.fs)))"

        for ch in chans; AP.C_iio_channel_enable(ch[1]); AP.C_iio_channel_enable(ch[2]); end
        buf = AP.C_iio_device_create_buffer(txDev, UInt(c.n), true)          # cyclic: the hardware repeats it
        buf == C_NULL && error("Could not create the TX buffer (n = $(c.n) too large?)")
        stp = AP.C_iio_buffer_step(buf)
        for (w, (chI, chQ)) in zip(waves, chans)
            pI = AP.C_iio_buffer_first(buf, chI); pQ = AP.C_iio_buffer_first(buf, chQ)
            for k in 0:c.n-1
                unsafe_store!(Ptr{Int16}(pI + k * stp), round(Int16, real(w[k+1]) * TX_FULL))
                unsafe_store!(Ptr{Int16}(pQ + k * stp), round(Int16, imag(w[k+1]) * TX_FULL))
            end
        end
        ret = AP.C_iio_buffer_push(buf)
        ret < 0 && error("TX buffer push failed (code $ret)")
        for p in phys; write_attr(p, "hardwaregain", @sprintf("%.2f", c.atten)); end
        rb = merge(rb, (gain = read_str(phy1, "hardwaregain"),))
        t0 = now(UTC)
        @printf("TRANSMITTING %s at %.6f MHz: TX LO %d Hz, rate %d, bandwidth %d, attenuation %s\n", mode, c.freq / 1e6, rb.lo, rb.fs, rb.bw, rb.gain)
        if c.secs > 0
            sleep(c.secs)
        else
            print("Press Enter to stop: "); readline()
        end
        log_line(c, mode, t0, now(UTC), txlo, fbb, rb, desc)
        println("Logged to ", c.log)
    finally
        quiet!((phy1, phy2))
        buf != C_NULL && AP.C_iio_buffer_destroy(buf)
        AP.C_iio_context_destroy(ctx)
        println("TX attenuation set to maximum and radio closed.")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    tx_main(ARGS)
end
