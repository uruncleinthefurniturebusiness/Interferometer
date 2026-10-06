include("dual_channel_capture.jl")
using Dates

# =================================================================
# Continuous dual-channel capture with concurrent disk writing.
#
# Producer (the captureDualRX! loop below) and consumer (a separate
# writer task) are decoupled via a bounded Channel, so a slow/stalled
# disk write can never block the next iio_buffer_refill call -- that
# is the actual mechanism that drops samples in a naive
# "capture -> write -> capture" serial pattern, not raw disk speed
# (at ~2.7 Msps/channel the combined raw rate is ~21.6 MB/s, well
# under any SSD's ceiling -- see Hardware Setup.md).
#
# IMPORTANT -- run this with at least 2 Julia threads:
#     julia -t 2 streaming_capture.jl
# (or set JULIA_NUM_THREADS=2 before launching). The producer's
# iio_buffer_refill call is a blocking ccall into libiio; Julia's
# cooperative `@async` scheduler does NOT yield during a blocking
# ccall on the same OS thread, so with the default single thread the
# writer task cannot make progress while a refill is in flight --
# Threads.@spawn only gives real concurrency here if there's a
# second OS thread for it to actually run on.
#
# STORAGE LOCATION -- do NOT point OUT_DIR at anything inside the
# OneDrive-synced project folder. OneDrive watches/locks files as
# they're written and can contend with a sustained high-rate write.
# Capture to a local, non-synced path and move finished files into
# the tracked project afterward if they need to be kept there.
# =================================================================

const SAMPLE_RATE  = Int(2_700_000)      # measured sustained ceiling with margin -- see Hardware Setup.md
const CENTER_FREQ  = Int(1420e6)
const GAIN         = 40
const BUFFER_SECS  = 0.5                  # size of each producer buffer, in seconds
const QUEUE_DEPTH  = 8                    # ~4s of slack absorbed between producer and writer before backpressure reaches the producer
const CAPTURE_SECS = 60                   # total capture duration
const OUT_DIR      = "C:/PlutoCaptures"   # OUTSIDE the OneDrive-synced project tree -- see note above
const PLUTO_ADDR   = "usb:1.2.5"

function write_metadata(path::String, n_samples_written::Int, start_time::DateTime)
    open(path, "w") do io
        println(io, "sample_rate_hz = $(SAMPLE_RATE)")
        println(io, "center_freq_hz = $(CENTER_FREQ)")
        println(io, "gain_db = $(GAIN)")
        println(io, "dtype = ComplexF32, raw (real, imag) float32 pairs, no header")
        println(io, "channels = rx1, rx2 (separate files, sample-for-sample aligned)")
        println(io, "n_samples_per_channel = $(n_samples_written)")
        println(io, "capture_start_local = $(start_time)")
    end
end

function run_streaming_capture()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    rx1_path  = joinpath(OUT_DIR, "capture_$(timestamp)_rx1.cf32")
    rx2_path  = joinpath(OUT_DIR, "capture_$(timestamp)_rx2.cf32")
    meta_path = joinpath(OUT_DIR, "capture_$(timestamp)_meta.txt")

    buffer_size = round(Int, SAMPLE_RATE * BUFFER_SECS)
    n_buffers   = ceil(Int, CAPTURE_SECS / BUFFER_SECS)

    println("Streaming capture: $(SAMPLE_RATE/1e6) Msps/channel, $(CAPTURE_SECS)s target -> $(OUT_DIR)")
    println("Julia threads available: $(Threads.nthreads()) -- need >=2 for the writer to progress while a refill blocks.")
    if Threads.nthreads() < 2
        println("WARNING: only 1 thread available. Relaunch with `julia -t 2 streaming_capture.jl` or this will behave like a serial capture-then-write loop.")
    end

    queue = Channel{Tuple{Vector{ComplexF32},Vector{ComplexF32}}}(QUEUE_DEPTH)

    writer_task = Threads.@spawn begin
        n_written = 0
        open(rx1_path, "w") do f1
            open(rx2_path, "w") do f2
                for (rx1, rx2) in queue
                    write(f1, rx1)
                    write(f2, rx2)
                    n_written += length(rx1)
                end
            end
        end 
        n_written
    end

    start_time = now()
    radio = openDualRX(CENTER_FREQ, SAMPLE_RATE, GAIN, buffer_size; addr = PLUTO_ADDR)
    dropped = false
    for i in 1:n_buffers
        try
            rx1, rx2 = captureDualRX!(radio)
            put!(queue, (rx1, rx2))
        catch e
            println("Buffer $(i): refill ERROR -- $(e)")
            dropped = true
            break
        end
        if i % 20 == 0
            println("  ... $(round(i * BUFFER_SECS, digits=1))s captured")
        end
    end
    closeDualRX!(radio)
    close(queue)

    n_written = fetch(writer_task)
    write_metadata(meta_path, n_written, start_time)

    expected = (dropped ? 0 : n_buffers) * buffer_size
    println(dropped ? "Capture stopped early due to an error." : "Capture complete.")
    println("Wrote $(n_written) samples/channel to $(rx1_path) / $(rx2_path)")
    println("Metadata: $(meta_path)")
    if !dropped && n_written != n_buffers * buffer_size
        println("WARNING: wrote $(n_written) samples/channel but expected $(n_buffers * buffer_size) -- writer may have lagged behind the producer, check queue depth / disk speed.")
    end
end

run_streaming_capture()
