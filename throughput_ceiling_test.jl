include("dual_channel_capture.jl")

# =================================================================
# Dual-channel throughput ceiling test
#
# Finds the real sustained sample rate this specific Pluto unit can
# hold on BOTH RX channels simultaneously over USB2, without dropped
# samples. Feng et al. (2021) measured ~2.7 Msps/stream as a reliable
# ceiling on comparable PlutoSDR hardware, and an earlier short run on
# this unit landed at ~3 Msps/channel, consistent with that figure --
# this test extends the check to a full 10-minute sustained capture
# per candidate rate, since a rate that holds for 30s isn't the same
# claim as one that holds for a real observing integration.
#
# Method: for each candidate rate, open the dual-RX radio ONCE and
# call captureDualRX! back-to-back for a sustained TEST_DURATION_S.
# openDualRX/captureDualRX! already support this -- the buffer is
# created once and reused every refill, so this is just a loop, no
# changes needed to dual_channel_capture.jl itself.
#
# Three signals are reported per rate, not just pass/fail:
#   1. Hard failure: captureDualRX! throws (iio_buffer_refill returned
#      a negative byte count -- the kernel-detected overrun signal
#      already wired up in dual_channel_capture.jl's error check), or
#      returns a short buffer.
#   2. Soft/marginal signal: worst-case wall-clock time per refill
#      running noticeably longer than buffer_size/sample_rate -- the
#      pipeline is keeping up but with little headroom.
#   3. Achieved throughput: the AVERAGE sample rate actually sustained
#      over the whole run (total samples / total wall-clock time) and
#      the MAX/MIN instantaneous per-buffer rate seen -- the requested
#      rate is what you asked the AD9361 for, these are what you
#      actually got, and they can differ even without a hard drop.
#
# NOTE: at 10 minutes/rate, a full sweep through every candidate that
# happens to succeed can take up to ~an hour (6 rates x 10 min). Since
# ~3 Msps/channel is already the working assumption, consider trimming
# candidate_rates below to just the ones actually worth 10 minutes of
# wall-clock time (e.g. confirming 3.0 holds, checking whether 4.0
# also would) rather than re-running the whole sweep at full length.
#
# Captured data is discarded, isolating throughput from the separate
# storage-architecture question (see streaming_capture.jl for that).
# =================================================================

pluto_addr       = "usb:1.2.5"      # confirm via `iio_info -s`
test_freq        = Int(104.9e6)     # reuse the known Heart FM signal -- content doesn't matter here, just needs to be receiving *something* consistently
gain             = 40
test_duration_s  = 300               # sustained -- 5 minutes, long enough to catch slow-onset USB/driver issues a 30s test would miss
buffer_secs      = 0.5               # size of each refill's buffer, in seconds -- small enough to be sensitive to overrun, large enough to keep Julia-loop overhead low
progress_every   = 40                # print a heartbeat every N buffers (~20s at buffer_secs=0.5) so a 10-minute run doesn't look hung

candidate_rates = [2_000_000, 2_700_000, 3_000_000, 4_000_000]#, 5_000_000, 6_000_000]

function test_rate(sample_rate::Int)
    buffer_size = round(Int, sample_rate * buffer_secs)
    n_buffers   = ceil(Int, test_duration_s / buffer_secs)

    println("\n--- Testing $(sample_rate/1e6) Msps/channel: $(buffer_size) samples/buffer x $(n_buffers) buffers (~$(round(n_buffers*buffer_secs/60, digits=1)) min) ---")

    radio = openDualRX(test_freq, sample_rate, gain, buffer_size; addr = pluto_addr)

    expected_buffer_time = buffer_size / sample_rate
    per_buffer_rates_msps = Float64[]   # buffer_size/dt per successful refill -- gives achieved max/avg/min throughput, not just the requested rate
    dropped = false
    n_ok = 0

    t_start = time()
    for i in 1:n_buffers
        t0 = time()
        try
            rx1, rx2 = captureDualRX!(radio)
            if length(rx1) != buffer_size || length(rx2) != buffer_size
                println("  buffer $(i): SHORT READ -- expected $(buffer_size), got RX1=$(length(rx1)) RX2=$(length(rx2))")
                dropped = true
            end
        catch e
            println("  buffer $(i): refill ERROR -- $(e)")
            dropped = true
            break
        end
        dt = time() - t0
        push!(per_buffer_rates_msps, buffer_size / dt / 1e6)
        n_ok += 1

        if i % progress_every == 0
            running_avg = sum(per_buffer_rates_msps) / length(per_buffer_rates_msps)
            println("  ... $(round(i*buffer_secs/60, digits=1)) min done, running avg $(round(running_avg, digits=3)) Msps")
        end
    end
    t_end = time()

    closeDualRX!(radio)

    total_wall_s = t_end - t_start
    samples_captured = n_ok * buffer_size
    avg_rate_msps = total_wall_s > 0 ? samples_captured / total_wall_s / 1e6 : 0.0
    max_rate_msps = isempty(per_buffer_rates_msps) ? 0.0 : maximum(per_buffer_rates_msps)
    min_rate_msps = isempty(per_buffer_rates_msps) ? 0.0 : minimum(per_buffer_rates_msps)
    max_refill_time = min_rate_msps > 0 ? buffer_size / (min_rate_msps * 1e6) : 0.0
    overrun_ratio = expected_buffer_time > 0 ? max_refill_time / expected_buffer_time : 0.0

    println("  Buffers completed: $(n_ok)/$(n_buffers) over $(round(total_wall_s/60, digits=2)) min")
    println("  Requested: $(sample_rate/1e6) Msps | Average achieved: $(round(avg_rate_msps, digits=3)) Msps | Max (peak buffer): $(round(max_rate_msps, digits=3)) Msps | Min (worst buffer): $(round(min_rate_msps, digits=3)) Msps")
    println("  Worst-case refill/buffer-time ratio: $(round(overrun_ratio, digits=2))x")

    if dropped
        println("  RESULT: DROPPED SAMPLES / ERROR at $(sample_rate/1e6) Msps")
    elseif overrun_ratio > 1.5
        println("  RESULT: MARGINAL at $(sample_rate/1e6) Msps -- some refills took $(round(overrun_ratio, digits=1))x longer than expected, close to the edge even without a hard failure")
    else
        println("  RESULT: CLEAN at $(sample_rate/1e6) Msps over the full $(round(total_wall_s/60, digits=1)) minute run")
    end

    return !dropped
end

println("Sustained dual-channel throughput ceiling test -- $(round(test_duration_s/60, digits=1)) min per rate")
for rate in candidate_rates
    ok = test_rate(rate)
    if !ok
        println("\nStopping sweep -- first failure at $(rate/1e6) Msps. Ceiling is below this.")
        break
    end
end
