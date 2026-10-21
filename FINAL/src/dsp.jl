# Signal processing: sample conversion, channel health, the FX correlator, band averages, tone fitting and fits.
# Convention everywhere: V = x1 * conj(x2); a positive phase means the signal reaches RX2 later.

wrap_pi(x) = mod(x + pi, 2pi) - pi
wrap180(d) = mod(d + 180, 360) - 180

function unwrap_deg(p::AbstractVector{<:Real})
    q = Float64.(p)
    for i in 2:length(q)
        q[i] = q[i-1] + wrap180(q[i] - q[i-1])
    end
    return q
end

# The ADC is 12-bit in a 16-bit container; whether the value is right-aligned (+-2048) or left-aligned (+-32768)
# is checked from the data rather than assumed.
function detect_fullscale(b::Block)
    m = 0
    low_zero = true
    for v in (b.i1, b.q1, b.i2, b.q2), s in v
        a = abs(Int(s)); a > m && (m = a)
        low_zero &= (s & Int16(15)) == 0
    end
    return (m > 2048 || (low_zero && m > 0)) ? 32768.0f0 : 2048.0f0
end

to_complex(i::Vector{Int16}, q::Vector{Int16}, fsc::Float32) = ComplexF32.(i ./ fsc, q ./ fsc)

function chan_stats(x::Vector{ComplexF32})
    n = length(x); p = 0.0; pk = 0.0f0; nclip = 0; sr = 0.0; si = 0.0
    @inbounds for v in x
        re = real(v); im = imag(v)
        p += re * re + im * im; sr += re; si += im
        a = max(abs(re), abs(im)); a > pk && (pk = a); a >= CLIP_LEVEL && (nclip += 1)
    end
    return (power = p / n, peak = Float64(pk), clip = nclip / n, dc = complex(sr / n, si / n))
end

# ---------------------------------------------------------------------
# FX correlator: each block is split into segments of nfft samples, windowed, transformed, and the cross and auto
# products are summed. finalize! divides by (segments x sum of w^2), so P summed over all bins / nfft is the mean
# power in full-scale^2 units (0 dBFS = a full-scale complex sine).
# ---------------------------------------------------------------------
mutable struct FX
    nfft::Int
    nsegb::Int
    win::Vector{Float32}
    wpow::Float64
    plan::Any
    V::Vector{ComplexF64}
    P1::Vector{Float64}
    P2::Vector{Float64}
    nseg::Int
    blocks::Int
    t_first::DateTime
    t_last::DateTime
    pw1::Float64; pw2::Float64
    clip1::Float64; clip2::Float64
    flags::UInt32
end

function FX(nfft::Int, nsegb::Int, window::String)
    w = window == "hann" ? Float32.(0.5 .- 0.5 .* cos.(2pi .* (0:nfft-1) ./ nfft)) : ones(Float32, nfft)
    plan = plan_fft(Matrix{ComplexF32}(undef, nfft, nsegb), 1)
    return FX(nfft, nsegb, w, sum(abs2, Float64.(w)), plan, zeros(ComplexF64, nfft), zeros(nfft), zeros(nfft),
              0, 0, DateTime(0), DateTime(0), 0.0, 0.0, 0.0, 0.0, 0x00)
end

function reset!(fx::FX)
    fill!(fx.V, 0); fill!(fx.P1, 0); fill!(fx.P2, 0)
    fx.nseg = 0; fx.blocks = 0
    fx.pw1 = fx.pw2 = fx.clip1 = fx.clip2 = 0.0
    fx.flags = 0x00
    return fx
end

function accumulate!(fx::FX, x1::Vector{ComplexF32}, x2::Vector{ComplexF32})
    X1 = fx.plan * (reshape(x1, fx.nfft, fx.nsegb) .* fx.win)
    X2 = fx.plan * (reshape(x2, fx.nfft, fx.nsegb) .* fx.win)
    fx.V  .+= vec(sum(X1 .* conj.(X2), dims = 2))
    fx.P1 .+= vec(sum(abs2.(X1), dims = 2))
    fx.P2 .+= vec(sum(abs2.(X2), dims = 2))
    fx.nseg += fx.nsegb
    return fx
end

# One block into the correlator, with its health statistics and flags.
function add_block!(fx::FX, b::Block, x1, x2, s1, s2, blk_ms::Int)
    fx.blocks == 0 && (fx.t_first = b.utc - Millisecond(blk_ms))
    fx.t_last = b.utc
    accumulate!(fx, x1, x2)
    fx.blocks += 1
    fx.pw1 += s1.power; fx.pw2 += s2.power
    fx.clip1 += s1.clip; fx.clip2 += s2.clip
    b.short && (fx.flags |= FLAG_SHORT)
    b.qfull && (fx.flags |= FLAG_QFULL)
    return fx
end

function finalize!(fx::FX)::Record
    s = fx.nseg * fx.wpow
    nb = max(fx.blocks, 1)
    flags = fx.flags
    p1 = fx.pw1 / nb; p2 = fx.pw2 / nb
    c1 = fx.clip1 / nb; c2 = fx.clip2 / nb
    (c1 > 1e-3 || c2 > 1e-3) && (flags |= FLAG_CLIP)
    (p1 < LOW_POWER || p2 < LOW_POWER) && (flags |= FLAG_LOWPOW)
    return Record(datetime2unix(fx.t_first), datetime2unix(fx.t_last), Int32(fx.nseg), flags,
                  Float32(p1), Float32(p2), Float32(c1), Float32(c2),
                  ComplexF32.(fftshift(fx.V ./ s)), Float32.(fftshift(fx.P1 ./ s)), Float32.(fftshift(fx.P2 ./ s)))
end

flagstr(f::Integer) = (s = strip((f & FLAG_CLIP != 0 ? "CLIP " : "") * (f & FLAG_LOWPOW != 0 ? "LOWPOW " : "") *
                                 (f & FLAG_SHORT != 0 ? "SHORT " : "") * (f & FLAG_QFULL != 0 ? "QFULL" : "")); isempty(s) ? "ok" : s)

# Baseband frequency of each bin in fftshifted order, and the absolute frequency.
bb_freqs(nfft, fs) = collect(((0:nfft-1) .- nfft ÷ 2) .* (fs / nfft))
bin_freqs(nfft, fs, f_lo) = f_lo .+ bb_freqs(nfft, fs)
dc_index(nfft) = nfft ÷ 2 + 1

# Bins used in band averages: all but +-dc_mask bins around DC (the LO leakage is common to both channels).
function band_keep(nfft::Int, dc_mask::Int)
    keep = trues(nfft); c = dc_index(nfft)
    keep[max(1, c - dc_mask):min(nfft, c + dc_mask)] .= false
    return keep
end

band_rho(rec::Record, keep) = sum(ComplexF64.(rec.V[keep])) / sqrt(sum(Float64, rec.P1[keep]) * sum(Float64, rec.P2[keep]))
rho_bins(rec::Record) = ComplexF64.(rec.V) ./ sqrt.(Float64.(rec.P1) .* Float64.(rec.P2))

# Band, DC-bin and strongest-bin coherence of a record.
function summarize(rec::Record, dc_mask::Int)
    n = length(rec.V); c = dc_index(n)
    keep = band_keep(n, dc_mask)
    rho = band_rho(rec, keep)
    rho_dc = rec.V[c] / sqrt(rec.P1[c] * rec.P2[c])
    pw = Float64.(rec.P1 .+ rec.P2); pw[.!keep] .= -Inf
    k = argmax(pw)
    return (rho = rho, rho_dc = ComplexF64(rho_dc), k = k, rho_pk = ComplexF64(rec.V[k] / sqrt(rec.P1[k] * rec.P2[k])))
end

# Power in full-scale^2 (dBFS) of bins `idx`, after taking off `noise_per_bin`.
bins_power(P, idx, noise_per_bin = 0.0) = max(sum(Float64, P[idx]) - noise_per_bin * length(idx), 1e-30) / length(P)
dbfs(p) = 10log10(max(p, 1e-30))

# 7-bin window around baseband frequency f (Hz); empty if it falls outside the band.
function tone_bins(nfft::Int, fs::Float64, f::Float64; half::Int = 3)
    k = round(Int, f / (fs / nfft)) + dc_index(nfft)
    return clamp(k - half, 1, nfft):clamp(k + half, 1, nfft)
end

# ---------------------------------------------------------------------
# Tone fit: FFT peak, parabolic interpolation on log power, three phase-slope refinements, then the phase of the
# mixed-down sum. Returns the baseband frequency (Hz) and the phase (rad) at the first sample. The search skips
# +-dc_hz around DC, so a weak tone is not mistaken for the LO leakage.
# ---------------------------------------------------------------------
function tone_fit(x::AbstractVector{ComplexF32}, fs::Float64; L::Int = 65536, dc_hz::Float64 = 2.5e3)
    L = min(L, length(x))
    s = ComplexF64.(@view x[1:L])
    p = abs2.(fft(s))
    kd = ceil(Int, dc_hz / (fs / L))
    ps = copy(p); ps[1:min(kd + 1, L)] .= 0; ps[max(1, L - kd + 1):L] .= 0
    k = argmax(ps)
    a = log(p[mod1(k - 1, L)] + 1e-300); b = log(p[k] + 1e-300); c = log(p[mod1(k + 1, L)] + 1e-300)
    d = 0.5 * (a - c) / (a - 2b + c)
    f = ((k - 1) + (isfinite(d) ? d : 0.0)) * fs / L
    f > fs / 2 && (f -= fs)
    n = collect(0.0:L-1)
    h = L ÷ 2
    for _ in 1:3
        r = cis.(-2pi * f / fs .* n) .* s
        f += angle(sum(@view r[h+1:L]) * conj(sum(@view r[1:h]))) * fs / (2pi * h)
    end
    return f, angle(sum(cis.(-2pi * f / fs .* n) .* s))
end

function circ_stats(deg::AbstractVector{<:Real})
    z = mean(cis.(deg2rad.(deg)))
    return (mean = rad2deg(angle(z)), std = rad2deg(sqrt(max(-2 * log(max(abs(z), 1e-12)), 0.0))), R = abs(z))
end

# Tone monitor: predicts each block's first tone phase from the previous block. A lost sample moves it by 2*pi*f/fs.
mutable struct ToneMon
    fs::Float64
    bufsize::Int
    prev_f::Vector{Float64}
    prev_ph::Vector{Float64}
    have::Bool
end
ToneMon(fs, bufsize) = ToneMon(fs, bufsize, zeros(2), zeros(2), false)

# Returns (f1, residual RX1, residual RX2, RX1-RX2 phase in deg); residuals are NaN for the first block.
function tone_update!(tm::ToneMon, x1, x2)
    f1, p1 = tone_fit(x1, tm.fs); f2, p2 = tone_fit(x2, tm.fs)
    r1 = NaN; r2 = NaN
    if tm.have
        r1 = wrap_pi(p1 - (tm.prev_ph[1] + 2pi * 0.5 * (f1 + tm.prev_f[1]) * tm.bufsize / tm.fs))
        r2 = wrap_pi(p2 - (tm.prev_ph[2] + 2pi * 0.5 * (f2 + tm.prev_f[2]) * tm.bufsize / tm.fs))
    end
    tm.prev_f .= (f1, f2); tm.prev_ph .= (p1, p2); tm.have = true
    return (f = f1, res1 = r1, res2 = r2, dphi = rad2deg(wrap_pi(p1 - p2)))
end

# Weighted straight line y = a + b x. Returns a, b, the standard error of b, and the residuals.
function fit_line(x::AbstractVector, y::AbstractVector, w::AbstractVector = ones(length(x)))
    W = sum(w); xm = sum(w .* x) / W; ym = sum(w .* y) / W
    Sxx = sum(w .* (x .- xm) .^ 2)
    b = sum(w .* (x .- xm) .* (y .- ym)) / Sxx
    a = ym - b * xm
    res = y .- (a .+ b .* x)
    n = length(x)
    se = n > 2 ? sqrt(sum(w .* res .^ 2) / (n - 2) / Sxx) : NaN
    return (a = a, b = b, se_b = se, res = res)
end

# Delay and phase from a cross spectrum: maximise |sum rho_k w_k exp(-j 2 pi f_k tau)| over a grid of tau, then refine.
# f in Hz (baseband is fine: the RF part only moves phi0), rho complex per bin, w weights.
function delay_search(f::AbstractVector, rho::AbstractVector, w::AbstractVector; tmax_ns = 1000.0, step_ns = 0.5)
    coh(t) = sum(w .* rho .* cis.(-2pi .* f .* t * 1e-9))
    best = argmax(t -> abs(coh(t)), -tmax_ns:step_ns:tmax_ns)
    st = step_ns
    for _ in 1:4
        st /= 10
        best = argmax(t -> abs(coh(t)), (best - 10st):st:(best + 10st))
    end
    z = coh(best)
    phi0 = rad2deg(angle(z))
    res = [rad2deg(wrap_pi(angle(rho[i]) - deg2rad(phi0) - 2pi * f[i] * best * 1e-9)) for i in eachindex(f)]
    fl = fit_line(f, res, w)          # the uncertainty of the slope, from the scatter of the residuals
    return (tau_ns = best, tau_err_ns = fl.se_b / 360 * 1e9, phi0_deg = phi0, coherent = abs(z) / sum(w .* abs.(rho)), res_deg = res)
end

# rms change of a phase series over a lag of `lag` samples (deg): the phase structure function divided by sqrt(2).
function phase_change_rms(ph::AbstractVector{<:Real}, lag::Int)
    lag >= length(ph) && return NaN
    d = ph[1+lag:end] .- ph[1:end-lag]
    return sqrt(mean(abs2, d) / 2)
end

# Persistent timing steps: median of the next 5 lags against the previous 5, so a one-off USB delay is ignored.
function lag_steps(lags::AbstractVector{<:Real}, step_ms::Real)
    n = length(lags); steps = Tuple{Int,Float64}[]; i = 2
    while i <= n
        d = median(lags[i:min(i + 4, n)]) - median(lags[max(1, i - 5):i-1])
        abs(d) > step_ms ? (push!(steps, (i, d)); i += 5) : (i += 1)
    end
    return steps
end
