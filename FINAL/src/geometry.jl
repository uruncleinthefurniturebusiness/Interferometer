# Sun position, baseline geometry and fringe stopping. Same frame and sign convention as the simulator in
# Real/pluto_interferometer.jl (sim_geometry) and observe.jl, so measured and simulated fringes compare directly.

# Bessel J1 (Numerical Recipes rational approximation, absolute error about 1e-8).
function besselj1(x::Float64)
    ax = abs(x)
    if ax < 8.0
        y = x * x
        n = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        d = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))))
        return n / d
    end
    z = 8.0 / ax; y = z * z; xx = ax - 2.356194491
    a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
    a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
    ans = sqrt(0.636619772 / ax) * (cos(xx) * a1 - z * sin(xx) * a2)
    return x < 0 ? -ans : ans
end

# Normalized visibility of a uniform disk of diameter diam_arcmin at projected baseline q (wavelengths); negative past the first null.
disk_factor(diam_arcmin::Float64, q::Float64) = (x = pi * deg2rad(diam_arcmin / 60) * q; x < 1e-9 ? 1.0 : 2besselj1(x) / x)

# Low-precision solar coordinates (Astronomical Almanac, about 0.01 deg from 1950 to 2050); refraction ignored.
function sun_radec(t::DateTime)
    n = datetime2julian(t) - 2451545.0
    L = mod(280.460 + 0.9856474 * n, 360.0)
    g = deg2rad(mod(357.528 + 0.9856003 * n, 360.0))
    lam = deg2rad(L + 1.915 * sin(g) + 0.020 * sin(2g))
    eps = deg2rad(23.439 - 4.0e-7 * n)
    return mod(atan(cos(eps) * sin(lam), cos(lam)), 2pi), asin(sin(eps) * sin(lam))
end

# Greenwich mean sidereal time (rad) from UT, to about 0.1 s over this century.
gmst_rad(t::DateTime) = deg2rad(mod(280.46061837 + 360.98564736629 * (datetime2julian(t) - 2451545.0), 360.0))

# Baseline (length, azimuth RX1 -> RX2) toward a source at hour angle H, declination dec: (u,v,w) in wavelengths at f,
# the geometric delay of channel 2 (s) and the altitude.
function baseline_uvw(geo::GeoCfg, f::Float64, H::Float64, dec::Float64)
    lat = deg2rad(geo.lat_deg); az = deg2rad(geo.baseline_az_deg)
    east = geo.baseline_m * sin(az); north = geo.baseline_m * cos(az)
    X = -north * sin(lat); Y = east; Z = north * cos(lat)                  # equatorial frame (m)
    bs = X * cos(dec) * cos(H) - Y * cos(dec) * sin(H) + Z * sin(dec)      # baseline . source direction
    lam = C_LIGHT / f
    u = (sin(H) * X + cos(H) * Y) / lam
    v = (-sin(dec) * cos(H) * X + sin(dec) * sin(H) * Y + cos(dec) * Z) / lam
    alt = asin(clamp(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(H), -1, 1))
    return (u = u, v = v, w = bs / lam, tau = -bs / C_LIGHT, alt = alt)
end

# The Sun at time t: hour angle, declination, altitude, (u,v,w), geometric delay of RX2 (s), fringe rate (Hz) and
# the predicted disk factor.
function sun_geometry(geo::GeoCfg, f::Float64, t::DateTime)
    ra, dec = sun_radec(t)
    H = mod(gmst_rad(t) + deg2rad(geo.lon_deg) - ra + pi, 2pi) - pi
    g0 = baseline_uvw(geo, f, H, dec); g1 = baseline_uvw(geo, f, H + OMEGA_E, dec)
    q = hypot(g0.u, g0.v)
    return (H = H, dec = dec, alt = g0.alt, u = g0.u, v = g0.v, w = g0.w, tau = g0.tau, fringe_hz = g1.w - g0.w,
            q = q, disk = disk_factor(geo.diam_arcmin, q))
end

need_baseline(geo::GeoCfg) = isfinite(geo.baseline_m) && isfinite(geo.baseline_az_deg) ||
    error("target=sun needs the baseline: baseline_m= and baseline_az_deg= (or target=none)")

# Fringe stopping: remove the predicted phase 2*pi*(f_lo + f_k)*tau from every kept bin, then band-average. The per-bin
# rotation also removes the delay slope across the band, so no coherence is lost to it.
function fringe_stop(rec::Record, fbb::Vector{Float64}, keep, f_lo::Float64, tau::Float64)
    ph = 2pi .* (f_lo .+ fbb[keep]) .* tau
    return sum(ComplexF64.(rec.V[keep]) .* cis.(-ph)) / sqrt(sum(Float64, rec.P1[keep]) * sum(Float64, rec.P2[keep]))
end
