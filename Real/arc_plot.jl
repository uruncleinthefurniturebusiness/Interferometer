# arc_plot.jl -- figures for the arc test, from the files written by `julia arc_test.jl fit`
#
# Usage: julia arc_plot.jl tag=A [dir=C:/PlutoCaptures]
#        julia arc_plot.jl path=C:/PlutoCaptures/arc_results_A_fit_points.csv
#        julia arc_plot.jl tag=A sim=../Simulation/figures/arc_..._points.csv     (overlays the arc_sim.jl run)
# Writes <base>_phase.pdf, _residual.pdf, _refs.pdf, _coherence.pdf, _fringe_re.pdf and _fringe_im.pdf (one panel each,
# for the report) and _summary.png.
# The fringe panels are the visibility V = |rho| exp(i(phase - phi0)) with the fitted instrument phase removed, so
# Re(V) and Im(V) swing between +-|rho| as the source moves, as in capture_sim.jl and arc_sim.jl.

using Plots, Printf
gr()

path_diff(t, B, R) = (s = sind(t); sqrt(R^2 + R * B * s + B^2 / 4) - sqrt(R^2 - R * B * s + B^2 / 4))

function plot_arc(args::Vector{String})
    d = Dict{String,String}()
    for a in args
        occursin('=', a) || error("Bad argument '$a' (expected key=value)")
        k, v = split(a, '='; limit = 2); d[String(k)] = String(v)
    end
    pts = haskey(d, "path") ? d["path"] :
          haskey(d, "tag")  ? joinpath(get(d, "dir", "C:/PlutoCaptures"), "arc_results_" * d["tag"] * "_fit_points.csv") :
          error("Give tag=... or path=...")
    base = replace(pts, "_fit_points.csv" => "")
    prm = Dict{String,String}()
    for l in readlines(base * "_fit_params.csv")[2:end]
        k, v = split(l, ','; limit = 2); prm[String(k)] = String(v)
    end
    num(k) = parse(Float64, prm[k])
    B = num("B_fit_m"); Bt = num("B_tape_m"); R = num("R_m"); lam = num("lambda_m")
    t0 = num("theta0_deg"); phi0 = num("phi0_deg"); rms = num("rms_deg"); tag = prm["tag"]

    lines = filter(l -> !isempty(strip(l)), readlines(pts))
    hdr = split(lines[1], ','); rows = [split(l, ',') for l in lines[2:end]]
    val(r, name) = parse(Float64, r[findfirst(==(name), hdr)])
    arc = filter(r -> r[1] == "arc", rows); ref = filter(r -> r[1] == "ref", rows)
    th = [val(r, "theta_deg") for r in arc]; o = sortperm(th); th = th[o]
    col(name) = [val(r, name) for r in arc][o]
    meas = col("meas_unwrapped_deg"); res = col("residual_deg")
    rho = col("rho"); rho_band = col("rho_band"); rho_band_exp = col("rho_band_exp")

    tf = range(minimum(th), maximum(th), length = 400)
    p1 = plot(tf, phi0 .+ 360 .* path_diff.(tf .+ t0, B, R) ./ lam, label = "exact geometry (fit)", lw = 2,
              xlabel = "Source angle theta (deg)", ylabel = "RX1 - RX2 phase (deg)", legend = :topleft,
              title = @sprintf("Arc test %s: B = %.1f mm (tape %.1f mm)", tag, 1e3B, 1e3Bt))
    plot!(p1, tf, phi0 .+ 360 .* B .* sind.(tf .+ t0) ./ lam, label = "far-field approximation", ls = :dash)
    scatter!(p1, th, meas, label = "measured", ms = 3)

    p2 = scatter(th, res, label = "", ms = 3, xlabel = "Source angle theta (deg)", ylabel = "Residual (deg)",
                 title = @sprintf("Residuals, rms %.1f deg", rms))
    hline!(p2, [0.0], color = :black, label = "")
    hline!(p2, [rms, -rms], ls = :dot, label = "+- rms")

    if isempty(ref)
        p3 = plot(title = "No broadside references", legend = false)
    else
        tr = [val(r, "t_min") for r in ref]; pr = [val(r, "phase_corr_deg") for r in ref]
        p3 = plot(tr, pr, marker = :circle, label = "", xlabel = "Time (min)", ylabel = "Phase change (deg)",
                  title = "Instrument phase at broadside")
    end

    p4 = scatter(th, rho, label = "|rho| at the tone", ms = 3, ylims = (0, 1.05),
                 xlabel = "Source angle theta (deg)", ylabel = "Coherence", title = "Coherence", legend = :bottomleft)
    scatter!(p4, th, rho_band, label = "band-average |rho|", ms = 3)
    plot!(p4, th, rho_band_exp, label = "SNR/(1+SNR)", ls = :dash)

    # Fringes: the visibility with the instrument phase removed, against the fitted geometry at the median |rho|
    geo = meas .- phi0
    geo_fit = 360 .* path_diff.(tf .+ t0, B, R) ./ lam
    rho_m = sort(rho)[cld(length(rho), 2)]
    p5 = plot(tf, rho_m .* cosd.(geo_fit), label = "exact geometry (fit)", lw = 2, ylims = (-1.05, 1.05),
              xlabel = "Source angle theta (deg)", ylabel = "Re(V)", title = "Fringes: real part of the visibility",
              legend = :bottomleft)
    scatter!(p5, th, rho .* cosd.(geo), label = "measured", ms = 3)
    p6 = plot(tf, rho_m .* sind.(geo_fit), label = "exact geometry (fit)", lw = 2, ylims = (-1.05, 1.05),
              xlabel = "Source angle theta (deg)", ylabel = "Im(V)", title = "Fringes: imaginary part of the visibility",
              legend = :bottomleft)
    scatter!(p6, th, rho .* sind.(geo), label = "measured", ms = 3)

    # Simulated run (arc_sim.jl) on top of the measured one
    if haskey(d, "sim")
        sl = filter(l -> !isempty(strip(l)), readlines(d["sim"]))
        sr = [parse.(Float64, split(l, ',')) for l in sl[2:end]]
        sth = [r[1] for r in sr]; sph = [r[2] for r in sr]; srho = [r[3] for r in sr]
        scatter!(p1, sth, phi0 .+ sph, label = "simulated", ms = 3, marker = :xcross)
        scatter!(p4, sth, srho, label = "simulated |rho| at the tone", ms = 3, marker = :xcross)
        scatter!(p5, sth, srho .* cosd.(sph), label = "simulated", ms = 3, marker = :xcross)
        scatter!(p6, sth, srho .* sind.(sph), label = "simulated", ms = 3, marker = :xcross)
    end

    for (p, name) in ((p1, "phase"), (p2, "residual"), (p3, "refs"), (p4, "coherence"), (p5, "fringe_re"), (p6, "fringe_im"))
        savefig(plot(p, size = (600, 400)), base * "_" * name * ".pdf")
    end
    savefig(plot(p1, p5, p2, p6, p3, p4, layout = (3, 2), size = (1400, 1300), margin = 5Plots.mm), base * "_summary.png")
    println("Wrote ", base, "_phase.pdf, _residual.pdf, _refs.pdf, _coherence.pdf, _fringe_re.pdf, _fringe_im.pdf and _summary.png")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_arc(ARGS)
end
