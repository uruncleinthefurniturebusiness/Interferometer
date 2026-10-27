# =====================================================================
# spectrum_plot.jl (FINAL) -- the spectral display of one record (AT-06): both auto spectra and the amplitude and
# phase of the cross spectrum against RF frequency, with three marked bins checked against the .vis file.
#
#     julia spectrum_plot.jl [path=<file>.vis] [rec=10] [f_mhz=1419.5,1420.405751,1420.9] [out=<file>.pdf]
#
# Without path= it uses the newest .vis in C:/PlutoCaptures. The three check values are read from the file a second
# time, straight from the bytes of the record, and compared with the values the plot was drawn from.
# =====================================================================

include(joinpath(@__DIR__, "src", "PlutoIF.jl"))
const P = PlutoIF
using Printf, Plots

Base.@kwdef mutable struct DispCfg
    path::String = ""
    dir::String = "C:/PlutoCaptures"
    rec::Int = 10                                   # record to show (1-based)
    f_mhz::String = "1419.5,1420.405751,1420.9"     # bins to check: nearest bin to each RF frequency
    out::String = ""
end

# One record's spectra read straight from the bytes: the independent path for the check.
function raw_bins(path, nfft, rec, idx)
    open(path, "r") do io
        base = (rec - 1) * P.record_bytes(nfft) + 40
        rd(T, off) = (seek(io, off); read(io, T))
        return [(V = rd(ComplexF32, base + 8(k - 1)), P1 = rd(Float32, base + 8nfft + 4(k - 1)),
                 P2 = rd(Float32, base + 12nfft + 4(k - 1))) for k in idx]
    end
end

function main(args)
    dc = DispCfg(); P.apply_kv!(args, dc)
    path = isempty(dc.path) ? P.newest_vis(dc.dir) : dc.path
    isempty(path) && error("No .vis files in $(dc.dir)")
    v = P.load_vis(path); n = v.nfft; fs = v.fs; flo = v.f_lo; recs = v.recs
    1 <= dc.rec <= length(recs) || error("rec must be 1..$(length(recs))")
    r = recs[dc.rec]
    f = P.bin_freqs(n, fs, flo) ./ 1e6
    a1 = 10log10.(Float64.(r.P1) ./ n); a2 = 10log10.(Float64.(r.P2) ./ n)            # dBFS per bin
    amp = abs.(ComplexF64.(r.V)) ./ sqrt.(Float64.(r.P1) .* Float64.(r.P2))            # |rho[k]|
    ph = rad2deg.(angle.(ComplexF64.(r.V)))                                            # deg
    idx = [argmin(abs.(f .- x)) for x in parse.(Float64, split(dc.f_mhz, ','))]

    gr(); default(fontfamily = "Computer Modern", framestyle = :box, grid = true, titlefontsize = 10, guidefontsize = 9,
                  legendfontsize = 7, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, markerstrokewidth = 0)
    xl = "RF frequency (MHz)"
    mk(p, y) = scatter!(p, f[idx], y[idx], ms = 4, color = :red, label = "checked bins")
    p1 = plot(f, a1, xlabel = xl, ylabel = "Power (dBFS per bin)", title = "Auto spectrum RX1", label = ""); mk(p1, a1)
    p2 = plot(f, a2, xlabel = xl, ylabel = "Power (dBFS per bin)", title = "Auto spectrum RX2", label = ""); mk(p2, a2)
    p3 = plot(f, amp, xlabel = xl, ylabel = "|ρ| (normalised)", title = "Cross-spectrum amplitude", label = "", yscale = :log10); mk(p3, amp)
    p4 = scatter(f, ph, ms = 1, xlabel = xl, ylabel = "Phase (deg)", title = "Cross-spectrum phase", label = "", ylims = (-180, 180)); mk(p4, ph)
    for p in (p1, p2, p3, p4); vline!(p, [1420.405751], ls = :dash, color = :gray, label = "HI rest frequency"); end
    fig = plot(p1, p2, p3, p4, layout = (2, 2), size = (1100, 700),
               plot_title = @sprintf("%s, record %d", basename(path), dc.rec), plot_titlefontsize = 10)
    out = isempty(dc.out) ? replace(path, r"\.vis$" => "_rec$(dc.rec)_display.pdf") : dc.out
    savefig(fig, out); savefig(fig, replace(out, r"\.pdf$" => ".png"))
    println("Wrote ", out)

    # values the plot was drawn from, taken back out of the figure's series
    plotted(p) = p.series_list[1][:y]
    raw = raw_bins(path, n, dc.rec, idx)
    worst = 0.0
    println("\n bin   RF (MHz)      quantity        plotted        from file      rel. diff")
    for (j, k) in enumerate(idx)
        x = raw[j]
        file = (10log10(Float64(x.P1) / n), 10log10(Float64(x.P2) / n), abs(ComplexF64(x.V)) / sqrt(Float64(x.P1) * Float64(x.P2)), rad2deg(angle(ComplexF64(x.V))))
        for (q, p, v) in zip(("RX1 dBFS", "RX2 dBFS", "|rho|", "phase deg"), (p1, p2, p3, p4), file)
            y = plotted(p)[k]
            d = abs(y - v) / max(abs(v), 1e-12)
            worst = max(worst, d)
            @printf(" %4d  %.6f  %-12s %14.6g %14.6g %12.2e\n", k, f[k], q, y, v, d)
        end
    end
    @printf("\nLargest relative difference %.2e: %s (limit 0.1%%)\n", worst, worst <= 1e-3 ? "PASS" : "FAIL")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
