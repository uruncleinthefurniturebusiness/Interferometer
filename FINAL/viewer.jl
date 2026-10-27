# =====================================================================
# viewer.jl (FINAL) -- live display of an interferometer run: fringes and visibility spectra
#
#     julia viewer.jl [path=<file>.vis] [key=value ...]
#
# Runs as its own process and follows the .vis file that pluto_interferometer.jl is writing, so plotting can never
# stall the capture. Without path= it opens the newest .vis in dir and moves to a newer one when a new run starts.
# It also replays a finished file. Panels:
#   fringe          band-average Re rho, Im rho and |rho| against time (the fringes); with target=sun also |rho| after
#                   fringe stopping
#   fringe phase    band-average phase against time, raw and (target=sun) fringe-stopped
#   |rho[k]|        visibility amplitude against frequency, averaged over the last `avg` records
#   arg rho[k]      visibility phase against frequency, same records
#   auto spectra    P1 and P2 (dBFS per bin), DC mask shaded, HI line marked when in the band
#   waterfall       arg rho (or |rho|) in wf_cols channels against record number
# snapshot=<file>.png renders the whole file once and saves the picture instead of opening a window.
# =====================================================================

include(joinpath(@__DIR__, "src", "PlutoIF.jl"))
const P = PlutoIF
using GLMakie, Dates, Printf, Statistics

Base.@kwdef mutable struct ViewCfg
    path::String       = ""
    dir::String        = "C:/PlutoCaptures"
    avg::Int           = 10          # records averaged in the spectra
    smooth::Int        = 8           # bins summed (complex) per point in the spectra, for display only
    span_min::Float64  = 15.0        # time shown in the fringe panels (minutes)
    wf_rows::Int       = 300         # records kept in the waterfall
    wf_cols::Int       = 256         # frequency channels in the waterfall
    wf::String         = "phase"     # phase | amp
    poll_s::Float64    = 0.5
    follow_new::Bool   = true        # without path=: switch to a newer .vis when one appears
    snapshot::String   = ""          # save a PNG of the whole file and exit
end

# Everything known about the file being shown.
mutable struct ViewState
    path::String
    nfft::Int
    fs::Float64
    f_lo::Float64
    dc_mask::Int
    keep::BitVector
    fbb::Vector{Float64}
    fmhz::Vector{Float64}
    geo::P.GeoCfg
    sun::Bool
    fol::P.VisFollower
    t::Vector{Float64}             # record mid times (unix s)
    rho::Vector{ComplexF64}        # band average
    rho_fs::Vector{ComplexF64}     # fringe-stopped band average (target=sun)
    last::Vector{P.Record}         # the last `avg` records
    lasttau::Vector{Float64}       # their predicted delays (target=sun), for fringe stopping before averaging
    wf::Matrix{Float32}            # wf_cols x wf_rows, newest row last
    nflag::Int
    fringe_s::Float64
end

getf(m, k, d) = haskey(m, k) ? parse(Float64, m[k]) : d

function open_state(path::String, vc::ViewCfg)
    mp = P.meta_path(path)
    for _ in 1:40                                  # the run writes .meta at its start; give it a moment
        isfile(mp) && break
        sleep(0.25)
    end
    isfile(mp) || error("No .meta next to $path")
    m = P.read_meta(mp)
    nfft = parse(Int, m["nfft"]); fs = parse(Float64, m["fs_hz"]); f_lo = parse(Float64, m["f_lo_hz"])
    dcm = parse(Int, get(m, "dc_mask", "4"))
    geo = P.GeoCfg(target = get(m, "target", "none"), lat_deg = getf(m, "lat_deg", -33.9), lon_deg = getf(m, "lon_deg", 18.46),
                   baseline_m = getf(m, "baseline_m", NaN), baseline_az_deg = getf(m, "baseline_az_deg", NaN),
                   cable_delay_ns = getf(m, "cable_delay_ns", 0.0), diam_arcmin = getf(m, "diam_arcmin", 32.0))
    sun = geo.target == "sun" && isfinite(geo.baseline_m) && isfinite(geo.baseline_az_deg)
    fbb = P.bb_freqs(nfft, fs)
    return ViewState(path, nfft, fs, f_lo, dcm, P.band_keep(nfft, dcm), fbb, (f_lo .+ fbb) ./ 1e6, geo, sun,
                     P.VisFollower(path, nfft), Float64[], ComplexF64[], ComplexF64[], P.Record[], Float64[],
                     fill(NaN32, vc.wf_cols, vc.wf_rows), 0, NaN)
end

# rho in wf_cols channels (DC bins left out), as phase (deg) or amplitude.
function wf_row(st::ViewState, rec::P.Record, vc::ViewCfg)
    edges = round.(Int, range(0, st.nfft, length = vc.wf_cols + 1))
    row = fill(NaN32, vc.wf_cols)
    for j in 1:vc.wf_cols
        idx = [k for k in (edges[j]+1):edges[j+1] if st.keep[k]]
        isempty(idx) && continue
        z = sum(ComplexF64.(rec.V[idx])) / sqrt(sum(Float64, rec.P1[idx]) * sum(Float64, rec.P2[idx]))
        row[j] = vc.wf == "amp" ? abs(z) : rad2deg(angle(z))
    end
    return row
end

function ingest!(st::ViewState, rec::P.Record, vc::ViewCfg)
    tm = 0.5 * (rec.t_start + rec.t_end)
    push!(st.t, tm)
    push!(st.rho, P.band_rho(rec, st.keep))
    tau = 0.0
    if st.sun
        g = P.sun_geometry(st.geo, st.f_lo, unix2datetime(tm))
        tau = g.tau + 1e-9 * st.geo.cable_delay_ns
        push!(st.rho_fs, P.fringe_stop(rec, st.fbb, st.keep, st.f_lo, tau))
        st.fringe_s = 1 / max(abs(g.fringe_hz), 1e-9)
    end
    push!(st.last, rec); push!(st.lasttau, tau)
    length(st.last) > vc.avg && (popfirst!(st.last); popfirst!(st.lasttau))
    st.wf[:, 1:end-1] .= st.wf[:, 2:end]; st.wf[:, end] .= wf_row(st, rec, vc)
    rec.flags != 0 && (st.nflag += 1)
    return st
end

pts(x, y) = isempty(x) ? [Point2f(0, NaN)] : Point2f.(x, y)

function build_figure(vc::ViewCfg)
    fig = Figure(size = (1500, 980), fontsize = 13)
    status = Label(fig[0, 1:3], "waiting for data", tellwidth = false, halign = :left, fontsize = 14)
    ax1 = Axis(fig[1, 1], title = "Fringe: band-average ρ against time", xlabel = "Time since start (min)", ylabel = "ρ")
    ax2 = Axis(fig[2, 1], title = "Fringe phase", xlabel = "Time since start (min)", ylabel = "Phase (deg)", yticks = -180:90:180)
    ax3 = Axis(fig[1, 2:3], title = "Visibility amplitude |ρ[k]| (bins summed in groups of $(vc.smooth))", ylabel = "|ρ|")
    ax4 = Axis(fig[2, 2:3], title = "Visibility phase arg ρ[k] (fringe-stopped when target=sun)", ylabel = "Phase (deg)", yticks = -180:90:180)
    ax5 = Axis(fig[3, 1], title = "Auto spectra", xlabel = "Frequency (MHz)", ylabel = "Power per bin (dB)")
    ax6 = Axis(fig[3, 2], title = vc.wf == "amp" ? "Waterfall: |ρ|" : "Waterfall: arg ρ", xlabel = "Frequency (MHz)", ylabel = "Record (newest at top)")
    ylims!(ax2, -185, 185); ylims!(ax4, -185, 185)
    o = (re = Observable(pts([], [])), im = Observable(pts([], [])), amp = Observable(pts([], [])), ampfs = Observable(pts([], [])),
         ph = Observable(pts([], [])), phfs = Observable(pts([], [])),
         sa = Observable(pts([], [])), sp = Observable(pts([], [])), a1 = Observable(pts([], [])), a2 = Observable(pts([], [])),
         wfx = Observable(collect(1.0:vc.wf_cols)), wfz = Observable(fill(NaN32, vc.wf_cols, vc.wf_rows)),
         status = status.text)
    lines!(ax1, o.re, color = :steelblue, label = "Re ρ")
    lines!(ax1, o.im, color = :darkorange, label = "Im ρ")
    lines!(ax1, o.amp, color = :black, linewidth = 2, label = "|ρ|")
    lines!(ax1, o.ampfs, color = :seagreen, linewidth = 2, linestyle = :dash, label = "|ρ| fringe-stopped")
    axislegend(ax1, position = :rt, framevisible = false, labelsize = 11)
    scatter!(ax2, o.ph, color = :steelblue, markersize = 5, label = "raw")
    scatter!(ax2, o.phfs, color = :seagreen, markersize = 5, label = "fringe-stopped")
    axislegend(ax2, position = :rt, framevisible = false, labelsize = 11)
    lines!(ax3, o.sa, color = :black)
    scatter!(ax4, o.sp, color = :steelblue, markersize = 3)
    lines!(ax5, o.a1, color = :steelblue, label = "RX1")
    lines!(ax5, o.a2, color = :darkorange, label = "RX2")
    axislegend(ax5, position = :rt, framevisible = false, labelsize = 11)
    hm = heatmap!(ax6, o.wfx, 1:vc.wf_rows, o.wfz, colormap = vc.wf == "amp" ? :viridis : :twilight,
                  colorrange = vc.wf == "amp" ? (0, 1) : (-180, 180), nan_color = :gray85)
    Colorbar(fig[3, 3], hm, label = vc.wf == "amp" ? "|ρ|" : "deg", width = 12)
    linkxaxes!(ax3, ax4); linkxaxes!(ax5, ax6)
    return fig, o, (ax1, ax2, ax3, ax4, ax5, ax6)
end

# Marks that depend on the file: DC mask and HI line.
function decorate!(axes, st::ViewState)
    w = (st.dc_mask + 0.5) * st.fs / st.nfft / 1e6; c = st.f_lo / 1e6
    for ax in axes[3:5]
        vspan!(ax, c - w, c + w, color = (:gray, 0.25))
        st.fmhz[1] <= P.HI_REST / 1e6 <= st.fmhz[end] && vlines!(ax, P.HI_REST / 1e6, color = :red, linestyle = :dot)
    end
end

function refresh!(o, axes, st::ViewState, vc::ViewCfg)
    isempty(st.t) && return
    t0 = st.t[1]; tmin = (st.t .- t0) ./ 60
    sel = findall(>=(tmin[end] - vc.span_min), tmin)
    x = tmin[sel]; z = st.rho[sel]
    o.re[] = pts(x, real.(z)); o.im[] = pts(x, imag.(z)); o.amp[] = pts(x, abs.(z))
    o.ph[] = pts(x, rad2deg.(angle.(z)))
    if st.sun
        zf = st.rho_fs[sel]
        o.ampfs[] = pts(x, abs.(zf)); o.phfs[] = pts(x, rad2deg.(angle.(zf)))
    end
    # With target=sun each record is fringe-stopped before the average, so the turning fringe does not cancel and the
    # phase left is the instrument's (bandpass) phase. Otherwise the plain average (it loses amplitude if the fringe turns).
    V = sum(ComplexF64.(r.V) .* cis.(-2pi .* (st.f_lo .+ st.fbb) .* t) for (r, t) in zip(st.last, st.lasttau)); P1 = sum(r -> Float64.(r.P1), st.last); P2 = sum(r -> Float64.(r.P2), st.last)
    V[.!st.keep] .= 0; P1k = copy(P1); P2k = copy(P2); P1k[.!st.keep] .= 0; P2k[.!st.keep] .= 0
    m = max(1, vc.smooth); g = st.nfft ÷ m
    grp(v) = vec(sum(reshape(v[1:g*m], m, g), dims = 1))
    rho = grp(V) ./ sqrt.(grp(P1k) .* grp(P2k))
    fg = vec(mean(reshape(st.fmhz[1:g*m], m, g), dims = 1))
    o.sa[] = pts(fg, abs.(rho)); o.sp[] = pts(fg, rad2deg.(angle.(rho)))
    n = length(st.last)
    o.a1[] = pts(st.fmhz, 10log10.(max.(P1 ./ n, 1e-30))); o.a2[] = pts(st.fmhz, 10log10.(max.(P2 ./ n, 1e-30)))
    edges = range(st.fmhz[1], st.fmhz[end], length = vc.wf_cols + 1)
    o.wfx[] = collect(0.5 .* (edges[1:end-1] .+ edges[2:end])); o.wfz[] = copy(st.wf)
    for ax in axes[[1, 3, 5]]; autolimits!(ax); end
    xlims!(axes[2], max(0, tmin[end] - vc.span_min), max(tmin[end], 0.1))
    zl = st.rho[end]; r = st.last[end]
    o.status[] = @sprintf("%s   |   %d records, last %s UTC   |   band |ρ| %.4f, phase %.1f°   |   P1 %.1f dBFS, P2 %.1f dBFS   |   flagged %d%s   |   avg %d records",
        basename(st.path), length(st.t), Dates.format(unix2datetime(st.t[end]), "HH:MM:SS"), abs(zl), rad2deg(angle(zl)),
        P.dbfs(r.pow1), P.dbfs(r.pow2), st.nflag,
        st.sun ? @sprintf("   |   Sun: stopped |ρ| %.4f, %.1f°, fringe period %.0f s", abs(st.rho_fs[end]), rad2deg(angle(st.rho_fs[end])), st.fringe_s) : "",
        n)
end

function main(args)
    vc = ViewCfg(); P.apply_kv!(args, vc)
    path = isempty(vc.path) ? P.newest_vis(vc.dir) : vc.path
    while isempty(path)
        println("No .vis in $(vc.dir) yet; waiting for a run to start..."); sleep(2)
        path = P.newest_vis(vc.dir)
    end
    st = open_state(path, vc)
    fig, o, axes = build_figure(vc)
    decorate!(axes, st)
    if !isempty(vc.snapshot)
        foreach(rec -> ingest!(st, rec, vc), P.poll!(st.fol))
        vc.span_min = Inf
        refresh!(o, axes, st, vc)
        save(vc.snapshot, fig)
        println("Saved ", vc.snapshot, " (", length(st.t), " records)")
        return
    end
    println("Showing ", path)
    screen = display(fig)
    while isopen(screen)
        new = P.poll!(st.fol)
        if !isempty(new)
            foreach(rec -> ingest!(st, rec, vc), new)
            refresh!(o, axes, st, vc)
        end
        if isempty(vc.path) && vc.follow_new
            p = P.newest_vis(vc.dir)
            if !isempty(p) && p != st.path
                println("New run: ", p)
                st = open_state(p, vc)
                fig, o, axes = build_figure(vc); decorate!(axes, st)
                screen = display(fig)
            end
        end
        sleep(vc.poll_s)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
