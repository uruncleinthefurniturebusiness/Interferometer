# Scratch preview: capture_sim.jl's disk-Sun model over the full ~40 min path, with the 3 m dish beam.
using Plots, Random, FFTW
include(joinpath(@__DIR__, "signal_shift_function 1.jl"))
gr(); Random.seed!(1234)

sample_rate = 2.5e6; f_LO = 1.42e9; c = 299792458.0; N_samples = 1000
baseline = 20.0; λ = c / f_LO
T_e = 24*60*60; ω_e = 2π / T_e

# Sun disk (as in capture_sim.jl)
sun_diam = deg2rad(0.6); R_sun = sun_diam / 2; n_grid = 41
grid = range(-R_sun, R_sun, length=n_grid)
sub_offsets = [x for x in grid, y in grid if x^2 + y^2 <= R_sun^2]
N_sub = length(sub_offsets)
SNR_linear = 0.5

# Earth rotation + 3 m fixed dish beam
δ_sun = deg2rad(-6.0); D_dish = 3.0
HPBW = 1.15 * λ / D_dish
θ_null = 1.22 * λ / D_dish
T_view = 2θ_null / (ω_e * cos(δ_sun))
t_range = range(-T_view/2, T_view/2, step=7.3)
theta_range = asin.(cos(δ_sun) .* sin.(ω_e .* t_range))
beam(θ) = exp(-4log(2) * (θ / HPBW)^2)
N_pos = length(t_range)
println("HPBW $(round(rad2deg(HPBW), digits=2)) deg, null $(round(rad2deg(θ_null), digits=2)) deg, ",
        "window $(round(T_view/60, digits=1)) min, $N_pos positions, $N_sub emitters")

function simulate(theta_range, sub_offsets, beam, SNR_linear, N_samples, baseline, c, f_LO, sample_rate)
    N_sub = length(sub_offsets)
    C = Vector{ComplexF64}(undef, length(theta_range))
    for (i, θ) in enumerate(theta_range)
        s1 = zeros(ComplexF64, N_samples); s2 = zeros(ComplexF64, N_samples)
        a = sqrt(beam(θ) * SNR_linear / N_sub)
        for d in sub_offsets
            τ = baseline * sin(θ + d) / c
            b = a .* randn(ComplexF64, N_samples)
            s1 .+= b
            s2 .+= shift_sampled_signal(b, τ, 1/sample_rate, "circular") .* exp(-2π*im*f_LO*τ)
        end
        x1 = s1 .+ randn(ComplexF64, N_samples); x2 = s2 .+ randn(ComplexF64, N_samples)
        C[i] = sum(x1 .* conj.(x2)) / sqrt(sum(abs2, x1) * sum(abs2, x2))
    end
    return C
end
C = simulate(theta_range, sub_offsets, beam, SNR_linear, N_samples, baseline, c, f_LO, sample_rate)

# Analytic: rho(θ) = A·SNR/(1+A·SNR) · grid disk factor, phase 2π f_LO B sinθ / c
V_grid = sum(cos.(2π * baseline / λ .* sub_offsets)) / N_sub
ρ_exp = [beam(θ)*SNR_linear / (1 + beam(θ)*SNR_linear) * V_grid for θ in theta_range]
phase = 2π * f_LO * baseline .* sin.(theta_range) ./ c
V_exp = ρ_exp .* cis.(phase)
tm = t_range ./ 60
σ = 1 / sqrt(2N_samples)

p1 = plot(tm, real.(V_exp), lw=2, label="analytic", color=:black,
    title="Fringes as the Sun crosses the beam (real part of V)",
    xlabel="Time from transit (min)", ylabel="Re(V)")
plot!(p1, tm, ρ_exp, ls=:dash, color=:red, label="beam envelope")
plot!(p1, tm, -ρ_exp, ls=:dash, color=:red, label="")
scatter!(p1, tm, real.(C), ms=2, color=:steelblue, msw=0, label="simulated (N=1000 per point)")
vspan!(p1, [-HPBW/2, HPBW/2] ./ (ω_e*cos(δ_sun)) ./ 60, alpha=0.08, color=:orange, label="HPBW")

p2 = plot(tm, ρ_exp, lw=2, color=:red, label="expected |V|",
    title="Visibility amplitude", xlabel="Time from transit (min)", ylabel="|V|")
scatter!(p2, tm, abs.(C), ms=2, color=:steelblue, msw=0, label="simulated")
hline!(p2, [σ*sqrt(π/2)], ls=:dot, color=:gray, label="noise-only mean |V|")

p3 = scatter(tm, rad2deg.(angle.(C)), ms=2, color=:steelblue, msw=0, label="simulated",
    title="Fringe phase", xlabel="Time from transit (min)", ylabel="Phase (deg)")
plot!(p3, tm, rad2deg.(angle.(V_exp)), color=:black, label="analytic")

p4 = plot(tm, beam.(theta_range), lw=2, color=:orange, legend=false,
    title="3 m dish power beam along the path", xlabel="Time from transit (min)", ylabel="A(θ)")

plt = plot(p1, p2, p3, p4, layout=@layout([a{0.5h}; b c d]), size=(1600, 1000), margin=6Plots.mm)
savefig(plt, joinpath(@__DIR__, "beam_fringes.png"))
println("saved")
