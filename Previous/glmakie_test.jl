using GLMakie
fig = Figure()
Axis(fig[1,1])
scatter!(fig[1,1], rand(10), rand(10))
display(fig)
readline()