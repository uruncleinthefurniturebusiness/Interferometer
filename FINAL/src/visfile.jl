# The .vis record file, its .meta text file, and a follower that reads records as a running capture appends them.
#
# Record layout (little-endian): Float64 t_start, Float64 t_end, Int32 nseg, UInt32 flags, Float32 pow1, pow2, clip1,
# clip2, then ComplexF32 V[nfft], Float32 P1[nfft], Float32 P2[nfft]; spectra fftshifted (DC at index nfft/2+1).

record_bytes(nfft::Int) = 40 + 16nfft

function write_record(io::IO, rec::Record)
    write(io, rec.t_start, rec.t_end, rec.nseg, rec.flags, rec.pow1, rec.pow2, rec.clip1, rec.clip2)
    write(io, rec.V); write(io, rec.P1); write(io, rec.P2)
end

function read_record(io::IO, nfft::Int)
    t0 = read(io, Float64); t1 = read(io, Float64); ns = read(io, Int32); fl = read(io, UInt32)
    p1 = read(io, Float32); p2 = read(io, Float32); c1 = read(io, Float32); c2 = read(io, Float32)
    V = Vector{ComplexF32}(undef, nfft); read!(io, V)
    P1 = Vector{Float32}(undef, nfft); read!(io, P1)
    P2 = Vector{Float32}(undef, nfft); read!(io, P2)
    return Record(t0, t1, ns, fl, p1, p2, c1, c2, V, P1, P2)
end

function read_records(path::String, nfft::Int)
    recs = Record[]
    n = filesize(path) ÷ record_bytes(nfft)
    open(path, "r") do io
        for _ in 1:n; push!(recs, read_record(io, nfft)); end
    end
    return recs
end

# Reads only complete records, so it can follow a file that is still being written.
mutable struct VisFollower
    path::String
    nfft::Int
    pos::Int
end
VisFollower(path::String, nfft::Int) = VisFollower(path, nfft, 0)

function poll!(f::VisFollower)
    recs = Record[]
    isfile(f.path) || return recs
    rb = record_bytes(f.nfft)
    n = (filesize(f.path) - f.pos) ÷ rb
    n <= 0 && return recs
    open(f.path, "r") do io
        seek(io, f.pos)
        for _ in 1:n; push!(recs, read_record(io, f.nfft)); end
    end
    f.pos += n * rb
    return recs
end

# The .meta file: "key = value" lines. Written at the start of a run (so a viewer can open the file at once) and
# appended at the end with the run summary.
function write_meta(path::String, pairs; append::Bool = false)
    open(path, append ? "a" : "w") do io
        for (k, v) in pairs
            println(io, k, " = ", v)
        end
    end
end

function read_meta(path::String)
    d = Dict{String,String}()
    for line in eachline(path)
        occursin(" = ", line) || continue
        k, v = split(line, " = "; limit = 2)
        d[strip(k)] = strip(v)
    end
    return d
end

meta_path(vis::String) = replace(vis, r"\.vis$" => ".meta")
