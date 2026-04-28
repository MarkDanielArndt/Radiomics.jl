"""
Undecimated (stationary) Haar wavelet decomposition for 2D and 3D images.

Filter kernels (periodic boundary conditions):
  Low-pass  (L): out[i] = 0.5 * x[i] + 0.5 * x[i+1]
  High-pass (H): out[i] = 0.5 * x[i] - 0.5 * x[i+1]

2D subbands: LL, LH, HL, HH
3D subbands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

where the letters denote the filter applied along dimension 1, 2 (and 3 for 3D).
"""

"""
Apply a length-2 Haar FIR filter to a 1D signal with periodic boundary conditions.
`lp = true` → low-pass,  `lp = false` → high-pass.
"""
function _haar_1d(signal::AbstractVector{<:Real}, lp::Bool)
    n = length(signal)
    out = Vector{Float64}(undef, n)
    k2 = lp ? 0.5 : -0.5          # second tap; first tap is always 0.5
    @inbounds for i in 1:n
        out[i] = 0.5 * signal[i] + k2 * signal[mod1(i + 1, n)]
    end
    return out
end

"""
Apply the Haar low- or high-pass filter along dimension `dim` of `arr`,
preserving the array size.
"""
function _apply_haar(arr::AbstractArray{<:Real}, lp::Bool, dim::Int)
    return mapslices(v -> _haar_1d(v, lp), arr; dims=dim)
end

"""
    haar_wavelet_2d(img; subband=nothing) -> Dict{String, Matrix{Float64}} or Matrix{Float64}

Compute the one-level undecimated Haar wavelet decomposition of a 2D image.

If `subband` is `nothing` (default), returns a `Dict` with all four subbands,
each the same size as `img`:
- `"LL"`, `"LH"`, `"HL"`, `"HH"` filter along dim 1 then dim 2

If `subband` is specified, only that single subband is computed and returned
directly as a `Matrix{Float64}`. Only the filter passes
needed for that subband are executed, which is more efficient when just one
subband is required.

# Arguments
- `img::AbstractMatrix{<:Real}`: Input 2D image.
- `subband::Union{String,Nothing}`: Optional 2-character subband name, e.g. `"LH"`.
  Each character must be `'L'` (low-pass) or `'H'` (high-pass), corresponding
  to the filter applied along dimension 1 and 2 respectively.

# Returns
- `Dict{String, Matrix{Float64}}`: All subband images (when `subband=nothing`).
- `Matrix{Float64}`: Single subband image (when `subband` is specified).

# Example

# All subbands
subbands = haar_wavelet_2d(img_slice)
ll = subbands["LL"]

# Single subband only
lh = haar_wavelet_2d(img_slice; subband="LH")
"""
function haar_wavelet_2d(img::AbstractMatrix{<:Real}; subband::Union{String,Nothing}=nothing)
    f64 = Float64.(img)
    if subband !== nothing
        length(subband) == 2 || error("2D subband must be 2 characters (e.g. \"LH\"), got \"$subband\"")
        all(c -> c == 'L' || c == 'H', subband) || error("Subband characters must be 'L' or 'H', got \"$subband\"")
        result = f64
        for (dim, c) in enumerate(subband)
            result = _apply_haar(result, c == 'L', dim)
        end
        return result
    end
    L = _apply_haar(f64, true,  1)
    H = _apply_haar(f64, false, 1)
    return Dict{String, Matrix{Float64}}(
        "LL" => _apply_haar(L, true,  2),
        "LH" => _apply_haar(L, false, 2),
        "HL" => _apply_haar(H, true,  2),
        "HH" => _apply_haar(H, false, 2),
    )
end

"""
    haar_wavelet_3d(img; subband=nothing) -> Dict{String, Array{Float64, 3}} or Array{Float64, 3}

Compute the one-level undecimated Haar wavelet decomposition of a 3D image.

If `subband` is `nothing` (default), returns a `Dict` with all eight subbands,
each the same size as `img`:
`"LLL"`, `"LLH"`, `"LHL"`, `"LHH"`, `"HLL"`, `"HLH"`, `"HHL"`, `"HHH"`.

Letters denote the filter applied along dimensions 1, 2, and 3 respectively.

If `subband` is specified, only that single subband is computed and returned
directly as an `Array{Float64, 3}`. 

# Arguments
- `img::AbstractArray{<:Real, 3}`: Input 3D image.
- `subband::Union{String,Nothing}`: Optional 3-character subband name, e.g. `"LLH"`.
  Each character must be `'L'` (low-pass) or `'H'` (high-pass), corresponding
  to the filter applied along dimensions 1, 2, and 3 respectively.

# Returns
- `Dict{String, Array{Float64, 3}}`: All subband images (when `subband=nothing`).
- `Array{Float64, 3}`: Single subband image (when `subband` is specified).

# Example

# All subbands
subbands = haar_wavelet_3d(img)
for (name, sub) in subbands
    features = extract_radiomic_features(sub, mask, spacing; features=[:first_order])
end

# Single subband only
llh = haar_wavelet_3d(img; subband="LLH")

"""
function haar_wavelet_3d(img::AbstractArray{<:Real, 3}; subband::Union{String,Nothing}=nothing)
    f64 = Float64.(img)
    if subband !== nothing
        length(subband) == 3 || error("3D subband must be 3 characters (e.g. \"LLH\"), got \"$subband\"")
        all(c -> c == 'L' || c == 'H', subband) || error("Subband characters must be 'L' or 'H', got \"$subband\"")
        result = f64
        for (dim, c) in enumerate(subband)
            result = _apply_haar(result, c == 'L', dim)
        end
        return result
    end

    L1 = _apply_haar(f64, true,  1)
    H1 = _apply_haar(f64, false, 1)

    LL2 = _apply_haar(L1, true,  2)
    LH2 = _apply_haar(L1, false, 2)
    HL2 = _apply_haar(H1, true,  2)
    HH2 = _apply_haar(H1, false, 2)

    return Dict{String, Array{Float64, 3}}(
        "LLL" => _apply_haar(LL2, true,  3),
        "LLH" => _apply_haar(LL2, false, 3),
        "LHL" => _apply_haar(LH2, true,  3),
        "LHH" => _apply_haar(LH2, false, 3),
        "HLL" => _apply_haar(HL2, true,  3),
        "HLH" => _apply_haar(HL2, false, 3),
        "HHL" => _apply_haar(HH2, true,  3),
        "HHH" => _apply_haar(HH2, false, 3),
    )
end

"""
    haar_wavelet(img; subband=nothing) -> Dict{String, Array{Float64}} or Array{Float64}

Dispatch to `haar_wavelet_2d` or `haar_wavelet_3d` based on the dimensionality of `img`.

If `subband` is `nothing` (default), all subbands are computed and returned as a `Dict`.
If `subband` is specified, only that single subband is computed and returned directly
as an array, skipping all unnecessary filter passes.

# Arguments
- `img::AbstractArray{<:Real}`: 2D or 3D input image.
- `subband::Union{String,Nothing}`: Optional subband name (`"LH"` for 2D, `"LLH"` for 3D).
  Each character must be `'L'` (low-pass) or `'H'` (high-pass).

# Returns
- `Dict{String, Array{Float64}}`: All subband images (when `subband=nothing`).
- `Array{Float64}`: Single subband image (when `subband` is specified).

# Example

# All subbands
subbands = haar_wavelet(img)           # Dict with "LLL".."HHH"

# Single subband only
llh = haar_wavelet(img; subband="LLH") # Array{Float64, 3}

"""
function haar_wavelet(img::AbstractArray{<:Real}; subband::Union{String,Nothing}=nothing)
    nd = ndims(img)
    nd == 2 && return haar_wavelet_2d(img; subband=subband)
    nd == 3 && return haar_wavelet_3d(img; subband=subband)
    error("haar_wavelet only supports 2D and 3D images, got $(nd)D.")
end
