# This supports nonbroadcasting math on NamedDimArrays

# Matrix product
valid_matmul_dims(a::Tuple{Symbol}, b::Tuple{Vararg{Symbol}}) = true
function valid_matmul_dims(a::Tuple{Symbol, Symbol}, b::Tuple{Vararg{Symbol}})
    a_dim = a[end]
    b_dim = b[1]

    return a_dim === b_dim || a_dim === :_ || b_dim === :_
end

matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b,)::Tuple{Symbol}) = (a1,)
matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b1, b2)::Tuple{Symbol,Symbol}) = (a1, b2)
matmul_names((a1,)::Tuple{Symbol,},(b1,b2)::Tuple{Symbol,Symbol}) = (a1, b2)

function matrix_prod_names(a, b)
    # 0 Allocations. See `@btime (()-> matrix_prod_names((:foo, :bar),(:bar,)))()`
    valid_matmul_dims(a, b) || throw(DimensionMismatch(
            "Cannot take matrix product of arrays with different inner dimension names. $a vs $b"
        ))
    res = matmul_names(a, b)
    compile_time_return_hack(res)
end


for (NA, NB) in ((1,2), (2,1), (2,2))  #Vector * Vector, is not allowed
    @eval function Base.:*(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
        L = matrix_prod_names(A,B)
        data = *(parent(a), parent(b))
        return NamedDimsArray{L}(data)
    end
end

"""
    @declare_matmul(MatrixT, VectorT=nothing)

This macro helps define matrix multiplication for the types
with 2D type parameterization `MatrixT` and 1D `VectorT`.
It defines the various overloads for `Base.:*` that are required.
It should be used at the top level of a module.
"""
macro declare_matmul(MatrixT, VectorT=nothing)
    dim_combos = VectorT === nothing ? ((2,2),) : ((1,2), (2,1), (2,2))
    codes = map(dim_combos) do (NA, NB)
        TA_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NA})
        TB_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NB})
        TA_other = (VectorT, MatrixT)[NA]
        TB_other = (VectorT, MatrixT)[NB]

        quote
            function Base.:*(a::$TA_named, b::$TB_other)
                return *(a, NamedDims.NamedDimsArray{NamedDims.names(b)}(b))
            end
            function Base.:*(a::$TA_other, b::$TB_named)
                return *(NamedDims.NamedDimsArray{NamedDims.names(a)}(a), b)
            end
        end
    end
    return esc(Expr(:block, codes...))
end

@declare_matmul(AbstractMatrix, AbstractVector)
@declare_matmul(Diagonal,)

# LinearAlgebra
#==
    Note on implementation of factorisition types:
    The general strategy is to make one of the fields of the factorization types
    into a NamedDimsArray.
    We can then dispatch on this as required, and strip it out with `parent` for accessing operations.
    However, the type of the field does not actually always match to what it should be
    from a mathematical perspective.
    Which is corrected in `Base.getproperty`, as required
==#

## lu

function LinearAlgebra.lu!(nda::NamedDimsArray{L}, args...; kwargs...) where L
    inner_lu = lu!(parent(nda), args...; kwargs...)
    factors = NamedDimsArray{L}(getfield(inner_lu, :factors))
    ipiv = getfield(inner_lu, :ipiv)
    info = getfield(inner_lu, :info)
    return LU(factors, ipiv, info)
end

function Base.parent(F::LU{T,<:NamedDimsArray{L}}) where {T, L}
    factors = parent(getfield(F, :factors))
    ipiv = getfield(F, :ipiv)
    info = getfield(F, :info)
    return LU(factors, ipiv, info)
end

function Base.getproperty(F::LU{T,<:NamedDimsArray{L}}, d::Symbol) where {T, L}
    inner = getproperty(parent(F), d)
    if d == :L || d == :U
        return NamedDimsArray{L}(inner)
    elseif d == :P
        perm_matrix_labels = (first(L), first(L))
        return NamedDimsArray{perm_matrix_labels}(inner)
    elseif d == :p
        perm_vector_labels = (first(L),)
        return NamedDimsArray{perm_vector_labels}(inner)
    else
        return inner
    end
end
>>>>>>> introduce lu!
