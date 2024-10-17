#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Computes the value of the one norm, or the Frobenius norm, or
/// the infinity norm, or the element of largest absolute value of
/// a real symmetric matrix A.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlansy</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_work >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_work>> > )
constexpr Decay<DerefTypeOf<T_Blk_A>> Sym_Norm(
  NormType normType, Half half,
  Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick return if possible.

  if( 0 == n ){ return {}; }

  auto value = undefined< Scalar >;

  const Stride A_cs = Lyt::ColStride( A_, A_ld );
  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_ds = Lyt::DiagStride( A_, A_ld );

  if( Half::Both == half )
  { return value; }

  switch( normType )
  {
  default: break;
  case NormType::Max:
    {
      // Find max(abs(A(i,j))).

      value = {};
      if( Half::Upper == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          for( Index i = 0; i <= j; ++i )
          {
            const auto aij = Abs( A(i,j) );
            if( ( value < aij ) || IsUndefined(aij) )
            { value = aij; }
          }
        }
      }
      else
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          for( Index i = j; i < (Index)n; ++i )
          {
            const auto aij = Abs( A(i,j) );
            if( ( value < aij ) || IsUndefined(aij) )
            { value = aij; }
          }
        }
      }
    }
    break;

  case NormType::Inf:
  case NormType::One:
    {
      // Find normI(A) ( = norm1(A), since A is symmetric).

      value = {};
      if( Half::Upper == half )
      {
        for( Index j = 0; j < n; ++j )
        {
          Scalar sum{};
          for( Index i = 0; i <= (j-1); ++i )
          {
            const auto aij = Abs(A(i,j));
            sum += aij;
            work[i] += aij;
          }
          work[j] = sum + Abs( A(j,j) );
        }
        for( Index i = 0; i < (Index)n; ++i )
        {
          const auto &u = work[i];
          if( ( value < u ) || IsUndefined(u) )
          { value = u; }
        }
      }
      else
      {
        for( Index i = 0; i < (Index)n; ++i )
        { work[i] = {}; }
        for( Index j = 0; j < (Index)n; ++j )
        {
          auto sum = work[j] + Abs( A(j,j) );
          for( Index i = j+1; i < (Index)n; ++i )
          {
            const auto aij = Abs(A(i,j));
            sum += aij;
            work[i] += aij;
          }
          if( ( value < sum ) || IsUndefined(sum) )
          { value = sum; }
        }
      }
    }
    break;

  case NormType::Frob:
    {
      // Find normF(A).
      // SSQ(1) is scale
      // SSQ(2) is sum-of-squares
      // For better accuracy, sum each column separately.

      Scalar ssq[2]{ {}, unit<Scalar> };
      Scalar colssq[2];

      // Sum off-diagonals

      if( Half::Upper == half )
      {
        for( Index j = 1; j < (Index)n; ++j )
        {
          colssq[0] = {};
          colssq[1] = unit<Scalar>;
          const auto A_col = Lyt::ColPtr( A_, 0, j, A_ld );
          Vec_SmSqr< Lyt >( (j-1)+1, A_col, A_cs, colssq[0], colssq[1] );
          Aux_CombSsq2( ssq, colssq );
        }
      }
      else
      {
        for( Index j = 0; j < (Index)(n-1); ++j )
        {
          colssq[0] = {};
          colssq[1] = unit<Scalar>;
          const auto A_col = Lyt::ColPtr( A_, j+1, j, A_ld );
          Vec_SmSqr< Lyt >( n-(j+1), A_col, A_cs, colssq[0], colssq[1] );
          Aux_CombSsq2( ssq, colssq );
        }
      }

      ssq[1] += ssq[1];

      // Sum diagonal

      const auto pA_diag = Lyt::DiagPtr( A_, 0, 0, A_ld );

      colssq[0] = {};
      colssq[1] = unit<Scalar>;
      Vec_SmSqr( n, pA_diag, A_ds, colssq[0], colssq[1] );
      Aux_CombSsq2( ssq, colssq );
      value = ssq[0]*Sqrt( ssq[1] );
    }
    break;
  }

  return value;
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif