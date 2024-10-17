#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Fills an m-by-n matrix A to BETA on the diagonal and
/// ALPHA on the offdiagonals.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlaset</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Mat_Fill( Half half,
  Size m, Size n,
  const T_Scalar &alpha, const T_Scalar &beta,
  T_Blk_A A_, Stride A_ld )
{
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  switch( half )
  {
  case Half::Upper:
    {
      // Set the strictly upper triangular or
      // trapezoidal part of the array to ALPHA.
      for( Index j = 1; j < (Index)n; ++j )
      { for( Index i = 0; i < Min(j-1,(Index)(m-1)); ++i )
      { A(i,j) = alpha; } }
    }
    break;

  case Half::Lower:
    {
      // Set the strictly lower triangular or trapezoidal part of the
      // array to ALPHA.
      for( Index j = 0; j < (Index)Min(m,n); ++j )
      { for( Index i = j+1; i < (Index)m; ++i )
      { A(i,j) = alpha; } }
    }
    break;

  default:
  case Half::Both:
    {
      // Set the leading m-by-n submatrix to ALPHA.
      for( Index j = 0; j < (Index)n; ++j )
      { for( Index i = 0; i < (Index)m; ++i )
      { A(i,j) = alpha; } }
    }
    break;
  }

  // Set the first min(M,N) diagonal elements to BETA.
  for( Index i = 0; i < (Index)Min(m,n); ++i )
  { A(i,i) = beta; }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif