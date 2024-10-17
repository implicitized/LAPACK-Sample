#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Index of the last non-zero column, or -1 if the matrix is all zero.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>iladc</c>.
/// </remarks>
template< typename Lyt = ColMajor, typename T_Blk_A >
constexpr Index Idx_LastCol(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld ) noexcept
{
  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick test for the common case where one corner is non-zero.
  if( 0 == n )
  { return n-1; }
  if( ! IsZero( A(0,n-1) ) || ! IsZero( A(m-1,n-1) ) )
  { return n-1; }

  // Now scan each column from the end, returning with the first non-zero.
  for( Index j = (Index)(n-1); j >= 0; --j )
  { for( Index i = 0; i < (Index)m; ++i )
  { if( ! IsZero( A(i,j) ) ){ return j; } } }

  return -1;
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif