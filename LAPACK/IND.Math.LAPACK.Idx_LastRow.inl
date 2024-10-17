#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Index of the last non-zero row, or -1 if the matrix is all zero.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>iladr</c>.
/// </remarks>
template< typename Lyt = ColMajor, typename T_Blk_A >
constexpr Index Idx_LastRow(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld ) noexcept
{
  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick test for the common case where one corner is non-zero.
  if( 0 == m )
  { return m-1; }
  if( (0.0 != A(m-1,0)) || (0.0 != A(m-1,n-1)) )
  { return m-1; }

  Index row = -1;

  // Scan up each column tracking the last zero row seen.
  for( Index j = 0; j < (Index)n; ++j )
  {
    Index i = (Index)(m-1);
    while( (0.0 == A(Max(i,0),j)) && (i >= 0) ){ --i; }
    row = Max( row, i );
  }

  return row;
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif