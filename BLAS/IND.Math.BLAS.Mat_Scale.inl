#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// B := A*d
/// or B := d*A
/// 
/// A and B are m by n
/// If side == Right, d is of length n, or scales the columns of A
/// If side == Left, d is of length m, or scales the rows of A
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_d,
  typename T_Blk_B >
requires( areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_d>>,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Mat_Scale( Side side,
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_d d,
  T_Blk_B B_, Stride B_ld )
{
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  auto B_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( B_, i, j, A_ld ); };
  auto B_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( B_, i, j, A_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Stride B_rs = Lyt::RowStride( B_, B_ld );
  const Stride B_cs = Lyt::ColStride( B_, B_ld );

  if( Side::Right == side )
  {
    // B := A*d
    for( Index j = 0; j < (Index)m; ++j )
    { Vec_Scale< Lyt >( n, d[j], A_Col(0,j), A_cs, B_Col(0,j), B_cs ); }
  }
  else
  {
    for( Index i = 0; i < (Index)n; ++i )
    { Vec_Scale< Lyt >( m, d[i], A_Row(i,0), A_rs, B_Row(i,0), B_rs ); }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif