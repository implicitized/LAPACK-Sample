#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Performs row exchanges driven by an index vector.
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_piv >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
         && isNativeSignedIntegral< Decay<DerefTypeOf<T_Arr_piv>> > )
constexpr void Mat_RowSwp(
  Size n,
  T_Blk_A A_, Stride A_ld,
  Index k0, Index k1,
  T_Arr_piv piv_ )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };

  if( 0 == n )
  { return; }

  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  for( Index i = k0; i <= k1; ++i )
  {
    const Index i1 = piv_[i];
    if( i == i1 )
    { continue; }

    Vec_Swap< Lyt >( n, A_Row(i,0), A_rs, A_Row(i1,0), A_rs );
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif