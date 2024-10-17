#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// RQ factorization of a real m by n matrix A.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dgerq2</c>
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_tau,
  typename T_Arr_work >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_tau>>,
  Decay<DerefTypeOf<T_Arr_work>> > )
constexpr void Mat_Fctr_RQ(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  const Size k = Min( m, n );

  for( Index i = (Index)(k-1); i >= 0; --i )
  {
    // Generate elementary reflector H(i) to annihilate A( (m-k)+i, 0:(n-k)+i-1 )

    Rfl_VecGen< Lyt >( (n-k)+(i+1), A( (m-k)+i, (n-k)+i ),
      A_Row( (m-k)+i, 0 ), A_rs, tau[i] );

    // Apply H(i) to A( 0:(m-k)+i-1, 0:(n-k)+i ) from the right

    const auto Aii = A( (m-k)+i, (n-k)+i );
    Rfl_MatMul< Lyt >( Side::Right, (m-k)+(i+1)-1, (n-k)+(i+1),
      A_Row( (m-k)+i, 0 ), A_rs, tau[i],
      A_, A_ld, work );
    A( (m-k)+i, (n-k)+i ) = Aii;
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif