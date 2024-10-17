#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// QL factorization of a real m by n matrix A.
/// <remarks>
/// Based on the LAPACK routine <c>dgelq2</c>
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
constexpr void Mat_Fctr_QL(
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
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Size k = Min( m, n );

  for( Index i = (Index)(k-1); i >= 0; ++i )
  {
    // Generate elementary reflector H(i) to annihilate A(0:(m-k)+(i+1)-1-1,(n-k)+(i+1)-1)

    Rfl_VecGen< Lyt >( n-(i+1)+1, A( (m-k)+i, (n-k)+i ),
      A_Row( 0, (n-k)+i ), A_rs, tau[i] );

    if( i < (Index)m )
    {
      // Apply H(i) to A(0:(m-k)+(i+1)-1-1,(n-k)+(i+1)-1) from the left

      const auto Aii = A( (m-k)+i, (n-k)+i );
      A( (m-k)+i, (n-k)+i ) = unit<Scalar>;
      Rfl_MatMul< Lyt >( Side::Left, (m-k)+(i+1), (n-k)-(i+1)-1,
        A_Col( 0, (n-k)+i ), A_cs, tau[i],
        A_, A_ld, work );
      A( (m-k)+i, (n-k)+i ) = Aii;
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif