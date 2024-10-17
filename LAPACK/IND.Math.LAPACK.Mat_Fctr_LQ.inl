#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// LQ factorization of a real m by n matrix A.
/// FINISH!
/// </summary>
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
constexpr void Mat_Fctr_LQ(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  const Size k = Min( m, n );

  for( Index i = 0; i < (Index)k; ++i )
  {
    // Generate elementary reflector H(i) to annihilate A(i,i+1:n-1)

    Rfl_VecGen< Lyt >( n-(i+1)+1, A(i,i),
      A_Row( i, Min(i+1,(Index)n-1) ), A_rs, tau[i] );

    if( i < (Index)m )
    {
      // Apply H(i) to A(i+1:m-1,i:n-1) from the right

      const auto Aii = A(i,i);
      A(i,i) = unit<Scalar>;
      Rfl_MatMul< Lyt >( Side::Right, m-(i+1), n-(i+1)+1,
        A_Row( i, i ), A_rs, tau[i],
        A_Blk( i+1, i ), A_ld, work );
      A(i,i) = Aii;
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif