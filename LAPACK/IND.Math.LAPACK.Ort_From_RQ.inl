#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_RQ_WorkSize( Size m, Size n, Size k ) noexcept
{
  IND_NOT_USED( m );
  IND_NOT_USED( n );
  return k;
}

/// <summary>
/// Generates an m by n real matrix Q with orthonormal columns,
/// which is defined as the first n columns of a product of k elementary
/// reflectors of order m.
///
///       Q  =  H(1) H(2) . . . H(k)
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dorgr2</c>.
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
constexpr void Ort_From_RQ(
  Size m, Size n, Size k,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };

  if( n < m ){ throw BadArgument{ "Ort_From_RQ", 2 }; }
  if( k > m ){ throw BadArgument{ "Ort_From_RQ", 3 }; }

  // Quick return if possible

  if( 0 == n ){ return; }

  const Scalar one = unit<Scalar>;
  const Scalar zero = {};

  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  if( k < (Index)m )
  {
    // Initialise rows 0:(m-k)-1 to rows of the unit matrix
    for( Index j = 0; j < (Index)n; ++j )
    {
      for( Index h = 0; h < (Index)(m-k); ++h )
      { A(h,j) = zero; }
      if( ( j >= (Index)(n-m) ) && ( j < (Index)(n-k) ) )
      { A( m-n+j, j ) = zero; }
    }
  }

  for( Index i = 0; i < (Index)k; ++i )
  {
    const auto ii = (Index)(m - k + i);

    // Apply H(i) to A(0:m-k+i,0:n-k+i) from the right
    A( ii, n-m+ii ) = one;
    Rfl_MatMul< Lyt >( Side::Right, ii, n-m+ii+1,
      A_Row( ii, 0 ), A_rs, tau[i],
      A_, A_ld, work );
    Vec_Scale< Lyt >( n-m+ii, -tau[i], A_Row( ii, 0 ), A_rs );
    A( ii, n-m+ii ) = one - tau[i];
    Vec_Zero< Lyt >( m-(ii+1), A_Row( ii, n-m+ii ), A_rs );
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif