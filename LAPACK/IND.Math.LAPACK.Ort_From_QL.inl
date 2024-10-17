#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_QL_WorkSize( Size m, Size n, Size k ) noexcept
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
/// Based on the LAPACK routine <c>dorg2l</c>.
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
constexpr void Ort_From_QL(
  Size m, Size n, Size k,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  if( n > m ){ throw BadArgument{ "Ort_From_QL", 2 }; }
  if( k > n ){ throw BadArgument{ "Ort_From_QL", 3 }; }

  // Quick return if possible

  if( 0 == n ){ return; }

  const Scalar zero = {};
  const Scalar one = unit<Scalar>;

  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  // Initialise columns 0:n-k to columns of the unit matrix

  for( Index j = 0; j < (Index)(n-k); ++j )
  {
    for( Index h = 0; h < (Index)m; ++h )
    { A(h,j) = zero; }
    A( m-n+j, j ) = one;
  }

  for( Index i = 0; i < (Index)k; ++i )
  {
    Index ii = n-k+i;

    // Apply H(i) to A(0:(m-k+(i+1))-1,1:(n-k+(i+1))-1) from the left
    A( m-n+ii, ii ) = one;
    Rfl_MatMul< Lyt >( Side::Left, i+1, ii,
      A_Col( 0, ii ), A_cs, tau[i],
      A_, A_ld, work );
    Vec_Scale< Lyt >( i, -tau[i], A_Col( 0, ii ), A_cs );
    A( i, ii ) = one - tau[i];
    if( i < (Index)(m-1) )
    { Vec_Zero< Lyt >( m-(i+1), A_Col( i+1, ii ), A_cs ); }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif