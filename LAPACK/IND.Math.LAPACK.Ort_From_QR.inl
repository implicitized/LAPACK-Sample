#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_QR_WorkSize( Size m, Size n, Size k ) noexcept
{
  IND_NOT_USED( m );
  IND_NOT_USED( n );
  return m;
}
/// <summary>
/// Generates an m by n real matrix Q with orthonormal columns,
/// which is defined as the first n columns of a product of k elementary
/// reflectors of order m.
///
///       Q  =  H(1) H(2) . . . H(k)
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dorg2r</c>.
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
constexpr void Ort_From_QR(
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
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  if( n > m ){ throw BadArgument{ "Ort_From_QR", 2 }; }
  if( k > n ){ throw BadArgument{ "Ort_From_QR", 3 }; }

  // Quick return if possible

  if( 0 == n ){ return; }

  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Scalar zero = {};
  const Scalar one = unit<Scalar>;

  // Initialise columns k+1:n-1 to columns of the unit matrix

  for( Index j = (Index)k; j < (Index)n; ++j )
  {
    for( Index h = 0; h < (Index)m; ++h )
    { A(h,j) = zero; }
    A(j,j) = one;
  }

  for( Index i = (Index)(k-1); i >= 0; --i )
  {
    // Apply H(i) to A(i:m-1,i:n-1) from the left
    if( i < (Index)(n-1) )
    {
      A(i,i) = one;
      Rfl_MatMul< Lyt >( Side::Left, m-i, n-(i+1),
        A_Col(i,i), A_cs, tau[i],
        A_Blk(i,i+1), A_ld, work );
    }
    if( i < (Index)(m-1) )
    { Vec_Scale< Lyt >( m-(i+1), -tau[i], A_Col(i+1,i), A_cs ); }
    A(i,i) = one - tau[i];
    Vec_Zero< Lyt >( i, A_Col(0,i), A_cs );
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif