#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_LQ_WorkSize( Size m, Size n, Size k ) noexcept
{
  IND_NOT_USED( n );
  IND_NOT_USED( k );
  return m;
}

/// <summary>
/// Generates an m by n real matrix Q with orthonormal columns,
/// which is defined as the first n columns of a product of k elementary
/// reflectors of order m.
///
///       Q  =  H(k) ...  H(2) H(1)
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dorgl2</c>.
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
constexpr void Ort_From_LQ(
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

  if( n < m ){ throw BadArgument{ "Ort_From_LQ", 2 }; }
  if( k > m ){ throw BadArgument{ "Ort_From_LQ", 3 }; }

  // Quick return if possible

  if( 0 == n ){ return; }

  const Scalar one = unit<Scalar>;
  const Scalar zero = {};

  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  // Initialise rows k+1:m-1 to rows of the unit matrix

  for( Index j = 0; j < (Index)n; ++j )
  {
    for( Index h = k+1; h < (Index)m; ++h )
    { A(h,j) = zero; }
    if( ( j > (Index)(k-1) ) && ( j < (Index)m ) )
    { A(j,j) = one; }
  }

  for( Index i = (Index)(k-1); i >= 0; --i )
  {
    // Apply H(i) to A(i:m-1,i:n-1) from the right
    if( i < (Index)(n-1) )
    {
      if( i < (Index)(m-1) )
      {
        A(i,i) = one;
        Rfl_MatMul< Lyt >( Side::Right, m-(i+1), n-i,
          A_Row(i,i), A_rs, tau[i],
          A_Blk(i+1,i), A_ld, work );
      }
      Vec_Scale< Lyt >( n-(i+1), -tau[i], A_Row(i,i+1), A_rs );
    }
    A(i,i) = one - tau[i];
    Vec_Zero< Lyt >( i, A_Row(i,0), A_rs );
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif