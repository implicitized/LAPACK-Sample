#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// QR factorization of a real m by n matrix A.
/// 
///    A = Q * ( R ),
///            ( 0 )
///
/// where:
///
///    Q is a m-by-m orthogonal matrix;
///    R is an upper-triangular n-by-n matrix;
///    0 is a (m-n)-by-n zero matrix, if m > n.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dgeqr2</c>
/// 
/// The matrix Q is represented as a product of elementary reflectors
///
///     Q = H(1) H(2) . . . H(k), where k = min(m,n).
///
///  Each H(i) has the form
///
///     H(i) = I - tau * v * v**T
///
///  where tau is a real scalar, and v is a real vector with
///  v(0:i-1) = 0 and v(i) = 1; v(i+1:m-1) is stored on exit in A(i+1:m-1,i),
///  and tau in TAU(i).
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
constexpr void Mat_Fctr_QR(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay< DerefTypeOf< T_Blk_A > >;

  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Size k = Min( m, n );

  for( Index i = 0; i < (Index)k; ++i )
  {
    // Generate elementary reflector H(i) to annihilate A(i+1:m-1,i)

    Rfl_VecGen< Lyt >( m-(i+1)+1, A(i,i),
      A_Col( Min(i+1,(Index)m-1), i ), A_cs, tau[i] );

    if( i < (Index)n )
    {
      // Apply H(i) to A(i:m-1,i+1:n-1) from the left

      const auto Aii = A(i,i);
      A(i,i) = unit<Scalar>;
      Rfl_MatMul< Lyt >( Side::Left, m-(i+1)+1, n-(i+1),
        A_Col( i, i ), A_cs, tau[i],
        A_Blk( i, i+1 ), A_ld, work );
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