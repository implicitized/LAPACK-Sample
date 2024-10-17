#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// A := alpha*x*(~y) + A.
///
/// For a general matrix A.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dger</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y,
  typename T_Blk_A >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_x>>,
  Decay<DerefTypeOf<T_Vec_y>>,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Mat_Rank1Upd(
  Size m, Size n,
  const T_Scalar &alpha,
  T_Vec_x x_, Stride x_s,
  T_Vec_y y_, Stride y_s,
  T_Blk_A A_, Stride A_ld )
{
  auto x = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( x_, i, x_s ); };
  auto y = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( y_, i, y_s ); };
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // A := 0*x*(~y) + A
  if( IsZero( alpha ) )
  { return; }

  for( Index j = 0; j < (Index)n; ++j )
  {
    const auto ayj = alpha*y(j);
    for( Index i = 0; i < (Index)m; ++i )
    { A(i,j) += x(i)*ayj; }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif