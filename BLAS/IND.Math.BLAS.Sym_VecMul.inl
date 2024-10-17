#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// y := alpha*A*x + beta*y.
///
/// For a symmetric matrix A.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dsymv</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A,
  typename T_Vec_x,
  typename T_Vec_y >
requires( areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Vec_x>>,
  Decay<DerefTypeOf<T_Vec_y>> > )
constexpr void Sym_VecMul( Half half,
  Size n,
  const T_Scalar &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Vec_x x_, Stride x_s,
  const T_Scalar &beta,
  T_Vec_y y_, Stride y_s )
{
  auto x = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( x_, i, x_s ); };
  auto y = [&]( auto i ) -> auto &
  { return Lyt::VecRef( y_, i, y_s ); };
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  if( 0 == n ){ return; }

  // y := y*beta
  Vec_Scale< Lyt >( n, beta, y_, y_s );

  if( IsZero( alpha ) )
  { return; }

  if( Half::Upper == half )
  {
    for( Index j = 0; j < (Index)n; ++j )
    {
      const auto u = alpha*x(j);
      T_Scalar v{};
      for( Index i = 0; i < j; ++i )
      {
        y(i) += u*A(i,j);
        v += A(i,j)*x(i);
      }
      y(j) += u*A(j,j) + alpha*v;
    }
  }
  else if( Half::Lower == half )
  {
    for( Index j = 0; j < (Index)n; ++j )
    {
      const auto u = alpha*x(j);
      T_Scalar v{};
      y(j) += u*A(j,j);
      for( Index i = j + 1; i < (Index)n; ++i )
      {
        y(i)+= u*A(i,j);
        v += A(i,j)*x(i);
      }
      y(j) += alpha*v;
    }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif