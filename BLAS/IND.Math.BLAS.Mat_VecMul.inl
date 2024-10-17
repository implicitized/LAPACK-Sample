#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// y := alpha*A*x + beta*y.
/// or y := alpha*(~A)*x + beta*y
///
/// For a general m by n matrix A
/// </summary>
/// <summary>
/// Based on the BLAS routine <c>dgemv</c>.
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Blk_A A_, T_Scalar u, T_Vec_x x, T_Vec_y y )
{ { (*y) = u*(*A_)*(*x) + u*(*y) }; } )
constexpr void Mat_VecMul(
  Trnsp A_trnsp,
  Size m, Size n,
  const T_Scalar &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Vec_x x_, Stride x_s,
  const T_Scalar &beta,
  T_Vec_y y_, Stride y_s )
{
  auto x = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( x_, i, x_s ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  switch( A_trnsp )
  {
  case Trnsp::No:
    {
      // y := beta*y
      Vec_Scale< Lyt >( m, beta, y_, y_s );

      if( IsZero( alpha ) )
      { return; }

      for( Index j = 0; j < (Index)n; ++j )
      {
        const auto axj = alpha*x(j);
        const auto A_col = Lyt::ColPtr( A_, 0, j, A_ld );
        Vec_AXPlusY< Lyt >( m, axj, A_col, A_cs, y_, y_s );
      }
    }
    break;

  case Trnsp::Yes:
    {
      // y := beta*y
      Vec_Scale< Lyt >( n, beta, y_, y_s );

      if( IsZero( alpha ) )
      { return; }

      for( Index i = 0; i < (Index)m; ++i )
      {
        const auto axi = alpha*x(i);
        const auto A_row = Lyt::RowPtr( A_, i, 0, A_ld );
        Vec_AXPlusY< Lyt >( n, axi, A_row, A_rs, y_, y_s );
      }
    }
    break;

  case Trnsp::Conj:
    {
      // y := beta*y
      Vec_Scale< Lyt >( n, beta, y_, y_s );

      if( IsZero( alpha ) )
      { return; }

      for( Index i = 0; i < (Index)m; ++i )
      {
        const auto axi = alpha*x(i);
        const auto A_row = Lyt::RowPtr( A_, i, 0, A_ld );
        Vec_AConjXPlusY< Lyt >( n, axi, A_row, A_rs, y_, y_s );
      }
    }
    break;
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif