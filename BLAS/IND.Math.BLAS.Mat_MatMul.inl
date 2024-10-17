#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// C := alpha*A*B + beta*C
/// or C := alpha*(~A)*B + beta*C
/// or C := alpha*A*(~B) + beta*C
/// or C := alpha*(~A)*(~B) + beta*C
/// 
/// A is m by k, ~A is k by m
/// B is k by n, ~B is n by k
/// C is m by n
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dgemm</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A,
  typename T_Blk_B,
  typename T_Blk_C >
requires( requires( T_Blk_A A_, T_Blk_B B_, T_Blk_C C_, T_Scalar u )
{ { (*C_) = u*(*A_)*(*B_) + u*(*C_) }; } )
constexpr void Mat_MatMul(
  Trnsp A_trnsp, Trnsp B_trnsp,
  Size m, Size n, Size k,
  const T_Scalar &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Blk_B B_, Stride B_ld,
  const T_Scalar &beta,
  T_Blk_C C_, Stride C_ld )
{
  if( (0 == m) || (0 == n) || (0 == k) ){ return; }
  if( IsZero( alpha ) && IsUnit( beta ) ){ return; }

  const Stride C_rs = Lyt::RowStride( C_, C_ld );
  const Stride C_cs = Lyt::ColStride( C_, C_ld );

  if( IsZero( alpha ) )
  {
    if( IsZero( beta ) )
    {
      if constexpr ( isColMajor< Lyt > )
      {
        // Zero-fill columns of C
        for( Index j = 0; j < (Index)n; ++j )
        {
          const auto C_col = Lyt::ColPtr( C_, 0, j, C_ld );
          Vec_Zero< Lyt >( m, C_col, 1 );
        }
      }
      else
      {
        // Zero-fill rows of C
        for( Index i = 0; i < (Index)m; ++i )
        {
          const auto C_row = Lyt::RowPtr( C_, i, 0, C_ld );
          Vec_Zero< Lyt >( n, C_row, 1 );
        }
      }
    }
    else
    {
      if constexpr ( isColMajor< Lyt > )
      {
        // Scale columns of C
        for( Index j = 0; j < (Index)n; ++j )
        {
          const auto C_col = Lyt::ColPtr( C_, 0, j, C_ld );
          Vec_Scale< Lyt >( n, beta, C_col, 1 );
        }
      }
      else
      {
        // Scale rows of C
        for( Index i = 0; i < (Index)m; ++i )
        {
          const auto C_row = Lyt::RowPtr( C_, i, 0, C_ld );
          Vec_Scale< Lyt >( m, beta, C_row, 1 );
        }
      }
    }

    return;
  }

  const Stride B_rs = Lyt::RowStride( B_, B_ld );
  const Stride B_cs = Lyt::ColStride( B_, B_ld );

  switch( B_trnsp )
  {
  case Trnsp::No:
    {
      // C := alpha*(%A)*B + beta*C

      // Transform columns of B
      for( Index j = 0; j < (Index)n; ++j )
      {
        const auto B_col = Lyt::ColPtr( B_, 0, j, B_ld );
        const auto C_col = Lyt::ColPtr( C_, 0, j, C_ld );
        Mat_VecMul< Lyt >( A_trnsp, m, k, alpha, A_, A_ld, B_col, B_cs, beta, C_col, C_cs );
      }
    }
    break;

  case Trnsp::Yes:
    {
      // C := alpha*(%A)*(~B) + beta*C

      // Transform rows of ~B
      for( Index i = 0; i < (Index)n; ++i )
      {
        const auto B_row = Lyt::RowPtr( B_, i, 0, B_ld );
        const auto C_col = Lyt::ColPtr( C_, 0, i, C_ld );
        Mat_VecMul< Lyt >( A_trnsp, m, k, alpha, A_, A_ld, B_row, B_rs, beta, C_col, C_cs );
      }
    }
    break;

  case Trnsp::Conj:
    {
      // C := alpha*(%A)*Conj(~B) + beta*C

      // Transform rows of ~B
      for( Index i = 0; i < (Index)n; ++i )
      {
        const auto B_row = Lyt::RowPtr( B_, i, 0, B_ld );
        const auto C_col = Lyt::ColPtr( C_, 0, i, C_ld );
        Mat_ConjVecMul< Lyt >( A_trnsp, m, k, alpha, A_, A_ld, B_row, B_rs, beta, C_col, C_cs );
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