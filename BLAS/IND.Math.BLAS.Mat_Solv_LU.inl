#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// DGETRS solves a system of linear equations
///   A * x = b  or  (~A)*x = b
/// with a general N-by-N matrix A using the LU factorization computed
/// by Mat_Fctr_LU.
/// 
/// b is overwritten by x on output.
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_piv,
  typename T_Vec_b
>
void Mat_Solv_LU( Trnsp A_trnsp,
  Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_piv piv_,
  T_Vec_b b_, Stride b_s )
{
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick return if possible
  if( 0 == n )
  { return; }

  switch( A_trnsp )
  {
  default:
    {}throw BadArgument{ "Mat_Solv_LU", 1 };
  case Trnsp::No:
    {
      // Solve A*X = B.

      // Apply row interchanges to the right hand sides.
      //Mat_RowSwp< Lyt >( 1, b_,b_s, 0, (Index)(n-1), piv_ );
      Vec_PivSwp< Lyt >( b_,b_s, 0, (Index)(n-1), piv_ );

      // Solve L*x = b, overwriting b with x.
      Tri_Solv_Vec< Lyt >( Half::Lower, Trnsp::No, Diag::IsUnit,
        n, A_, A_ld, b_, b_s );

      // Solve U*X = B, overwriting B with X.
      Tri_Solv_Vec< Lyt >( Half::Upper, Trnsp::No, Diag::NotUnit,
        n, A_, A_ld, b_, b_s );
    }
    break;

  case Trnsp::Yes:
  case Trnsp::Conj:
    {
      // Solve (~A)*X = B or Conj(~A)*X = B

      // Solve (~U)*X = B, overwriting B with X.
      Tri_Solv_Vec< Lyt >( Half::Upper, A_trnsp, Diag::NotUnit,
        n, A_, A_ld, b_, b_s );

      // Solve L**T *X = B, overwriting B with X.
      Tri_Solv_Vec< Lyt >( Half::Lower, A_trnsp, Diag::IsUnit,
        n, A_, A_ld, b_, b_s );

      // Apply row interchanges to the solution vectors.
      //Mat_RowSwp< Lyt >( 1, b_, b_s, 0, (Index)(n-1), piv_ );
      Vec_PivSwp< Lyt >( b_,b_s, 0, (Index)(n-1), piv_ );
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