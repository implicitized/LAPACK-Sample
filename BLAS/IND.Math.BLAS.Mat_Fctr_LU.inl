#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

struct Mat_Fctr_LU_Result
{
  bool success = false;

  // If i >= 0, U(i,i) is exactly zero. The factorization
  // has been completed, but the factor U is
  // exactly singular, and division by zero 
  // occur if it is used
  // to solve a system of equations.
  Index i = -1;

  constexpr inline operator bool () const noexcept
  { return this->success; }

  IND_NOTHROW_VITAE( Mat_Fctr_LU_Result );

  constexpr Mat_Fctr_LU_Result( bool success, Index i = -1 ) noexcept
  : success{ success }, i{ i }
  {}
};

/// <summary>
///
/// Computes an LU factorization of a general M-by-N matrix A
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
///    A = P * L * U
/// where P is a permutation matrix, L is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and U is upper
/// triangular (upper trapezoidal if m < n).
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
/// </summary>
///<returns>
/// A <see cref="Mat_Fctr_LU_Result"/> describing the status of the factorization.
/// </returns>
/// <remarks>
/// Based on the BLAS routine <c>dgetrf2</v>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_piv >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
         && isNativeSignedIntegral< Decay<DerefTypeOf<T_Arr_piv>> > )
constexpr Mat_Fctr_LU_Result Mat_Fctr_LU(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_piv piv_ )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  if( 0 == m || 0 == n )
  { return { true }; }

  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  if( 1 == m )
  {
    // Use unblocked code for one row case
    // Just need to handle IPIV and INFO

    piv_[0] = 0;

    if( IsZero( A(0,0) ) )
    { return { true, 0 }; }else
    { return { true }; }
  }
  else if( 1 == n )
  {
    // One column case.

    // Find pivot and test for singularity
    const auto A_col = A_Col( 0, 0 );
    Index k = 0;
    Scalar A_k0 = Lyt::VecRef( A_col, 0, A_cs );
    for( Index i = 1; i < (Index)m; ++i )
    {
      const Scalar &A_i0 = Lyt::VecRef( A_col, i, A_cs );
      if( Abs( A_i0 ) > Abs( A_k0 ) )
      { A_k0 = A_i0; k = i; }
    }
    piv_[0] = k;

    if( IsZero( A_k0 ) )
    {
      return { true, 0 };
    }

    // Apply the interchange.
    if( 0 != k )
    { Swap( A(0,0), A(k,0) ); }

    // Compute elements 1:(M-1) of the column
    //if( Abs( A(0,0) ) >= minValue< Scalar > )
    if( Abs( A_k0 ) >= minValue< Scalar > )
    {
      const Scalar rA_00 = Inv( A_k0 );
      for( Index i = 1; i < (Index)m; ++i )
      { Lyt::VecRef( A_col, i, A_cs ) *= rA_00; }
    }
    else
    {
      for( Index i = 1; i < (Index)m; ++i )
      { Lyt::VecRef( A_col, i, A_cs ) /= A_k0; }
    }

    return { true };
  }

  // Use recursive code

  //        [ A00 ]
  // Factor [ --- ]
  //        [ A10 ]

  const Size piv_n = Min(m,n);

  const Size n1 = Min( m, n )/2;
  const Size n2 = n - n1;

  auto Fctr_00 = Mat_Fctr_LU< Lyt >( m, n1, A_, A_ld, piv_ );

  //                       [ A01 ]
  // Apply interchanges to [ --- ]
  //                       [ A11 ]
  Mat_RowSwp< Lyt >( n2, A_Blk(0,n1), A_ld, 0, (Index)(n1-1), piv_ );

  // Solve A01
  Tri_Solv_Mat< Lyt >( Side::Left, Half::Lower, Trnsp::No, Diag::IsUnit,
    n1, n2, unit< Scalar >,
    A_, A_ld,
    A_Blk( 0, n1 ), A_ld );

  // Update A11
  Mat_MatMul< Lyt >( Trnsp::No, Trnsp::No,
    m-n1, n2, n1, -unit< Scalar >,
    A_Blk( n1, 0 ), A_ld,
    A_Blk( 0, n1 ), A_ld, unit< Scalar >,
    A_Blk( n1, n1 ), A_ld );

  // Factor A11
  auto Fctr_11 = Mat_Fctr_LU< Lyt >( m-n1, n2,
    A_Blk( n1, n1 ), A_ld, piv_ + n1 );

  // Adjust INFO (Fctr_00.i) and the pivot indices
  Fctr_00.success = Fctr_11.success;
  if( 0 == Fctr_00.i )
  { Fctr_00.i = Fctr_11.i + n1; }
  for( Index i = (Index)n1; i < (Index)piv_n; ++i )
  { piv_[i] += (Index)n1; }

  // Apply interchanges to A10
  Mat_RowSwp< Lyt >( n1, A_, A_ld, (Index)n1, (Index)(piv_n-1), piv_ );

  return Fctr_00;
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif