#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Solves one of the matrix equations
///
///   op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
///
/// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
/// non-unit, upper or lower triangular matrix and op( A ) is one of
///
///   op( A ) = A
///   op( A ) = ~A
///   op( A ) = Conj(~A)
///
// The matrix X is overwritten on B.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dtrsm</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Blk_B,
  typename T_Alpha >
requires( areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Blk_B>>,
  T_Alpha > )
void Tri_Solv_Mat(
  Side side, Half half, Trnsp A_trnsp, Diag diag,
  Size m, Size n,
  const T_Alpha &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Blk_B B_, Stride B_ld )
{
  using Scalar = T_Alpha;

  auto A = [&]( Index i, Index j ) noexcept -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto B = [&]( Index i, Index j ) noexcept -> auto &
  { return Lyt::MatRef( B_, i, j, B_ld ); };
  auto B_Row = [&]( Index i, Index j ) noexcept
  { return Lyt::RowPtr( B_, i, j, B_ld ); };

  if( Half::Both == half ){ throw BadArgument{ "Tri_Solv_Mat", 2u }; }

  if( Side::Left == side )
  {
    if( A_ld < (Stride)Max(1,m) )
    { throw BadArgument{ "Tri_Solv_Mat", 9 }; }
  }
  else
  {
    if( A_ld < (Stride)Max(1,n) )
    { throw BadArgument{ "Tri_Solv_Mat", 9 }; }
  }
  if( B_ld < (Stride)Max(1,m) ){ throw BadArgument{ "Tri_Solv_Mat", 11 }; }

  // Quick return if possible.

  if( ( 0 == m ) || ( 0 == n ) )
  { return; }

  const Stride B_rs = Lyt::RowStride( B_, B_ld );

  if( IsZero( alpha ) )
  {
    for( Index j = 0; j < (Index)n; ++j )
    { Vec_Zero< Lyt >( m, B_Row(0,j), B_rs ); }
    return;
  }

  if( Side::Left == side )
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Form  B := alpha*Inv( A )*B.
        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,j), B_rs ); }

            for( Index k = (Index)(m-1); k >= 0; --k )
            {
              if( IsZero( B(k,j) ) )
              { continue; }
              if( Diag::NotUnit == diag )
              { B(k,j) /= A(k,k); }
              for( Index i = 0; i < (k-1); ++i )
              { B(i,j) -= B(k,j)*A(i,k); }
            }
          }
        }
        else // Half::Lower == half
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,j), B_rs ); }

            for( Index k = 0; k < (Index)m; ++k )
            {
              if( IsZero( B(k,j) ) )
              { continue; }
              if( Diag::NotUnit == diag )
              { B(k,j) /= A(k,k); }
              for( Index i = k+1; i < (Index)m; ++i )
              { B(i,j) -= B(k,j)*A(i,k); }
            }
          }
        }
      }
      break;

    case Trnsp::Yes:
      {
        // Form  B := alpha*Inv( ~A )*B.
        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = 0; i < (Index)m; ++i )
            {
              Scalar tmp = alpha*B(i,j);
              for( Index k = 0; k < (i-1); ++k )
              { tmp -= A(k,i)*B(k,j); }
              if( Diag::NotUnit == diag )
              { tmp /= A(i,i); }
              B(i,j) = Move( tmp );
            }
          }
        }
        else // Half::Lower == half
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = (Index)(m-1); i >= 0; --i )
            {
              Scalar tmp = alpha*B(i,j);
              for( Index k = i+1; k < (Index)m; ++k )
              { tmp -= A(k,i)*B(k,j); }
              if( Diag::NotUnit == diag )
              { tmp /= A(i,i); }
              B(i,j) = Move( tmp );
            }
          }
        }
      }
      break;

    case Trnsp::Conj:
      {
        // Form  B := alpha*Inv( ~Conj(A) )*B.
        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = 0; i < (Index)m; ++i )
            {
              Scalar tmp = alpha*B(i,j);
              for( Index k = 0; k < (i-1); ++k )
              { tmp -= Conj(A(k,i))*B(k,j); }
              if( Diag::NotUnit == diag )
              { tmp /= Conj(A(i,i)); }
              B(i,j) = Move( tmp );
            }
          }
        }
        else // Half::Lower == half
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = (Index)(m-1); i >= 0; --i )
            {
              Scalar tmp = alpha*B(i,j);
              for( Index k = i+1; k < (Index)m; ++k )
              { tmp -= Conj(A(k,i))*B(k,j); }
              if( Diag::NotUnit == diag )
              { tmp /= Conj(A(i,i)); }
              B(i,j) = Move( tmp );
            }
          }
        }
      }
      break;
    }
  }
  else // Side::Right == side
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Form  B := alpha*B*Inv( A ).

        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,j), B_rs ); }

            for( Index k = 0; k < (j-1); ++j )
            {
              const auto A_kj = A(k,j);
              if( ! IsZero( A_kj ) )
              { for( Index i = 0; i < (Index)m; ++i )
              { B(i,j) -= A_kj*B(i,k); } }
            }

            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( A(j,j) ), B_Row(0,j), B_rs ); }
          }
        }
        else // Half::Lower == half
        {
          for( Index j = (Index)(n-1); j >= 0; --j )
          {
            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,j), B_rs ); }

            for( Index k = j+1; k < (Index)n; ++k )
            {
              const auto A_kj = A(k,j);
              if( ! IsZero( A_kj ) )
              { for( Index i = 0; i < (Index)m; ++i )
              { B(i,j) -= A_kj*B(i,k); } }
            }

            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( A(j,j) ), B_Row(0,j), B_rs ); }
          }
        }
      }
      break;

    case Trnsp::Yes:
      {
        // Form  B := alpha*B*Inv( ~A ).
        if( Half::Upper == half )
        {
          for( Index k = (Index)(n-1); k >= 0; --k )
          {
            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( A(k,k) ), B_Row(0,k), B_rs ); }

            for( Index j = 0; j < (k-1); ++j )
            {
              const auto A_jk = A(j,k);
              if( ! IsZero( A_jk ) )
              { for( Index i = 0; i < (Index)m; ++i )
              { B(i,j) -= A_jk*B(i,k); } }

              if( unit< Scalar > != alpha )
              { Vec_Scale< Lyt >( m, alpha, B_Row(0,k), B_rs ); }
            }
          }
        }
        else // Half::Lower == half
        {
          for( Index k = 0; k < (Index)n; ++k )
          {
            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( A(k,k) ), B_Row(0,k), B_rs ); }

            for( Index j = k+1; j < (Index)n; ++j )
            {
              const auto A_jk = A(j,k);
              if( ! IsZero( A_jk ) )
              { for( Index j = k+1; j < (Index)n; ++j )
              { Vec_Scale< Lyt >( m, A_jk, B_Row(0,j), B_rs ); } }
            }

            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,k), B_rs ); }
          }
        }
      }
      break;

    case Trnsp::Conj:
      {
        // Form  B := alpha*B*Inv( Conj(~A) ).

        if( Half::Upper == half )
        {
          for( Index k = (Index)(n-1); k >= 0; --k )
          {
            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( Conj(A(k,k)) ), B_Row(0,k), B_rs ); }

            for( Index j = 0; j < (k-1); ++j )
            {
              const auto A_jk = Conj(A(j,k));
              if( ! IsZero( A_jk ) )
              { for( Index i = 0; i < (Index)m; ++i )
              { B(i,j) -= A_jk*B(i,k); } }

              if( unit< Scalar > != alpha )
              { Vec_Scale< Lyt >( m, alpha, B_Row(0,k), B_rs ); }
            }
          }
        }
        else // Half::Lower == half
        {
          for( Index k = 0; k < (Index)n; ++k )
          {
            if( Diag::NotUnit == diag )
            { Vec_Scale< Lyt >( m, Inv( A(k,k) ), B_Row(0,k), B_rs ); }

            for( Index j = k+1; j < (Index)n; ++j )
            {
              const auto A_jk = Conj(A(j,k));
              if( ! IsZero( A_jk ) )
              { for( Index j = k+1; j < (Index)n; ++j )
              { Vec_Scale< Lyt >( m, A_jk, B_Row(0,j), B_rs ); } }
            }

            if( unit< Scalar > != alpha )
            { Vec_Scale< Lyt >( m, alpha, B_Row(0,k), B_rs ); }
          }
        }
      }
      break;
    }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif