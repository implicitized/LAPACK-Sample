#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Performs one of the matrix-vector operations
///
///    x := A*x,   or   x := (~A)*x,
///
/// where x is an n element vector and  A is an n by n unit, or non-unit,
/// upper or lower triangular matrix.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dtrmv</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Vec_x >
requires( areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Vec_x>> > )
constexpr void Tri_VecMul(
  Half half, Trnsp A_trnsp, Diag diag, 
  Size n,
  T_Blk_A A_, Stride A_ld,
  T_Vec_x x_, Stride x_s )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto x = [&]( auto i ) -> auto &
  { return Lyt::VecRef( x_, i, x_s ); };
  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Validate arguments.

  if( Half::Both == half ){ throw BadArgument{ "Tri_VecMul", 1 }; }
  if( A_ld < (Stride)Max(1,n) ){ throw BadArgument{ "Tri_VecMul", 6 }; }

  // Quick return if possible.

  if( 0 == n ){ return; }

  // Start the operations. In this version the elements of A are
  // accessed sequentially with one pass through A.

  switch( A_trnsp )
  {
  case Trnsp::No:
    {
      // Form  x := A*x.

      if( Half::Upper == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          const auto xj = x(j);
          if( ! IsZero( xj ) )
          {
            for( Index i = 0; i <= j-1; ++i )
            { x(i) += xj*A(i,j); }
            if( Diag::NotUnit == diag )
            { x(j) *= A(j,j); }
          }
        }
      }
      else if( Half::Lower  == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          const auto xj = x(j);
          if( ! IsZero( xj ) )
          {
            for( Index i = (Index)(n-1); i >= j+1; --i )
            { x(i) += xj*A(i,j); }
            if( Diag::NotUnit == diag )
            { x(j) *= A(j,j); }
          }
        }
      }
    }
    break;

  case Trnsp::Yes:
    {
      // Form  x := (~A)*x.

      if( Half::Upper == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          auto &xj = x(j);
          if( Diag::NotUnit == diag )
          { xj *= A(j,j); }
          for( Index i = j-1; i >= 0; --i )
          { xj += A(i,j)*x(i); }
        }
      }
      else if( Half::Lower  == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          auto &xj = x(j);
          if( Diag::NotUnit == diag )
          { xj *= A(j,j); }
          for( Index i = j+1; i < (Index)n; ++i )
          { xj += A(i,j)*x(i); }
        }
      }
    }
    break;

  case Trnsp::Conj:
    {
      // Form  x := Conj(~A)*x.

      if( Half::Upper == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          auto &xj = x(j);
          if( Diag::NotUnit == diag )
          { xj *= Conj(A(j,j)); }
          for( Index i = j-1; i >= 0; --i )
          { xj += Conj(A(i,j))*x(i); }
        }
      }
      else if( Half::Lower  == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          auto &xj = x(j);
          if( Diag::NotUnit == diag )
          { xj *= Conj(A(j,j)); }
          for( Index i = j+1; i < (Index)n; ++i )
          { xj += Conj(A(i,j))*x(i); }
        }
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