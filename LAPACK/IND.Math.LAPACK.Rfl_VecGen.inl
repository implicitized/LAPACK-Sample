#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Generates a real elementary reflector H of order n, such that
///
///       H * [ alpha ] = [ beta ],   (~H)*H = I.
///           [   x   ]   [   0  ]
///
/// where alpha and beta are scalars, and x is an (n-1)-element real
/// vector. H is represented in the form
///
///       H = I - tau * [ 1 ] * [ 1 ~v ]
///                     [ v ]
///
/// where tau is a real scalar and v is a real (n-1)-element
/// vector.
///
/// If the elements of x are all zero, then tau = 0 and H is taken to be
/// the unit matrix.
///
/// Otherwise  1 &lt;= tau &lt;= 2.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlarfg</c>.
/// </remarks>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x >
requires( ! isComplex< T_Scalar > && areTheSame< T_Scalar, Decay<DerefTypeOf<T_Vec_x>> > )
constexpr void Rfl_VecGen( Size n, T_Scalar &alpha, T_Vec_x x_, Stride x_s, T_Scalar &tau ) noexcept
{
  if( 0 == n ){ return; }

  if( 1 == n )
  {
    tau = {};
    return;
  }

  auto xnorm = Vec_Norm2< Lyt >( n-1, x_, x_s );

  if( IsZero( xnorm ) )
  {
    // H  =  I
    tau = {};
  }
  else
  {
    // General case

    auto beta = -CopySign( Hypot( alpha, xnorm ), alpha );
    auto safmin = minValue< T_Scalar >;
    Index knt = 0;
    if( Abs( beta ) < safmin )
    {
      // XNORM, BETA may be inaccurate; scale X and recompute them

      auto rsafmn = Inv( safmin );
      do
      {
        ++knt;
        Vec_Scale< Lyt >( n-1, rsafmn, x_, x_s );
        beta *= rsafmn;
        alpha *= rsafmn;
      }
      while( ( Abs(beta) < safmin ) && ( knt < 20 ) );

      // New BETA is at most 1, at least SAFMIN

      xnorm = Vec_Norm2< Lyt >( n-1, x_, x_s );
      beta = -CopySign( Hypot( alpha, xnorm ), alpha );
    }
    tau = ( beta - alpha )/beta;
    Vec_Scale< Lyt >( n-1, Inv( alpha - beta ), x_, x_s );

    // If ALPHA is subnormal, it may lose relative accuracy

    for( Index j = 0; j < knt; ++j )
    { beta *= safmin; }
    alpha = beta;
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif