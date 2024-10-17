#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Computes the values  scl  and  smsq  such that
///
///    ( scl**2 )*smsq = x( 1 )^2 +...+ x( n )^2 + ( scale^2 )*sumsq,
///
/// where  x( i ) = X( 1 + ( i - 1 )*INCX ). The value of  sumsq  is
/// assumed to be non-negative and  scl  returns the value
///
///    scl = max( scale, abs( x( i ) ) ).
///
/// scale and sumsq must be supplied in SCALE and SUMSQ and
/// scl and smsq are overwritten on SCALE and SUMSQ respectively.
///
/// The routine makes only one pass through the vector x.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlassq</c>.
/// </remarks>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_x>> > )
constexpr void Vec_SmSqr(
  Size n, T_Vec_x x, Stride x_s,
  T_Scalar &scale, T_Scalar &sumsq )
{
  if( 0 == n ){ return; }

  for( Index i = 0; i < (Index)n; ++i )
  {
    const auto absxi = Abs( Lyt::VecRef( x, i, x_s ) );
    if( ( IntSignOrZero( absxi ) > 0 ) || IsUndefined( absxi ) )
    {
      if( scale < absxi )
      {
        sumsq = unit<T_Scalar> + sumsq*Sqr( scale / absxi );
        scale = absxi;
      }
      else
      {
        sumsq += Sqr( absxi / scale );
      }
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif