#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Generates a plane rotation so that
///
///    [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS^2 + SN^2 = 1.
///    [ -SN  CS  ]     [ G ]     [ 0 ]
///
/// This is a slower, more accurate version of the BLAS1 routine DROTG,
/// with the following other differences:
///    F and G are unchanged on return.
///    If G=0, then CS=1 and SN=0.
///    If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any
///       floating point operations (saves work in DBDSQR when
///       there are zeros on the diagonal).
///
/// If F exceeds G in magnitude, CS will be positive.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlartg</c>.
/// </remarks>
template< typename T_Scalar >
requires( ! isComplex< T_Scalar > )
constexpr void Aux_PlnRot2(
  const T_Scalar &f, const T_Scalar &g,
  T_Scalar &cs, T_Scalar &sn, T_Scalar &r )
{
  if( IsZero( g ) )
  {
    cs = unit<T_Scalar>;
    sn = {};
    r = f;
    return;
  }

  if( IsZero( f ) )
  {
    cs = {};
    sn = unit<T_Scalar>;
    r = g;
    return;
  }

  if constexpr ( isExact< T_Scalar > )
  {
    r = Hypot( f, g );
    cs = f/r;
    sn = g/r;
  }
  else
  {
    Size count = 0;

    const auto &safmin = minValue< T_Scalar >;
    const auto &safmax = maxValue< T_Scalar >;

    auto f1 = f;
    auto g1 = g;
    auto scale = Max( Abs(f), Abs(g) );
    if( scale >= safmax )
    {
      do
      {
        ++count;
        f1 *= safmin;
        g1 *= safmin;
        scale = Max( Abs(f1), Abs(g1) );
      }
      while( safmax >= scale );

      r = Hypot( f1, g1 );
      cs = f1/r;
      sn = g1/r;
      while( count-- )
      { r *= safmax; }
    }
    else if( scale <= safmin )
    {
      do
      {
        ++count;
        f1 *= safmax;
        g1 *= safmax;
        scale = Max( Abs(f1), Abs(g1) );
      }
      while( scale <= safmin );

      r = Hypot( f1, g1 );
      cs = f1/r;
      sn = g1/r;
      while( count-- )
      { r *= safmin; }
    }
    else
    {
      r = Hypot( f1, g1 );
      cs = f1/r;
      sn = g1/r;
    }

    if( ( Abs(f) > Abs(g) ) && ( IntSignOrZero(cs) < 0 ) )
    {
      cs = -cs;
      sn = -sn;
      r = -r;
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif