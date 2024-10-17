#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Real eigenvalues and eigenvector of:
///
/// [ a b ]
/// [ b c ]
/// </summary>
/// <remarks>
/// This is based on the LAPACK routine <c>dlaev2</c>.
/// </remarks>
template< typename T_Scalar >
requires( ! isComplex< T_Scalar > )
constexpr void Aux_EigVec2(
  const T_Scalar &a, const T_Scalar &b, const T_Scalar &c,
  T_Scalar &rt1, T_Scalar &rt2, T_Scalar &cs1, T_Scalar &sn1 )
{
  using _n_Impl::_oneHalf;

  // Compute the eigenvalues

  const auto sm = a + c;
  const auto df = a - c;
  const auto adf = Abs( df );
  const auto tb = b + b;
  const auto ab = Abs( tb );

  auto acmx = c;
  auto acmn = a;

  if( Abs(a) > Abs(c) )
  {
    acmx = a;
    acmn = c;
  }

  T_Scalar rt{};
  T_Scalar cs{};
  T_Scalar sn{};

  if( adf > ab )
  {
    rt = adf*Hypot( unit<T_Scalar>, ab/adf );
  }
  else if( adf < ab )
  {
    rt = ab*Hypot( unit<T_Scalar>, adf/ab );
  }
  else
  {
    // Includes case AB=ADF=0

    //rt = ab*Sqrt2< T_Scalar >();
    rt = ab*Sqrt( T_Scalar( 2.0 ) );
  }

  int sgn1 = 1;
  int sgn2 = 1;

  switch( IntSignOrZero( sm ) )
  {
  case -1:
    {
      rt1 = _oneHalf< T_Scalar >*( sm - rt );
      sgn1 = -1;
      rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
    }
    break;

  case 1:
    {
      rt1 = _oneHalf< T_Scalar >*( sm + rt );
      rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
    }
    break;

  default:
    {
      // Includes case RT1 = RT2 = 0
      rt1 = _oneHalf< T_Scalar >*rt;
      rt2 = -_oneHalf< T_Scalar >*rt;
    }
    break;
  }

  // Compute the eigenvector

  if( IntSignOrZero( df ) >= 0 )
  {
    cs = df + rt;
  }
  else
  {
    cs = df - rt;
    sgn2 = -1;
  }

  const auto acs = Abs(cs);

  if( acs > ab )
  {
    const auto ct = -tb/cs;
    sn1 = Inv( Hypot( unit<T_Scalar>, ct) );
    cs1 = ct*sn1;
  }
  else
  {
    if( IsZero( ab ) )
    {
      cs1 = unit< T_Scalar >;
      sn1 = {};
    }
    else
    {
      const auto tn = -cs/tb;
      cs1 = Inv( Hypot( unit<T_Scalar>, tn ) );
      sn1 = tn*cs1;
    }
  }
  if( sgn1 == sgn2 )
  {
    auto tn = cs1;
    cs1 = -sn1;
    sn1 = tn;
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif