#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Real eigenvalues of:
///
/// [ a b ]
/// [ b c ]
/// </summary>
/// <remarks>
/// This is based on the LAPACK routine <c>dlae2</c>.
/// </remarks>
template< typename T_Scalar >
requires( ! isComplex< T_Scalar > )
constexpr void Aux_Eig2(
  const T_Scalar &a, const T_Scalar &b, const T_Scalar &c,
  T_Scalar &rt1, T_Scalar &rt2 )
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

  switch( IntSignOrZero( sm ) )
  {
  case -1:
    {
      rt1 = _oneHalf< T_Scalar >*( sm - rt );
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
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif