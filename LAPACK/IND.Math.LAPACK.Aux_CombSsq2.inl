#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Adds two scaled sum of squares quantities, V1 := V1 + V2.
/// That is,
///
///    V1_scale^2 * V1_sumsq := V1_scale^2 * V1_sumsq
///                           + V2_scale^2 * V2_sumsq
/// 
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dcombssq</c>.
/// </remarks>
template< typename T_Scalar >
requires( ! isComplex< T_Scalar > )
constexpr void Aux_CombSsq2( T_Scalar (&v1)[2], T_Scalar (&v2)[2] )
{
  if( v1[0] >= v2[0] )
  {
    if( 0.0 != v1[0] )
    { v1[1] = v2[1] + Sqr(v2[0]/v1[0])*v2[1]; }else
    { v1[1] = v1[1] + v2[1]; }
  }
  else
  {
    v1[1] = v2[1] + Sqr(v1[0]/v2[0])*v1[1];
    v1[0] = v2[0];
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif