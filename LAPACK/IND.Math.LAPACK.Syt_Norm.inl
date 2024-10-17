#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Returns the value of the one norm,  or the Frobenius norm, or
/// the  infinity norm,  or the  element of  largest absolute value of a
/// real symmetric tridiagonal matrix A given as vectors d and e.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlanst</c>.
/// </remarks>
template< typename Lyt = Flat,
  typename T_Arr_d,
  typename T_Arr_e >
requires( ! isComplex< Decay<DerefTypeOf<T_Arr_d>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Arr_d>>,
  Decay<DerefTypeOf<T_Arr_e>> > )
constexpr Decay<DerefTypeOf<T_Arr_d>> Syt_Norm( NormType normType, Size n, T_Arr_d d, T_Arr_e e )
{
  using Scalar = Decay<DerefTypeOf<T_Arr_d>>;
  // Quick return if possible.

  if( 0 == n ){ return {}; }

  Scalar sum{};
  Scalar anorm{};

  switch( normType )
  {
  case NormType::Max:
    {
      anorm = Abs( d[n-1] );
      for( Index i = 0; i < (Index)(n-1); ++i )
      {
        sum = Abs( d[i] );
        if( ( anorm < sum ) || IsUndefined( sum ) ){ anorm = sum; }
        sum = Abs( e[i] );
        if( ( anorm < sum ) || IsUndefined( sum ) ){ anorm = sum; }
      }
    }
    break;

  case NormType::One:
  case NormType::Inf:
    {
      if( 1 == n )
      {
        anorm = Abs( d[0] );
      }
      else
      {
        anorm = Abs( d[0] ) + Abs( e[0] );
        sum = Abs( e[n-2] ) + Abs( d[n-1] );
        if( ( anorm < sum ) || IsUndefined( sum ) ){ anorm = sum; }

        for( Index i = 1; i < (Index)(n-1); ++i )
        {
          sum = Abs( d[i] ) + Abs( e[i] ) + Abs( e[i-1] );
          if( ( anorm < sum ) || IsUndefined( sum ) ){ anorm = sum; }
        }
      }
    }
    break;

  case NormType::Frob:
    {
      Scalar scale{};
      if( n > 1 )
      {
        Vec_SmSqr< Lyt >( n-1, e, 1, scale, sum );
        sum *= Scalar{2};
      }
      Vec_SmSqr< Lyt >( n, d, 1, scale, sum );
      anorm = scale*Sqrt( sum );
    }
    break;
  }

  return anorm;
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif