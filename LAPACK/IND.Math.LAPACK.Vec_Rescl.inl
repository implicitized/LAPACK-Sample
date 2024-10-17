#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Multiplies the size n real vector X by the real scalar
/// CTO/CFROM.  This is done without over/underflow as long as the final
/// result CTO*A(I,J)/CFROM does not over/underflow.
/// </summary>
/// <remarks>
///
/// </remarks>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_x>> > )
constexpr void Vec_Rescl(
  const T_Scalar &cfrom, const T_Scalar &cto,
  Size n, T_Vec_x x, Size x_s )
{
  // Argument validation

  if( IsZero(cfrom) || IsUndefined(cfrom) )
  { throw BadArgument{ "Vec_Rescl", 1 }; }

  if( IsUndefined(cto) )
  { throw BadArgument{ "Vec_Rescl", 2 }; }

  // Quick return if possible

  if( 0 == n ){ return; }
  if( cfrom == cto ){ return; }

  auto cfromc = cfrom;
  auto ctoc = cto;

  bool done = false;

  do
  {
    T_Scalar alpha{};

    if constexpr ( isExact< T_Scalar > )
    {
      alpha = cto / cfrom;
      done = true;
    }
    else
    {
      const auto smlnum = minValue< T_Scalar >;
      const auto bignum = Inv( smlnum );

      T_Scalar cto1{};
      auto cfrom1 = cfromc*smlnum;

      if( cfrom1 == cfromc )
      {
        // CFROMC is an inf.  Multiply by a correctly signed zero for
        // finite CTOC, or a NaN if CTOC is infinite.
        alpha = ctoc / cfromc;
        done = true;
        cto1 = ctoc;
      }
      else
      {
        cto1 = ctoc / bignum;
        if( cto1 == ctoc )
        {
          // CTOC is either 0 or an inf.  In both cases, CTOC itself
          // serves as the correct multiplication factor.
          alpha = ctoc;
          done = true;
          cfromc = unit<T_Scalar>;
        }
        else if( (Abs(cfrom1) > Abs(ctoc)) && ! IsZero(ctoc) )
        {
          alpha = smlnum;
          done = false;
          cfromc = cfrom1;
        }
        else if( Abs(cto1) > Abs(cfromc) )
        {
          alpha = bignum;
          done = false;
          ctoc = cto1;
        }
        else
        {
          alpha = ctoc / cfromc;
          done = true;
        }
      }
    }

    Vec_Scale< Lyt >( n, alpha, x, x_s );
  }
  while( ! done );
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif