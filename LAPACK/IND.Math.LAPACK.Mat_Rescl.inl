#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Multiplies the m by n real matrix A by the real scalar
/// CTO/CFROM.  This is done without over/underflow as long as the final
/// result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
/// A may be full, upper triangular, lower triangular, upper Hessenberg,
/// or banded.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlascl</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Mat_Rescl( MatType type,
  Size kl, Size ku,
  const T_Scalar &cfrom, const T_Scalar &cto,
  Size m, Size n,
  T_Blk_A A_, Stride A_ld )
{
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Argument validation

  if( IsZero(cfrom) || IsUndefined(cfrom) )
  { throw BadArgument{ "Mat_Rescl", 4 }; }

  if( IsUndefined(cto) )
  { throw BadArgument{ "Mat_Rescl", 5 }; }

  if( ( (int)type < (int)MatType::UpperHess ) && (A_ld < Max(1,m)) )
  { throw BadArgument{ "Mat_Rescl", 9 }; }

  if( (int)type >= (int)MatType::LowerBand )
  {
    if( kl > Max(m-1,0) )
    { throw BadArgument{ "Mat_Rescl", 2 }; }

    if( ku > Max(n-1,0) ||
      ((( MatType::LowerBand == type ) ||
        ( MatType::UpperBand == type )) &&
      (kl != ku)) )
    { throw BadArgument{ "Mat_Rescl", 3 }; }

    if( (( MatType::LowerBand == type ) && (A_ld < (kl+1))) ||
        (( MatType::UpperBand == type ) && (A_ld < (ku+1))) ||
        (( MatType::Banded == type ) &&
        (A_ld < 2*(kl+ku+1))) )
    { throw BadArgument{ "Mat_Rescl", 9 }; }
  }

  // Quick return if possible

  if( (0 == n) || (0 == m )){ return; }
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

    switch( type )
    {
    // Technically should not get here.
    default: throw InternalError{ "Mat_Rescl" };

    case MatType::Full:
      {
        for( Index j = 0; j < (Index)n; ++j )
        { for( Index i = 0; i < (Index)m; ++i )
        { A(i,j) *= alpha; } }
      }
      break;

    case MatType::LowerTri:
      {
        for( Index j = 0; j < (Index)n; ++j )
        { for( Index i = j; i < (Index)m; ++i )
        { A(i,j) *= alpha; } }
      }
      break;

    case MatType::UpperTri:
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          const Index nn = Min(j,(Index)(m-1));
          for( Index i = 0; i <= nn; ++i )
          { A(i,j) *= alpha; }
        }
      }
      break;

    case MatType::UpperHess:
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          const Index nn = Min(j+1,(Index)(m-1));
          for( Index i = 0; i <= nn; ++i )
          { A(i,j) *= alpha; }
        }
      }
      break;

    case MatType::LowerBand:
      {
        const Index k3 = (Index)( kl + 1 );
        const Index k4 = (Index)( n + 1 );
        for( Index j = 0; j < (Index)n; ++j )
        {
          const Index nn = Min(k3,k4-(j+1))-1;
          for( Index i = 0; i <= nn; ++i )
          { A(i,j) *= alpha; }
        }
      }
      break;

    case MatType::UpperBand:
      {
        const Index k1 = (Index)( ku + 2 );
        const Index k3 = (Index)( ku + 1 );
        for( Index j = 0; j < (Index)n; ++j )
        {
          for( Index i = Max(k1-(j+1),1)-1; i < k3; ++i )
          { A(i,j) *= alpha; }
        }
      }
      break;

    case MatType::Banded:
      {
        const Index k1 = (Index)( kl + ku + 2 );
        const Index k2 = (Index)( kl + 1 );
        const Index k3 = (Index)( 2*kl + ku + 1 );
        const Index k4 = (Index)( kl + ku + 1 + m );
        for( Index j = 0; j < (Index)n; ++j )
        {
          const Index nn = Min(k3,k4-(j+1));
          for( Index i = Max(k1-(j+1),k2)-1; i < nn; ++i )
          { A(i,j) *= alpha; }
        }
      }
      break;
    }
  }
  while( ! done );
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif