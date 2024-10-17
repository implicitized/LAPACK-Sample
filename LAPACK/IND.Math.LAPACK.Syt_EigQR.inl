#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Eigenvalue solver for Symmetric Tridiagonal matrices.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dsterf</c>, and corrected
/// to use the logic in <c>dsteqr</c> (more accurate).
/// </remarks>
template< typename T_Scalar, typename DefaultLyt = Flat >
requires( ! isComplex< T_Scalar > )
class Syt_EigQR
{
public:

  using Scalar = T_Scalar;

  struct Config
  {
    Size maxIterationCount = 64;
    Scalar zeroTol = std::numeric_limits< T_Scalar >::epsilon(); //Epsilon< Scalar >();
  };

private:

  Config _config;

public:

  constexpr const Config &config() const noexcept
  { return this->_config; }
  constexpr void SetConfig( const Config &config ) noexcept
  { this->_config = config; }

  IND_NOTHROW_VITAE( Syt_EigQR );

  template< typename Lyt = DefaultLyt,
    typename T_Arr_d,
    typename T_Arr_e >
  requires( areTheSame< Scalar,
    Decay<DerefTypeOf<T_Arr_d>>,
    Decay<DerefTypeOf<T_Arr_e>> > )
  constexpr bool Solve( Size n, T_Arr_d d, T_Arr_e e ) const
  {
    if( 0 == n ){ return true; }

    Size count = 0;
    Size maxCount = n*this->_config.maxIterationCount;
    Index k1 = 0;

    const Scalar eps2 = Sqr( this->_config.zeroTol );
    const Scalar safmin = minValue<Scalar>;
    const Scalar safmax = Inv( safmin );
    const Scalar ssfmin = Sqrt( safmin )/eps2;
    const Scalar ssfmax = Sqrt( safmax )/3;

    const Scalar zero = {};
    const Scalar one = unit<Scalar>;

    while( count < maxCount )
    {
      if( k1 > (Index)(n-1) )
      {
        // All eigenvalues and vector found.
        // Converged.
        break;
      }

      // Determine where the matrix splits and choose QL or QR iteration
      // for each block, according to whether top or bottom diagonal
      // element is smaller.

      if( k1 > 0 )
      { e[k1-1] = zero; }

      Index k0 = k1;
      for( ; k0 < (Index)(n-1); ++k0 )
      {
        const Scalar ek0 = Abs( e[k0] );
        if( IsZero( ek0 ) ){ break; }

        const Scalar errk0 = this->_config.zeroTol
                           * Sqrt( Abs(d[k0]) )
                           * Sqrt( Abs(d[k0+1]) );

        if( ek0 <= errk0 )
        {
          e[k0] = {};
          break;
        }
      }

      Index k = k1;
      Index k1_prev = k1;
      Index kend = k0;
      Index kend_prev = kend;
      k1 = k0+1;
      if( kend == k )
      { continue; }

      // Scale submatrix in rows and columns k to kend

      const auto anorm = Syt_Norm< Lyt >( NormType::Max, kend-k+1, d+k, e+k );
      if( IsZero( anorm ) )
      { continue; }

      const auto scale = Clamp( anorm, ssfmin, ssfmax );

      Vec_Rescl< Lyt >( anorm, scale, kend-k+1, d+k, 1 );
      Vec_Rescl< Lyt >( anorm, scale, kend-k, e+k, 1 );

      // Choose between QL and QR iteration

      if( Abs(d[kend]) < Abs(d[k]) )
      {
        kend = k1_prev;
        k = kend_prev;
      }

      if( kend >= k )
      {
        // QL iteration
        while( k <= kend )
        {
          // Look for a small subdiagonal element.

          if( k != kend )
          {
            for( k0 = k; k0 <= (kend-1); ++k0 )
            {
              const Scalar ek02 = Sqr(e[k0]);
              const Scalar errk02 = eps2 * Abs(d[k0]*d[k0+1]);
              if( ek02 <= ( errk02 + safmin ) )
              { break; }
            }
          }
          else
          {
            k0 = kend;
          }

          if( k0 < kend )
          { e[k0] = zero; }

          if( k0 == k )
          {
            // Eigenvalue found.
            ++k;
            continue;
          }

          // If remaining matrix is 2 by 2, use Aux_Eig2 to
          // compute its eigensystem.

          if( k0 == k+1 )
          {
            Aux_Eig2( d[k], e[k], d[k+1], d[k], d[k+1] );
            e[k] = {};
            k += 2;
            continue;
          }

          if( count == maxCount )
          {
            // Failed to converge.
            return false;
          }
          ++count;

          // Form shift.
          Scalar f = ( d[k+1] - d[k] )/( 2*e[k] );
          Scalar r = Hypot( f, one );
          Scalar g = d[k0] - d[k] + ( e[k]/( f + CopySign(r,f) ) );
          Scalar c = one;
          Scalar s = one;
          Scalar p = {};

          for( Index i = k0-1; i >= k; --i )
          {
            f = s*e[i];
            const Scalar b = c*e[i];
            Aux_PlnRot2( g, f, c, s, r );
            if( i != k0-1 )
            { e[i+1] = r; }
            g = d[i+1] - p;
            r = ( d[i] - g )*s + 2*c*b;
            p = s*r;
            d[i+1] = g + p;
            g = c*r - b;
          }

          d[k] -= p;
          e[k] = g;
        }
        // while( g <= gend )
      }
      else // End of QL block
      {
        // QR iteration
        while( k >= kend )
        {
          // Look for a small superdiagonal element.
          if( k != kend )
          {
            for( k0 = k; k0 >= (kend+1); --k0 )
            {
              const Scalar ek02 = Sqr(e[k0-1]);
              const Scalar errk02 = eps2 * Abs(d[k0]*d[k0-1]);
              if( ek02 <= ( errk02 + safmin ) )
              { break; }
            }
          }
          else
          {
            k0 = kend;
          }

          if( k0 > kend )
          { e[k0-1] = zero; }
          if( k0 == k )
          {
            // Eigenvalue found.
            --k;
            continue;
          }

          // If remaining matrix is 2 by 2, use Aux_Eig2 to
          // compute its eigensystem.

          if( k0 == k-1 )
          {
            Aux_Eig2( d[k-1], e[k-1], d[k], d[k-1], d[k] );
            e[k-1] = {};
            k -= 2;
            continue;
          }

          if( count == maxCount )
          {
            // Failed to converge.
            return false;
          }
          ++count;

          // Form shift.

          Scalar f = ( d[k-1] - d[k] )/( 2*e[k-1] );
          Scalar r = Hypot( f, one );
          Scalar g = d[k0] - d[k] + ( e[k-1]/( f + CopySign(r,f) ) );
          Scalar c = one;
          Scalar s = one;
          Scalar p = {};

          for( Index i = k0; i <= (k-1); ++i )
          {
            f = s*e[i];
            const Scalar b = c*e[i];
            Aux_PlnRot2( g, f, c, s, r );
            if( i != k0 )
            { e[i-1] = r; }
            g = d[i] - p;
            r = ( d[i+1] - g )*s + 2*c*b;
            p = s*r;
            d[i] = g + p;
            g = c*r - b;
          }

          d[k] -= p;
          e[k-1] = g;
        }
        // while( g >= gend )
      }
      // End of QL block

      Vec_Rescl< Lyt >( scale, anorm, (kend_prev+1)-(k1_prev+1)+1, d+k1_prev, 1 );
    }
    // while( count < maxCount )

    // Converged.
    return true;
  }
};

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif