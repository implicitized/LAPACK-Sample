#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// x = 0
/// </summary>
template< typename Lyt = Flat, typename T_Vec_x >
requires( requires( T_Vec_x x ){ { *x = {} }; } )
constexpr void Vec_Zero( Size n, T_Vec_x x, Stride x_s )
{ while( n-- ){ *x = {}; Lyt::VecInc( x, x_s ); } }

/// <summary>
/// x = alpha*(x/alpha), or just set all elements of x to alpha.
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Scalar = Decay<DerefTypeOf<T_Vec_x>> >
requires( requires( const T_Scalar &alpha, T_Vec_x x ){ { *x = alpha }; } )
constexpr void Vec_Fill( Size n, const T_Scalar &alpha, T_Vec_x x, Stride x_s )
{
  while( n-- )
  {
    *x = alpha;
    Lyt::VecInc( x, x_s ); }
}

/// <summary>
/// y := x
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( areTheSame< Decay<DerefTypeOf<T_Vec_x>>, Decay<DerefTypeOf<T_Vec_y>> > )
constexpr void Vec_Copy( Size n, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  while( n-- )
  {
    *y = *x;
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
}

/// <summary>
/// y := Conj(x)
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x, T_Vec_y y ){ { *y = Conj(*x) }; } )
constexpr void Vec_Conj( Size n, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  while( n-- )
  {
    *y = Conj(*x);
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
}

/// <summary>
/// x &lt;-&gt; y
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( areTheSame< Decay<DerefTypeOf<T_Vec_x>>, Decay<DerefTypeOf<T_Vec_y>> > )
constexpr void Vec_Swap( Size n, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  while( n-- )
  {
    Swap( *x, *y );
    x += x_s;
    y += y_s;
  }
}

template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Arr_piv >
requires( isNativeSignedIntegral< Decay<DerefTypeOf<T_Arr_piv>> > )
constexpr void Vec_PivSwp( T_Vec_x x, Stride x_s, Index k0, Index k1, T_Arr_piv piv_ )
{
  for( Index i = k0; i <= k1; ++i )
  {
    const Index i1 = piv_[i];
    if( i != i1 )
    { Swap( Lyt::VecRef( x, i, x_s ), Lyt::VecRef( x, i1, x_s ) ); }
  }
}

/// <summary>
/// x := alpha*x
/// </summary>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x >
constexpr void Vec_Scale( Size n, const T_Scalar &alpha, T_Vec_x x, Stride x_s )
{
  if( IsZero( alpha ) ){ Vec_Zero< Lyt >( n, x, x_s ); }
  if( ! IsUnit( alpha ) )
  {
    while( n-- )
    {
      (*x) *= alpha;
      Lyt::VecInc( x, x_s );
    }
  }
}

/// <summary>
/// y := alpha*x
/// </summary>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y >
constexpr void Vec_Scale( Size n, const T_Scalar &alpha, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  if( IsZero( alpha ) )
  { Vec_Zero< Lyt >( n, y, y_s ); }

  if( ! IsUnit( alpha ) )
  {
    while( n-- )
    {
      (*y) = alpha*(*x);
      Lyt::VecInc( x, x_s );
      Lyt::VecInc( y, y_s );
    }
  }
}

template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { Conj(*y_)*(*x_) }; } )
constexpr Decay<DerefTypeOf<T_Vec_x>> Vec_Dot( Size n, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  if( 0 == n )
  { return {}; }
  auto sum = Conj(*x)*(*y);
  Lyt::VecInc( x, x_s );
  Lyt::VecInc( y, y_s );
  while( --n )
  {
    sum += Conj(*x)*(*y);
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
  return sum;
}

template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { (*y_)*(*x_) }; } )
constexpr Decay<DerefTypeOf<T_Vec_x>> Vec_DotU( Size n, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  if( 0 == n )
  { return {}; }
  auto sum = (*x)*(*y);
  Lyt::VecInc( x, x_s );
  Lyt::VecInc( y, y_s );
  while( --n )
  {
    sum += (*x)*(*y);
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
  return sum;
}

/// <summary>
/// y := y + x
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { (*y_) += (*x_) }; } )
constexpr void Vec_Add( Size n, T_Vec_x x_, Stride x_s, T_Vec_y y_, Stride y_s )
{
  while( n-- )
  {
    (*y_) += (*x_);
    Lyt::VecInc( x_, x_s );
    Lyt::VecInc( y_, y_s );
  }
}

/// <summary>
/// y := y + Conj(x)
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { (*y_) += Conj(*x_) }; } )
constexpr void Vec_AddConj( Size n, T_Vec_x x_, Stride x_s, T_Vec_y y_, Stride y_s )
{
  while( n-- )
  {
    (*y_) += Conj(*x_);
    Lyt::VecInc( x_, x_s );
    Lyt::VecInc( y_, y_s );
  }
}

/// <summary>
/// y := y - x
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { (*y_) += (*x_) }; } )
constexpr void Vec_Sub( Size n, T_Vec_x x_, Stride x_s, T_Vec_y y_, Stride y_s )
{
  while( n-- )
  {
    (*y_) += (*x_);
    Lyt::VecInc( x_, x_s );
    Lyt::VecInc( y_, y_s );
  }
}

/// <summary>
/// y := y - Conj(x)
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x,
  typename T_Vec_y >
requires( requires( T_Vec_x x_, T_Vec_y y_ ){ { (*y_) += Conj(*x_) }; } )
constexpr void Vec_SubConj( Size n, T_Vec_x x_, Stride x_s, T_Vec_y y_, Stride y_s )
{
  while( n-- )
  {
    (*y_) += Conj(*x_);
    Lyt::VecInc( x_, x_s );
    Lyt::VecInc( y_, y_s );
  }
}

/// <summary>
/// y := alpha*x + y
/// </summary>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y >
constexpr auto Vec_AXPlusY( Size n, const T_Scalar &alpha, T_Vec_x x, Stride x_s, T_Vec_y y, Stride y_s )
{
  while( n-- )
  {
    *y += alpha*(*x);
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
}

/// <summary>
/// y := alpha*Conj(x) + y
/// </summary>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y >
constexpr auto Vec_AConjXPlusY( Size n, const T_Scalar &alpha, T_Vec_x x, const Stride x_s, T_Vec_y y, const Stride y_s )
{
  while( n-- )
  {
    *y += alpha*Conj(*x);
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
}

/// <summary>
/// Numerically stable 2-norm of a vector, where applicable.
/// </summary>
template< typename Lyt = Flat,
  typename T_Vec_x >
constexpr Decay<DerefTypeOf<T_Vec_x>> Vec_Norm2( Size n, T_Vec_x x_, Stride x_s )
{
  using Scalar = Decay<DerefTypeOf<T_Vec_x>>;

  if( 0 == n ){ return {}; }
  if( 1 == n ){ return Abs(*x_); }

  Scalar norm{};

  if constexpr ( isExact< Scalar > )
  {
    while( n-- )
    {
      norm += Sqr( *x_ );
      Lyt::VecInc( x_, x_s );
    }
  }
  else
  {
    Scalar scale{};
    Scalar ssq{ unit<Scalar> };

    while( n-- )
    {
      const auto xi = Abs( *x_ );
      if( ! IsZero( xi ) )
      {
        if( scale >= xi )
        {
          ssq += Sqr( xi/scale );
        }
        else
        {
          ssq = unit<Scalar> + ssq*Sqr( scale/xi );
          scale = xi;
        }
      }
      norm = scale*Sqrt( ssq );
      Lyt::VecInc( x_, x_s );
    }
  }

  return norm;
}

/// <summary>
/// Applies a plane rotation between two vectors x and y
/// given by a cosine (c) and sine (c).
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>drot</c>.
/// </remarks>
template< typename Lyt = Flat,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_x>>,
  Decay<DerefTypeOf<T_Vec_y>> > )
constexpr void Vec_PlnRot(
  Size n,
  T_Vec_x x, Stride x_s,
  T_Vec_y y, Stride y_s,
  const T_Scalar &c, const T_Scalar &s )
{
  while( n-- )
  {
    T_Scalar &x0 = *x;
    T_Scalar &y0 = *y;
    const T_Scalar x1 = c*x0 + s*y0;
    const T_Scalar y1 = c*y0 - s*x0;
    x0 = x1;
    y0 = y1;
    Lyt::VecInc( x, x_s );
    Lyt::VecInc( y, y_s );
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif