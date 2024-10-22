/******************************************************************************
 * COPYRIGHT (C) 2024 Christopher Gary
 * All Rights Reserved.
 *
 * This code is provided for the sole purpose of evaluating the technical
 * skills of the author/applicant in the context of a job application. 
 * Redistribution, modification, use, or incorporation of this code in any 
 * product or project, in part or in whole, is strictly prohibited without 
 * prior written consent from the author.
 *
 * THIS CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL 
 * THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING 
 * FROM, OUT OF, OR IN CONNECTION WITH THE CODE OR THE USE OR OTHER DEALINGS 
 * IN THE CODE.
 *
 * By using this code, you agree to comply with the terms of this license.
 *****************************************************************************/

#pragma once

#include <string>
#include <type_traits>
#include <cmath>
#include <memory>
#include <numeric>

namespace IND {

// Misc utilities imported from the parent library.

// Doppelgangers here are defensive coding against
// inconsistent standard library implementations.

#define IND_NOT_USED( x ) (void)(x)

// Clang and VS tend to disagree on "implicit deletion",
// so this just defines them all.
#define IND_NOTHROW_VITAE( Type )\
  constexpr Type() noexcept = default;\
  constexpr Type( const Type & ) noexcept = default;\
  constexpr Type( Type && ) noexcept = default;\
  constexpr Type &operator = ( const Type & ) noexcept = default;\
  constexpr Type &operator = ( Type && ) noexcept = default;\
  constexpr ~Type() noexcept = default

using UIntSize = std::size_t;
using IntSize = std::make_signed_t< std::size_t >;

using Float32 = float;
using Float64 = double;

template< typename T_Type >
using Decay = std::decay_t< T_Type >;

template< typename T_PointerOrIterator >
using DerefTypeOf = decltype( * std::declval< const T_PointerOrIterator & >() );

template< typename T_Value >
constexpr T_Value Clamp( const T_Value &value, const T_Value &lo, const T_Value &hi )
{ return (value > hi)? hi : (value < lo)? lo : value; }

template< typename T_Left, typename T_Right >
constexpr auto Min( const T_Left &a, const T_Right &b )
{ return (a < b) ? a : b; }
template< typename T_Value, typename ...T_Values >
constexpr auto Min( const T_Value &a, const T_Values &...b )
{
  const auto &c = Min( b... );
  return (a < c) ? a : c;
}

template< typename T_Left, typename T_Right >
constexpr auto Max( const T_Left &a, const T_Right &b )
{ return (a > b) ? a : b; }
template< typename T_Value, typename ...T_Values >
constexpr auto Max( const T_Value &a, const T_Values &...b )
{
  const auto &c = Max( b... );
  return (a > c) ? a : c;
}

template< typename T_Arg >
inline constexpr T_Arg &&Forward( std::remove_reference_t<T_Arg> &arg ) noexcept
{ return ( static_cast<T_Arg &&>(arg) ); }

template< typename T_Arg >
inline constexpr T_Arg &&Forward( std::remove_reference_t<T_Arg> &&arg ) noexcept
{ return ( static_cast<T_Arg &&>(arg) ); }

template< typename T_Arg >
inline constexpr std::remove_reference_t<T_Arg> &&Move( T_Arg &&arg ) noexcept
{ return ( static_cast<std::remove_reference_t<T_Arg> &&>(arg) ); }

template< typename T_First, typename T_Second >
  inline constexpr void Swap( T_First &a, T_Second &b ) noexcept
  { std::swap( a, b ); }

// This actually turned out to be more vague than helpful...
template< typename T_Value >
inline constexpr bool isExact = false;

// I'm really not a fan of having to specialize
// a complete numeric_limits for everything supporting
// even just a subset of the related concepts.
template< typename T_Value >
inline constexpr auto minValue = std::numeric_limits< T_Value >::min();

template< typename T_Value >
inline constexpr auto maxValue = std::numeric_limits< T_Value >::max();

template< typename T_Type >
inline constexpr bool isSigned = std::is_signed< T_Type >::value;
template< typename T_Type >
inline constexpr bool isUnsigned = std::is_unsigned< T_Type >::value;

/// <summary>
/// True for types that DO NOT have +/- infinity
/// or unbounded representations.
/// </summary>
template< typename T_Type >
inline constexpr bool isFinite = std::is_integral<T_Type>::value;

// False for bool - bool is semantically not an integer.
template< typename T_Type >
inline constexpr bool isIntegral = std::is_integral<T_Type>::value && ( ! std::is_same<T_Type, bool>::value );
template< typename T_Type >
inline constexpr bool isSignedIntegral = isIntegral<T_Type> && isSigned<T_Type>;
template< typename T_Type >
inline constexpr bool isUnsignedIntegral = isIntegral<T_Type> && isUnsigned<T_Type>;

// False for bool - bool is semantically not an integer.
template< typename T_Type >
inline constexpr bool isNativeIntegral = std::is_integral<T_Type>::value && ( ! std::is_same<T_Type, bool>::value );
template< typename T_Type >
inline constexpr bool isNativeSignedIntegral = isNativeIntegral<T_Type> && isSigned<T_Type>;
template< typename T_Type >
inline constexpr bool isNativeUnsignedIntegral = isNativeIntegral<T_Type> && isUnsigned<T_Type>;

template< typename T_Type >
inline constexpr bool isFiniteIntegral = isFinite<T_Type> && isIntegral<T_Type>;
template< typename T_Type >
inline constexpr bool isFiniteSignedIntegral = isFiniteIntegral<T_Type> && isSigned<T_Type>;
template< typename T_Type >
inline constexpr bool isFiniteUnsignedIntegral = isFiniteIntegral<T_Type> && isUnsigned<T_Type>;

namespace _n_Impl {

  template< typename T_First, typename T_Second, typename ...T_Others >
  struct _s_AreTheSame
  {
    static constexpr bool value = _s_AreTheSame< T_First, T_Second >::value
                               && _s_AreTheSame< T_Second, T_Others... >::value;
  };

  template< typename T_First, typename T_Second >
  struct _s_AreTheSame< T_First, T_Second > : std::false_type{};

  template< typename T_Type >
  struct _s_AreTheSame< T_Type, T_Type > : std::true_type{};

}// namespace _n_Impl

template< typename T_First, typename T_Second, typename ...T_Others >
inline constexpr bool areTheSame = _n_Impl::_s_AreTheSame< T_First, T_Second, T_Others... >::value;

namespace Math {

  template< typename T_Value >
  inline constexpr auto undefined = std::numeric_limits< T_Value >::quiet_NaN();

  template< typename T_Value >
  inline constexpr auto infinity = std::numeric_limits< T_Value >::infinity();

  // This looks silly at first, but it
  // helps to deal with cases where default
  // construction of a wrapper type might
  // not actually zero-default-initialize,
  // then also handle (through specialization)
  // int == 0 in specialized constructors for
  // things that have no business accepting an
  // int, or use it as an argument type for
  // other purposes.
  template< typename T_Value >
  inline constexpr T_Value zero = T_Value( 0 );

  // This is also used (in the parent codebase)
  // where the value type may consist of low-dimension
  // matrices.
  template< typename T_Value >
  inline constexpr T_Value unit = T_Value( 1 );

  // Scalar tests for zero-equivalence, if compiled
  // against strict IEEE conformance, cannot compile
  // to simple integer comparisons.
  // 
  // This is meant to side-step that issue explicitly,
  // where performance can be improved while reliably
  // handling NaNs in some other part of the algorithm.
  //
  // Moreover, this also avoids having to construct
  // a "zero" element of some arbitrary base field
  // (or providing overrides for all base fields for
  // int, though only for the specific case of "is/not 0").
  //
  // There were many dependencies on an IEEE-754 utility
  // library, which includes support for expansions,
  // that would really just clutter this small demo.
  //
  // So, this does the obvious...
  template< typename T_Value >
  inline constexpr bool IsZero( const T_Value &x ) noexcept
  { return T_Value(0) == x; }

  // Similar to IsZero. This is faster
  // for things like a rational than
  // constructing a rational and comparing
  // generically.
  template< typename T_Value >
  inline constexpr bool IsUnit( const T_Value &x ) noexcept
  { return unit< T_Value > == x; }

  template< typename T_Value >
  inline constexpr int IntSignOrZero( const T_Value &x ) noexcept
  {
    if( x > 0 ){ return  1; }
    if( x < 0 ){ return -1; }
    return 0;
  }

  template< typename T_Value >
  inline constexpr int IntSignOrZero( const T_Value &x, const T_Value &tolerance ) noexcept
  {
    if( x >  tolerance ){ return  1; }
    if( x < -tolerance ){ return -1; }
    return 0;
  }

  inline bool IsUndefined( Float32 x ) noexcept
  { return std::isnan( x ); }
  inline bool IsUndefined( Float64 x ) noexcept
  { return std::isnan( x ); }

  // The version of Abs used in the parent library
  // is based on a branch-free NaN check and bit masking.
  inline Float32 Abs( Float32 x ) noexcept
  { return std::abs( x ); }
  inline Float32 Sqr( Float32 x ) noexcept
  { return x*x; }
  inline Float32 Sqrt( Float32 x ) noexcept
  { return std::sqrt( x ); }
  inline Float32 Hypot( Float32 x, Float32 y ) noexcept
  { return std::hypot( x, y ); }
  inline Float32 CopySign( Float32 to, Float32 from ) noexcept
  { return std::copysign( to, from ); }

  inline Float64 Abs( Float64 x ) noexcept
  { return std::abs( x ); }
  inline Float64 Sqr( Float64 x ) noexcept
  { return x*x; }
  inline Float64 Sqrt( Float64 x ) noexcept
  { return std::sqrt( x ); }
  inline Float64 Hypot( Float64 x, Float64 y ) noexcept
  { return std::hypot( x, y ); }
  inline Float64 CopySign( Float64 to, Float64 from ) noexcept
  { return std::copysign( to, from ); }

  template< typename T_Value >
  inline constexpr T_Value Inv( const T_Value &x ) noexcept
  { return T_Value(1)/x; }

  template< typename T_Value >
  constexpr bool IsWithinBound( const T_Value &x, const T_Value &tolerance ) noexcept
  { return Abs(x) <= tolerance; }

  // Complex not making an appearance here.
  template< typename T_Value >
  inline constexpr bool isComplex = false;

  template< typename T_Value >
  inline constexpr T_Value Conj( const T_Value &x ) noexcept
  { return x; }

}// namespace Math
}// namespace IND


