#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Rfl_MatMul_WorkSize( Size m, Size n ) noexcept
{
  IND_NOT_USED( m );
  return n;
}

/// <summary>
/// Applies a real elementary reflector H to a real m by n matrix
/// C, from either the left or the right. H is represented in the form
///
///       H = I - tau * v * (~v)
///
/// where tau is a real scalar and v is a real vector.
///
/// If tau = 0, then H is taken to be the unit matrix.
/// </summary>
/// <param name="work">
/// Must point to a buffer of at least <paramref name="n"/> elements.
/// </param>
/// <remarks>
/// Based on the LAPACK routine <c>dlarf</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Vec_v,
  typename T_Blk_C,
  typename T_Arr_work >
requires( ! isComplex< T_Scalar >
  && areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_v>>,
  Decay<DerefTypeOf<T_Blk_C>>,
  Decay<DerefTypeOf<T_Arr_work>> > )
constexpr void Rfl_MatMul( Side side,
  Size m, Size n,
  T_Vec_v v, Stride V_cs,
  const T_Scalar &tau,
  T_Blk_C C_, Stride C_ld,
  T_Arr_work work )
{
  // Quick return if possible.
  if(  IsZero( tau ) ){ return; }

  // Set up variables for scanning V.  LASTV begins pointing to the end
  // of V.

  Index i = 0;
  Index lastv = 0;
  Index lastc = 0;

  if( Side::Left == side )
  { lastv = m; }else
  { lastv = n; }
  if( V_cs > 0 )
  { i = (1+(lastv-1)*V_cs)-1; }

  // Look for the last non-zero row in V.
  while( (lastv > 0) && IsZero( v[i] ) )
  { --lastv; i -= V_cs; }

  // Scan for the last non-zero column in C(0:lastv-1,:).
  if( Side::Left == side )
  { lastc = Idx_LastCol< Lyt >( lastv, n, C_, C_ld ) + 1; }

  // Scan for the last non-zero row in C(:,0:lastv-1).
  else
  { lastc = Idx_LastRow< Lyt >( m, lastv, C_, C_ld ) + 1; }

  const T_Scalar one = unit<T_Scalar>;

  // Note that lastc == 0 renders the BLAS operations null; no special
  // case is needed at this level.
  if( Side::Left == side )
  {
    // Form H*C
    if( lastv > 0 )
    {
      Mat_VecMul< Lyt >( Trnsp::Yes, lastv, lastc, one, C_, C_ld, v, V_cs, {}, work, 1 );
      Mat_Rank1Upd< Lyt >( lastv, lastc, -tau, v, V_cs, work, 1, C_, C_ld );
    }
  }
  else
  {
    // Form C*H
    if( lastv > 0 )
    {
      Mat_VecMul< Lyt >( Trnsp::No, lastc, lastv, one, C_, C_ld, v, V_cs, {}, work, 1 );
      Mat_Rank1Upd< Lyt >( lastc, lastv, -tau, work, 1, v, V_cs, C_, C_ld );
    }
  }
}
}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif