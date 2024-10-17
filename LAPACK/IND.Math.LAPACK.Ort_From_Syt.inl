#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_Syt_WorkSize( Size n ) noexcept
{ return n-1; }

/// <summary>
/// Generates a real orthogonal matrix Q which is defined as the
/// product of n-1 elementary reflectors of order N, as returned by
/// <see cref="Sym_Rdto_Sytd"/> <c>(DSYTRD)</c>:
/// 
/// if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
/// 
/// if UPLO = 'Lyt', Q = H(1) H(2) . . . H(n-1).
/// 
/// The work buffer must be at least n*n in size.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dorgtr</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_tau,
  typename T_Arr_work >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_tau>>,
  Decay<DerefTypeOf<T_Arr_work>> > )
constexpr void Ort_From_Syt( Half half,
  Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_tau tau,
  T_Arr_work work )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Scalar zero = {};
  const Scalar one = unit<Scalar>;

  // Quick return if possible

  if( 0 == n ){ return; }

  if( Half::Upper == half )
  {
    // Q was determined by a call to DSYTRD with UPLO = 'Upper'
    //
    // Shift the vectors which define the elementary reflectors one
    // column to the left, and set the last row and column of Q to
    // those of the unit matrix

    for( Index j = 0; j < (Index)(n-1); ++j )
    { Vec_Copy< Lyt >( j, A_Col(0,j+1), A_cs, A_Col(0,j), A_cs ); }
    Vec_Zero< Lyt >( n-1, A_Row(n-1,0), A_rs );
    Vec_Zero< Lyt >( n-1, A_Col(0,n-1), A_cs );
    A(n-1,n-1) = one;

    // Generate Q(0:n-2,0:n-2)
    Ort_From_QL< Lyt >( n-1, n-1, n-1, A_, A_ld, tau, work );
  }
  else if( Half::Lower == half )
  {
    // Q was determined by a call to DSYTRD with UPLO = 'Lower'.
    //
    // Shift the vectors which define the elementary reflectors one
    // column to the right, and set the first row and column of Q to
    // those of the unit matrix

    for( Index j = (Index)(n-1); j >= 1; --j )
    {
      A(0,j) = zero;
      for( Index i = j+1; i < (Index)n; ++i )
      { A(i,j) = A(i,j-1); }
    }
    A(0,0) = one;
    Vec_Zero< Lyt >( n-1, A_Col(1,0), A_cs );
#if 0
    for( Index j = (Index)(n-1); j >= 1; --j )
    {
      for( Index i = j+1; i < (Index)n; ++i )
      { A(i,j) = A(i,j-1); }
    }
    A(0,0) = one;
    Vec_Zero< Lyt >( n-1, A_Row(0,1), A_rs );
    Vec_Zero< Lyt >( n-1, A_Col(1,0), A_cs );
#endif

    // Generate Q(0:n-2,0:n-2)
    Ort_From_QR< Lyt >( n-1, n-1, n-1, A_Blk(1,1), A_ld, tau, work );
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif