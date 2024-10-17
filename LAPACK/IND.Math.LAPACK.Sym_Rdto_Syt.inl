#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Tridiagonal reduction of a real symmetric matrix.
///
/// If half == Half::Upper, the matrix Q is represented as a product of elementary
/// reflectors
/// 
///    Q = H(n-1) . . . H(2) H(1).
/// 
/// Each H(i) has the form
/// 
///    H(i) = I - tau * v * v**T
/// 
/// where tau is a real scalar, and v is a real vector with
/// v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
/// A(1:i-1,i+1), and tau in TAU(i).
/// 
/// If half == Half::Lower, the matrix Q is represented as a product of elementary
/// reflectors
/// 
///    Q = H(1) H(2) . . . H(n-1).
/// 
/// Each H(i) has the form
/// 
///    H(i) = I - tau * v * v**T
/// 
/// where tau is a real scalar, and v is a real vector with
/// v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
/// and tau in TAU(i).
/// 
/// The contents of A on exit are illustrated by the following examples
/// with n = 5:
/// 
/// if Half::Upper:                       if Half::Lower:
/// 
///   [  d   e   v2  v3  v4 ]              [  d                  ]
///   [      d   e   v3  v4 ]              [  e   d              ]
///   [          d   e   v4 ]              [  v1  e   d          ]
///   [              d   e  ]              [  v1  v2  e   d      ]
///   [                  d  ]              [  v1  v2  v3  e   d  ]
/// 
/// where d and e denote diagonal and off-diagonal elements of T, and vi
/// denotes an element of the vector defining H(i).
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dsytd2</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_d,
  typename T_Arr_e,
  typename T_Arr_tau >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_A>> >
    && areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_d>>,
  Decay<DerefTypeOf<T_Arr_e>>,
  Decay<DerefTypeOf<T_Arr_tau>> > )
constexpr void Sym_Rdto_Syt( Half half,
  Size n, T_Blk_A A_, Stride A_ld,
  T_Arr_d d, T_Arr_e e, T_Arr_tau tau )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick return if possible
  if( 0 == n ){ return; }

  const auto overTwo = Inv( 2*unit<Scalar> );

  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  if( Half::Upper == half )
  {
    // Reduce the upper triangle of A

    for( Index i = (Index)( n-2 ); i >= 0; --i )
    {
      // Generate elementary reflector H(i) = I - tau*v*(~v)
      // to annihilate A(0:i-1,i+1).

      const auto v = Lyt::ColPtr( A_, 0, i+1, A_ld );

      Scalar taui;
      Rfl_VecGen< Lyt >( i+1, A(i,i+1), v, A_cs, taui );
      e[i] = A(i,i+1);

      if( ! IsZero( taui ) )
      {
        // Apply H(i) from both sides to A(0:i,0:i)

        A(i,i+1) = unit<Scalar>;

        // Compute  x := tau*A*v  storing x in TAU(0:i)
        Sym_VecMul< Lyt >( Half::Upper, i+1, taui,
          A_, A_ld, v, A_cs, {}, tau, 1 );

        // Compute  w := x - ( (1/2)*tau*dot(x,v) )*v
        const auto alpha = -overTwo*taui*Vec_Dot< Lyt >( i+1, tau, 1, v, A_cs );
        Vec_AXPlusY< Lyt >( i+1, alpha, v, A_cs, tau, 1 );

        // Apply the transformation as a rank-2 update:
        // A := A - v*(~w) - w*(~v)
        Sym_Rank2Upd< Lyt >( Half::Upper, i+1, -unit<Scalar>,
          v, A_cs, tau, 1, A_, A_ld );
        A(i,i+1) = e[i];
      }
      d[i+1] = A(i+1,i+1);
      tau[i] = taui;
    }
    d[0] = A(0,0);
  }
  else if( Half::Lower == half )
  {
    // Reduce the lower triangle of A

    for( Index i = 0; i < (Index)(n-1); ++i )
    {
      // Generate elementary reflector H(i) = I - tau*v*(~v)
      // to annihilate A(i+2:n-1,i)

      const auto v1 = Lyt::BlkPtr( A_, Min(i+2,(Index)(n-1)), i, A_ld );

      Scalar taui;
      Rfl_VecGen< Lyt >( n-(i+1), A(i+1,i), v1, A_cs, taui );
      e[i] = A(i+1,i);

      if( 0.0 != taui )
      {
        // Apply H(i) from both sides to A(i+1:n-1,i+1:n-1)

        // Compute  x := tau*A*v  storing x in TAU(i:n-2)

        const auto v = Lyt::ColPtr( A_, i+1, i, A_ld );
        v[0] = unit<Scalar>;

        const T_Blk_A pA_sub = Lyt::BlkPtr( A_, i+1, i+1, A_ld );
        Sym_VecMul< Lyt >( Half::Lower, n-(i+1), taui,
          pA_sub, A_ld, v, A_cs, {}, tau+i, 1 );

        // Compute  w := x - ((1/2) * tau*dot(x,v)) * v
        const auto alpha = -overTwo*taui*Vec_Dot< Lyt >( n-(i+1), tau+i, 1, v, A_cs );
        Vec_AXPlusY< Lyt >( n-(i+1), alpha, v, A_cs, tau+i, 1 );

        // Apply the transformation as a rank-2 update:
        // A := A - v*(~w) - w*(~v)
        Sym_Rank2Upd< Lyt >( Half::Lower, n-(i+1), -unit<Scalar>,
          v, A_cs, tau+i, 1, pA_sub, A_ld );
        v[0] = e[i];
      }
      d[i] = A(i,i);
      tau[i] = taui;
    }
    d[n-1] = A(n-1,n-1);
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif