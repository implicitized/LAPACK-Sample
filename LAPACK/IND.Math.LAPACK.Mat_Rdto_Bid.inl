#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Mat_Rdto_Bid_WorkSize( Size m, Size n ) noexcept
{ return Rfl_MatMul_WorkSize( m, n ); }

/// <summary>
/// Reduces a general real M-by-N matrix A to upper or lower
/// bidiagonal form B by an orthogonal transformation: (~Q)*A*P = B.
///
/// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
///
/// The matrices Q and P are represented as products of elementary
/// reflectors:
///
/// If m >= n,
///
///    Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
///
/// Each H(i) and G(i) has the form:
///
///    H(i) = I - q_tau*v*(~v)  and G(i) = I - p_tau*u*(~u)
///
/// where q_tau and p_tau are real scalars, and v and u are real vectors;
/// v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
/// u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
/// q_tau is stored in Q_tau(i) and p_tau in P_tau(i).
///
/// If m < n,
///
///    Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
///
/// Each H(i) and G(i) has the form:
///
///    H(i) = I - q_tau*v*(~v)  and G(i) = I - p_tau*u*(~u)
///
/// where q_tau and p_tau are real scalars, and v and u are real vectors;
/// v(0:i) = 0, v(i+1) = 1, and v(i+2:m-1) is stored on exit in A(i+2:m,i);
/// u(0:i-1) = 0, u(i) = 1, and u(i+1:n-1) is stored on exit in A(i,i+1:n);
/// q_tau is stored in Q_tau(i) and p_tau in P_tau(i).
///
/// The contents of A on exit are illustrated by the following examples:
///
/// m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
///
///   [  d   e   u0  u0  u0 ]           [  d   u0  u0  u0  u0  u0 ]
///   [  v0  d   e   u1  u1 ]           [  e   d   u1  u1  u1  u1 ]
///   [  v0  v1  d   e   u2 ]           [  v0  e   d   u2  u2  u2 ]
///   [  v0  v1  v2  d   e  ]           [  v0  v1  e   d   u3  u3 ]
///   [  v0  v1  v2  v3  d  ]           [  v0  v1  v2  e   d   u4 ]
///   [  v0  v1  v2  v3  v4 ]
///
/// where d and e denote diagonal and off-diagonal elements of B, vi
/// denotes an element of the vector defining H(i), and ui an element of
/// the vector defining G(i).
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dgebd2</c>.
/// </remarks>
/// 
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Arr_d,
  typename T_Arr_e,
  typename T_Arr_Q_tau,
  typename T_Arr_P_tau,
  typename T_Arr_work >
requires( areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Arr_d>>,
  Decay<DerefTypeOf<T_Arr_e>>,
  Decay<DerefTypeOf<T_Arr_Q_tau>>,
  Decay<DerefTypeOf<T_Arr_P_tau>>,
  Decay<DerefTypeOf<T_Arr_work>> > )
constexpr void Mat_Rdto_Bid(
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Arr_d d, T_Arr_e e,
  T_Arr_Q_tau Q_tau,
  T_Arr_P_tau P_tau,
  T_Arr_work work )
{
  using Scalar = Decay< DerefTypeOf< T_Blk_A > >;

  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto A_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( A_, i, j, A_ld ); };
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  const Stride A_cs = Lyt::ColStride( A_, A_ld );
  const Stride A_rs = Lyt::RowStride( A_, A_ld );

  const Scalar one = unit<Scalar>;
  const Scalar zero = {};

  if( m >= n )
  {
    // Reduce to upper bidiagonal form

    for( Index i = 0; i < (Index)n; ++i )
    {
      // Generate elementary reflector H(i) to annihilate A(i+1:m-1,i)

      Rfl_VecGen< Lyt >( m-i, A(i,i),
        A_Col( Min(i+1,(Index)m-1 ), i ), A_cs,
        Q_tau[i] );
      d[i] = A(i,i);

      // Apply H(i) to A(i:m-1,i+1:n-1) from the left

      if( i < (Index)(n-1) )
      {
        A(i,i) = one;
        Rfl_MatMul< Lyt >( Side::Left, m-i, n-(i+1),
          A_Col( i,   i ), A_cs, Q_tau[i],
          A_Blk( i, i+1 ), A_ld, work );
        A(i,i) = d[i];

        // Generate elementary reflector G(i) to annihilate A(i,i+2:n-1)
        Rfl_VecGen< Lyt >( n-(i+1), A(i,i+1),
          A_Row( i, Min(i+2,(Index)n-1) ), A_rs,
          P_tau[i] );
        e[i] = A(i,i+1);

        // Apply G(i) to A(i+1:m-1,i+1:n-1) from the right
        A(i,i+1) = one;
        Rfl_MatMul< Lyt >( Side::Right, m-(i+1), n-(i+1),
          A_Row( i,   i+1 ), A_rs, P_tau[i],
          A_Blk( i+1, i+1 ), A_ld, work );
        A(i,i+1) = e[i];
      }
      else
      {
        P_tau[i] = zero;
      }
    }
  }
  else
  {
    // Reduce to lower bidiagonal form

    for( Index i = 0; i < (Index)m; ++i )
    {
      // Generate elementary reflector G(i) to annihilate A(i,i+1:n-1)

      Rfl_VecGen< Lyt >( n-i, A(i,i),
        A_Row( i, Min(i+1,(Index)n-1 ) ), A_rs,
        P_tau[i] );
      d[i] = A(i,i);

      // Apply G(i) to A(i+1:m-1,i:n-1) from the right

      if( i < (Index)(m-1) )
      {
        A(i,i) = one;
        Rfl_MatMul< Lyt >( Side::Right, m-(i+1), n-i,
          A_Row( i,   i ), A_rs, P_tau[i],
          A_Blk( i+1, i ), A_ld, work );
        A(i,i) = d[i];

        // Generate elementary reflector H(i) to annihilate A(i+2:m-1,i)

        Rfl_VecGen< Lyt >( m-(i+1), A(i+1,i),
          A_Col( Min(i+2,(Index)m-1), i ), A_cs,
          Q_tau[i] );
        e[i] = A(i+1,i);

        // Apply H(i) to A(i+1:m-1,i+1:n-1) from the left
        A(i+1,i) = one;
        Rfl_MatMul< Lyt >( Side::Left, m-(i+1), n-(i+1),
          A_Col( i+1,   i ), A_cs, Q_tau[i],
          A_Blk( i+1, i+1 ), A_ld, work );
        A(i+1,i) = e[i];
      }
      else
      {
        Q_tau[i] = zero;
      }
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif