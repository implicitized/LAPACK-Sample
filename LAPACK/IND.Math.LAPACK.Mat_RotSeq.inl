#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Applies a sequence of plane rotations to a real m by n matrix A,
/// from either the left or the right.
///
/// The c and s arrays contain the m-1 cosine and sine pairs.
///
/// When SIDE = Side::Left, the transformation takes the form
///
///    A := P*A
///
/// and when SIDE = Side::Right, the transformation takes the form
///
///    A := A*(~P)
///
/// where P is an orthogonal matrix consisting of a sequence of z plane
/// rotations, with z = M when SIDE = 'Lyt' and z = N when SIDE = 'R',
/// and (~P) is the transpose of P.
///
/// When DIRECT = Direct::Fwd, then
///
///    P = P(z-1) * ... * P(2) * P(1)
///
/// and when DIRECT = Direct::Bwd, then
///
///    P = P(1) * P(2) * ... * P(z-1)
///
/// where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
///
///    R(k) = [  c(k)  s(k) ]
///         = [ -s(k)  c(k) ].
///
/// When PIVOT = Pivot::Var, the rotation is performed
/// for the plane (k,k+1), i.e., P(k) has the form
///
///    P(k) = [  1                                            ]
///           [       ...                                     ]
///           [              1                                ]
///           [                   c(k)  s(k)                  ]
///           [                  -s(k)  c(k)                  ]
///           [                                1              ]
///           [                                     ...       ]
///           [                                            1  ]
///
/// where R(k) appears as a rank-2 modification to the identity matrix in
/// rows and columns k and k+1.
///
/// When PIVOT = Pivot::Top, the rotation is performed for the
/// plane (1,k+1), so P(k) has the form
///
///    P(k) = [  c(k)                    s(k)                 ]
///           [         1                                     ]
///           [              ...                              ]
///           [                     1                         ]
///           [ -s(k)                    c(k)                 ]
///           [                                 1             ]
///           [                                      ...      ]
///           [                                             1 ]
///
/// where R(k) appears in rows and columns 1 and k+1.
///
/// Similarly, when PIVOT = Pivot::Btm, the rotation is
/// performed for the plane (k,z), giving P(k) the form
///
///    P(k) = [ 1                                             ]
///           [     ...                                       ]
///           [             1                                 ]
///           [                  c(k)                    s(k) ]
///           [                         1                     ]
///           [                              ...              ]
///           [                                     1         ]
///           [                 -s(k)                    c(k) ]
///
/// where R(k) appears in rows and columns k and z.  The rotations are
/// performed without ever forming P(k) explicitly.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlasr</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename c_Ptr_t,
  typename s_Ptr_t,
  typename T_Blk_A >
requires( ! isComplex< Decay<DerefTypeOf<c_Ptr_t>> >
  && areTheSame<
  Decay<DerefTypeOf<c_Ptr_t>>,
  Decay<DerefTypeOf<s_Ptr_t>>,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Mat_RotSeq(
  Side side, Pivot pivot, Direct direct,
  Size m, Size n,
  c_Ptr_t c, s_Ptr_t s,
  T_Blk_A A_, Stride A_ld )
{
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  // Quick return if possible

  if( 0 == m || 0 == n ){ return; }

  if( Side::Left == side )
  {
    // Form  P * A

    switch( pivot )
    {
    default: break;
    case Pivot::Var:
      {
        if( Direct::Fwd == direct )
        {
          for( Index j = 0; j < (Index)(m-1); ++j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)n; ++i )
              {
                const auto aji = A(j+1,i);
                A(j+1,i) = cj*aji - sj*A(j,i);
                A(j,i) = sj*aji + cj*A(j,i);
              }
            }
          }
        }
        else if( Direct::Bwd == direct )
        {
          for( Index j = (Index)(m-1)-1; j >= 0; --j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)n; ++i )
              {
                const auto aji = A(j+1,i);
                A(j+1,i) = cj*aji - sj*A(j,i);
                A(j,i) = sj*aji + cj*A(j,i);
              }
            }
          }
        }
      }
      break;

    case Pivot::Top:
      {
        if( Direct::Fwd == direct )
        {
          for( Index j = 1; j < (Index)m; ++j )
          {
            const auto &cj = c[j-1];
            const auto &sj = s[j-1];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)n; ++i )
              {
                const auto aji = A(j,i);
                A(j,i) = cj*aji - sj*A(0,i);
                A(0,i) = sj*aji + cj*A(0,i);
              }
            }
          }
        }
        else if( Direct::Bwd == direct )
        {
          for( Index j = (Index)(m-1); j >= 1; --j )
          {
            const auto &cj = c[j-1];
            const auto &sj = s[j-1];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)n; ++i )
              {
                const auto aji = A(j,i);
                A(j,i) = cj*aji - sj*A(0,i);
                A(0,i) = sj*aji + cj*A(0,i);
              }
            }
          }
        }
      }
      break;

    case Pivot::Btm:
      {
          if( Direct::Fwd == direct )
          {
            for( Index j = 0; j < (Index)(m-1); ++j )
            {
              const auto &cj = c[j];
              const auto &sj = s[j];
              if( ! IsUnit( cj ) || ! IsUnit( sj ) )
              {
                for( Index i = 0; i < (Index)n; ++i )
                {
                  const auto aji = A(j,i);
                  A(j,i) = sj*A(m-1,i) + cj*aji;
                  A(m-1,i) = cj*A(m-1,i) - sj*aji;
                }
              }
            }
          }
          else if( Direct::Bwd == direct )
          {
            for( Index j = (Index)((m-1)-1); j >= 0; --j )
            {
              const auto &cj = c[j];
              const auto &sj = s[j];
              if( ! IsUnit( cj ) || ! IsUnit( sj ) )
              {
                for( Index i = 0; i < (Index)n; ++i )
                {
                  const auto aji = A(j,i);
                  A(j,i) = sj*A(m-1,i) + cj*aji;
                  A(m-1,i) = cj*A(m-1,i) - sj*aji;
                }
              }
            }
          }
      }
      break;
    }
  }
  else if( Side::Right == side )
  {
    // Form A * (~P)

    switch( pivot )
    {
    default: break;
    case Pivot::Var:
      {
        if( Direct::Fwd == direct )
        {
          for( Index j = 0; j < (Index)(n-1); ++j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j+1);
                A(i,j+1) = cj*aij - sj*A(i,j);
                A(i,j) = sj*aij + cj*A(i,j);
              }
            }
          }
        }
        else if( Direct::Bwd == direct )
        {
          for( Index j = (Index)((n-1)-1); j >= 0; --j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j+1);
                A(i,j+1) = cj*aij - sj*A(i,j);
                A(i,j) = sj*aij + cj*A(i,j);
              }
            }
          }
        }
      }
      break;

    case Pivot::Top:
      {
        if( Direct::Fwd == direct )
        {
          for( Index j = 1; j < (Index)n; ++j )
          {
            const auto &cj = c[j-1];
            const auto &sj = s[j-1];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j);
                A(i,j) = cj*aij - sj*A(i,0);
                A(i,0) = sj*aij + cj*A(i,0);
              }
            }
          }
        }
        else if( Direct::Bwd == direct )
        {
          for( Index j = n-1; j >= 1; --j )
          {
            const auto &cj = c[j-1];
            const auto &sj = s[j-1];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j);
                A(i,j) = cj*aij - sj*A(i,0);
                A(i,0) = sj*aij + cj*A(i,0);
              }
            }
          }
        }
      }
      break;

    case Pivot::Btm:
      {
        if( Direct::Fwd == direct )
        {
          for( Index j = 0; j < (Index)(n-1); ++j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j);
                A(i,j) = sj*A(i,n-1) + cj*aij;
                A(i,n-1) = cj*A(i,n) - sj*aij;
              }
            }
          }
        }
        else if( Direct::Bwd == direct )
        {
          for( Index j = (Index)(n-1)-1; j >= 0; --j )
          {
            const auto &cj = c[j];
            const auto &sj = s[j];
            if( ! IsUnit( cj ) || ! IsUnit( sj ) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              {
                const auto aij = A(i,j);
                A(i,j) = sj*A(i,n-1) + cj*aij;
                A(i,n-1) = cj*A(i,n-1) - sj*aij;
              }
            }
          }
        }
      }
      break;
    }
  }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif