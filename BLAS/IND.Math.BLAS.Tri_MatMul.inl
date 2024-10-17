#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
/// 
/// B := alpha*A*B
/// or B := alpha*(~A)*B
/// or B := alpha*B*A
/// or B := alpha*B*(~A)
///
/// B is m by n, A is a unit or non-unit upper or lower triangular matrix.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dtrmm</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A,
  typename T_Blk_B >
requires( areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Blk_B>> > )
constexpr void Tri_MatMul(
  Side side, Half half, Trnsp A_trnsp, Diag diag,
  Size m, Size n,
  const T_Scalar &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Blk_B B_, Stride B_ld )
{
  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto B = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( B_, i, j, B_ld ); };

  // Test the input parameters.

  const auto A_nrow = ( Side::Left == side )? m : n;

  if( Half::Both == half ){ throw BadArgument{ "Tri_MatMul", 2 }; }
  if( A_ld < Max(1,A_nrow) ){ throw BadArgument{ "Tri_MatMul", 9 }; }
  if( B_ld < Max(1,m) ){ throw BadArgument{ "Tri_MatMul", 11 }; }

  // Quick return if possible.

  if( (0 == m) || (0 == n) ){ return; }

  if( Side::Left == side )
  {

    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // B := alpha*A*B.

        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index k = 0; k < (Index)m; ++k )
            {
              if( ! IsZero(B(k,j)) )
              {
                auto u = alpha*B(k,j);
                for( Index i = 0; i <= k-1; ++i )
                { B(i,j) += u*A(i,k); }
                if( Diag::NotUnit == diag )
                { u *= A(k,k); }
                B(k,j) = u;
              }
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index k = (Index)(m-1); k >= 0; --k )
            {
              if( ! IsZero(B(k,j)) )
              {
                auto u = alpha*B(k,j);
                B(k,j) = u;
                if( Diag::NotUnit == diag )
                { B(k,j) *= A(k,k); }
                for( Index i = k+1; i < (Index)m; ++i )
                { B(i,j) += u*A(i,k); }
              }
            }
          }
        }
      }
      break;

    case Trnsp::Yes:
      {
        // B := alpha*(~A)*B.

        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = (Index)(m-1); i >= 0; --i )
            {
              auto u = B(i,j);
              if( Diag::NotUnit == diag )
              { u *= A(i,i); }
              for( Index k = 0; k <= i-1; ++k )
              { u += A(k,i)*B(k,j); }
              B(i,j) = alpha*u;
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = 0; i < (Index)m; ++i )
            {
              auto u = B(i,j);
              if( Diag::NotUnit == diag )
              { u *= A(i,i); }
              for( Index k = (Index)(i+1); k < (Index)m; ++k )
              { u += A(k,i)*B(k,j); }
              B(i,j) = alpha*u;
            }
          }
        }
      }
      break;

    case Trnsp::Conj:
      {
        // B := alpha*Conj(~A)*B.

        if( Half::Upper == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = (Index)(m-1); i >= 0; --i )
            {
              auto u = B(i,j);
              if( Diag::NotUnit == diag )
              { u *= Conj(A(i,i)); }
              for( Index k = 0; k <= i-1; ++k )
              { u += Conj(A(k,i))*B(k,j); }
              B(i,j) = alpha*u;
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            for( Index i = 0; i < (Index)m; ++i )
            {
              auto u = B(i,j);
              if( Diag::NotUnit == diag )
              { u *= Conj(A(i,i)); }
              for( Index k = (Index)(i+1); k < (Index)m; ++k )
              { u += Conj(A(k,i))*B(k,j); }
              B(i,j) = alpha*u;
            }
          }
        }
      }
      break;

    }
  }
  else if( Side::Right == side )
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // B := alpha*B*A.

        if( Half::Upper == half )
        {
          for( Index j = (Index)(n-1); j >= 0; --j )
          {
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= A(j,j); }
            for( Index i = 0; i < (Index)m; ++i )
            { B(i,j) = u*B(i,j); }
            for( Index k = 0; k <= j-1; ++k )
            {
              if( ! IsZero(A(k,j)) )
              {
                u = alpha*A(k,j);
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index j = 0; j < (Index)n; ++j )
          {
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= A(j,j); }
            for( Index i = 0; i < (Index)m; ++i )
            { B(i,j) = u*B(i,j); }
            for( Index k = j+1; k < (Index)n; ++k )
            {
              if( ! IsZero(A(k,j)) )
              {
                u = alpha*A(k,j);
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
          }
        }
      }
      break;

    case Trnsp::Yes:
      {
        // B := alpha*B*(~A).

        if( Half::Upper == half )
        {
          for( Index k = 0; k < (Index)n; ++k )
          {
            for( Index j = 0; j <= k-1; ++j )
            {
              if( ! IsZero(A(j,k)) )
              {
                auto u = alpha*A(j,k);
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= A(k,k); }
            if( ! IsUnit(u) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              { B(i,k) = u*B(i,k); }
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index k = (Index)(n-1); k >= 0; --k )
          {
            for( Index j = k+1; j < (Index)n; ++j )
            {
              if( ! IsZero(A(j,k)) )
              {
                auto u = alpha*A(j,k);
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= A(k,k); }
            if( ! IsUnit(u) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              { B(i,k) = u*B(i,k); }
            }
          }
        }
      }
      break;

    case Trnsp::Conj:
      {
        // B := alpha*B*Conj(~A).

        if( Half::Upper == half )
        {
          for( Index k = 0; k < (Index)n; ++k )
          {
            for( Index j = 0; j <= k-1; ++j )
            {
              if( ! IsZero(A(j,k)) )
              {
                auto u = alpha*Conj(A(j,k));
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= Conj(A(k,k)); }
            if( ! IsUnit(u) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              { B(i,k) = u*B(i,k); }
            }
          }
        }
        else if( Half::Lower == half )
        {
          for( Index k = (Index)(n-1); k >= 0; --k )
          {
            for( Index j = k+1; j < (Index)n; ++j )
            {
              if( ! IsZero(A(j,k)) )
              {
                auto u = alpha*Conj(A(j,k));
                for( Index i = 0; i < (Index)m; ++i )
                { B(i,j) += u*B(i,k); }
              }
            }
            auto u = alpha;
            if( Diag::NotUnit == diag )
            { u *= Conj(A(k,k)); }
            if( ! IsUnit(u) )
            {
              for( Index i = 0; i < (Index)m; ++i )
              { B(i,k) = u*B(i,k); }
            }
          }
        }
      }
      break;
    }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif