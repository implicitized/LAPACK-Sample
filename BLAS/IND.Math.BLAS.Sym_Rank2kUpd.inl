#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// C := alpha*A*(~B) + alpha*B*(~A) + beta*C
/// or C := alpha*(~A)*B + alpha*(~B)*A + beta*C
/// 
/// For n x k matrices A, B, and n x n symmetric matrix C.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dsyr2k</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Blk_A,
  typename T_Blk_B,
  typename T_Blk_C >
requires( areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Blk_B>>,
  Decay<DerefTypeOf<T_Blk_C>> > )
constexpr void Sym_Rank2kUpd(
  Half half, Trnsp AB_trnsp,
  Size n, Size k,
  const T_Scalar &alpha,
  T_Blk_A A_, Stride A_ld,
  T_Blk_B B_, Stride B_ld,
  const T_Scalar &beta,
  T_Blk_C C_, Stride C_ld )
{
  auto A = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto B = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( B_, i, j, B_ld ); };
  auto C = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( C_, i, j, C_ld ); };

  if( Half::Both == half ){ throw BadArgument{ "Sym_Rank2kUpd", 1 }; }

  if( 0 == n ){ return; }

  if( IsZero( alpha ) )
  {
    if( Half::Upper == half )
    {
      for( Index j = 0; j < n; ++j )
      { for( Index i = 0; i <= j; ++i )
      { C(i,j) *= beta; } }
    }
    else if( Half::Lower == half )
    {
      for( Index j = 0; j < n; ++j )
      { for( Index i = j; j < n; ++i )
      { C(i,j) *= beta; } }
    }
    return;
  }

  if( Trnsp::Yes == AB_trnsp )
  {
    // C := alpha*A*(~B) + alpha*B*(~A) + C.
    if( Half::Upper == half )
    {
      for( Index j = 0; j < n; ++j )
      {
        if( IsZero( beta ) )
        { for( Index i = 0; i <= j; ++i )
        { C(i,j) = {}; } }
        else if( ! IsUnit( beta ) )
        { for( Index i = 0; i <= j; ++j )
        { C(i,j) *= beta; } }
        for( Index h = 0; h <= j; ++h )
        {
          auto u = A(j,h);
          auto v = B(j,h);
          if( ! IsZero( u ) || ! IsZero( v ) )
          {
            u *= alpha;
            v *= alpha;
            for( Index i = 0; i <= j; ++i )
            { C(i,j) += A(i,h)*v + B(i,h)*u; }
          }
        }
      }
    }
    else if( Half::Lower == half )
    {
      for( Index j = 0; j < n; ++j )
      {
        if( IsZero( beta ) )
        { for( Index i = j; i < n; ++i )
        { C(i,j) = {}; } }
        else if( ! IsUnit( beta ) )
        { for( Index i = j; i < n; ++i )
        { C(i,j) *= beta; } }
        for( Index h = 0; h < k; ++h )
        {
          auto u = A(j,h);
          auto v = B(j,h);
          if( ! IsZero( u ) || ! IsZero( v ) )
          {
            u *= alpha;
            v *= alpha;
            for( Index i = j; i < n; ++i )
            { C(i,j) += A(i,h)*v + B(i,h)*u; }
          }
        }
      }
    }
  }
  else if( Trnsp::No == AB_trnsp )
  {
    // C := alpha*(~A)*B + alpha*(~B)*A + C.
    if( Half::Upper == half )
    {
      for( Index j = 0; j < n; ++j )
      {
        for( Index i = 0; i <= j; ++i )
        {
          T_Scalar u{};
          T_Scalar v{};
          for( Index h = 0; h < k; ++h )
          {
            u += A(h,i)*B(h,j);
            v += B(h,i)*A(h,j);
          }
          if( IsZero( beta ) )
          { C(i,j) = alpha*u + alpha*v; }else
          { C(i,j) = beta*C(i,j) + alpha*u + alpha*v; }
        }
      }
    }
    else if( Half::Lower == half )
    {
      for( Index j = 0; j < n; ++j )
      {
        for( Index i = j; i < n; ++i )
        {
          T_Scalar u{};
          T_Scalar v{};
          for( Index h = 0; h < k; ++h )
          {
            u += A(h,i)*B(h,j);
            v += B(h,i)*A(h,j);
          }
          if( IsZero( beta ) )
          { C(i,j) = alpha*u + alpha*v; }else
          { C(i,j) = beta*C(i,j) + alpha*u + alpha*v; }
        }
      }
    }
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif