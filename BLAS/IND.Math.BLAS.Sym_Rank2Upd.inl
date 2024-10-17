#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Computes:
///
/// A := alpha*x*(~y) + alpha*y*(~x) + A.
///
/// For a symmetric matrix A.
/// </summary>
/// <remarks>
/// Based on the BLAS routine <c>dsyr2</c>
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Scalar,
  typename T_Vec_x,
  typename T_Vec_y,
  typename T_Blk_A >
requires( areTheSame< T_Scalar,
  Decay<DerefTypeOf<T_Vec_x>>,
  Decay<DerefTypeOf<T_Vec_y>>,
  Decay<DerefTypeOf<T_Blk_A>> > )
constexpr void Sym_Rank2Upd( Half half,
  Size n,
  const T_Scalar &alpha,
  T_Vec_x x_, Stride x_s,
  T_Vec_y y_, Stride y_s,
  T_Blk_A A_, Stride A_ld )
{
  auto x = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( x_, i, x_s ); };
  auto y = [&]( auto i ) -> const auto &
  { return Lyt::VecRef( y_, i, y_s ); };
  auto A = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( A_, i, j, A_ld ); };

  if( Half::Both == half ){ throw BadArgument{ "Sym_Rank2Upd", 1 }; }

  // A := 0*x*(~y) + 0*y*(~x) + A
  if( IsZero( alpha ) )
  { return; }

  if( Half::Upper == half )
  {
    for( Index j = 0; j < (Index)n; ++j )
    {
      auto u = y(j);
      auto v = x(j);
      if( ! IsZero( u ) || ! IsZero( v ) )
      {
        u *= alpha;
        v *= alpha;
        for( Index i = 0; i <= j; ++i )
        { A(i,j) += x(i)*u + y(i)*v; }
      }
    }
  }
  else if( Half::Lower == half )
  {
    for( Index j = 0; j < (Index)n; ++j )
    {
      auto u = y(j);
      auto v = x(j);
      if( ! IsZero( u ) || ! IsZero( v ) )
      {
        u *= alpha;
        v *= alpha;
        for( Index i = j; i < (Index)n; ++i )
        { A(i,j) += x(i)*u + y(i)*v; }
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