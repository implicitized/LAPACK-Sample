#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Solves one of the systems of equations
///
///   A*x = b,   or   (~A)*x = b,
///
/// where b and x are n element vectors and A is an n by n unit, or
/// non-unit, upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// </summary>
template<
  typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Vec_x
>
requires( areTheSame<
  Decay<DerefTypeOf<T_Blk_A>>,
  Decay<DerefTypeOf<T_Vec_x>> > )
constexpr void Tri_Solv_Vec(
  Half half, Trnsp A_trnsp, Diag diag,
  Size n,
  T_Blk_A A_, Stride A_ld,
  T_Vec_x x_, Stride x_s )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_A>>;

  auto A = [&]( Index i, Index j ) noexcept -> const Scalar &
  { return Lyt::MatRef( A_, i, j, A_ld ); };
  auto x = [&]( Index i ) noexcept -> Scalar &
  { return Lyt::VecRef( x_, i, x_s ); };

  if( Half::Both == half ){ throw BadArgument{ "Tri_Solv_Vec", 1u }; }
  if( A_ld < (Stride)Max(1,n) ){ throw BadArgument{ "Tri_Solv_Vec", 6 }; }

  switch( A_trnsp )
  {
  case Trnsp::No:
    {
      // Form x := Inv(A)*x

      if( Half::Upper == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          if( IsZero( x(j) ) ){ continue; }
          if( Diag::NotUnit == diag )
          { x(j) /= A(j,j); }
          const auto xj = x(j);
          for( Index i = j-1; i >= 0; --i )
          { x(i) -= xj*A(i,j); }
        }
      }
      else if( Half::Lower == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          if( IsZero( x(j) ) ){ continue; }
          if( Diag::NotUnit == diag )
          { x(j) /= A(j,j); }
          const auto xj = x(j);
          for( Index i = j+1; i < (Index)n; ++i )
          { x(i) -= xj*A(i,j); }
        }
      }
    }
    break;

  case Trnsp::Yes:
    {
      // Form x := Inv(~A)*x

      if( Half::Upper == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          auto xj = x(j);
          for( Index i = 0; i < j; ++i )
          { xj -= A(i,j)*x(i); }
          if( Diag::NotUnit == diag )
          { xj /= A(j,j); }
          x(j) = Move( xj );
        }
      }
      else if( Half::Lower == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          auto xj = x(j);
          for( Index i = (Index)(n-1); i > j; --i )
          { xj -= A(i,j)*x(i); }
          if( Diag::NotUnit == diag )
          { xj /= A(j,j); }
          x(j) = Move( xj );
        }
      }
    }
    break;

  case Trnsp::Conj:
    {
      // Form x := Inv(Conj(~A))*x

      if( Half::Upper == half )
      {
        for( Index j = 0; j < (Index)n; ++j )
        {
          auto xj = x(j);
          for( Index i = 0; i < j; ++i )
          { xj -= Conj(A(i,j))*x(i); }
          if( Diag::NotUnit == diag )
          { xj /= Conj(A(j,j)); }
          x(j) = Move( xj );
        }
      }
      else if( Half::Lower == half )
      {
        for( Index j = (Index)(n-1); j >= 0; --j )
        {
          auto xj = x(j);
          for( Index i = (Index)(n-1); i > j; --i )
          { xj -= Conj(A(i,j))*x(i); }
          if( Diag::NotUnit == diag )
          { xj /= Conj(A(j,j)); }
          x(j) = Move( xj );
        }
      }
    }
    break;
  }
}

}// namespace BLAS
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.BLAS.h> instead.
#endif