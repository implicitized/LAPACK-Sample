#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// Matrix copy with optional conjugate transpose.
/// 
/// B := A
/// B := (~A)
/// B := Conj(~A)
/// 
/// The A and B buffers must be distinct.
/// 
/// B is m by n.
/// A is m by n, and (~A) is n by m.
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlacpy</c> with extended functionality.
/// This has been moved to into the BLAS layer, as it is fundamental.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Blk_B >
requires( requires( const T_Blk_A A_, const T_Blk_B B_ )
{ { B_[0] = A_[0] }; } )
constexpr void Mat_Copy(
  Half half, Trnsp A_trnsp,
  Size m, Size n,
  T_Blk_A A_, Stride A_ld,
  T_Blk_B B_, Stride B_ld )
{
  auto A_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( A_, i, j, A_ld ); };
  auto A_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( A_, i, j, A_ld ); };

  auto B_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( B_, i, j, B_ld ); };
  auto B_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( B_, i, j, B_ld ); };

  const Stride A_rs = Lyt::RowStride( A_, A_ld );
  const Stride A_cs = Lyt::ColStride( A_, A_ld );

  const Stride B_rs = Lyt::RowStride( B_, B_ld );
  const Stride B_cs = Lyt::ColStride( B_, B_ld );

  switch( half )
  {
  case Half::Upper:
    {
      switch( A_trnsp )
      {
      case Trnsp::No:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( (Size)(m-j),
                A_Col( 0, j ), A_cs,
                B_Col( 0, j ), B_cs );
            }
          }
          else
          {
            // Copy rows
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( (Size)(n-i),
                A_Row( i, i ), A_rs,
                B_Row( i, i ), B_rs );
            }
          }
        }
        break;

      case Trnsp::Yes:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( (Size)(m-j),
                A_Col( 0, j ), A_cs,
                B_Row( j, j ), B_rs );
            }
          }
          else
          {
            // Copy rows into columns
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( (Size)(n-i),
                A_Row( i, i ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;

      case Trnsp::Conj:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows with conjugation
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Conj< Lyt >( (Size)(m-j),
                A_Col( 0, j ), A_cs,
                B_Row( j, j ), B_rs );
            }
          }
          else
          {
            // Copy rows into columns with conjugation
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Conj< Lyt >( (Size)(n-i),
                A_Row( i, i ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;
      }
    }
    break;

  case Half::Lower:
    {
      switch( A_trnsp )
      {
      case Trnsp::No:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( (Size)(m-j),
                A_Col( j, j ), A_cs,
                B_Col( j, j ), B_cs );
            }
          }
          else
          {
            // Copy rows
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( (Size)(n-i),
                A_Row( i, 0 ), A_rs,
                B_Row( i, 0 ), B_rs );
            }
          }
        }
        break;

      case Trnsp::Yes:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( (Size)(m-j),
                A_Col( 0, j ), A_cs,
                B_Row( j, 0 ), B_rs );
            }
          }
          else
          {
            // Copy rows into columns
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( (Size)(n-i),
                A_Row( i, 0 ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;

      case Trnsp::Conj:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows with conjugation
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Conj< Lyt >( (Size)(m-j),
                A_Col( 0, j ), A_cs,
                B_Row( j, 0 ), B_rs );
            }
          }
          else
          {
            // Copy rows into columns with conjugation
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Conj< Lyt >( (Size)(n-i),
                A_Row( i, 0 ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;
      }
    }
    break;

  case Half::Both:
    {
      switch( A_trnsp )
      {
      case Trnsp::No:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( m,
                A_Col( 0, j ), A_cs,
                B_Col( 0, j ), B_cs );
            }
          }
          else
          {
            // Copy rows
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( n,
                A_Row( i, 0 ), A_rs,
                B_Row( i, 0 ), B_rs );
            }
          }
        }
        break;

      case Trnsp::Yes:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Copy< Lyt >( m,
                A_Col( 0, j ), A_cs,
                B_Row( j, 0 ), B_rs );
            }
          }
          else
          {
            // Copy rows into columns
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Copy< Lyt >( n,
                A_Row( i, 0 ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;

      case Trnsp::Conj:
        {
          if constexpr ( isColMajor< Lyt > )
          {
            // Copy columns into rows with conjugation
            for( Index j = 0; j < (Index)n; ++j )
            {
              Vec_Conj< Lyt >( m,
                A_Col( 0, j ), A_cs,
                B_Row( j, 0 ), B_cs );
            }
          }
          else
          {
            // Copy rows into columns with conjugation
            for( Index i = 0; i < (Index)m; ++i )
            {
              Vec_Conj< Lyt >( n,
                A_Row( i, 0 ), A_rs,
                B_Col( 0, i ), B_cs );
            }
          }
        }
        break;
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