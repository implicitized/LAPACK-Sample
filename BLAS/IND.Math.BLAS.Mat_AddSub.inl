#ifdef __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

/// <summary>
/// B := B + A
/// or B := B + (~A)
/// 
/// B is m by n
/// A is m by n
/// (~A) is n by m
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Blk_B >
requires( requires( T_Blk_A A_, T_Blk_B B_ ){ { A_[0] += B_[0] }; } )
constexpr void Mat_Add( Trnsp A_trnsp,
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

  if constexpr ( isColMajor< Lyt > )
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Add columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_Add< Lyt >( n, A_Col(0,j), A_cs, B_Col(0,j), B_cs ); }
      }
      break;
    case Trnsp::Yes:
      {
        // Add rows to columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_Add< Lyt >( n, A_Row(j,0), A_rs, B_Col(0,j), B_cs ); }
      }
      break;
    case Trnsp::Conj:
      {
        // Add row conjugates to columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_AddConj< Lyt >( n, A_Row(j,0), A_rs, B_Col(0,j), B_cs ); }
      }
      break;
    }
  }
  else
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Add rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_Add< Lyt >( m, A_Row(i,0), A_rs, B_Row(i,0), B_rs ); }
      }
      break;
    case Trnsp::Yes:
      {
        // Add columns to rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_Add< Lyt >( m, A_Col(0,i), A_cs, B_Row(i,0), B_rs ); }
      }
      break;
    case Trnsp::Conj:
      {
        // Add column conjugates to rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_AddConj< Lyt >( m, A_Col(0,i), A_cs, B_Row(i,0), B_rs ); }
      }
      break;
    }
  }
}

/// <summary>
/// B := B - A
/// or B := B - (~A)
/// 
/// B is m by n
/// A is m by n
/// (~A) is n by m
/// </summary>
template< typename Lyt = ColMajor,
  typename T_Blk_A,
  typename T_Blk_B >
requires( requires( T_Blk_A A_, T_Blk_B B_ ){ { A_[0] -= B_[0] }; } )
constexpr void Mat_Sub( Trnsp A_trnsp,
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

  if constexpr ( isColMajor< Lyt > )
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Subtract columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_Sub< Lyt >( n, A_Col(0,j), A_cs, B_Col(0,j), B_cs ); }
      }
      break;
    case Trnsp::Yes:
      {
        // Subtract rows from columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_Sub< Lyt >( n, A_Row(j,0), A_rs, B_Col(0,j), B_cs ); }
      }
      break;
    case Trnsp::Conj:
      {
        // Subtract row conjugates from columns
        for( Index j = 0; j < (Index)m; ++j )
        { Vec_SubConj< Lyt >( n, A_Row(j,0), A_rs, B_Col(0,j), B_cs ); }
      }
      break;
    }
  }
  else
  {
    switch( A_trnsp )
    {
    case Trnsp::No:
      {
        // Subtract rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_Sub< Lyt >( m, A_Row(i,0), A_rs, B_Row(i,0), B_rs ); }
      }
      break;
    case Trnsp::Yes:
      {
        // Subtract columns from rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_Sub< Lyt >( m, A_Col(0,i), A_cs, B_Row(i,0), B_rs ); }
      }
      break;
    case Trnsp::Conj:
      {
        // Subtract column conjugates from rows
        for( Index i = 0; i < (Index)n; ++i )
        { Vec_AddConj< Lyt >( m, A_Col(0,i), A_cs, B_Row(i,0), B_rs ); }
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