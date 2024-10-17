#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

inline constexpr Size Ort_From_Bid_WorkSize( Vect vect, Size m, Size n, Size k ) noexcept
{
  if( Vect::Q == vect )
  { return Ort_From_QR_WorkSize( m, n, k ); }
  else if( Vect::Pt == vect )
  { return Ort_From_LQ_WorkSize( m, n, k ); }

  return 0;
}

/// <summary>
/// 
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dorgbr</c> with the block logic removed.
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
constexpr void Ort_From_Bid( Vect vect,
  Size m, Size n, Size k,
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

   const Scalar zero = {};
   const Scalar one = unit<Scalar>;

   const Stride A_rs = Lyt::RowStride( A_, A_ld );
   const Stride A_cs = Lyt::ColStride( A_, A_ld );

   switch( vect )
   {
   default:
     {
       throw BadArgument{ "Ort_From_Bid", 1 };
     }
     break;

   case Vect::Q:
     {
       if( m >= k )
       {
         Ort_From_QR< Lyt >( m, n, k, A_, A_ld, tau, work );
       }
       else
       {
         // If m < k, assume m = n
         //
         // Shift the vectors which define the elementary reflectors one
         // column to the right, and set the first row and column of Q
         // to those of the unit matrix.

         A(0,0) = one;
         if( m > 1 )
         {
           for( Index j = (Index)(m-1); j >= 1; --j )
           { Vec_Copy< Lyt >( m-(j+1), A_Col(j+1,j-1), A_cs, A_Col(j+1,j), A_cs ); }
           Vec_Zero< Lyt >( m-1, A_Row(0,1), A_rs );
           Vec_Zero< Lyt >( m-1, A_Col(1,0), A_cs );
           Ort_From_QR< Lyt >( m-1, m-1, m-1, A_Blk(1,1), A_ld, tau, work );
         }
       }
     }
     break;

   case Vect::Pt:
     {
       if( k < n )
       {
         Ort_From_LQ< Lyt >( m, n, k, A_, A_ld, tau, work );
       }
       else
       {
         // If k >= n, assume m = n
         //
         // Shift the vectors which define the elementary reflectors one
         // row downward, and set the first row and column of P**T to
         // those of the unit matrix.

         A(0,0) = one;
         if( n > 1 )
         {
           for( Index i = (Index)(n-1); i >= 1; --i )
           { Vec_Copy< Lyt >( n-(i+1), A_Row(i-1,i+1), A_rs, A_Row(i,i+1), A_rs ); }
           Vec_Zero< Lyt >( n-1, A_Row(0,1), A_rs );
           Vec_Zero< Lyt >( n-1, A_Col(1,0), A_cs );
           Ort_From_LQ< Lyt >( n-1, n-1, n-1, A_Blk(1,1), A_ld, tau, work );
         }
       }
     }
     break;
   }
}

}// namespace LAPACK
}// namespace Math
}// namespace IND

#else
#error This file must not be included directly. #include <IND.Math.LAPACK.h> instead.
#endif