#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Forms the triangular factor T of a real block reflector H
/// of order n, which is defined as a product of k elementary reflectors.
///
/// If DIRECT = Fwd, H = H(1) H(2) . . . H(k) and T is upper triangular;
///
/// If DIRECT = Bwd, H = H(k) . . . H(2) H(1) and T is lower triangular.
///
/// If STOREV = ByCol, the vector which defines the elementary reflector
/// H(i) is stored in the i-th column of the array V, and
///
///    H  =  I - V * T * (~V)
///
/// If STOREV = ByRow, the vector which defines the elementary reflector
/// H(i) is stored in the i-th row of the array V, and
///
/// H  =  I - (~V) * T * V
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlarft</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_V,
  typename T_Arr_tau,
  typename T_Blk_T >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_V>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Blk_V>>,
  Decay<DerefTypeOf<T_Arr_tau>>,
  Decay<DerefTypeOf<T_Blk_T>> > )
constexpr void Rfl_BlkGen(
  Direct direct, Store storev,
  Size n, Size k, T_Blk_V V_, Size V_ld,
  T_Arr_tau tau,
  T_Blk_T T_, Size T_ld )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_V>>;

  auto V = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( V_, i, j, V_ld ); };
  auto T = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( T_, i, j, T_ld ); };

  // Quick return if possible
  if( 0 == n ){ return; }

  Index lastv;
  const Stride T_cs = Lyt::ColStride( T_, T_ld );
  const Stride V_cs = Lyt::ColStride( V_, V_ld );

  if( Direct::Fwd == direct )
  {
    Index prevlastv = (Index)(n-1);
    for( Index i = 0; i < (Index)k; ++i )
    {
      prevlastv = Max( i, prevlastv );

      if( IsZero( tau[i] ) )
      {
        // H(i) = I
        for( Index j = 0; j <= i; ++j )
        { T(j,i) = {}; }
      }
      else
      {
        // General case
        if( Store::ByCol == storev )
        {
          // Skip any trailing zeros.
          
          for( lastv = (Index)(n-1); lastv >= i+1; --lastv )
          { if( ! IsZero( V(lastv,i) ) ){ break; } }

          for( Index j = 0; j <= i-1; ++j )
          { T(j,i) = -tau[i]*V(i,j); }

          auto j = Min( lastv, prevlastv );

          // T(0:i-1,i) := - tau(i) * (~V(i:j,0:i-1)) * V(i:j,i)
          const auto V_blk = Lyt::BlkPtr( V_, i+1, 0, V_ld );
          const auto V_col = Lyt::BlkPtr( V_, i+1, i, V_ld );
          const auto T_col = Lyt::BlkPtr( T_, 0, i, T_ld );
          Mat_VecMul< Lyt >( Trnsp::Yes,
            j-i, (i-1)+1, -tau[i], V_blk, V_ld,
            V_col, V_cs, unit<Scalar>, T_col, T_cs );
        }
        else if( Store::ByRow == storev )
        {
          // Skip any trailing zeros.
          for( lastv = (Index)(n-1); lastv >= i+1; --lastv )
          { if( ! IsZero( V(i,lastv) ) ){ break; } }

          for( Index j = 0; j <= i-1; ++j )
          { T(j,i) = -tau[i]*V(j,i); }

          auto j = Min( lastv, prevlastv );

          // T(0:i-1,i) := - tau(i) * V(0:i-1,i:j) * (~V(i,i:j))
          const auto pV_sub = Lyt::BlkPtr( V_, 0, i+1, V_ld );
          const auto V_row = Lyt::BlkPtr( V_, i, i+1, V_ld );
          const auto T_col = Lyt::BlkPtr( T_, 0, i, T_ld );
          Mat_VecMul< Lyt >( Trnsp::No,
            (i-1)+1, j-i, -tau[i], pV_sub, V_ld,
            V_row, V_ld, unit<Scalar>, T_col, T_cs );
        }

        // T(0:i-1,i) := T(0:i-1,1:i-1) * T(0:i-1,i)
        const auto T_col = Lyt::BlkPtr( T_, 0, i, T_ld );
        Tri_VecMul< Lyt >( Half::Upper, Trnsp::No, Diag::NotUnit,
          (i-1)+1, T_, T_ld, T_col, T_cs );

        T(i,i) = tau[i];
        if( i > 0 )
        { prevlastv = Max( prevlastv, lastv ); }else
        { prevlastv = lastv; }
      }
    }
  }
  else if( Direct::Bwd == direct )
  {
    Index prevlastv = 0;
    for( Index i = (Index)(k-1); i >= 0; --i )
    {
      if( IsZero( tau[i] ) )
      {
        // H(i) = I
        for( Index j = i; j < (Index)k; ++j )
        { T(j,i) = 0.0; }
      }
      else
      {
        // General case
        if( i < k )
        {
          if( Store::ByCol == storev )
          {
            // Skip any leading zeros.
            for( lastv = 0; lastv <= i-1; ++lastv )
            { if( ! IsZero( V(lastv,i) ) ){ break; } }

            for( Index j = i+1; j < (Index)k; ++j )
            { T(j,i) = -tau[i]*V((n-k+(i+1))-1,j); }

            auto j = Max( lastv, prevlastv );

            // T(i+1:k-1,i) = -tau(i) * (~V(j:n-k+i-1,i+1:k-1))*V(j:n-k+i-1,i)
            const auto V_blk = Lyt::BlkPtr( V_, j, i+1, V_ld );
            const auto V_col = Lyt::BlkPtr( V_, j, i, V_ld );
            const auto T_col = Lyt::BlkPtr( T_, i+1, i, T_ld );
            Mat_VecMul< Lyt >( Trnsp::Yes,
              (n-k+(i+1)-(j+1))-1, k-(i+1), -tau[i], V_blk, V_ld,
              V_col, V_cs, unit<Scalar>, T_col, T_cs );
          }
          else if( Store::ByRow == storev )
          {
            // Skip any leading zeros.
            for( lastv = 0; lastv <= i-1; ++lastv )
            { if( ! IsZero( V(i,lastv) ) ){ break; } }

            for( Index j = i+1; j < (Index)k; ++j )
            { T(j,i) = -tau[i]*V(j,(n-k+(i+1))-1); }

            auto j = Max( lastv, prevlastv );

            // T(i+1:k-1,i) = -tau(i) * V(i+1:k-1,j:(n-k+(i+1))-1) * (~V(i,j:(n-k+(i+1))-1))
            const auto V_blk = Lyt::BlkPtr( V_, i+1, j, V_ld );
            const auto V_row = Lyt::BlkPtr( V_, i, j, V_ld );
            const auto T_col = Lyt::BlkPtr( T_, i+1, i, T_ld );
            Mat_VecMul< Lyt >( Trnsp::No,
              k-(i+1), (n-k+(i+1)-(j+1))-1, -tau[i], V_blk, V_ld,
              V_row, V_ld, unit<Scalar>, T_col, T_cs );
          }

          // T(i+1:k-1,i) := T(i+1:k-1,i+1:k-1) * T(i+1:k-1,i)
          const auto T_blk = Lyt::BlkPtr( T_, i+1, i+1, T_ld );
          const auto T_col = Lyt::BlkPtr( T_, i+1, i, T_ld );
          Tri_VecMul< Lyt >( Half::Lower, Trnsp::No, Diag::NotUnit,
            k-(i+1), T_blk, T_ld, T_col, T_cs );

          if( i > 1 )
          { prevlastv = Min( prevlastv, lastv ); }else
          { prevlastv = lastv; }
        }
        T(i,i) = tau[i];
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
