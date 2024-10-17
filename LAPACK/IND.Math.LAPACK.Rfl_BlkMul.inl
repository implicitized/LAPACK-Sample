#ifdef __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

/// <summary>
/// Applies a real block reflector H or its transpose (~H) to a
/// real m by n matrix C, from either the left or the right.
/// 
///
/// The shape of the matrix V and the storage of the vectors which define
/// the H(i) is best illustrated by the following example with n = 5 and
/// k = 3. The elements equal to 1 are not stored; the corresponding
/// array elements are modified but restored on exit. The rest of the
/// array is not used.
///
/// DIRECT = Fwd and STOREV = ByCol:         DIRECT = Fwd and STOREV = ByRow:
///
///             V = [  1       ]                 V = [  1 v1 v1 v1 v1 ]
///                 [ v1  1    ]                     [     1 v2 v2 v2 ]
///                 [ v1 v2  1 ]                     [        1 v3 v3 ]
///                 [ v1 v2 v3 ]
///                 [ v1 v2 v3 ]
///
/// DIRECT = Bwd and STOREV = ByCol:         DIRECT = Bwd and STOREV = ByRow:
///
///              V = [ v1 v2 v3 ]                 V = [ v1 v1  1       ]
///                  [ v1 v2 v3 ]                     [ v2 v2 v2  1    ]
///                  [  1 v2 v3 ]                     [ v3 v3 v3 v3  1 ]
///                  [     1 v3 ]
///                  [        1 ]
/// </summary>
/// <remarks>
/// Based on the LAPACK routine <c>dlarfb</c>.
/// </remarks>
template< typename Lyt = ColMajor,
  typename T_Blk_V,
  typename T_Blk_T,
  typename T_Blk_C,
  typename T_Blk_W >
requires( ! isComplex< Decay<DerefTypeOf<T_Blk_V>> >
  && areTheSame<
  Decay<DerefTypeOf<T_Blk_V>>,
  Decay<DerefTypeOf<T_Blk_T>>,
  Decay<DerefTypeOf<T_Blk_C>>,
  Decay<DerefTypeOf<T_Blk_W>> > )
constexpr void Rfl_BlkMul(
  Side side, Trnsp H_trnsp, Direct direct, Store storev,
  Size m, Size n, Size k,
  T_Blk_V V_, Size V_ld,
  T_Blk_T T_, Size T_ld,
  T_Blk_C C_, Stride C_ld,
  T_Blk_W W_, Size W_ld )
{
  using Scalar = Decay<DerefTypeOf<T_Blk_V>>;

  auto T = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( T_, i, j, T_ld ); };

  auto W = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( W_, i, j, W_ld ); };

  auto C = [&]( auto i, auto j ) -> auto &
  { return Lyt::MatRef( C_, i, j, C_ld ); };
  auto C_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( C_, i, j, C_ld ); };
  auto C_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( C_, i, j, C_ld ); };
  auto C_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( C_, i, j, C_ld ); };

  auto V = [&]( auto i, auto j ) -> const auto &
  { return Lyt::MatRef( V_, i, j, V_ld ); };
  auto V_Blk = [&]( auto i, auto j ) -> auto
  { return Lyt::BlkPtr( V_, i, j, V_ld ); };
  auto V_Row = [&]( auto i, auto j ) -> auto
  { return Lyt::RowPtr( V_, i, j, V_ld ); };
  auto V_Col = [&]( auto i, auto j ) -> auto
  { return Lyt::ColPtr( V_, i, j, V_ld ); };

  if( Trnsp::Conj == H_trnsp ){ throw BadArgument{ "Aux_BlkReflMul", 2 }; }

  // Quick return if possible

  if( (0 == m) || (0 == n) ){ return; }

  const Stride C_rs = Lyt::RowStride( C_, C_ld );
  const Stride C_cs = Lyt::ColStride( C_, C_ld );

  const Stride W_rs = Lyt::RowStride( W_, W_ld );
  const Stride W_cs = Lyt::ColStride( W_, W_ld );

  const Trnsp T_trnsp = ( Trnsp::No == H_trnsp ) ? Trnsp::Yes : Trnsp::No;

  if( Store::ByCol == storev )
  {
    if( Direct::Fwd == direct )
    {
      // Let  V = [ V1 ]    (first K rows)
      //          [ V2 ]
      //
      //  where V1 is unit lower triangular.

      if( Side::Left == side )
      {
        // Form  H*C  or  (~H)*C  where  C = [ C1 ]
        //                                   [ C2 ]
        //
        // W := (~C)*V = ((~C1)*V1 + (~C2)*V2)  (stored in WORK)
        //
        // W := (~C1)
        Mat_Copy< Lyt >( Half::Both, Trnsp::Yes, n, k, C_, C_ld, W_, W_ld );

        // W := W*V1

        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::No, Diag::IsUnit,
          n, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        if( m > k )
        {
          // W := W + (~C2)*V2
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::No, n, k, m-k,
            unit<Scalar>, C_Blk(k,0), C_ld,
            V_Blk(k,0), V_ld,
            unit<Scalar>, W_, W_ld );

          // W := W*(~T)  or  W*T
          Tri_MatMul< Lyt >( Side::Right, Half::Upper, T_trnsp, Diag::NotUnit,
            n, k, unit<Scalar>, T_, T_ld, W_, W_ld );

          // C := C - V*(~W)

          if( m > k )
          {
            Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m-k, n, k,
              -unit<Scalar>, V_Blk(k,0), V_ld,
              W_, W_ld,
              unit<Scalar>, C_Blk(k,0), C_ld );
          }

          // W := W*(~V1)
          Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::Yes, Diag::IsUnit,
            n, k, unit<Scalar>, V_, V_ld, W_, W_ld );

          // C1 := C1 - (~W)
          Mat_Sub< Lyt >( Trnsp::Yes, k, n, W_, W_ld, C_, C_ld );
        }
      }
      else if( Side::Right == side )
      {
        // Form  C*H  or  C*(~H) where  C = [ C1  C2 ]
        //
        // W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
        //
        // W := C1
        Mat_Copy< Lyt >( Half::Both, Trnsp::No, m, k, C_, C_ld, W_, W_ld );

        // W := W*V1
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::No, Diag::IsUnit,
          m, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // W := W + C2*V2
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::No, m, k, n-k,
            unit<Scalar>, C_Blk(0,k), C_ld,
            V_Blk(k,0), V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*T  or  W*(~T)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, H_trnsp, Diag::NotUnit,
          m, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - W*(~V)

        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m, n-k, k,
            -unit<Scalar>, W_, W_ld,
            V_Blk(k,0), V_ld,
            unit<Scalar>, C_Blk(0,k), C_ld );
        }

        // W := W*(~V1)
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::Yes, Diag::IsUnit,
          m, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // C1 := C1 - W
        Mat_Sub< Lyt >( Trnsp::No, m, k, W_, W_ld, C_, C_ld );
      }
    }
    else if( Direct::Bwd == direct )
    {
      // Let  V = [ V1 ]
      //          [ V2 ]    (last K rows)
      //
      // where V2 is unit upper triangular.

      if( Side::Left == side )
      {
        // Form  H*C  or  (~H)*C  where  C = [ C1 ]
        //                                   [ C2 ]
        //
        // W := (~C)*V = ((~C1)*V1 + (~C2)*V2)  (stored in WORK)

        // W := (~C2)
        Mat_Copy< Lyt >( Half::Both, Trnsp::Yes, n, k, C_Blk(m-k,0), C_ld, W_, W_ld );

        // W := W*V2
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::No, Diag::IsUnit,
          n, k, unit<Scalar>, V_Blk(m-k, 0), V_ld, W_, W_ld );

        // W := W + (~C1)*V1
        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::No, n, k, m-k,
            unit<Scalar>, C_, C_ld,
            V_, V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*(~T) or W*T
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, T_trnsp, Diag::NotUnit,
          n, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - V*(~W)
        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m-k, n, k,
            -unit<Scalar>, V_, V_ld,
            W_, W_ld,
            unit<Scalar>, C_, C_ld );
        }

        // W := W*(~V2)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::Yes, Diag::IsUnit,
          n, k, unit<Scalar>, V_Blk(m-k,0), V_ld, W_, W_ld );

        // C2 := C2 - (~W)
        Mat_Sub< Lyt >( Trnsp::Yes, k, n, W_, W_ld, C_Blk(m-k,0), C_ld );
      }
      else if( Side::Right == side )
      {
        // Form C*H or C*(~H) where C = [ C1  C2 ]
        //
        // W := C*V = (C1*V1 + C2*V2) (stored in WORK)
        //
        // W := C2
        Mat_Copy< Lyt >( Half::Both, Trnsp::No, m, k, C_Blk(0,n-k), C_ld, W_, W_ld );

        // W := W*V2
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::No, Diag::IsUnit,
          m, k, unit<Scalar>, V_Blk(n-k,0), V_ld, W_, W_ld );

        // W := W + C1*V1
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::No, m, k, n-k,
            unit<Scalar>, C_, C_ld,
            V_, V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*T or W*(~T)
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, H_trnsp, Diag::NotUnit,
          m, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - W*(~V)
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m, n-k, k,
            -unit<Scalar>, W_, W_ld,
            V_, V_ld,
            unit<Scalar>, C_, C_ld );
        }

        // W := W*(~V2)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::Yes, Diag::IsUnit,
          m, k, unit<Scalar>, V_Blk(n-k,0), V_ld, W_, W_ld );

        // C2 := C2 - W
        Mat_Sub< Lyt >( Trnsp::No, m, k, W_, W_ld, C_Blk(0,n-k), C_ld );
      }
    }
  }
  else if( Store::ByRow == storev )
  {
    if( Direct::Fwd == direct )
    {
      // Let  V = [ V1  V2 ]    (V1: first K columns)
      // where  V1  is unit upper triangular.

      if( Side::Left == side )
      {
        // Form  H * C  or  H**T * C  where  C = [ C1 ]
        //                                       [ C2 ]
        //
        // W := (~C)*(~V) = ((~C1)*(~V1) + (~C2)*(~V2)) (stored in WORK)

        // W := (~C1)
        Mat_Copy< Lyt >( Half::Both, Trnsp::Yes, n, k, C_, C_ld, W_, W_ld );

        // W := W*(~V1)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::Yes, Diag::IsUnit,
          n, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // W := W + (~C2)*(~V2)
        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::Yes, n, k, m-k,
            unit<Scalar>, C_Blk(k,0), C_ld,
            V_Blk(0,k), V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*(~T) or W*T
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, T_trnsp, Diag::NotUnit,
          n, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - (~V)*(~W)

        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::Yes, m-k, n, k,
            -unit<Scalar>, V_Blk(0,k), V_ld,
            W_, W_ld,
            unit<Scalar>, C_Blk(k,0), C_ld );
        }

        // W := W*V1
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::No, Diag::IsUnit,
          n, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // C1 := C1 - (~W)
        Mat_Sub< Lyt >( Trnsp::Yes, k, n, W_, W_ld, C_, C_ld );
      }
      else if( Side::Right == side )
      {
        // Form C*H or C*(~H)  where  C = [ C1  C2 ]
        //
        // W := C*(~V) = (C1*(~V1) + C2*(~V2)) (stored in WORK)

        // W := C1
        Mat_Copy< Lyt >( Half::Both, Trnsp::No, m, k, C_, C_ld, W_, W_ld );

        // W := W*(~V1)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::Yes, Diag::IsUnit,
          m, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // W := W + C2*(~V2)
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m, k, n-k,
            unit<Scalar>, C_Blk(0,k), C_ld,
            V_Blk(0,k), V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*T or W*(~T)
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, H_trnsp, Diag::NotUnit,
          m, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - W*V2
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::No, m, n-k, k,
            -unit<Scalar>, W_, W_ld,
            V_Blk(0,k), V_ld,
            unit<Scalar>, C_Blk(0,k), C_ld );
        }

        // W := W*V1
        Tri_MatMul< Lyt >( Side::Right, Half::Upper, Trnsp::No, Diag::IsUnit,
          m, k, unit<Scalar>, V_, V_ld, W_, W_ld );

        // C1 := C1 - W
        Mat_Sub< Lyt >( Trnsp::No, m, k, W_, W_ld, C_, C_ld );
      }
    }
    else if( Direct::Bwd == direct )
    {
      // Let  V =  [ V1  V2 ]    (V2: last K columns)
      //
      // where  V2  is unit lower triangular.

      if( Side::Left == side )
      {
        // Form H*C or (~H)*C where C = [ C1 ]
        //                              [ C2 ]
        //
        // W := (~C)*(~V) = ((~C1)*(~V1) + (~C2)*(~V2)) (stored in WORK)

        // W := (~C2)
        Mat_Copy< Lyt >( Half::Both, Trnsp::Yes, n, k, C_Blk(m-k,0), C_ld, W_, W_ld );

        // W := W*(~V2)
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::Yes, Diag::IsUnit,
          n, k, unit<Scalar>, V_Blk(0,m-k), V_ld, W_, W_ld );

        // W := W + (~C1)*(~V1)
        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::Yes,
            n, k, m-k, unit<Scalar>, C_, C_ld,
            V_, V_ld, unit<Scalar>, W_, W_ld );
        }

        // W := W*(~T) or W*T
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, T_trnsp, Diag::NotUnit,
          n, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - (~V)*(~W)
        if( m > k )
        {
          Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::Yes,
            m-k, n, k, -unit<Scalar>, V_, V_ld,
            W_, W_ld, unit<Scalar>, C_, C_ld );
        }

        // W := W*V2
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::No, Diag::IsUnit,
          n, k, unit<Scalar>, V_Blk(0,m-k), V_ld, W_, W_ld );

        // C2 := C2 - (~W)
        Mat_Sub< Lyt >( Trnsp::Yes, k, n, W_, W_ld, C_Blk(m-k,0), C_ld );
      }
      else if( Side::Right == side )
      {
        // Form C*H or C*(~H) where C = [ C1  C2 ]
        //
        // W := C*(~V) = (C1*(~V1) + C2*(~V2))  (stored in WORK)

        // W := C2
        Mat_Copy< Lyt >( Half::Both, Trnsp::No, m, k, C_Blk(0,n-k), C_ld, W_, W_ld );

        // W := W*(~V2)
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::Yes, Diag::IsUnit,
          m, k, unit<Scalar>, V_Blk(0,n-k), V_ld, W_, W_ld );

        // W := W + C1*(~V1)
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m, k, n-k,
            unit<Scalar>, C_, C_ld,
            V_, V_ld,
            unit<Scalar>, W_, W_ld );
        }

        // W := W*T or W*(~T)
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, H_trnsp, Diag::NotUnit,
          m, k, unit<Scalar>, T_, T_ld, W_, W_ld );

        // C := C - W*V1
        if( n > k )
        {
          Mat_MatMul< Lyt >( Trnsp::No, Trnsp::No, m, n-k, k,
            -unit<Scalar>, W_, W_ld,
            V_, V_ld,
            unit<Scalar>, C_, C_ld );
        }

        // W := W*V2
        Tri_MatMul< Lyt >( Side::Right, Half::Lower, Trnsp::No, Diag::IsUnit,
          m, n, unit<Scalar>, V_Blk(0,n-k), V_ld, W_, W_ld );

        // C1 := C1 - W
        Mat_Sub< Lyt >( Trnsp::No, m, k, W_, W_ld, C_Blk(0,n-k), C_ld );
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