/******************************************************************************
 * COPYRIGHT (C) 2024 Christopher Gary
 * All Rights Reserved.
 *
 * This code is provided for the sole purpose of evaluating the technical
 * skills of the author/applicant in the context of a job application. 
 * Redistribution, modification, use, or incorporation of this code in any 
 * product or project, in part or in whole, is strictly prohibited without 
 * prior written consent from the author.
 *
 * THIS CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL 
 * THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING 
 * FROM, OUT OF, OR IN CONNECTION WITH THE CODE OR THE USE OR OTHER DEALINGS 
 * IN THE CODE.
 *
 * By using this code, you agree to comply with the terms of this license.
 *****************************************************************************/

#ifndef __IND_MATH_LAPACK_H__
#define __IND_MATH_LAPACK_H__

#include <IND.Math.BLAS.h>

#ifdef __IND_MATH_LAPACK_H_CONTENTS__
#error __IND_MATH_LAPACK_H_CONTENTS__ is a reserved token.
#endif

#define __IND_MATH_LAPACK_H_CONTENTS__

namespace IND {
namespace Math {
namespace LAPACK {

using namespace BLAS;

enum class MatType
{
  Full,
  LowerTri,
  UpperTri,
  UpperHess,
  LowerBand,
  UpperBand,
  Banded
};

enum class NormType
{
  Max,
  One,
  Inf,
  Frob
};

enum class Store
{
  ByRow,
  ByCol
};

enum class Pivot
{
  Btm,
  Top,
  Var
};

enum class Direct
{
  Fwd,
  Bwd
};

enum class Vect
{
  Q,
  Pt
};

namespace _n_Impl {

// Simple and readable, without syntactical
// binding to a particular value type.
template< typename T_Scalar >
inline constexpr T_Scalar _oneHalf = Inv( T_Scalar{2} );

//template< typename T_Scalar >
//requires( isIEEE754< T_Scalar > || IND::IEEE754::Ex::isSum< T_Scalar > )
//inline constexpr T_Scalar _oneHalf< T_Scalar > = T_Scalar( 0.5 );

}// namespace _n_Impl
}// namespace LAPACK
}// namespace Math
}// namespace IND

#include <IND.Math.LAPACK.Idx_LastRow.inl>   // iladlr
#include <IND.Math.LAPACK.Idx_LastCol.inl>   // iladlc

#include <IND.Math.LAPACK.Aux_CombSsq2.inl>  // xcombssq
#include <IND.Math.LAPACK.Aux_PlnRot2.inl>   // xlartg
#include <IND.Math.LAPACK.Aux_Eig2.inl>      // xlae2
#include <IND.Math.LAPACK.Aux_EigVec2.inl>   // xlaev2

#include <IND.Math.LAPACK.Vec_Rescl.inl>     // <-------- extension
#include <IND.Math.LAPACK.Vec_SmSqr.inl>     // xlassq

#include <IND.Math.LAPACK.Rfl_VecGen.inl>    // xlarfg
#include <IND.Math.LAPACK.Rfl_BlkGen.inl>    // xlarft
#include <IND.Math.LAPACK.Rfl_BlkMul.inl>    // xlarb
#include <IND.Math.LAPACK.Rfl_MatMul.inl>    // xlarf

#include <IND.Math.LAPACK.Mat_RotSeq.inl>    // xlasr
#include <IND.Math.LAPACK.Mat_Fill.inl>      // xlaset
#include <IND.Math.LAPACK.Mat_Rescl.inl>     // xlascl
#include <IND.Math.LAPACK.Mat_Rdto_Bid.inl>  // xgebrd
#include <IND.Math.LAPACK.Mat_Fctr_QL.inl>   // xgeql2
#include <IND.Math.LAPACK.Mat_Fctr_QR.inl>   // xgeqr2
#include <IND.Math.LAPACK.Mat_Fctr_LQ.inl>   // xgelq2
#include <IND.Math.LAPACK.Mat_Fctr_RQ.inl>   // xgerq2

#include <IND.Math.LAPACK.Sym_Norm.inl>      // xlansy
#include <IND.Math.LAPACK.Sym_Rdto_Syt.inl>  // xsytd2 | xsytrd

#include <IND.Math.LAPACK.Syt_Norm.inl>      // xlanst
#include <IND.Math.LAPACK.Syt_EigQR.inl>     // xsterf
#include <IND.Math.LAPACK.Syt_EigVecQR.inl>  // xsteqr

#include <IND.Math.LAPACK.Ort_From_LQ.inl>   // xorgl2
#include <IND.Math.LAPACK.Ort_From_RQ.inl>   // xorgr2
#include <IND.Math.LAPACK.Ort_From_QL.inl>   // xorg2l
#include <IND.Math.LAPACK.Ort_From_QR.inl>   // xorg2l
#include <IND.Math.LAPACK.Ort_From_Syt.inl>  // xorgtr
#include <IND.Math.LAPACK.Ort_From_Bid.inl>  // xorgbr <- simplified

#undef __IND_MATH_LAPACK_H_CONTENTS__

#endif //__IND_MATH_LAPACK_H__