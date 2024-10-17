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

#ifndef __IND_MATH_BLAS_H__
#define __IND_MATH_BLAS_H__

#include <Common.h>

#ifdef __IND_MATH_BLAS_H_CONTENTS__
#error __IND_MATH_BLAS_H_CONTENTS__ is a reserved token.
#endif

#define __IND_MATH_BLAS_H_CONTENTS__

namespace IND {
namespace Math {
namespace BLAS {

using Size = UIntSize;
using Index = IntSize;
using Stride = Index;

class BadArgument /*: public System::BadArgument*/
{
public:

  // The parent library uses UTF-8 strings with an option
  // to support char-strings as a direct subset (assuming
  // char is ASCII). Moreover, the constructor does not
  // throw unless explicitly instructed to allocate.
  //
  // This is merely preference, very common, though unfortunately
  // not reality in some cases.

  using Message = std::string;

  //using System::BadArgument::Message;

  Message message;

  // 1-based argument index.
  Index arg;

  BadArgument( Message &&message, Index arg ) noexcept
  /*: System::BadArgument{ message }*/
  : message{ std::move( message ) }
  , arg{ arg }
  {}

  constexpr ~BadArgument() noexcept = default;
};

class InternalError
{
public:

  using Message = std::string;

  Message message;

  InternalError( Message &&message ) noexcept
  : message{ std::move( message ) }
  {}

  constexpr ~InternalError() noexcept = default;
};

//IND_DEFINE_SIMPLE_EXCEPTION( InternalError, System::Exception );

struct Flat
{
  template< typename T_Vec_x >
  static constexpr auto &VecRef( const T_Vec_x &x, Index i, Stride x_s )
  { return x[i*x_s]; }

  template< typename T_Vec_x >
  static constexpr auto VecPtr( const T_Vec_x &x, Index i, Stride x_s )
  { return x + i*x_s; }

  template< typename T_Vec_x >
  static constexpr void VecInc( T_Vec_x &x, Stride x_s )
  { x += x_s; }

  template< typename T_Vec_x >
  static constexpr void VecDec( T_Vec_x &x, Stride x_s )
  { x -= x_s; }
};

struct ColMajor : Flat
{
  template< typename T_Blk_A >
  static constexpr Stride ColStride( const T_Blk_A &, Stride A_ld )
  { return 1; }

  template< typename T_Blk_A >
  static constexpr Stride RowStride( const T_Blk_A &, Stride A_ld )
  { return A_ld; }

  template< typename T_Blk_A >
  static constexpr Stride DiagStride( const T_Blk_A &, Stride A_ld )
  { return A_ld + 1; }

  template< typename T_Blk_A >
  static constexpr auto &MatRef( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return A_[ i + j*A_ld ]; }

  template< typename T_Blk_A >
  static constexpr auto BlkPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return A_ + ( i + j*A_ld ); }

  template< typename T_Blk_A >
  static constexpr auto RowPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }

  template< typename T_Blk_A >
  static constexpr auto ColPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }

  template< typename T_Blk_A >
  static constexpr auto DiagPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }
};

struct RowMajor : Flat
{
  template< typename T_Blk_A >
  static constexpr Stride ColStride( const T_Blk_A &, Stride A_ld )
  { return A_ld; }

  template< typename T_Blk_A >
  static constexpr Stride RowStride( const T_Blk_A &, Stride A_ld )
  { return 1; }

  template< typename T_Blk_A >
  static constexpr Stride DiagStride( const T_Blk_A &, Stride A_ld )
  { return A_ld + 1; }

  template< typename T_Blk_A >
  static constexpr auto &MatRef( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return A_[ i*A_ld + j ]; }

  template< typename T_Blk_A >
  static constexpr auto BlkPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return A_ + ( i*A_ld + j ); }

  template< typename T_Blk_A >
  static constexpr auto RowPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }

  template< typename T_Blk_A >
  static constexpr auto ColPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }

  template< typename T_Blk_A >
  static constexpr auto DiagPtr( const T_Blk_A &A_, Index i, Index j, Stride A_ld )
  { return BlkPtr( A_, i, j, A_ld ); }
};

template< typename T_Layout >
inline constexpr bool isColMajor = areTheSame< T_Layout, ColMajor >;

template< typename T_Layout >
inline constexpr bool isRowMajor = areTheSame< T_Layout, RowMajor >;

enum class Trnsp
{
  No = 0,
  Yes,
  Conj
};

enum class Half
{
  Upper = 1,
  Lower = 2,
  Both  = 3
};

enum class Diag
{
  IsUnit,
  NotUnit
};

enum class Side
{
  Left = 0,
  Right
};

}// namespace BLAS
}// namespace Math
}// namespace IND

//----------------------------------------------------------------
// Many BLAS APIs seem redundant with basic system utilities
// such as memset or ZeroFill.
//
// These exist to provide low-level binding points for
// operations that affect the performance of the higher-level
// LAPACK APIs without creating any undue coupling with how the
// system primitives may have been specialized elsewhere.
// 
// Insomuch as possible, the use cases and entry points match
// the BLAS "standard", with some additional primitives that
// provide some operational symmetry within the BLAS layer itself.
//----------------------------------------------------------------

#include <IND.Math.BLAS.Vec_X.inl>
#include <IND.Math.BLAS.Sym_Rank2Upd.inl>     // xsyr2
#include <IND.Math.BLAS.Sym_Rank2kUpd.inl>    // xsyr2k
#include <IND.Math.BLAS.Sym_VecMul.inl>       // xsymv

#include <IND.Math.BLAS.Tri_VecMul.inl>       // xtrmv
#include <IND.Math.BLAS.Tri_MatMul.inl>       // xtrmm
#include <IND.Math.BLAS.Tri_Solv_Vec.inl>     // xtrsv
#include <IND.Math.BLAS.Tri_Solv_Mat.inl>     // xtrsm

#include <IND.Math.BLAS.Mat_Copy.inl>         // xlacpy with extended functionality
#include <IND.Math.BLAS.Mat_Scale.inl>        // <-------- extension
#include <IND.Math.BLAS.Mat_AddSub.inl>       // <-------- extension
#include <IND.Math.BLAS.Mat_Rank1Upd.inl>     // xger
#include <IND.Math.BLAS.Mat_VecMul.inl>       // xgemv
#include <IND.Math.BLAS.Mat_ConjVecMul.inl>   // <-------- extension
#include <IND.Math.BLAS.Mat_MatMul.inl>       // xgemm

#include <IND.Math.BLAS.Mat_RowSwp.inl>        // xlaswp
#include <IND.Math.BLAS.Mat_Fctr_LU.inl>      // xgetrf | xgetrf2
#include <IND.Math.BLAS.Mat_Solv_LU.inl>      // xgetsv | xgetrs

#undef __IND_MATH_BLAS_H_CONTENTS__

#endif  // __IND_MATH_BLAS_H__