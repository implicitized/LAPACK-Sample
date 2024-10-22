/*
LAPACK Sample

Copyright (C) 2024 Christopher Gary

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <random>
#include <vector>
#include <IND.Math.LAPACK.h>

void Example_Inverse();
void Example_Eigensystem();
void Example_Bidiagonal();

int main( int argc, char **argv )
{
  using namespace IND;

  Example_Inverse();
  Example_Eigensystem();
  Example_Bidiagonal();

  return 0;
}

void Example_Inverse()
{
  using namespace std;

  using namespace IND;
  using namespace Math;
  using namespace LAPACK;

  using Lyt = RowMajor;
  using Scalar = Float64;

  cout << "-------- Inverse Example" << endl;

  Scalar A[25]
  {
     1, 0, 0, 0,-10,
     0, 1, 4,-5, 0,
     20, 0, 1, 0,-20,
     0, 0, 2, 1, 0,
     8, 3, 0, 0, 1
  };

  Scalar LU[25];
  copy( A, A+25, LU );

  Index piv[5];
  for( auto &k : piv ){ k = -1; }

  const auto factored = Mat_Fctr_LU< Lyt >( 5,5, LU,5, piv );
  if( ! factored )
  {
    cout << "ERROR: Mat_Fctr_LU failed." << endl;
    return;
  }

  Scalar b[5]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
  Scalar x[5];
  copy( b, b+5u, x );

  Mat_Solv_LU< Lyt >( Trnsp::No, 5, LU,5, piv, x,1 );

  Scalar y[5];
  Mat_VecMul< Lyt >( Trnsp::No, 5,5, 1.0f, A,5, x,1, 0.0f,y,1 );

  for( Index i = 0; i < 5; ++i )
  {
    if( Abs( y[i] - b[i] ) > 1.0e-5f )
    {
      cout << "ERROR: With y = A*x, and A*x = b, y != b." << endl;
      return;
    }
  }

  cout << "-------- SUCCESS!" << endl;
}

void Example_Eigensystem()
{
  using namespace std;

  using namespace IND;
  using namespace Math;
  using namespace LAPACK;

  cout << "-------- Eigensystem Example" << endl;

  using Lyt = ColMajor;
  using Scalar = Float64;

  uniform_real_distribution< Scalar > dist{ -100.0f, 100.0f };
  mt19937 gen{};

  Size n = 200;
  Size n2 = n*n;

  cout << "Solving " << n << " x " << n << " random symmetric problem..." << endl;

  vector< Scalar > bfr;

  bfr.resize( 4*n2 + ( 2*n + 3*(n-1) ) + Max(
    Ort_From_Syt_WorkSize( n ),
    Syt_EigVecQR_WorkSize( n ) ) );

  auto * A = bfr.data();
  auto * B = A + n2;
  auto * C = B + n2;
  auto * S = C + n2;
  auto * d = S + n2;
  auto * d1 = d + n;
  auto * e = d1 + n;
  auto * e1 = e + (n-1);
  auto * tau = e1 + (n-1);
  auto * work = tau + (n-1);

  for( Index i = 0; i < (Index)n2; ++i )
  { A[i] = dist(gen); }

  // S = (~A)*A
  Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::No, n, n, n, 1.0f, A,n, A,n, 0.0f, S,n );

  // A := S
  copy( S, S+n2, A );

  Sym_Rdto_Syt< Lyt >( Half::Lower, n, S,n, d,e,tau );
  Ort_From_Syt< Lyt >( Half::Lower, n, S,n, tau, work );

  copy( d, d+n, d1 );
  copy( e, e+(n-1), e1 );

  Syt_EigVecQR< Scalar > VQR{};
  if( ! VQR.Solve< Lyt >( n, d,e, S,n, work ) )
  {
    cout << "ERROR: Syt_EigVecQR failed to converge!" << endl;
    return;
  }

  // B = S*d
  Mat_Scale< Lyt >( Side::Right, n,n, S,n, d, B,n );
  // C = B*(~S) = S*d*(~S)
  Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, n,n,n, 1.0f, B,n, S,n, 0.0f, C,n );

  const Scalar dtol = 1.0e-10f;
  const Scalar Ztol = 1.0e-5f;

  for( Index i = 0; i < (Index)n2; ++i )
  {
    if( ! IsWithinBound( A[i] - C[i], Ztol ) )
    {
      cout << "ERROR: Syt_EigVecQR eigensystem did not round-trip!" << endl;
      break;
    }
  }

  Syt_EigQR< Scalar > QR{};
  if( ! QR.Solve< Lyt >( n, d1, e1 ) )
  {
    cout << "ERROR: Syt_EigQR failed to converge!" << endl;
    return;
  }

  sort( d, d+n );
  sort( d1, d1+n );

  for( Index i = 0; i < (Index)n; ++i )
  {
    if( ! IsWithinBound( d1[i] - d[i], dtol ) )
    {
      cout << "ERROR: Eigenvalues from Syt_EigQR did not match Syt_EigVecQR! " << d1[i] << " = " << d[i] << endl;
      break;
    }
  }

  cout << "-------- SUCCESS!" << endl;
}

void Example_Bidiagonal()
{
  using namespace std;

  using namespace IND;
  using namespace Math;
  using namespace LAPACK;

  cout << "-------- Bidiagonal Example" << endl;

  using Lyt = ColMajor;
  using Scalar = Float64;

  uniform_real_distribution< Scalar > dist{ -100.0f, 100.0f };
  mt19937 gen{};

  Size m = 93;
  Size n = 317;

  cout << "Reducing " << m << " x " << n << " random matrix..." << endl;

  Size m2 = m*m;
  Size n2 = n*n;
  Size mn = m*n;
  Size k = Min( m, n );

  vector< Scalar > bfr;
  bfr.resize( 5*mn + k + (k-1) + 2*Max(n,m) + ( Mat_Rdto_Bid_WorkSize( m, n ) + mn ) );

  auto * A = bfr.data();
  auto * B = A + mn;
  auto * C = B + mn;
  auto * Q = C + mn;
  auto * Pt = Q + mn;
  auto * d = Pt + mn;
  auto * e = d + k;
  auto * Q_tau = e + (k-1);
  auto * P_tau = Q_tau + Max(n,m);
  auto * work = P_tau + Max(n,m); // at least mn in size

  for( Index i = 0; i < (Index)mn; ++i )
  { A[i] = dist(gen); }

  auto A_ld = m;
  auto B_ld = m;
  auto C_ld = m;
  auto Pt_ld = m;
  auto Q_ld = m;

  copy( A, A+ mn, B );
  Mat_Rdto_Bid< Lyt >( m, n, B,B_ld, d,e,Q_tau,P_tau, work );

  copy( B, B+mn, Q );
  copy( B, B+mn, Pt );
  Ort_From_Bid< Lyt >( Vect::Q, m, n, n, Q,Q_ld, Q_tau, work );
  Ort_From_Bid< Lyt >( Vect::Pt, m, n, m, Pt,Pt_ld, P_tau, work );

  // ZeroFill( B, mn );
  memset( B, 0, mn*sizeof(Scalar) );

  const Stride B_ds = Lyt::DiagStride( B, B_ld );

  auto * B_d = Lyt::DiagPtr( B, 0, 0, C_ld );
  auto * B_e = ( m >= n )?
    // Upper bidiagonal
    Lyt::DiagPtr( B, 0, 1, B_ld ) :
    // Lower bidiagonal
    Lyt::DiagPtr( B, 1, 0, B_ld );

  Vec_Copy< Lyt >( k, d, 1, B_d, B_ds ); 
  Vec_Copy< Lyt >( k-1, e, 1, B_e, B_ds );

  // C = (~Q)*A
  Mat_MatMul< Lyt >( Trnsp::Yes, Trnsp::No, m, n, k, 1.0f, Q,Q_ld, A,A_ld, 0.0f, C,C_ld );
  // A = C*(~Pt) == B
  Mat_MatMul< Lyt >( Trnsp::No, Trnsp::Yes, m, k, n, 1.0f, C,C_ld, Pt,Pt_ld, 0.0f, A,A_ld );

  const Stride A_ds = Lyt::DiagStride( A, A_ld );

  auto * A_d = Lyt::DiagPtr( A, 0, 0, A_ld );
  auto * A_e = ( m >= n )?
    // Upper bidiagonal
    Lyt::DiagPtr( A, 0, 1, A_ld ) :
    // Lower bidiagonal
    Lyt::DiagPtr( A, 1, 0, A_ld );

  const Scalar diagTol = 1.0e-5f;

  for( Index i = 0; i < (Index)k; ++i )
  {
    const auto ad = Lyt::VecRef( A_d, i, A_ds );
    const auto bd = Lyt::VecRef( B_d, i, B_ds );
    if( ! IsWithinBound( ad - bd, diagTol ) )
    {
      cout << "ERROR: Ort_From_Bid - diagonal element mismatch! " << ad << " != " << bd << endl;
      break;
    }
  }

  for( Index i = 0; i < (Index)(k-1); ++i )
  {
    const auto ae = Lyt::VecRef( A_e, i, A_ds );
    const auto be = Lyt::VecRef( B_e, i, B_ds );
    if( ! IsWithinBound( ae - be, diagTol ) )
    {
      cout << "ERROR: Ort_From_Bid - off-diagonal element mismatch! " << ae << " != " << be << endl;
      break;
    }
  }

  cout << "-------- SUCCESS!" << endl;
}
