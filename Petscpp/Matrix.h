#ifndef PMATRIX_H_
#define PMATRIX_H_

#include <type_traits>
#include <initializer_list>
#include <ostream>
#include <Eigen/Dense>
#include <vector>
#include "Vector.h"

struct _p_Mat;
typedef struct _p_Mat* Mat;

namespace Petscpp{

  class Matrix;

  class MatrixSlice
  {
  public:
    MatrixSlice(Matrix &matrix,
                std::vector<int> const& rowIndices,
                std::vector<int> const& colIndices)
      : matrix_(matrix), rowIndices_(rowIndices), colIndices_(colIndices)
    { }
    // TODO: Remove redundant entries from specified indicies
    
    template <size_t M, size_t N>
    MatrixSlice(Matrix &matrix,
                std::array<int,M> const& rowIndices,
                std::array<int,N> const& colIndices)
      : matrix_(matrix)
    {
      rowIndices_.insert(rowIndices_.end(), rowIndices.begin(), rowIndices.end());
      colIndices_.insert(colIndices_.end(), colIndices.begin(), colIndices.end());
    }


    /*! Assign selected rows and indices to specified matrix
     *
     * Dimensions of matrix must be >= than row/column indicies
     * specified to instatiate this proxy class
     *
     * Getting values, as well as increment/decrement and assignment
     * operations must be preceded by PVector::assemble()
     */
    Matrix& operator=(Eigen::MatrixXd const& mat);

    /*! Increment selected rows and indices by specified matrix
     *
     * Dimensions of matrix must be >= than row/column indicies
     * specified to instatiate this proxy class.
     *
     * Getting values, as well as increment/decrement and assignment
     * operations must be preceded by PVector::assemble()
     */
    Matrix& operator+=(Eigen::MatrixXd const& values);

    /*! Set all values within range to specified value;
     *
     * Getting values, as well as increment/decrement and assignment
     * operations must be preceded by PVector::assemble()
     */
    Matrix& operator=(double val);

    // Experimental
    Eigen::MatrixXd toMatrix() const;

    // Number of rows of sliced matrix
    size_t rowCount() const { return rowIndices_.size(); }

    // Number of columns of sliced matrix
    size_t colCount() const { return colIndices_.size(); }

  private:
    Mat petscMatrix() const;

    Matrix &matrix_;
    std::vector<int> rowIndices_;
    std::vector<int> colIndices_;
  };


  /* Proxy class for chaining arithmetic add operations
   *
   * Do not manually utilize this class.
   */
  template <typename LHS, typename RHS>
  class MatrixAddOp
  {
  public:
    MatrixAddOp(LHS lhs, RHS rhs)
      : lhs_(lhs), rhs_(rhs)
    { }

    // Evaluate the expression
    Matrix eval();

    template <typename L, typename R>
    friend MatrixAddOp<L, R> operator*(MatrixAddOp<L, R> addOp, double alpha);

    template <typename L, typename R>
    friend MatrixAddOp<L, R> operator*(double alpha, MatrixAddOp<L, R> addOp);

    template <typename L, typename R>
    friend MatrixAddOp<L, R> operator/(MatrixAddOp<L, R> addOp, double alpha);

    friend class Matrix;
    friend class MatrixProxy;
    friend class MatrixScaleOp;

  private:
    // Recursively add LHS and RHS to v
    void addTo(Matrix& v);

    // create a duplicate of the left-most internal child matrix
    Mat duplicateMat(bool copyValues = true) const{
      lhs_.duplicateMat(copyValues);
    }

    LHS lhs_;
    RHS rhs_;
  };


  /* Proxy class for chaining arithmetic add operations
   *
   * Do not manually utilize this class.
   */
  template <typename LHS, typename RHS>
  class MatMatMultOp
  {
  public:
    MatMatMultOp(LHS lhs, RHS rhs)
      : lhs_(lhs), rhs_(rhs)
    { }

    // create a duplicate of the left-most internal child matrix
    Mat duplicateMat(bool copyValues = true) const{
      lhs_.duplicateMat(copyValues);
    }

    Matrix eval() const;


    MatMatMultOp<LHS,RHS>&
    operator*=(double alpha){
      // scale only the lhs
      lhs_ *= alpha;
      return *this;
    }

    MatMatMultOp<LHS,RHS>&
    operator/=(double alpha){
      // scale only the lhs
      lhs_ /= alpha;
      return *this;
    }

    // Untested
    // template <typename LHS_, typename RHS_>
    // friend MatMatMultOp<LHS_, RHS_> operator*(MatMatMultOp<LHS_, RHS_> addOp, double alpha);

    // // Untested
    // template <typename LHS_, typename RHS_>
    // friend MatMatMultOp<LHS_, RHS_> operator*(double alpha, MatMatMultOp<LHS_, RHS_> addOp);

    friend class MatrixProxy;
    template <typename L> friend class MatVecMultOp;
    template <typename L, typename R> friend class MatMatMultOp;

  private:
    // Evaluate the matrix product _without scaling_
    MatObj evalNoScale() const;

    // Total scaling factor
    double scaleFactor() const;

    // Multiply with input vector and return a new Vec object
    VecObj vecMultiplyNoScale(VecObj const& v) const;

    LHS lhs_;
    RHS rhs_;
  };


  /*! Proxy class for scaling of Matrix
   *
   * Do not manually utilize this class.
   */
  class MatrixScaleOp
  {
  public:
    MatrixScaleOp() = delete;
    MatrixScaleOp(Matrix& mat, double alpha);

    MatrixScaleOp& operator*=(double alpha);
    MatrixScaleOp& operator/=(double alpha);

    // Duplicate the matrix and its values. Ownership is passed to the
    // caller
    Mat duplicateMat(bool copyValues = true) const;

    // Evaluate the operator and return a new Matrix
    Matrix eval() const;

    template <typename LHS> friend class MatVecMultOp;
    template <typename L, typename R> friend class MatrixAddOp;
    template <typename L, typename R> friend class MatMatMultOp;

  private:
    // Return the stored scaling factor
    double scaleFactor() const;

    /* Helper for recursive matrix multiplication
     * Returns the stored internal petsc matrix without scaling
     */
    MatObj evalNoScale() const;

    /* Multiply with input vector without scaling */
    VecObj vecMultiplyNoScale(VecObj const& v) const;

    // Modify input matrix by adding internal matrix to it
    void addTo(Matrix& vec) const;

    Matrix &mat_;
    double alpha_;
  };


  template <typename LHS>
  class MatVecMultOp
  {
  public:
    MatVecMultOp(LHS lhs, VectorScaleOp rhs)
      : lhs_(lhs), rhs_(rhs)
    { }

    Vector eval() const{
      VecObj res = lhs_.vecMultiplyNoScale( VecObj(rhs_.petscVec(), false) );
      double scale = lhs_.scaleFactor();
      scale *= rhs_.scaleFactor();
      VecScale(res.get(), scale);
      Vector ret;
      ret.petscVec() = res.release();
      return ret;
    }

  private:
    LHS lhs_;
    VectorScaleOp rhs_;
  };


  class MatrixElement
  {
  public:
    MatrixElement(Matrix& mat, int m, int n)
      : mat_(mat), m_(m), n_(n)
    { }

    Matrix& operator=(double val);

    double value() const;

  private:
    Matrix &mat_;
    int m_;
    int n_;
  };


  class MatrixProxy
  {
  public:
    MatrixProxy(Matrix &m) : mat_(m) { }

    template <typename LHS, typename RHS>
    MatrixProxy& operator+=(MatrixAddOp<LHS, RHS> op);

    template <typename LHS, typename RHS>
    MatrixProxy& operator=(MatMatMultOp<LHS, RHS> op);

    // MatrixProxy& operator=(MatrixScaleOp op);

  private:
    Matrix &mat_;
  };


  /*! Wrapper for Petsc's Mat */
  class Matrix
  {
  public:
    Matrix();

    /* Perform a deep copy*/
    Matrix(Matrix const& other);

    /*! Allocate space for an  MxN matrix
     *
     * \param globSizes if true will use M and N as global sizes,
     *           other wise will use as local sizes
     * \param M number of rows
     * \param N number of columns
     * \param nzCount number of non-zero entries per row. Defaults to
     *           10% of the max(M,N) if specified as negative
     *
     * Note: if nzCount is not expicitely specified, the instance will
     * not be able to represent dense matrices
     */
    Matrix(int M, int N, bool globalSizes = true, int nzCount = -1);

    Matrix(MatrixScaleOp op);
    Matrix& operator=(Matrix const& other);
    Matrix& operator=(Matrix && other);

    template <typename LHS, typename RHS>
    Matrix(MatrixAddOp<LHS,RHS> addOp)
      : mat_(nullptr)
    {
      *this = addOp.eval();
    }

    template <typename LHS, typename RHS>
    Matrix(MatMatMultOp<LHS,RHS> op)
      : mat_(nullptr)
    {
      *this = op.eval();
    }

    ~Matrix();

    void setSize(int m, int n);

    MatrixProxy noAlias(){
      return MatrixProxy(*this);
    }


    /*! Flush matrix contents
     *
     * This function must be called whenever
     *
     * - Values of the matrix are set and want to get access to
     *     elements
     * - We wish to mix assign and increment/decrement operations
     *
     * ALL processors with access to the matrix must call this
     * function when one processor makes the call.
     */
    void assemble();

    /*! Return a proxy element to a single element of matrix
     *
     * \param row global row index of element
     * \param col global col index of element
     */
    MatrixElement operator()(int row, int col);


    /* Return a proxy class for accessing a portions of the matrix
     *
     * Convenience function
     */
    MatrixSlice operator()(std::initializer_list<int> const& rows,
                           std::initializer_list<int> const& cols);


    Matrix& operator*=(double alpha);


    /* Return a proxy class for accessing a portions of the matrix
     *
     * \param rows global indices of rows
     * \param cols global indices of columns
     */
    template <size_t M, size_t N>
    MatrixSlice operator()(std::array<int, M> const& rows,
                           std::array<int, N> const& cols){
      return MatrixSlice(*this, rows, cols);
    }

    /*! Return the global dimensions of the matrix <rows, cols> */
    std::pair<int,int> globalSize() const;

    /*! Return local dims of the matrix accessible by current proc
     *
     * Row count is stored as first entry, and column count is stored
     * as second entry. This information may be implementation
     * dependent. See Petsc's MatGetLocalSize().
     */
    std::pair<int,int> localSize() const;

    /*! local row count
     *
     * This is a convenience function
     */
    int localRowCount() const;

    /*! Global column count
     *
     * This is a convenience function
     */
    int localColumnCount() const;

    /*! Global row count
     *
     * This is a convenience function
     */
    int globalRowCount() const;

    /*! Global column count
     *
     * This is a convenience function
     */
    int globalColumnCount() const;

    /*! Return the starting row current processor has access to.
     *
     * For certain parallel layouts, this range may not be
     * well-defined. See Petsc's MatGetOwnershipRange()
     */
    int startingRow() const;

    friend class MatrixSlice;
    // friend class MatrixProxy;
    // friend class MatrixHandler;

    // Returns the underlying Petsc matrix
    Mat& petscMat();
    Mat const& petscMat() const;

  private:
    void cleanup();

    Mat mat_;
  };


  inline
  MatrixAddOp<MatrixScaleOp, MatrixScaleOp>
  operator+(Matrix &m1, Matrix &m2){
    using mso = MatrixScaleOp;
    return MatrixAddOp<mso, mso>(mso(m1,1), mso(m2,1));
  }


  inline
  MatrixAddOp<MatrixScaleOp, MatrixScaleOp>
  operator+(MatrixScaleOp m1, Matrix &m2){
    using mso = MatrixScaleOp;
    return MatrixAddOp<mso, mso>(m1, mso(m2,1));
  }


  inline
  MatrixAddOp<MatrixScaleOp, MatrixScaleOp>
  operator+(Matrix &m1, MatrixScaleOp m2){
    using mso = MatrixScaleOp;
    return MatrixAddOp<mso, mso>(mso(m1,1), m2);
  }


  inline
  MatMatMultOp<MatrixScaleOp, MatrixScaleOp>
  operator*(Matrix &m1, Matrix &m2){
    using mso = MatrixScaleOp;
    return MatMatMultOp<mso, mso>(mso(m1,1), mso(m2,1));
  }


  template <typename A, typename B>
  MatMatMultOp<MatMatMultOp<A,B>, MatrixScaleOp>
  operator*(MatMatMultOp<A,B> op, Matrix &m2){
    using mso = MatrixScaleOp;
    return MatMatMultOp<MatMatMultOp<A,B>, mso>(op, mso(m2,1));
  }


  template <typename A, typename B>
  MatMatMultOp<MatMatMultOp<A,B>, MatrixScaleOp>
  operator*(MatMatMultOp<A,B> mop, MatrixScaleOp sop){
    return MatMatMultOp<MatMatMultOp<A,B>, MatrixScaleOp>(mop, sop);
  }


  template <typename A, typename B>
  MatMatMultOp<MatrixScaleOp, MatMatMultOp<A,B>>
  operator*(MatrixScaleOp sop, MatMatMultOp<A,B> mop){
    return MatMatMultOp<MatrixScaleOp, MatMatMultOp<A,B>>(sop, mop);
  }


  template <typename A, typename B>
  MatMatMultOp<A,B>
  operator*(MatMatMultOp<A,B> mop, double alpha){
    mop *= alpha;
    return mop;
  }


  template <typename A, typename B>
  MatMatMultOp<A,B>
  operator*(double alpha, MatMatMultOp<A,B> mop){
    mop *= alpha;
    return mop;
  }


  template <typename A, typename B>
  MatMatMultOp<A,B>
  operator/(MatMatMultOp<A,B> mop, double alpha){
    mop /= alpha;
    return mop;
  }


  template <typename A, typename B>
  MatMatMultOp<MatrixScaleOp, MatMatMultOp<A,B>>
  operator*(Matrix &m2, MatMatMultOp<A,B> op){
    using mso = MatrixScaleOp;
    return MatMatMultOp<mso, MatMatMultOp<A,B>>(mso(m2,1), op);
  }


  inline
  MatMatMultOp<MatrixScaleOp, MatrixScaleOp>
  operator*(Matrix &m1, MatrixScaleOp m2){
    using mso = MatrixScaleOp;
    return MatMatMultOp<mso, mso>(mso(m1,1), m2);
  }


  inline
  MatMatMultOp<MatrixScaleOp, MatrixScaleOp>
  operator*(MatrixScaleOp m1, Matrix &m2){
    using mso = MatrixScaleOp;
    return MatMatMultOp<mso, mso>(m1, mso(m2,1));
  }


  inline
  MatVecMultOp<MatrixScaleOp>
  operator*(Matrix &mat, Vector& vec){
    return MatVecMultOp<MatrixScaleOp>(MatrixScaleOp(mat,1), VectorScaleOp(vec,1));
  }


  inline
  MatVecMultOp<MatrixScaleOp>
  operator*(MatrixScaleOp op, Vector& vec){
    return MatVecMultOp<MatrixScaleOp>(op, VectorScaleOp(vec,1));
  }


  template <typename LHS, typename RHS>
  MatVecMultOp< MatMatMultOp<LHS, RHS> >
  operator*(MatMatMultOp<LHS, RHS> op, Vector& vec){
    return MatVecMultOp< MatMatMultOp<LHS, RHS> >(op, VectorScaleOp(vec,1));
  }


  template <typename LHS, typename RHS>
  void
  MatrixAddOp<LHS, RHS>::
  addTo(Matrix& v){
    lhs_.addTo(v);
    rhs_.addTo(v);
  }


  template <typename LHS, typename RHS>
  Matrix
  MatrixAddOp<LHS, RHS>::
  eval(){
    Matrix m;
    m.petscMat() = lhs_.duplicateMat(false);
    addTo(m);
    return m;
  }


  template <typename LHS, typename RHS>
  MatObj
  MatMatMultOp<LHS, RHS>::
  evalNoScale() const{
    // FIXME: no control over matrix construct
    MatReuse use = MAT_INITIAL_MATRIX;
    // double fill = 1.0;
    Mat out;
    MatObj leftMat = lhs_.evalNoScale();
    MatObj rightMat = rhs_.evalNoScale();
    MatMatMult(leftMat.get(), rightMat.get(), use, PETSC_DEFAULT, &out);
    return MatObj(out, true);
  }


  template <typename LHS, typename RHS>
  Matrix
  MatMatMultOp<LHS, RHS>::
  eval() const{
    MatObj noscale = evalNoScale();
    Mat mat = noscale.release();
    MatScale(mat, scaleFactor());
    Matrix ret;
    ret.petscMat() = mat;
    return ret;
  }


  template <typename LHS, typename RHS>
  double
  MatMatMultOp<LHS, RHS>::
  scaleFactor() const{
    double scale = 1.0;
    scale *= lhs_.scaleFactor();
    scale *= rhs_.scaleFactor();
    return scale;
  }


  template <typename LHS, typename RHS>
  MatrixProxy&
  MatrixProxy::
  operator+=(MatrixAddOp<LHS, RHS> op){
    op.addTo(mat_);
    return *this;
  }

  template <typename LHS, typename RHS>
  MatrixAddOp<LHS, RHS>
  operator*(MatrixAddOp<LHS, RHS> addOp, double alpha){
    addOp.lhs_ *= alpha;
    addOp.rhs_ *= alpha;
    return addOp;
  }


  template <typename LHS, typename RHS>
  MatrixAddOp<LHS, RHS>
  operator*(double alpha, MatrixAddOp<LHS, RHS> addOp){
    addOp.lhs_ *= alpha;
    addOp.rhs_ *= alpha;
    return addOp;
  }


  template <typename LHS, typename RHS>
  MatrixAddOp<LHS, RHS>
  operator/(MatrixAddOp<LHS, RHS> addOp, double alpha){
    addOp.lhs_ /= alpha;
    addOp.rhs_ /= alpha;
    return addOp;
  }


  template <typename LHS, typename RHS>
  MatrixProxy&
  MatrixProxy::
  operator=(MatMatMultOp<LHS, RHS> op){
    // Evaluate the product without scaling
    MatObj res = op.evalNoScale();
    Mat m = res.release();

    // Evlauate combined scaling factor
    double scale = op.scaleFactor();

    // Scale the matrix product
    MatScale(m, scale);

    // Get ride of the old mat
    if (mat_.petscMat())
      MatDestroy(&mat_.petscMat());

    // Give ownership of the new matrix to mat_
    mat_.petscMat() = m;

    return *this;
  }


  template <typename LHS, typename RHS>
  VecObj
  MatMatMultOp<LHS, RHS>::
  vecMultiplyNoScale(VecObj const& v) const{
    VecObj v1 = rhs_.vecMultiplyNoScale(v);
    VecObj v2 = lhs_.vecMultiplyNoScale(v1);
    return v2;
  }


  template <typename LHS>
  Vector::
  Vector(MatVecMultOp<LHS> const& op){
    *this = op.eval();
  }


  MatrixScaleOp operator*(double alpha, Matrix &mat);
  MatrixScaleOp operator*(Matrix &mat, double alpha);
  MatrixScaleOp operator/(Matrix &mat, double alpha);
  MatrixScaleOp operator*(MatrixScaleOp s, double alpha);
  MatrixScaleOp operator/(MatrixScaleOp s, double alpha);
  MatrixScaleOp operator*(double alpha, MatrixScaleOp s);
  MatrixScaleOp operator/(double alpha, MatrixScaleOp s);


  std::ostream& operator<<(std::ostream& stream, MatrixSlice const& slice);


} // namespace Petscpp


#endif /* PMATRIX_H_ */
