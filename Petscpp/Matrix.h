#ifndef PMATRIX_H_
#define PMATRIX_H_

#include <type_traits>
#include <initializer_list>
#include <ostream>
#include "MatrixProxy.h"
#include <Eigen/Dense>

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

    // TODO: careful with row-major/col-major
    Matrix& operator=(OmMatrix_d<double> const& values);

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
    OmMatrix_d<double> toMatrix() const;

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
     *           10% of the max(M,N) if specified as zero
     */
    Matrix(int M, int N, bool globalSizes = true, int nzCount = 0);

    Matrix& operator=(Matrix const& other);
    Matrix& operator=(Matrix && other);
    ~Matrix();

    void setSize(int m, int n);

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
    Mat mat_;
  };


  inline std::ostream& operator<<(std::ostream& stream, MatrixSlice const& slice){
    stream << slice.toMatrix() << "\n";
    return stream;
  }


} // namespace Petscpp


#endif /* PMATRIX_H_ */
