#ifndef PMATRIXPROXY_H_
#define PMATRIXPROXY_H_

#include <initializer_list>
#include <vector>
#include <ostream>
#include <array>

// Temporary experminatal matrix for experimenting with MatrixProxy interface
// (row-major, stack-allocated)
template <typename T, size_t rowCount_, size_t colCount_>
class OmMatrix
{
  static_assert(std::is_arithmetic<T>::value, "Not Implemented for non-arithmetic types");
public:
  OmMatrix() : data_{{0}} { }

  static constexpr size_t rowCount() { return rowCount_; };
  static constexpr size_t colCount() { return colCount_; };

  T const& operator()(size_t row, size_t col) const{
    return data_[col + row*colCount_];
  }

  std::array<T, rowCount_*colCount_> const& data() const{
    return data_;
  }

private:
  std::array<T, rowCount_*colCount_> data_;
};


// Temporary experminatal matrix for experimenting with MatrixProxy interface
// (row-major, dynamically allocated) 
template <typename T>
class OmMatrix_d
{
  static_assert(std::is_arithmetic<T>::value, "Not Implemented for non-arithmetic types");
public:
  OmMatrix_d(size_t m, size_t n) : rowCount_(m), colCount_(n) {
    data_.resize(m*n);
  }

  constexpr size_t rowCount() const { return rowCount_; };
  constexpr size_t colCount() const { return colCount_; };

  T& operator()(size_t row, size_t col){
    return data_[col + row*colCount_];
  }

  T const& operator()(size_t row, size_t col) const{
    return data_[col + row*colCount_];
  }

  std::vector<double> const& data() const{
    return data_;
  }

  std::vector<double>& data(){
    return data_;
  }

private:
  int rowCount_;
  int colCount_;
  std::vector<T> data_;
};


template <typename T>
std::ostream& operator<<(std::ostream& stream, OmMatrix_d<T> const& matrix){
  for (size_t i=0; i<matrix.rowCount(); ++i){
    for (size_t j=0; j<matrix.colCount(); ++j){
      stream << matrix(i,j) << " ";
    }
    stream << "\n";
  }
  return stream;
}


#endif /* PMATRIXPROXY_H_ */
