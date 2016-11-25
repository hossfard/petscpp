#ifndef PUTILITY_H_
#define PUTILITY_H_

#include <vector>
#include "Petscpp.h"
#include <mpi.h>

namespace Petscpp{

  template <typename T>
  struct MpiType{ };

  template <>
  struct MpiType<int>
  {
    static const MPI_Datatype type = MPI_INT;
  };

  template <>
  struct MpiType<double>
  {
    static const MPI_Datatype type = MPIU_SCALAR;
  };

  
  /*! Convenience function for sending an integral type to target processor */
  template <typename T>
  void send(T value, int target, int tag=0){
    MPI_Send(&value, 1, MpiType<T>::type, target,tag, PETSC_COMM_WORLD);
  }


  /*! Convenience function for receiving an integral type from specified processor */
  template <typename T>
  T receive(int source, int tag=0){
    T val;
    MPI_Status status;
    MPI_Recv(&val, 1, MpiType<T>::type, source,tag, PETSC_COMM_WORLD, &status);
    return val;
  }


  /*! Return specified processor's appropriate share of a quantity */
  inline
  int divide(int quantity, int procId = Petscpp::procId(),
             int procCount = Petscpp::procCount()){
    return quantity/procCount + ((quantity % procCount) > procId);
  }


  /*! Divide the vector equally per partition and send it to other processors
   *
   * Other processors _must_ call receive_vec<T> to receive the vector
   * to avoid locking
   */
  template <typename T>
  std::vector<T>
  divideAndSend(std::vector<T> const& data){
    int const procCount = Petscpp::procCount();
    int const dataCount_p0 = divide(data.size(), Petscpp::procId());
    int offset = dataCount_p0;

    for (int p=0; p<procCount; ++p){
      if (p == Petscpp::procId())
        continue;

      int localDataCount = divide(data.size(), p);
      std::vector<T> localData(data.begin()+offset, data.begin()+offset+localDataCount);
      send<int>(localData.size(), p);
      MPI_Send(&localData[0], (int)localData.size(), MpiType<T>::type, p,0, PETSC_COMM_WORLD);
      offset += localDataCount;
    }

    return std::vector<T>(data.begin(), data.begin()+dataCount_p0);
  }


  /*! Send the contents of vector to target processor
   * Receiving processor must call receive_vec()
   */
  template <typename T>
  void send_vec(std::vector<T> const& data, int targetProc, int tag = 0){
    send<int>((int)data.size(), targetProc);
    MPI_Send(&data[0], (int)data.size(), MpiType<T>::type,
             targetProc, tag, PETSC_COMM_WORLD);
  }


  /*! Receive vector sent by send_vec() */
  template <typename T>
  std::vector<T>
  receive_vec(int source, int tag = 0){
    int const size = receive<int>(source);
    std::vector<T> data(size);
    MPI_Status status;
    MPI_Recv(&data[0], size, MpiType<T>::type, source, tag,
             PETSC_COMM_WORLD, &status);
    return data;
  }


  // Structure for representing blocked data passed using MPI
  template <typename T>
  class BlockVector
  {
  public:
    using iterator = typename std::vector<T>::iterator;
    using citerator = typename std::vector<T>::const_iterator;

    BlockVector() = default;

    template <typename Container>
    BlockVector(std::vector<Container> const& data, int begin, int end){
      int const N = end-begin;
      data_.reserve(N*4);
      sIndex_.reserve(N+1);
      blockSize_.reserve(N);

      sIndex_.push_back( 0 );
      for (int i=begin; i<end; ++i){
        Container const& v = data[i];
        for (auto d : v)
          data_.push_back(d);

        int const s = (int)v.size();
        blockSize_.push_back(s);
        sIndex_.push_back( sIndex_.back() + s );
      }
      sIndex_.pop_back();
    }

    std::pair<iterator, iterator> range(int i){
      return std::make_pair(begin(i), end(i));
    }

    std::pair<citerator, citerator> crange(int i) const{
      return std::make_pair(cbegin(i), cend(i));
    }

    // Number of blocks
    size_t size() const{
      return blockSize_.size();
    }

    // Size of block
    int blockSize(int i) const{
      return blockSize_[i];
    }

  private:
    template <typename U>
    friend BlockVector<U> receive_bvec(int source);

    iterator begin(int i){
      return data_.begin()+sIndex_[i];
    }

    iterator end(int i){
      return begin(i)+blockSize_[i];
    }

    citerator cbegin(int i) const{
      return data_.cbegin()+sIndex_[i];
    }

    citerator cend(int i) const{
      return cbegin(i)+blockSize_[i];
    }

    std::vector<T> data_;
    std::vector<int> sIndex_;
    std::vector<int> blockSize_;
  };


  template <typename T>
  BlockVector<T>
  receive_bvec(int source){
    BlockVector<T> ret;
    ret.data_ = receive_vec<T>(source);
    ret.blockSize_ = receive_vec<int>(source);
    ret.sIndex_ = receive_vec<int>(source);

    return ret;
  }


  namespace helper{
    template <typename Container>
    std::vector<typename Container::value_type >
    vectorize(std::vector<Container> const& vec, size_t begin, size_t end){
      using T = typename Container::value_type;

      std::vector<T> ret;
      // Some approximate for the final vector capacity
      ret.reserve(vec.size()*4);

      for (size_t i=begin; i<end; ++i){
        for (auto const& y : vec[i])
          ret.push_back(y);
      }
      return ret;
    }
  }


  template <typename Container>
  void
  send_bvec(std::vector<Container> const& vec, int target){
    size_t const N = vec.size();
    using value_type = typename Container::value_type;
    std::vector<value_type> const data = helper::vectorize(vec, 0, N);

    std::vector<int> sIndex;
    std::vector<int> blockSizes;
    sIndex.reserve(N+1);
    blockSizes.reserve(N);
    sIndex.push_back( 0 );
    for (auto const& v : vec){
      int const s = (int)v.size();
      blockSizes.push_back(s);
      sIndex.push_back( sIndex.back() + s );
    }
    sIndex.pop_back();

    send_vec(data, target);
    send_vec(blockSizes, target);
    send_vec(sIndex, target);
  }

} /* Petscpp */


#endif /* PUTILITY_H_ */
