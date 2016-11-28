#ifndef PVECTOR_H_
#define PVECTOR_H_

#include "MatrixProxy.h"
#include <Eigen/Dense>
#include "IndexSet.h"
#include <vector>

struct _p_Vec;
typedef struct _p_Vec* Vec;

namespace Petscpp{

  class Vector;
  template <typename lhs, typename rhs> class VectorAddOp;


  /*! Proxy class for lazy arithmetic operations on Vector
   *
   * Do not manually utilize this class.
   */
  class VectorProxy{
  public:
    VectorProxy(Vector& vec);

    // This operation is only valid if (other != vec_)
    Vector& operator+=(Vector const& other);

    template <typename LHS, typename RHS>
    VectorProxy& operator+=(VectorAddOp<LHS, RHS> other);

    void addTo(Vector& vec);

  private:
    Vector &vec_;
  };


  /*! Proxy class for accessing sections of a Vector
   *
   * Do not manually utilize this class.
   */
  class VectorSlice
  {
  public:
    VectorSlice(Vector& vec, std::vector<int> const& index);

    template <size_t N>
    VectorSlice(Vector& vec, std::array<int, N> const& index)
      : vec_(vec)
    {
      indices_.insert(indices_.end(), index.begin(), index.end());
    }

    Vector& operator=(double val);
    Vector& operator+=(Eigen::VectorXd const& vec);

    OmMatrix_d<double> values() const;

  private:
    Vector &vec_;
    std::vector<int> indices_;
  };


  class VectorSliceConst
  {
  public:
    VectorSliceConst(Vector const& vec, std::vector<int> const& index);

    template <size_t N>
    VectorSliceConst(Vector const& vec, std::array<int, N> const& index)
      : vec_(vec)
    {
      indices_.insert(indices_.end(), index.begin(), index.end());
    }

    OmMatrix_d<double> values() const;

  private:
    Vector const& vec_;
    std::vector<int> indices_;
  };


  // -----------------------------------------------------------------
  // Experimental

  class VectorElement
  {
  public:
    VectorElement(Vector& vec, int const& index);

    Vector& operator=(double val);
    Vector& operator-=(double val);
    double value() const;

  private:
    Vector &vec_;
    int index_;
  };

  class VectorElementConst
  {
  public:
    VectorElementConst(Vector const& vec, int const& index);
    double value() const;

  private:
    Vector const& vec_;
    int index_;
  };


  // -----------------------------------------------------------------

  /*! Proxy class for scaling of vectors
   *
   * Do not manually utilize this class.
   */
  class VectorScaleOp
  {
  public:
    VectorScaleOp(Vector& vec, double alpha) : vec_(vec), alpha_(alpha) { }

    VectorScaleOp& operator*=(double alpha);
    void addTo(Vector& vec);

    // create a duplicate of the vector
    Vec duplicateVec() const;

    // Evaluate the operator and return a new Vector
    Vector eval() const;

    friend VectorScaleOp operator*(VectorScaleOp s, double alpha);
    friend VectorScaleOp operator/(VectorScaleOp s, double alpha);
  private:
    Vector &vec_;
    double alpha_;
  };


  /* Proxy class for chaining arithmetic add operations
   *
   * Do not manually utilize this class.
   */
  template <typename LHS, typename RHS>
  class VectorAddOp
  {
  public:
    VectorAddOp(LHS v1, RHS v2);

    // Recursively add LHS and RHS to v
    void addTo(Vector& v);

    // create a duplicate of the left-most internal child vector
    Vec duplicateVec() const;

    // Untested
    template <typename LHS_, typename RHS_>
    friend VectorAddOp<LHS_, RHS_> operator*(VectorAddOp<LHS_, RHS_> addOp, double alpha);

    // Untested
    template <typename LHS_, typename RHS_>
    friend VectorAddOp<LHS_, RHS_> operator*(double alpha, VectorAddOp<LHS_, RHS_> addOp);
    //   addOp.lhs_ *= alpha;
    //   addOp.rhs_ *= alpha;
    //   return addOp;
    // }

  private:
    LHS lhs_;
    RHS rhs_;
  };


  // Untested
  template <typename LHS_, typename RHS_>
  VectorAddOp<LHS_, RHS_> operator*(VectorAddOp<LHS_, RHS_> addOp, double alpha){
    addOp.lhs_ *= alpha;
    addOp.rhs_ *= alpha;
    return addOp;
  }


  // Untested
  template <typename LHS_, typename RHS_>
  VectorAddOp<LHS_, RHS_> operator*(double alpha, VectorAddOp<LHS_, RHS_> addOp){
    addOp.lhs_ *= alpha;
    addOp.rhs_ *= alpha;
    return addOp;
  }


  /* For now, specialize VectorAddOp to disable Vector as template
     parameters: using Vector as template parameter will result in
     deep copy of Vector's interals */
  template <typename LHS>
  class VectorAddOp<LHS, Vector> {};

  template <typename RHS>
  class VectorAddOp<Vector, RHS> {};

  // -----------------------------------------------------------------

  class VectorScatterProxyConst
  {
  public:
    // If indexSet is uninitialized, scatter targets the entire vectror
    VectorScatterProxyConst(Vector const& vec, IndexSet indexSet)
      : vec_(vec), indexSet_(indexSet)
    { }

    VectorScatterProxyConst(VectorScatterProxyConst && other)
      : vec_(other.vec_), indexSet_(other.indexSet_)
    { }

    Vector const& petscVec() const{
      return petscVec();
    }

    friend class VectorScatterProxy;

  private:
    Vector const& vec_;
    IndexSet indexSet_;
  };


  // TODO: remove unnecessary copies of IndexSet
  class VectorScatterProxy
  {
  public:
    // If indexSet is uninitialized, scatter targets the entire vectror
    VectorScatterProxy(Vector& vec, IndexSet indexSet)
      : vec_(vec), indexSet_(indexSet)
    { }

    VectorScatterProxy(VectorScatterProxy && other)
      : vec_(other.vec_), indexSet_(other.indexSet_)
    { }

    Vector& operator=(VectorScatterProxy && other);
    Vector& operator=(VectorScatterProxyConst && other);
    Vector& petscVec();

  private:
    Vector &vec_;
    IndexSet indexSet_;
  };


  // -----------------------------------------------------------------

  enum class VectorType{
    Sequential, MPI, Standard
  };

  enum class SizeType{
    Global, Local
  };


  /*! Wrapper for Petsc's Vec
   *
   *
   */
  class Vector{
  public:
    Vector();
    Vector(size_t size, VectorType type = VectorType::MPI,
           SizeType t = SizeType::Global);
    Vector(size_t size, std::vector<int> const& ghostNodes,
           SizeType t = SizeType::Global);
    Vector(size_t size, double alpha);
    Vector(Vector const& other);
    Vector& operator=(Vector const& other);
    Vector& operator=(Vector && other);

    template <typename LHS, typename RHS>
    Vector(VectorAddOp<LHS,RHS> const& op){
      vector_ = op.duplicateVec();
      noAlias() += op;
    }

    Vector(VectorScaleOp const& op);

    ~Vector();

    /*! Access vector entrie(s)
     *
     * Each processor can only access its own share. Indices are in
     * terms of global index numbers. Negative indices are
     * ignored. Calling processor must respect what it has access to
     *
     * \sa localSize
     * \sa startingIndex
     *
     * \return Proxies to this vector with 'value()' accessors
     */
    VectorElement operator[](int index);
    VectorSlice operator()(std::initializer_list<int> const& indices);

    VectorElementConst operator[](int index) const;
    VectorSliceConst operator()(std::initializer_list<int> const& indices) const;

    template <size_t N>
    VectorSlice operator()(std::array<int, N> const& indices){
      return VectorSlice(*this, indices);
    }

    template <size_t N>
    std::vector<double> operator()(std::array<int, N> const& indices) const{
      std::vector<double> ret(indices.size());
      VecGetValues(vector_, indices.size(), indices.data(), &ret[0]);
      return ret;
    }

    /*! Return a proxy which can perform arithmetic operations */
    VectorProxy noAlias();

    /* Return a proxy object which may be used for setting values
       outside local range */
    VectorScatterProxy operator()(IndexSet indexSet){
      return VectorScatterProxy(*this, indexSet);
    }

    /* Return a proxy object which may be used for setting values
       outside local range */
    VectorScatterProxyConst operator()(IndexSet indexSet) const{
      return VectorScatterProxyConst(*this, indexSet);
    }

    /* Experimental
     *
     * Make required call to petsc's assembly functions if values have
     * been set, and we wish to get values.
     *
     * If a single processor calls this, _ALL_ other must call it too
     */
    void assemble();

    /*! Total size of the vector */
    int globalSize() const;

    /*! Size of vector on current processor */
    int localSize() const;

    /*! Return the starting index calling processor has access to */
    int startingIndex() const;

    friend Vector duplicate(Vector const& vec);
    friend class VectorProxy;
    friend class VectorSlice;
    friend class VectorSliceConst;
    friend class VectorElement;
    friend class VectorElementConst;
    friend class VectorScaleOp;
    template <typename lhs, typename rhs> friend class VectorAddOp;

    Vec& petscVec();
    Vec const& petscVec() const;

    bool isGhostPadded() const;

    // Experimental
    template <typename T >
    class VectorIterator : public std::iterator<std::random_access_iterator_tag,
                                                T,
                                                std::ptrdiff_t,
                                                T*,
                                                T&>
    {
    public:

      using diff_type =
        typename std::iterator<std::random_access_iterator_tag, T>::difference_type;

      VectorIterator(Vec vec, int size, bool ) : vec_(vec), index_(size), count_(nullptr) { }


      VectorIterator(Vec vec, int *count)
        : vec_(vec), index_(0), count_(count)
      {
        // std::cout << "VectorIterator(Vector)" << std::endl;
        if (count_ != nullptr){
          VecGetArray(vec_, &data_);
          (*count_)++;
        }
        else{
          VecGetLocalSize(vec, &index_);
        }
      }

      VectorIterator(VectorIterator && other)
        : vec_(nullptr), index_(0), count_(nullptr), data_(nullptr)
      {
        std::swap(data_,  other.data_);
        std::swap(index_, other.index_);
        std::swap(vec_,   other.vec_);
        std::swap(count_, other.count_);
      }

      VectorIterator(VectorIterator const& other)
        : data_(other.data_), index_(other.index_),
          count_(other.count_), vec_(other.vec_)
      {
        // std::cout << "VectorIterator(cosnt& other)" << std::endl;
        if (count_ != nullptr)
          (*count_)++;
      }

      VectorIterator& operator=(VectorIterator const& other)
      {
        data_  = other.data_;
        index_ = other.index_;
        vec_   = other.vec_;
        count_ = other.count_;

        if (count_)
          ++(*count_);
        return *this;
      }

      // VectorIterator& operator=(VectorIterator && other)
      // {
      //   data_  = other.data_;
      //   index_ = other.index_;
      //   size_  = other.size_;
      //   vec_   = other.vec_;
      //   count_ = other.count_;
      //   return *this;
      // }

      ~VectorIterator(){
        // std::cout << "~VectorIterator()" << std::endl;
        if ( (count_ != nullptr) && (vec_ != nullptr) ){
          --(*count_);
          if ( *count_ == 0){
            VecRestoreArray(vec_, &data_);
            // std::cout << "restoring" << std::endl;
          }
        }
      }

      // pre-increment
      VectorIterator& operator++(){
        ++index_;
        return *this;
      }

      // post-increment
      VectorIterator operator++(int){
        VectorIterator tmp(*this);
        ++index_;
        return tmp;
      }

      // pre-decrement
      VectorIterator& operator--(){
        --index_;
        return *this;
      }

      // post-decrement
      VectorIterator operator--(int){
        VectorIterator tmp(*this);
        --index_;
        return tmp;
      }

      bool operator==(VectorIterator const& other) const{
        return ( (index_ == other.index_) && (vec_ == other.vec_) );
      }

      bool operator!=(VectorIterator const& other) const{
        // return !(operator==(other));
        return ( (index_ != other.index_) && (vec_ == other.vec_) );
      }

      T& operator*() const{
        return data_[index_];
      }

      T& operator->() const{
        return data_[index_];
      }

      friend inline VectorIterator<T> operator+(VectorIterator<T> const& other,
                                                diff_type n){
        VectorIterator<T> ret(other);
        ret.index_ += n;
        return ret;
      }

      diff_type operator-(VectorIterator const& rhs) const{
        return index_-rhs.index_;
      }

      bool operator<(VectorIterator<T> const& other) const{
        return index_ < other.index_;
      }

      // cast to const-iterator
      // operator VectorIterator<T const>()
      // { return VectorIterator<T const>(vec_, count_); }

    private:
      Vec vec_;
      int index_;
      int *count_;
      PetscReal* data_;
    };

    using iterator = VectorIterator<PetscReal>;
    using const_iterator = VectorIterator<PetscReal const>;
    using citerator = const_iterator;

    inline iterator begin(){
      return iterator(vector_, &iteratorCount_);
    }

    inline citerator cbegin() const{
      return const_iterator(vector_, &iteratorCount_);
    }

    inline iterator end(){
      return iterator(vector_, nullptr);
    }

    inline citerator cend() const{
      return const_iterator(vector_, nullptr);
    }

  private:
    // Add current vector to other (used by proxies)
    void addTo(Vector &vec);

    // Duplicate internal vector properties (used by proxies)
    Vec duplicateVec() const;

    Vec vector_;
    mutable int iteratorCount_;
  };


  /* Write the local content of calling processor to octave format
   *
   * For ghost-padded vectors, all processors with access to the input
   * vector must call this function
   *
   * \param v vector to print
   * \param filename name of the file to write to
   * \param varName variable name assigned to the content of data
   */
  void octavePrintLocal(Vector const& v, std::string const& filename,
                        std::string const& varName = "data");


  // -----------------------------------------------------------------

  // Needs to be deprecated with the introduction of Vector iterator
  class VectorArray
  {
  public:
    VectorArray(Vector &vec) : pvec_(vec){
      VecGetArray(pvec_.petscVec(), &data_);
    }

    ~VectorArray(){
      VecRestoreArray(pvec_.petscVec(), &data_);
    }

    VectorArray(VectorArray const&) = delete;
    VectorArray& operator=(VectorArray const&) = delete;

    /* Return the size of the array. This corresponds to the local
       size of the vector calling processor has access to */
    int size() const{
      return pvec_.localSize();
    }

    double const& operator[](int i) const{ return data_[i]; }
    double& operator[](int i) { return data_[i]; }

  private:
    PetscScalar *data_;
    Vector &pvec_;
  };


  // -----------------------------------------------------------------
  // Template member impls

  template <typename LHS, typename RHS>
  VectorProxy&
  VectorProxy::
  operator+=(VectorAddOp<LHS, RHS> op){
    op.addTo(vec_);
    return *this;
  }


  template <typename LHS, typename RHS>
  void
  VectorAddOp<LHS, RHS>::
  addTo(Vector& vec){
    lhs_.addTo(vec);
    rhs_.addTo(vec);
  }


  template <typename LHS, typename RHS>
  Vec
  VectorAddOp<LHS, RHS>::
  duplicateVec() const{
    return lhs_.duplicateVec();
  }


  template <typename LHS, typename RHS>
  VectorAddOp<LHS,RHS>::
  VectorAddOp(LHS v1, RHS v2) : lhs_(v1), rhs_(v2) { }


  // -----------------------------------------------------------------
  // Free functions

  // Explicitly writing out the acceptable operator to avoid abiguities
  template<typename LHS, typename B, typename C>
  VectorAddOp<LHS, VectorAddOp<B,C> > operator+(LHS&& lhs, VectorAddOp<B,C>&& rhs){
    return VectorAddOp<LHS, VectorAddOp<B,C> >( lhs, rhs );
  }

  template<typename A, typename B, typename RHS>
  VectorAddOp< VectorAddOp<A,B>, RHS > operator+(VectorAddOp<A,B>&& lhs, RHS&& rhs){
    return VectorAddOp< VectorAddOp<A,B>, RHS >( lhs, rhs );
  }

  template<typename B, typename C>
  VectorAddOp<Vector, VectorAddOp<B,C> > operator+(Vector&& lhs, VectorAddOp<B,C>&& rhs){
    return VectorAddOp<Vector, VectorAddOp<B,C> >( lhs, rhs );
  }

  template<typename A, typename B>
  VectorAddOp<VectorAddOp<A,B>, Vector > operator+(VectorAddOp<A,B>&& lhs, Vector&& rhs){
    return VectorAddOp<VectorAddOp<A,B>, Vector >( lhs, rhs );
  }

  template<typename A, typename B>
  VectorAddOp<VectorAddOp<A,B>, Vector > operator+(VectorAddOp<A,B>&& lhs,
                                                   VectorScaleOp&& rhs){
    return VectorAddOp<VectorAddOp<A,B>, VectorScaleOp >( lhs, rhs );
  }

  template<typename A, typename B>
  VectorAddOp<VectorScaleOp, VectorAddOp<A,B> >operator+(VectorScaleOp && lhs,
                                                         VectorAddOp<A,B>&& rhs){
    return VectorAddOp<VectorScaleOp, VectorAddOp<A,B> >( lhs, rhs );
  }

  inline VectorAddOp<VectorScaleOp, VectorScaleOp> operator+(VectorScaleOp && lhs,
                                                             VectorScaleOp && rhs){
    return VectorAddOp<VectorScaleOp, VectorScaleOp>( lhs, rhs );
  }
  // -----------------------------------------------------------------

  // experimental
  template<typename LHS>
  VectorAddOp<LHS, VectorScaleOp> operator-(LHS&& lhs, Vector& rhs){
    return VectorAddOp<LHS, VectorScaleOp>( lhs, VectorScaleOp(rhs, -1) );
  }

  template<typename LHS>
  VectorAddOp<LHS, VectorScaleOp> operator-(LHS&& lhs, VectorScaleOp rhs){
    rhs *= -1;
    return VectorAddOp<LHS, VectorScaleOp>( lhs, rhs );
  }

  // Vector ddition and subtraction
  // TODO: Remove unnecessary scaling
  VectorAddOp<VectorScaleOp, VectorScaleOp> operator+(Vector& lhs, Vector& rhs);
  VectorAddOp<VectorScaleOp, VectorScaleOp> operator-(Vector& lhs, Vector& rhs);

  // Vector scalar multiplication
  VectorScaleOp operator*(double alpha, Vector &vec);
  VectorScaleOp operator*(Vector &vec, double alpha);

  // Untested
  inline VectorScaleOp operator*(VectorScaleOp s, double alpha){
    s.alpha_ *= alpha;
    return s;
  }

  // Untested
  inline VectorScaleOp operator/(VectorScaleOp s, double alpha){
    s.alpha_ /= alpha;
    return s;
  }


  // Vector Negation
  VectorScaleOp operator-(Vector &vec);

  // Return new vector with duplicate 'settings' of input vector
  Vector duplicate(Vector const& vec1);

  // -----------------------------------------------------------------


} // namespace Petscpp


#endif /* PVECTOR_H_ */
