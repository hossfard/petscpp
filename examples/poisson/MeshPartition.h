#ifndef MESHPARTITION_H_
#define MESHPARTITION_H_

#include <vector>
#include <array>
#include <Petscpp/Vector.h>


struct MeshData{
  using vec_int = std::vector<int>;
  using vec_dbl = std::vector<double>;

  std::vector<vec_dbl> vertices;
  std::vector<vec_int> connection;
  std::vector<int> vertexAttribute;
};


/*
 *
 */
struct PartitionedMeshData{
  using vec_int = std::vector<int>;

  std::array<Petscpp::Vector,2> coords;
  std::vector<vec_int> connection;
  std::vector<int> ghostNodes;

  // locally owned vertices, in original labeling
  std::vector<int> localVertices;
  Petscpp::IndexSet cellProcAssignment;
};


/*
 * Returned vertex connectivity is relabeled such that its indices
 * correspond to the locally accessible vector indices of the
 * coordinate vectors. Indices of shared vertices correspond to the
 * ghost-padded index of the local vector.
 *
 * Current implementation requires rank 0 to call this function
 */
PartitionedMeshData partitionMesh_sender(MeshData const& data);

/* Receive mesh partition
 *
 * Current implementation requires all participating ranks besides 0
 * to call this function
 */
PartitionedMeshData partitionMesh_receiver();


/* Send specified data to specified global index locations
 *
 * This function must be called by all processors having access to the
 * underlying data. The sum of the local data sizes passed by all
 * processors must be equal to the (local index sizes) times
 * blockSizes:
 *
 * \sum_p(localData.size()) == \sum_p(localVertices.size()*blockSize);
 *
 * \param localVertices vertices own by calling processor. Must be in
 *    original labeling
 * \param localData data assigned to current processor
 * \param blockSize constant block size of the local data vector
 * \return MPI vector with data at specified locations
 */
template <typename T>
Petscpp::Vector
distributeVertexAttribute(std::vector<T> const& localData,
                          std::vector<int> const& localVertices,
                          int blockSize){

  Petscpp::Vector newDataVec(localVertices.size()*blockSize,
                              Petscpp::VectorType::Sequential);
  Petscpp::Vector oldDataVec(localData.size(), Petscpp::VectorType::MPI,
                              Petscpp::SizeType::Local);

  int offset = oldDataVec.startingIndex();
  for (int i=offset; i<offset+oldDataVec.localSize(); ++i){
    oldDataVec[i] = localData[i-offset];
  }
  oldDataVec.assemble();

  newDataVec(Petscpp::IndexSet()) =
    oldDataVec(Petscpp::IndexSet(localVertices, blockSize));
  newDataVec.assemble();

  return newDataVec;
}


/* Send specified data to specified global indices
 *
 * Distribute vertex attributes to specified global index locations,
 * and return calling processor's share of data. This is a convenience
 * function for \sa distributeVertexAttribute().
 *
 * \sa distributeVertexAttribute()
 */
template <typename T>
std::vector<T>
distributeVertexAttribute_vec(std::vector<T> const& localData,
                              std::vector<int> const& localVertices,
                              int blockSize){

  Petscpp::Vector data = distributeVertexAttribute(localData, localVertices, blockSize);
  std::vector<T> ret(data.localSize());
  int offset = data.startingIndex();
  int j = 0;
  for (int i=offset; i<offset+data.localSize(); ++i){
    ret[j] = data[i].value() ;
    ++j;
  }
  return ret;
}


/* TODO:
 *
 */
template <typename T>
std::vector<T>
distributeVertexAttribute_vec(std::vector<T> const& localData,
                              std::vector<int> const& localVertices,
                              int blockSize,
                              std::vector<int> const& ghostNodes){

  std::vector<int> ret =
    distributeVertexAttribute_vec(localData, localVertices, blockSize);

  // Store the required ghost-padded values
  Petscpp::Vector scatVec(localVertices.size(), ghostNodes, Petscpp::SizeType::Local);
  int offset = scatVec.startingIndex();
  int k=0;
  for (int i=offset; i<offset+scatVec.localSize(); ++i){
    scatVec[i] = ret[k];
    ++k;
  }

  ret.resize(scatVec.localSize()+ghostNodes.size());
  offset = scatVec.startingIndex();
  VecGhostUpdateBegin(scatVec.petscVec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(scatVec.petscVec(), INSERT_VALUES, SCATTER_FORWARD);
  k = 0;
  for (size_t i=offset; i<offset+scatVec.localSize()+ghostNodes.size(); ++i){
    ret[k] = scatVec[i].value();
    ++k;
  }

  return ret;
}


/* TODO: comments
 *
 * \param newNumbering location on the new vector for the data to be sent to
 * \param data data to be sent to other processors
 * \param newSize local size of new vector on current processor
 */
template <typename T>
Petscpp::Vector
distributeCellAttribute(std::vector<T> const& data,
                        Petscpp::IndexSet const& newNumbering,
                        int newSize, int blockSize){

  int const oldSize = (int)data.size();

  Petscpp::Vector newDataVec(newSize, Petscpp::VectorType::MPI,
                              Petscpp::SizeType::Local);
  Petscpp::Vector oldDataVec(oldSize, Petscpp::VectorType::Sequential);

  for (size_t i=0; i<data.size(); ++i)
    oldDataVec[i] = data[i];

  oldDataVec.assemble();

  newDataVec(Petscpp::IndexSet(newNumbering, blockSize)) =
    oldDataVec(Petscpp::IndexSet());

  return newDataVec;
}



#endif /* MESHPARTITION_H_ */
