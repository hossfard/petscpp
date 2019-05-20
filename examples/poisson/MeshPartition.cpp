#include "MeshPartition.h"
#include "Adjacency.h"
#include <Petscpp/Petscpp.h>
#include <Petscpp/Vector.h>
#include <Petscpp/BitArray.h>
#include <Petscpp/Map.h>
#include <Petscpp/MatrixPartitioning.h>
#include <Petscpp/Utility.h>

#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>

/*
 * Example approach for partitioning 2-dimensional meshes
 */


struct MeshCon{
  std::vector<int> connectivity; // vectorized connectivity
  std::vector<int> vertexCount;  // number of vertices of given cell
  int conBlockSize;              // blocksize of the connectivity vector (maximum value)
};


struct VertData{
  std::vector<double> coords;
  std::vector<int> coordCount;
  int blockSize;                // blocksize of the coord vector
};


struct PartitionedMeshData_p{
  using vec_int = std::vector<int>;
  using vec_dbl = std::vector<double>;

  std::vector< vec_dbl > vertices;
  std::vector< vec_int > connection;
  std::vector<int> ghostNodes;
  std::vector<int> localVertices; // locally owned vertices, in original labeling
  Petscpp::IndexSet cellProcAssignment;

  std::vector<int> eqnum;
};


/* Vectorize a vector in the interval [begin,end) */
template <typename T>
std::vector<T> vectorize(std::vector<std::vector<T>> const& vec,
                         size_t begin, size_t end){
  std::vector<T> ret;
  // Some approximate for the final vector capacity
  ret.reserve(vec.size()*4);

  for (size_t i=begin; i<end; ++i){
    for (auto const& y : vec[i])
      ret.push_back(y);
  }

  return ret;
}


/* Vectorize a vector in the interval [begin,end), using a fixed block
 * size
*/
template <typename T>
std::vector<T> vectorize(std::vector< std::vector<T>> const& vec,
                         size_t begin, size_t end, int blockSize){

  std::vector<T> ret;
  ret.reserve((end-begin)*blockSize);

  int idx = 0;
  for (size_t i=begin; i<end; ++i){
    ret.insert(ret.begin() + idx*blockSize, vec[i].begin(), vec[i].end());
    ++idx;
  }

  return ret;
}


/* Vectorize a vector of arrays in the interval [begin,end) */
template <typename T, size_t N>
std::vector< T >
vectorize(std::vector< std::array<T,N> > const& vec, size_t begin, size_t end){
  std::vector<T> ret;
  ret.reserve(vec.size()*N);

  for (size_t i=begin; i<end; ++i){
    for (auto const& y : vec[i])
      ret.push_back(y);
  }
  return ret;
}


// Return a vector containing the sizes of a vector<vector<T>>
template <typename T>
std::vector<int>
subvectorSizes(std::vector< std::vector<T> > const& vecvec, int begin, int end){
  std::vector<int> sizes;
  sizes.reserve(end-begin);

  for (int i=begin; i<end; ++i)
    sizes.push_back((int)vecvec[i].size());

  return sizes;
}


Petscpp::MatrixPartitioning
createAdjacencyMatrix(int localCellCount, int globalCellCount,
                      std::vector<int> const& rowIndices,
                      std::vector<int> const& colIndices){

  Mat adj;

  /* MatCreateMPIAdj takes ownership of the row,col data. Manually
     allocate the data on the freestore */

  int *ii, *jj;
  PetscMalloc(sizeof(int)*(int)rowIndices.size(), &ii);
  PetscMalloc(sizeof(int)*(int)colIndices.size(), &jj);

  std::copy(rowIndices.begin(), rowIndices.end(), ii);
  std::copy(colIndices.begin(), colIndices.end(), jj);

  // MatCreateMPIAdj takes ownership of ii and jj -- do not free
  PetscErrorCode err =
    MatCreateMPIAdj(PETSC_COMM_WORLD, localCellCount, globalCellCount,
                    ii, jj, PETSC_NULL, &adj);

  MatSetOption(adj, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);

  // TODO: Does MatCreateMPIAdj clean up if it fails?
  if (err != 0){
    PetscFree(ii);
    PetscFree(jj);
  }

  return Petscpp::MatrixPartitioning(std::move(adj));
}


Petscpp::MatrixPartitioning
createAdjacencyMatrix(int globalCellCount,
                      Petscpp::BlockVector<int> const& adjData){

  Petscpp::MatrixPartitioning ret(globalCellCount);

  for (int i=0; i<(int)adjData.size(); ++i)
    ret.setNeighbors(i, std::vector<int>(adjData.crange(i).first,
                                         adjData.crange(i).second));

  return ret;
}


Petscpp::MatrixPartitioning
createAdjacencyMatrix(int globalVertCount, MeshCon const& meshCon){

  /* MatCreateMPIAdj takes ownership of the row,col data. Manually
     allocate the data on the freestore */
  int *ii, *jj;

  int const cellCount = meshCon.vertexCount.size();
  int const iiCount = cellCount + 1;
  int const jjCount = std::accumulate(meshCon.vertexCount.begin(),
                                      meshCon.vertexCount.end(), 0);
  PetscMalloc(sizeof(int)*iiCount, &ii);
  PetscMalloc(sizeof(int)*jjCount, &jj);

  ii[0] = 0;
  size_t offset = 0;
  for (int i=0; i<cellCount; ++i){
    int vCount = meshCon.vertexCount[i];
    std::vector<int> verts(meshCon.connectivity.begin()+offset,
                           meshCon.connectivity.begin()+offset+vCount);
    std::sort(verts.begin(), verts.end());
    std::copy(verts.begin(), verts.end(), jj+offset);
    ii[i+1] = ii[i] + vCount;
    offset += vCount;
  }

  Mat mesh;
  Mat dual;
  int const commonnodes = 2;    // ****

  MatCreateMPIAdj(MPI_COMM_WORLD, cellCount, globalVertCount, ii, jj, nullptr, &mesh);
  MatMeshToCellGraph(mesh, commonnodes, &dual);
  MatDestroy(&mesh);

  return Petscpp::MatrixPartitioning(std::move(dual));
}

struct CellPartitionData{
  // New connectivity table
  std::vector< std::vector<int> > connectivity;

  // Processor ID assigned to local cells
  Petscpp::IndexSet newNumbering;
};

/* Return calling processor's share of vertex connectivity data
 *
 */
CellPartitionData
partitionMesh_share(MeshCon const& meshCon,
                    Petscpp::BlockVector<int> const& adjacencyData,
                    int globalCellCount){

  CellPartitionData partitionData;

  // Create adjacency matrix
  Petscpp::MatrixPartitioning partitioningMatrix =
    createAdjacencyMatrix(globalCellCount, adjacencyData);

  // Get the processor number assigned to each cell
  Petscpp::IndexSet partitionedIndexSet = partitioningMatrix.partitioning();

  /* Move cells to right processors */

  // New number of cells assigned to each processor
  std::vector<int> newProcCellCount = partitionedIndexSet.partitionCount();
  // New number of cells assigned to current processor
  int const newCellCount = newProcCellCount[Petscpp::procId()];

  /* New index numbering of the cells. This along with
   * 'newProcCellCount' says which processor has which cells. Use this
   * along with scatter to rearrange vector entries
   */
  partitionData.newNumbering = partitionToNumbering(partitionedIndexSet);

  // -----------------------------------------------------------------
  auto newVCount =
    distributeCellAttribute(meshCon.vertexCount, partitionData.newNumbering,
                            newCellCount, 1);

  int newConVertCount = 0;
  int offset=newVCount.startingIndex();
  for (int i=offset; i<offset+newVCount.localSize(); ++i){
    newConVertCount += newVCount[i].value();
  }

  int blockSize = meshCon.conBlockSize;
  auto newMeshVerts =
    distributeCellAttribute(meshCon.connectivity,
                            partitionData.newNumbering,
                            newCellCount*blockSize, blockSize);

  // Restructure the connectivity array
  Petscpp::VectorArray vcountArray(newVCount);
  int idx = 0;
  for (int i=newMeshVerts.startingIndex();
       i<newMeshVerts.startingIndex()+newProcCellCount[Petscpp::procId()]*blockSize-1;
       i+=blockSize){

    int const vCount = vcountArray[idx];

    std::vector<int> indices;
    for (int c=0; c<vCount; ++c){
      indices.push_back( (int)newMeshVerts[i+c].value() );
    }

    partitionData.connectivity.push_back(indices);
    ++idx;
  }

  return partitionData;
}


// Called by all processors getting mesh data
std::vector< std::vector<int> >
partitionMesh_share(MeshCon const& meshCon, int globalVertCount){

  Petscpp::MatrixPartitioning partitioningMatrix =
    createAdjacencyMatrix(globalVertCount, meshCon);

  // The new processor-id local cells have been assigned to
  Petscpp::IndexSet partitionedIndexSet = partitioningMatrix.partitioning();

  // print(partitionedIndexSet);

  // New number of cells assigned to each processor
  std::vector<int> newProcCellCount = partitionedIndexSet.partitionCount();
  // New number of cells assigned to current processor
  int const newCellCount = newProcCellCount[Petscpp::procId()];

  /* New index numbering of the cells. This along with
   * 'newProcCellCount' says which processor has which cells. Use this
   * along with scatter to rearrange vector entries
   */
  Petscpp::IndexSet newNumbering = partitionToNumbering(partitionedIndexSet);

  // print(newNumbering);

  // -----------------------------------------------------------------
  // Keep information about blocksize of the vertex ids
  Petscpp::Vector newVCount =
    distributeCellAttribute(meshCon.vertexCount, newNumbering,
                            newCellCount, 1);

  int newConVertCount = 0;
  int offset=newVCount.startingIndex();
  for (int i=offset; i<offset+newVCount.localSize(); ++i){
    newConVertCount += newVCount[i].value();
  }

  // The block-size used for the vertices
  int const blockSize = meshCon.conBlockSize;
  auto newMeshVerts = distributeCellAttribute(meshCon.connectivity,
                                              newNumbering,
                                              newCellCount*blockSize,
                                              blockSize);

  // Restructure the connectivity array
  std::vector< std::vector<int> > ret;
  Petscpp::VectorArray vcountArray(newVCount);
  int idx = 0;
  offset = newMeshVerts.startingIndex();
  for (int i=offset; i<offset+newCellCount*blockSize-1; i+=blockSize){
    int const vCount = vcountArray[idx];
    std::vector<int> indices;
    for (int c=0; c<vCount; ++c){
      indices.push_back( (int)newMeshVerts[i+c].value() );
    }
    ret.push_back(indices);
    ++idx;
  }

  return ret;
}


template <typename T>
size_t
maxSubvectorSize(std::vector< std::vector<T> > const& vecvec){
  size_t maxLength = 0;
  for (auto const& vec : vecvec)
    maxLength = std::max<size_t>(maxLength, vec.size());

  return maxLength;
}


/* Send each processor's initial share of vertex connectivity
 *
 * Given a mesh composed of vertices, and the connection between the
 * vertices, sends each processor its initial estimate of mesh
 * connectivity
 *
 * Connectivity is vectorized with a blocksize corresponding to the
 * maximum vertex count of cells
 *
 * Only the processor doing the partitioning should call this function
 */
MeshCon
broadcastMeshCon(MeshData const& meshData){

  int maxBlockSize = (int)maxSubvectorSize(meshData.connection);

  // Estimate current processor's number of cells (sender)
  int const localCellCount = Petscpp::divide(meshData.connection.size(), 0);

  // Assign other processors' _initial_ share of grid
  std::vector< std::vector<int> > const& con = meshData.connection;

  MeshCon ret;
  ret.connectivity = vectorize(con, 0, localCellCount, maxBlockSize);
  ret.conBlockSize = maxBlockSize;

  for (int i=0; i<localCellCount; ++i){
    ret.vertexCount.push_back(con[i].size());
  }

  // int indexOffset = ret.startIndex.back() + con[localCellCount-1].size();
  int k = localCellCount;

  MPI_Bcast(&maxBlockSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  for (int p=1; p<Petscpp::procCount(); ++p){

    // Number of cells initially assigned to processor 'p'
    int const procCellCount = Petscpp::divide(meshData.connection.size(), p);

    // Number of vertices for each cell (for a uniform 2d grid, always 4)
    // std::vector<int> startIndex;
    std::vector<int> vertexCount;
    // vectorized grid data

    // Vectorize the connectivity, and store the number of vertices per cell
    std::vector<int> localGrid = vectorize(con, k, k+procCellCount, maxBlockSize);
    for (int i=0; i<procCellCount; ++i){
      vertexCount.push_back( con[k].size() );
      // indexOffset = sindex + con[k].size();
    }

    // Send the vectorized connectivity data to the designated processor
    Petscpp::send_vec(vertexCount, p);
    Petscpp::send_vec(localGrid, p);

    k += procCellCount;
  }

  // Return this processors share of the connectivity
  return ret;
}


MeshCon
receiveMeshCon(){
  MeshCon ret;

  // Block-size of connectivity table
  int conBlockSize = 0;
  MPI_Bcast(&conBlockSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  ret.conBlockSize = conBlockSize;

  // number of vertices of each cell (always 4 for uniform 2d grids)
  ret.vertexCount = Petscpp::receive_vec<int>(0);

  // size of the local mesh vector
  ret.connectivity = Petscpp::receive_vec<int>(0);

  return ret;
}


/* Create and share initial distribution of mesh
 *
 * Must be called only by the processor initiating partitioning
 *
 */
Petscpp::BlockVector<int>
broadcastAdjacency(MeshData const& meshData){

  if (Petscpp::procId() != 0 )
    std::cerr << "broadcastAdjacency was written assuming that proc 0 is the caller\n";

  // Create adjacency data for the full mesh connectivity
  std::vector< std::set<int> > adj = openfem::neighborList2d(meshData.connection);

  // Global cell count
  size_t const cellCount = meshData.connection.size();

  // Current processor's assigned cells
  size_t const proc0CellCount = Petscpp::divide(cellCount, 0);

  size_t offset = proc0CellCount;
  for (int pId=1; pId<Petscpp::procCount(); ++pId){
    // pId's processor's share of cells
    int const procCellCount = Petscpp::divide(cellCount, pId);

    // Get row and column data for the partition matrix
    std::vector< std::set<int> > subjec(adj.begin()+offset,
                                        adj.begin()+procCellCount+offset);
    Petscpp::send_bvec(subjec, pId);

    offset += procCellCount;
  }

  return Petscpp::BlockVector<int>(adj, 0, proc0CellCount);
}


Petscpp::BlockVector<int> receiveAdjacency(int source){
  return Petscpp::receive_bvec<int>(source);
}


/* Called only by processor distributing the mesh data
 *
 */
VertData
broadcastVertCoords(MeshData const& meshData){

  int const procCount = Petscpp::procCount();
  int const vertexCount_p0 = Petscpp::divide(meshData.vertices.size(), Petscpp::procId());

  int maxBlockSize = (int)maxSubvectorSize(meshData.vertices);
  MPI_Bcast(&maxBlockSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  int offset = vertexCount_p0;

  for (int p=0; p<procCount; ++p){
    if (p == Petscpp::procId())
      continue;

    int localVertexCount = Petscpp::divide(meshData.vertices.size(), p);
    std::vector<double> coords = vectorize(meshData.vertices, offset,
                                           offset+localVertexCount, maxBlockSize);

    Petscpp::send_vec(coords, p);
    std::vector<int> sizes = subvectorSizes(meshData.vertices, offset,
                                            offset+localVertexCount);
    Petscpp::send_vec(sizes, p);

    offset += localVertexCount;
  }

  VertData ret;
  ret.coords = vectorize(meshData.vertices, 0, vertexCount_p0, maxBlockSize);
  ret.coordCount = subvectorSizes(meshData.vertices, 0, vertexCount_p0);
  ret.blockSize = maxBlockSize;

  return ret;
}


VertData
receiveVertCoords(int sender = 0){
  VertData ret;
  MPI_Bcast(&ret.blockSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  ret.coords = Petscpp::receive_vec<double>(sender);
  ret.coordCount = Petscpp::receive_vec<int>(sender);

  return ret;
}


/* Return indices of vertices assigned to calling process
 *
 * Must be called by all processes
 */
std::vector<int>
distributeVertices(int globalVertCount,
                   PartitionedMeshData_p const& gridData){

  using namespace Petscpp;

  BitArray bitArray(globalVertCount);
  int maxProcVertCount = globalVertCount / Petscpp::procCount();
  int procId = Petscpp::procId();
  int procCount = Petscpp::procCount();
  bool isLastProc = (procId == (procCount-1));

  // Receive data from previous rank
  MPI_Status status;
  if ( procId != 0){
    MPI_Recv(bitArray.petscBT(), PetscBTLength(globalVertCount), MPI_CHAR,
             procId-1, 0, PETSC_COMM_WORLD, &status);
  }

  // last processor
  int localVertCount = 0;
  if ( isLastProc ){
    for (int i=0; i<globalVertCount; ++i){
      if ( bitArray.value(i) == false ){
        ++localVertCount;
      }
    }
    maxProcVertCount = localVertCount;
  }

  std::vector<int> localVertices;
  localVertices.reserve(maxProcVertCount);
  // if not last processor
  if ( !isLastProc ){
    for (size_t i=0; i<gridData.connection.size(); ++i){
      for (auto v : gridData.connection[i]){
        if (bitArray.getAndSet(v) == false){
          localVertices.push_back(v);
          // Perhaps processors it's wiser to let processors take
          // local all their vertices instead?
          if ((int)localVertices.size() >= maxProcVertCount )
            goto out;
        }
      }
    }
  out:;
  }
  else{  // last processor gets the remainder
    for (int i=0; i<globalVertCount; ++i){
      if (bitArray.value(i) == false)
        localVertices.push_back(i);
    }
  }

  // Send bit mask on to next processor
  if ( !isLastProc ){
    MPI_Send(bitArray.petscBT(), PetscBTLength(globalVertCount),
             MPI_CHAR, Petscpp::procId()+1, 0, PETSC_COMM_WORLD);
  }

  return localVertices;
}


/*
 * All functions getting mesh data should call this function
 */
void
partitionVertices_share(int globalVertCount, PartitionedMeshData_p &gridData,
                        VertData const& localCoords){

  // Vertices assigned to current processor based on partitioned mesh
  std::vector<int> localVertices = distributeVertices(globalVertCount, gridData);
  gridData.localVertices = localVertices;

  // Create mapping from specified indices to vector index numbering (0,1,2,..)
  Petscpp::Map appOrdering(localVertices);

  // Vectorize for labeling of vertex indices
  std::vector<int> vectorizedMesh = vectorize(gridData.connection, 0,
                                              gridData.connection.size());

  // Map mesh connectivity indices to Petsc ordering
  appOrdering.toPetsc(vectorizedMesh);

  /* Re-structure the mesh, now written in petsc numbering
   *
   * With this, we can now write to vector/matrices refering to
   *  locally owned portions of vectors/matrices
   */
  size_t k=0;
  for (size_t i=0; i<gridData.connection.size(); ++i){
    size_t vCount =  gridData.connection[i].size();
    std::vector<int> vec;
    for (size_t j=0; j<vCount; ++j, ++k){
      vec.push_back(vectorizedMesh[k]);
    }
    gridData.connection[i] = vec;
  }

  Petscpp::Vector newVertsVec =
    distributeVertexAttribute(localCoords.coords, localVertices, localCoords.blockSize);

  Petscpp::Vector newVertsCoordCount =
    distributeVertexAttribute(localCoords.coordCount, localVertices, 1);

  // -----------------------------------------------------------------

  appOrdering.toPetsc(localVertices);

  gridData.vertices.clear();
  gridData.vertices.reserve(localVertices.size());
  // gridData.vertices.resize(localVertices.size());

  int idx = 0;
  for (PetscReal const& vCount : newVertsCoordCount){
    std::vector<double> vec;
    for (int c=0; c<(int)vCount; ++c){
      vec.push_back(newVertsVec[idx].value());
      ++idx;
    }
    gridData.vertices.push_back(vec);
  }

  // Get ghost nodes
  std::map<int, int> localVertMap;
  int padLocation = *(std::max_element(localVertices.begin(), localVertices.end())) + 1;
  for (auto v : localVertices){
    localVertMap[v] = -1;       // flag indices owned by current processor
  }
  for (auto & array : gridData.connection){
    for (auto &v : array){
      auto iter = localVertMap.find(v);
      if (iter != localVertMap.end()){
        if (iter->second != -1) // set location to padded location
          v = iter->second;
      }
      else{
        gridData.ghostNodes.push_back(v);

        localVertMap[v] = padLocation;
        v = padLocation;
        ++padLocation;
      }
    }
  }
}


/* Example function to be performed by one processor: distribute input
 * (global) mesh to other processors
 *
 * This function would be more representative if the grid was an
 * unstructured mesh.
 */
PartitionedMeshData_p distributeMesh_s(MeshData const& meshData){

  PartitionedMeshData_p ret;

  /* Send each processor's initial share of vertex connectivity, and
     return this processor's share
  */
  MeshCon con = broadcastMeshCon(meshData);

  // Send each processor's adjacency data

  // broadcast global cell and vertex counts to all ranks
  int globalCellCount = meshData.connection.size();
  int globalVertCount = meshData.vertices.size();
  MPI_Bcast(&globalCellCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&globalVertCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  // Given the initial connection and adjacency, update the connectivity table

  /* Option 1 -- create the adjacency manually */
  Petscpp::BlockVector<int> adjData = broadcastAdjacency(meshData);
  CellPartitionData partitionData = partitionMesh_share(con, adjData, globalCellCount);
  ret.connection = partitionData.connectivity;
  ret.cellProcAssignment = std::move(partitionData.newNumbering);

  /* Option 2 -- create the adjacency using petsc and dual graphs */
  // ret.connection = partitionMesh_share(con, globalVertCount);

  // Send each processor's share of vertex coordinates
  VertData localCoords = broadcastVertCoords(meshData);

  // Partition the vertices and extract ghost nodes
  partitionVertices_share(globalVertCount, ret, localCoords);

  return ret;
}


// Example function to receive the mesh data, called by all processors
// except the processor initially distributing the data
PartitionedMeshData_p receiveMesh_r(int sender = 0){
  PartitionedMeshData_p ret;

  MeshCon con = receiveMeshCon();

  // Get grid cell count from sending processor (proc 0)
  int globalCellCount = 0;
  int globalVertCount = 0;
  MPI_Bcast(&globalCellCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&globalVertCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  // Option 1 --
  // MeshNeighborData adjacencyData = receiveAdjacency(0);
  // ret.connection = partitionMesh_share_opt1(con, adjacencyData, globalCellCount);

  // Option 2 --
  // ret.connection = partitionMesh_share(con, globalVertCount);

  // Option 3
  Petscpp::BlockVector<int> adjData = receiveAdjacency(0);
  // ret.connection = partitionMesh_share(con, adjData, globalCellCount);
  CellPartitionData partitionData = partitionMesh_share(con, adjData, globalCellCount);
  ret.connection = partitionData.connectivity;
  ret.cellProcAssignment = std::move(partitionData.newNumbering);

  // Receive this processors' vectorized share of vertex coordinates
  VertData localCoords = receiveVertCoords(sender);
  partitionVertices_share(globalVertCount, ret, localCoords);

  return ret;
}


PartitionedMeshData
partitionMesh_sender(MeshData const& meshData){

  PartitionedMeshData_p data = distributeMesh_s(meshData);

  PartitionedMeshData data_;
  data_.connection = data.connection;
  data_.ghostNodes = data.ghostNodes;
  data_.localVertices = data.localVertices;
  data_.cellProcAssignment = std::move(data.cellProcAssignment);
  Petscpp::Vector x_coord(data.localVertices.size(),
                           data.ghostNodes, Petscpp::SizeType::Local);
  Petscpp::Vector y_coord(data.localVertices.size(),
                           data.ghostNodes, Petscpp::SizeType::Local);
  int offset=x_coord.startingIndex();
  int k=0;
  for (int i=offset; i<x_coord.localSize()+offset; ++i){
    x_coord[i] = data.vertices[k][0];
    y_coord[i] = data.vertices[k][1];
    ++k;
  }

  data_.coords[0] = std::move(x_coord);
  data_.coords[1] = std::move(y_coord);

  return data_;
}


PartitionedMeshData
partitionMesh_receiver(){
  PartitionedMeshData_p data = receiveMesh_r();

  PartitionedMeshData data_;
  data_.connection = data.connection;
  data_.ghostNodes = data.ghostNodes;
  data_.localVertices = data.localVertices;
  data_.cellProcAssignment = std::move(data.cellProcAssignment);
  Petscpp::Vector x_coord(data.localVertices.size(),
                           data.ghostNodes, Petscpp::SizeType::Local);
  Petscpp::Vector y_coord(data.localVertices.size(),
                           data.ghostNodes, Petscpp::SizeType::Local);
  int offset=x_coord.startingIndex();
  int k=0;
  for (int i=offset; i<x_coord.localSize()+offset; ++i){
    x_coord[i] = data.vertices[k][0];
    y_coord[i] = data.vertices[k][1];
    ++k;
  }
  data_.coords[0] = std::move(x_coord);
  data_.coords[1] = std::move(y_coord);

  return data_;
}
