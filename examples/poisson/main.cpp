/* Solve the PDE $ \nabla^2 \phi = -2\phi^2 \cos(\pi x) \cos(\pi y) $
 * over $[0,1] \times [0,1]$
 *
 * subjected to the homogeneous boundary conditions $\phi = 1$ along
 * the boundary.
 *
 * The above equation admits the analytical solution
 *   $ \phi = \cos(\pi x) \cdot \cos(\pi y) $.
 *
 * In this example, the domain is descritized using uniform bilinear
 * elements. The mesh partitioning approach in 'meshpartition' example
 * is used in this implementation. To demonstrate distributing vertex
 * attributes using petscpp, the boundary conditions are excluded from
 * the LSE. After solving for the solution, the full solution vector
 * (i.e. including dirichlet boundary conditions) are constructed.
 *
 * Unless otherwise specified in 'petsc_settings.conf', the direct
 * solver 'superlu_dist' is used for solving the LSE (settings in
 * 'petsc_settings.conf' take precedence)
 *
 *
 * Usage:
 *
 * $ mpiexec -n number_of_proc ./poisson
 * $ ./plot_solution.sh number_of_proc
 *
 * Viewing results requires octave
 *
 */

#include <fstream>
#include <Eigen/Dense>
#include <Openfem/Bases.h>
#include <Openfem/Grids.h>
#include <Openfem/DiffOp.h>
#include <Openfem/MeshPartition.h>
#include <Petscpp/Petscpp.h>
#include <Petscpp/Matrix.h>
#include <Petscpp/Vector.h>
#include <Petscpp/Ksp.h>
#include <Petscpp/Utility.h>


// LHS and RHS of the linear system of equations (LSE)
struct LSE{
  Petscpp::Matrix lhs;
  Petscpp::Vector rhs;
};


/* Create the LHS of the equation, and adjust the RHS for the boundary
 * conditions.
 *
 * \param f functor returning localized discretized diff operators
 * \param r functor returning homoegenous boundary conditions
 * \param grid partitioned mesh corresponding to calling processor's share
 * \param eqNum equation number of the vertices owned by calling processor
 * \param N total number of equations assigned to calling processor
 * \param lse structure containing uninitalized LSE
 *
 */
template <typename lhs_functor, typename bc_functor>
void createLHS(lhs_functor f, bc_functor r,
               PartitionedMeshData const& grid,
               std::vector<int> eqNum, int N, LSE &lse);


/* Integrate the right hand side for the LSE
 *
 * \param func functor returning localized RHS integral
 * \param grid partitioned mesh corresponding to calling processor's share
 * \param eqNum equation number of the vertices owned by calling processor
 * \param rhs rhs of the LSE
 */
template <typename Functor>
void
createRHS(Functor func, PartitionedMeshData const& grid,
          std::vector<int> const& eqNum, Petscpp::Vector& rhs);


/* Create the reduced linear system of equations
 *
 * \param lhs_functor functor returning localized discretized diff operators
 * \param rhs_functor functor returning localized RHS integral
 * \param bc_functor functor returning homoegenous boundary conditions
 * \param pmesh partitioned mesh corresponding to calling processor's share
 * \param N total number of equations assigned to calling processor
 * \return reduced linear system of equations
 */
template <typename LHS_functor, typename RHS_functor, typename BC_functor>
LSE createLSE(LHS_functor lhs_functor, RHS_functor rhs_functor,
                      BC_functor bc_functor, PartitionedMeshData const& pmesh,
                      std::vector<int> const& eqNum, int N);


// Solve the linear system of equations (defaults to LU)
Petscpp::Vector solve(LSE const& lse);


// rhs function
double rhs(double x, double y){
  double const pi = 3.14159265359;
  double const pi2 = 9.86960440109;
  return -2*pi2*(std::cos(pi*x)*std::cos(pi*y));
}


// Homogeneous boundary conditions
struct bc_functor{
  double operator()(double /*x*/, double /*y*/) const{
    return 1;
  }
};


/* Return the inner product v1 and v2 */
template <typename Vec>
double innerProduct(Vec v1, Vec v2);


/* Local integral of the rhs function */
template <typename Interpolator>
struct poisson_rhs{
private:
  static size_t const size = Interpolator::size;
  using Vec = Eigen::Matrix<double, size, 1 >;
public:
  Vec operator()(Vec const& X, Vec const& Y) const{
    // rhs_functor rhs;
    auto integrand = [&X, &Y](std::array<double,2> const& gp){
      Interpolator interp;
      Vec q = interp.q(gp);
      double const x = innerProduct(q, X);
      double const y = innerProduct(q, Y);

      double const f = rhs(x,y);
      auto jac = Bases::jacobian(interp, X, Y, gp);
      double detj = jac.determinant();

      return (q * f * detj).eval();
    };
    return openfem::gq_integrate(integrand, GP::gp2x2);
  }
};


/* Assign an equation number to the vertices of the uniform
 * grid. Vertices on the boundary of the grid are assigned a value of
 * -1 to indicate a homogeneous boundary condition
 */
std::vector<int> gridEqNum(openfem::UniformQuadGrid const& grid);


// Have each processor write their portion of mesh data to file
void outputMeshData(PartitionedMeshData const& data);


/* Return full mesh solution vector, including at places where
 *  Dirichlet boundary conditions are imposed
 *
 *  \param condensed solution vector
 *  \param vector containing local equation numbers
 *  \param mdata partitioned mesh data
*/
Petscpp::Vector fullSolution(Petscpp::Vector const& sol,
                              std::vector<int> const& localEqNumbers,
                              PartitionedMeshData const& mdata);


int main(int argc, char *argv[])
{
  Petscpp::App app(argc, argv, "petsc_settings.conf");

  // Create a uniform grid over [0,1]x[0,1]
  int const nx = 20;
  int const ny = 20;
  std::vector<int> eqNum_l;

  // Create and partition a uniform grid over [0,1]x[0,1]
  PartitionedMeshData data;

  // Processor 0 to create mesh and distribute to other procs
  if (Petscpp::procId() == 0){
    openfem::UniformQuadGrid grid({0.0, 1.0}, nx, {0.0, 1.0}, ny);

    // Store in different format for partitioning
    MeshData meshDataActual;
    for (auto x : grid.vertices()){
      std::vector<double> coord = {x[0], x[1]};
      meshDataActual.vertices.push_back( coord );
    }

    for (auto x : grid.connectivity()){
      std::vector<int> con = {x[0], x[1], x[2], x[3]};
      meshDataActual.connection.push_back( con );
    }

    data = partitionMesh_sender(meshDataActual);

    // Assign an equation number to each vertex for the linear system
    // of equationp
    std::vector<int> eqNum_g = gridEqNum(grid);
    eqNum_l = Petscpp::divideAndSend<int>(eqNum_g);
  }
  else{
    data = partitionMesh_receiver();
    eqNum_l = Petscpp::receive_vec<int>(0);
  }

  // local processor's vertex equation number
  eqNum_l = distributeVertexAttribute_vec(eqNum_l, data.localVertices,
                                          1, data.ghostNodes);

  // Write mesh to file
  outputMeshData(data);

  // Size of the matrix system of equations (square) with homogenous
  // boundary conditions taken out
  int numEq = 0;
  for (size_t i=0; i<data.localVertices.size(); ++i){
    if (eqNum_l[i] > -1)
      ++numEq;
  }

  // Create system of equations
  openfem::diffop::Laplacian<Bases::Q1> laplacian_q1;
  LSE sys_petscpp =
    createLSE(laplacian_q1, poisson_rhs<Bases::Q1>(), bc_functor(), data, eqNum_l,
              numEq);


  // Solve system of equations
  Petscpp::Vector sol_petsc = solve(sys_petscpp);

  // Recover full solution (including boundary conditions)
  Petscpp::Vector sol_glob = fullSolution(sol_petsc, eqNum_l, data);


  // Dump the result to file
  std::string const solFilename =
    "output/sol_" + std::to_string(Petscpp::procId()) + ".m";
  Petscpp::octavePrintLocal(sol_glob, solFilename, "sol");

  return 0;
}


Petscpp::Vector fullSolution(Petscpp::Vector const& sol,
                              std::vector<int> const& localEqNumbers,
                              PartitionedMeshData const& mdata){

  // Recover full solution (including boundary conditions)
  Petscpp::Vector sol_glob(mdata.coords[0].localSize(),
                            mdata.ghostNodes, Petscpp::SizeType::Local);
  int offset = mdata.coords[0].startingIndex();
  for (size_t i=0; i<mdata.localVertices.size(); ++i){
    if (localEqNumbers[i] < 0){
      double x = (mdata.coords[0])[i+offset].value();
      double y = (mdata.coords[1])[i+offset].value();
      sol_glob[i+offset] = bc_functor()(x,y);
    }
  }

  std::vector<int> from, to;
  offset = sol_glob.startingIndex();
  for (size_t i=0; i<mdata.localVertices.size(); ++i){
    if (localEqNumbers[i] > -1){
      to.push_back(i+offset);
      from.push_back(localEqNumbers[i]);
    }
  }
  sol_glob(Petscpp::IndexSet(to)) = sol(Petscpp::IndexSet(from));

  return sol_glob;
}


void outputMeshData(PartitionedMeshData const& data){
  // Filenames for coordinates
  std::string const xposFilename = "output/x_" + std::to_string(Petscpp::procId()) + ".m";
  std::string const yposFilename = "output/y_" + std::to_string(Petscpp::procId()) + ".m";

  // Filename for mesh connectivity
  std::string const conFilename =
    "output/mesh_" + std::to_string(Petscpp::procId()) + ".m";

  // Write (local) mesh vertices to file
  Petscpp::octavePrintLocal(data.coords[0], xposFilename, "x");
  Petscpp::octavePrintLocal(data.coords[1], yposFilename, "y");

  // Write (local) mesh connectivty to file
  int offset = data.coords[0].startingIndex();
  std::ofstream meshStream(conFilename.c_str());
  if (meshStream.is_open()){
    meshStream << "con = [...\n";
    for (auto const& cell : data.connection){
      for (auto v : cell){
        meshStream << v-offset+1 << ' ';
      }
      meshStream << '\n';
    }
    meshStream << "];\n";
  }
}


template <typename lhs_functor, typename bc_functor>
void createLHS(lhs_functor f,
               bc_functor r,
               PartitionedMeshData const& grid,
               std::vector<int> eqNum, int N, LSE &lse){
  lse.lhs = Petscpp::Matrix(N,N, false, 10);
  lse.rhs = Petscpp::Vector(N, Petscpp::VectorType::MPI, Petscpp::SizeType::Local);

  using CellVector = Eigen::Matrix<double, 4, 1>;

  auto const& con = grid.connection;
  int offset = grid.coords[0].startingIndex();

  Petscpp::Vector const& X = grid.coords[0];
  Petscpp::Vector const& Y = grid.coords[1];

  for (auto const& cell : con){
    std::array<int, 4> locs = {{ eqNum[cell[0]-offset], eqNum[cell[1]-offset],
                                 eqNum[cell[2]-offset], eqNum[cell[3]-offset]}};
    std::array<int, 4> xlocs = {{ cell[0], cell[1], cell[2], cell[3]}};

    auto xv = X(xlocs);
    auto yv = Y(xlocs);

    CellVector x(&xv.data()[0]);
    CellVector y(&yv.data()[0]);
    auto mat = f(x,y);
    lse.lhs( locs, locs ) += mat;

    // Adjust the rhs for homogeneous bcs
    for (int i=0; i<4; ++i){
      int p = locs[i];
      if (p < 0){
        double bc_val = r(x[i],y[i]);
        for (int j=0; j<4; ++j)
          lse.rhs[ locs[j] ] -= mat(j,i) * bc_val;
      }
    }
  }

  lse.lhs.assemble();
}


template <typename Functor>
void
createRHS(Functor func,
          PartitionedMeshData const& grid,
          std::vector<int> const& eqNum, Petscpp::Vector& rhs){
  auto const& con = grid.connection;
  int offset = grid.coords[0].startingIndex();

  Petscpp::Vector const& X = grid.coords[0];
  Petscpp::Vector const& Y = grid.coords[1];

  for (auto const& cell : con){
    std::array<int, 4> locs = {{ eqNum[cell[0]-offset], eqNum[cell[1]-offset],
                                 eqNum[cell[2]-offset], eqNum[cell[3]-offset]}};
    std::array<int, 4> xlocs = {{ cell[0], cell[1], cell[2], cell[3]}};

    auto xv = X(xlocs);
    auto yv = Y(xlocs);

    Eigen::Vector4d const x(&xv.data()[0]);
    Eigen::Vector4d const y(&yv.data()[0]);

    rhs(locs) += func(x,y);
  }

  rhs.assemble();
}



template <typename LHS_functor, typename RHS_functor, typename BC_functor>
LSE createLSE(LHS_functor lhs_functor,
                      RHS_functor rhs_functor,
                      BC_functor bc_functor,
                      PartitionedMeshData const& pmesh,
                      std::vector<int> const& eqNum, int N){
  LSE sys;
  createLHS(lhs_functor, bc_functor, pmesh, eqNum, N, sys);
  createRHS(rhs_functor, pmesh, eqNum, sys.rhs);
  return sys;
}


Petscpp::Vector
solve(LSE const& lse){
  Petscpp::Ksp ksp;

  /* Use 'superlu_dist' as direct solver (Petsc must configured and
     built with superlu_dist) */
  ksp.setExternalPackage(Petscpp::SolverPackage::SuperLU_dist);

  /* Set options from 'petsc_settings.conf'. Duplicate settings above
     are overwritten with those in 'petsc_settings.conf' */
  ksp.loadFromOptions();

  // Display info regarding preconditioner and solver
  Petscpp::printInfo(ksp);

  return ksp.solve(lse.lhs, lse.rhs);
}


template <typename Vec>
double innerProduct(Vec v1, Vec v2){
  double res = 0;
  for (int i=0; i<v1.size(); ++i)
    res += v1[i] * v2[i];
  return res;
}


std::vector<int> gridEqNum(openfem::UniformQuadGrid const& grid){
  std::vector<int> boundary_indices;
  int const nx = grid.nx();
  int const ny = grid.ny();
  for (int i=0; i<nx+1; ++i){
    boundary_indices.push_back(i);
    boundary_indices.push_back((nx+1)*i);
    boundary_indices.push_back(nx+(nx+1)*i);
    boundary_indices.push_back((nx+1)*(ny+1)-i-1);
  }
  std::vector<int> eqNum(grid.vertexCount());
  for (auto i : boundary_indices){
    eqNum[i] = -1;
  }
  int num = 0;
  for (int& e : eqNum){
    if (e>=0){
      e = num;
      ++num;
    }
  }
  return eqNum;
}
