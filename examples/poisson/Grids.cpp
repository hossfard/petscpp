#include "Grids.h"
#include "Utility.h"
#include "Petscpp/Petscpp.h"

using array2d = std::array<double, 2>;
using array4i = std::array<int, 4>;
using namespace openfem;

UniformQuadGrid::
UniformQuadGrid(array2d rangex, size_t nx,
                array2d rangey, size_t ny)
  : nx_(nx), ny_(ny)
{
  rangex_ = openfem::sort(rangex);
  rangey_ = openfem::sort(rangey);

  vertices_.reserve((nx+1)*(ny+1));
  connection_.reserve(nx*ny);

  // vertex coordinates
  double const dx = (rangex[1]-rangex[0])/nx;
  double const dy = (rangey[1]-rangey[0])/ny;
  for (size_t i=0; i<ny+1; ++i)
    for (size_t j=0; j<nx+1; ++j)
      vertices_.push_back({{rangex[0] + j*dx, rangey[0] + i*dy}});

  // vertex connections
  for (int i=0; i<(int)ny; ++i){
    for (int j=0; j<(int)nx; ++j){
      connection_.push_back({{j + i*(int)(nx+1),
              j + i*(int)(nx+1) + 1,
              j + i*(int)(nx+1) + 1 + (int)(nx+1),
              j + i*(int)(nx+1) + (int)(nx+1)  }});
    }
  }
}


std::vector< array2d > const&
UniformQuadGrid::
vertices() const{
  return vertices_;
}


std::vector< array4i > const&
UniformQuadGrid::
connectivity() const{
  return connection_;
}


size_t
UniformQuadGrid::
nx() const{
  return nx_;
}


size_t
UniformQuadGrid::
ny() const{
  return ny_;
}


size_t
UniformQuadGrid::
vertexCount() const{
  return vertices_.size();
}


size_t
UniformQuadGrid::
cellCount() const{
  return connection_.size();
}


std::array<double, 2>
UniformQuadGrid::
xRange() const{
  return rangex_;
}


std::array<double, 2>
UniformQuadGrid::
yRange() const{
  return rangey_;
}


void
openfem::
print(UniformQuadGrid const& grid, std::ostream &stream /*= std::cout*/){

  auto const& vertices = grid.vertices();
  stream << "vertices: \n";
  for (size_t i=0; i<vertices.size(); ++i)
    stream << i << ": {" << vertices[i][0] << ", " << vertices[i][1] << "}\n";

  stream << "connections: \n";
  auto const& con = grid.connectivity();
  for (size_t i=0; i<con.size(); ++i)
    stream << i << ": {"
           << con[i][0] << ", "
           << con[i][1] << ", "
           << con[i][2] << ", "
           << con[i][3] << "}\n";
}


std::vector< std::vector<int> >
openfem::
adjacency(openfem::UniformQuadGrid const& grid){
  std::vector< std::array<int, 4> > const& con = grid.connectivity();

  std::vector< std::vector<int> > ret(con.size());
  int const nx = (int)grid.nx();
  int const ny = (int)grid.ny();

  auto isCorner = [nx,ny](int cell){
    if ( (cell == 0) || (cell == (nx-1)) || (cell == (nx*ny-1)) || (cell == nx*ny-nx) )
      return true;
    else
      return false;
  };
  auto isBottom = [nx](int cell){
    if ( (cell >= 0) && (cell < nx) )
      return true;
    else
      return false;
  };
  auto isLeft = [nx](int cell){
    if ( (cell%nx) == 0 )
      return true;
    else
      return false;
  };
  auto isRight = [nx](int cell){
    if ( ((cell+1)%nx) == 0 )
      return true;
    else
      return false;
  };
  auto isTop = [nx,ny](int cell){
    if ( (cell < nx*ny) && (cell >= (nx*ny-nx)) )
      return true;
    else
      return false;
  };

  for (size_t i=0; i<con.size(); ++i){

    if (isCorner(i)){
      if (isLeft(i) && isBottom(i)){
        ret[i].push_back(i+1);
        ret[i].push_back(i+nx);
      }
      else if (isRight(i) && isBottom(i)){
        ret[i].push_back(i-1);
        ret[i].push_back(i+nx);
      }
      else if (isLeft(i) && isTop(i)){
        ret[i].push_back(i-nx);
        ret[i].push_back(i+1);
      }
      else if (isRight(i) && isTop(i)){
        ret[i].push_back(i-1);
        ret[i].push_back(i-nx);
      }
    }
    else if (isBottom(i)){
      ret[i].push_back(i-1);
      ret[i].push_back(i+1);
      ret[i].push_back(i+nx);
    }
    else if (isLeft(i)){
      ret[i].push_back(i-nx);
      ret[i].push_back(i+1);
      ret[i].push_back(i+nx);
    }
    else if (isRight(i)){
      ret[i].push_back(i-nx);
      ret[i].push_back(i-1);
      ret[i].push_back(i+nx);
    }
    else if (isTop(i)){
      ret[i].push_back(i-nx);
      ret[i].push_back(i-1);
      ret[i].push_back(i+1);
    }
    else{
      ret[i].push_back(i-nx);
      ret[i].push_back(i-1);
      ret[i].push_back(i+1);
      ret[i].push_back(i+nx);
    }
  }

  return ret;
}


experimental::UniformQuadGrid_::
UniformQuadGrid_(array2d rangex, size_t nx,
                 array2d rangey, size_t ny){
  rangex_ = openfem::sort(rangex);
  rangey_ = openfem::sort(rangey);

  vertices_.reserve((nx+1)*(ny+1));
  connection_.reserve(nx*ny);

  // vertex coordinates
  double const dx = (rangex[1]-rangex[0])/nx;
  double const dy = (rangey[1]-rangey[0])/ny;
  for (size_t i=0; i<ny+1; ++i)
    for (size_t j=0; j<nx+1; ++j)
      vertices_.push_back({{rangex[0] + j*dx, rangey[0] + i*dy}});

  // vertex connections
  for (int i=0; i<(int)ny; ++i){
    for (int j=0; j<(int)nx; ++j){
      connection_.push_back({{j + i*(int)(nx+1),
              j + i*(int)(nx+1) + 1,
              j + i*(int)(nx+1) + 1 + (int)(nx+1),
              j + i*(int)(nx+1) + (int)(nx+1)  }});
    }
  }
}
