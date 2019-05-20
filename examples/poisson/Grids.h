#ifndef GRIDS_H_
#define GRIDS_H_

#include <array>
#include <vector>
#include <ostream>

/*! Uniform rectangular grid
 *
 */
namespace openfem{

  class UniformQuadGrid
  {
  public:
    using array2d = std::array<double, 2>;
    using array4i = std::array<int, 4>;

    UniformQuadGrid(array2d rangex, size_t nx,
                    array2d rangey, size_t ny);

    std::vector< array2d > const& vertices() const;
    std::vector< array4i > const& connectivity() const;

    // number of divisions of the grid in the horizontal direction
    size_t nx() const;
    // number of divisions of the grid in the vertical direction
    size_t ny() const;

    size_t vertexCount() const;
    size_t cellCount() const;

    array2d xRange() const;
    array2d yRange() const;

  private:
    std::vector< array2d > vertices_;
    std::vector< array4i > connection_;
    array2d rangex_, rangey_;
    size_t nx_;
    size_t ny_;
  };

  void print(UniformQuadGrid const& grid, std::ostream &stream = std::cout);

  // Return the list of neighbors of cells of input grid
  std::vector< std::vector<int> > adjacency(openfem::UniformQuadGrid const& grid);

}


namespace experimental{

  template <typename Derived, size_t Dimension, size_t cell_size>
  struct BaseGrid{

    using coordArray = std::array<double, Dimension>;
    using cellArray = std::array<int, cell_size>;

    std::vector<coordArray> const& vertices() const{
      return vertices_;
    }

    std::vector<cellArray> const& connectivity() const{
      return connection_;
    }

    size_t vertexCount() const{
      return vertices_.size();
    }

    size_t cellCount() const{
      return connection_.count();
    }

  protected:
    std::vector<coordArray> vertices_;
    std::vector<cellArray> connection_;
  };


  /*! Uniform rectangular grid
   *
   */
  class UniformQuadGrid_ : public BaseGrid<UniformQuadGrid_, 2, 4>
  {
  public:
    using array2d = std::array<double, 2>;
    using array4i = std::array<int, 4>;

    UniformQuadGrid_(array2d rangex, size_t nx,
                     array2d rangey, size_t ny);

    // number of divisions of the grid in the horizontal direction
    size_t nx() const;
    // number of divisions of the grid in the vertical direction
    size_t ny() const;

  private:
    array2d rangex_, rangey_;
    size_t nx_;
    size_t ny_;
  };


  template <typename T, size_t dimension, size_t cellSize>
  void print(BaseGrid<T,dimension, cellSize> const& grid,
             std::ostream &stream = std::cout){

    auto const& vertices = grid.vertices();
    stream << "vertices: \n";
    for (size_t i=0; i<vertices.size(); ++i){
      stream << i << ": {";
      for (auto const& vertex : vertices[i] )
        stream << vertex << ", ";

      stream <<  "\b\b}\n";
    }

    stream << "connections: \n";
    auto const& con = grid.connectivity();
    for (size_t i=0; i<con.size(); ++i){
      stream << i << ": {";
      for (auto const& id : con[i])
        stream << id << ", ";

      stream << "\b\b}\n";
    }
  }
}


#endif /* GRIDS_H_ */
