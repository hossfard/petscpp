#include "Adjacency.h"
#include <map>
#include <algorithm>

class Edge
{
public:
  Edge(size_t a, size_t b){
    auto minmax = std::minmax(a,b);
    begin_ = minmax.first;
    end_ = minmax.second;
  }

  size_t start() const{
    return begin_;
  };

  size_t end() const{
    return end_;
  }

  friend struct EdgeCompare;

private:
  size_t begin_;
  size_t end_;
};


struct EdgeCompare{
  bool operator()(Edge const& e1, Edge const& e2) const{
    if (e1.begin_ != e2.begin_)
      return e1.begin_ < e2.begin_;
    else
      return e1.end_ < e2.end_;
  }
};

/* Mapping between edges and parent faces */
class EdgeElements
{
public:
  using FaceId = size_t;
  using FaceSet = std::set<FaceId>;

  EdgeElements() : maxFaceId_(0) { }

  void insert(Edge edge, FaceId id){
    FaceSet& set = edgeFaceMap_[edge];
    set.insert(id);
    maxFaceId_ = std::max(maxFaceId_, id);
  }

  void insert(int begin, int end, FaceId id){
    insert(Edge(begin, end), id);
  }

private:
  friend std::vector< std::set<int> >
  neighborList(EdgeElements const& edges);

  std::map<Edge, FaceSet, EdgeCompare> edgeFaceMap_;
  size_t maxFaceId_;
};


std::vector< std::set<int> >
neighborList(EdgeElements const& edges){
  std::vector< std::set<int> > ret(edges.maxFaceId_+1);

  for (auto it=edges.edgeFaceMap_.begin(); it!=edges.edgeFaceMap_.end(); ++it){
    std::set<size_t> const& faces = it->second;
    std::vector<size_t> vec;
    std::copy(faces.begin(), faces.end(), std::back_inserter(vec));
    for (size_t i=0; i<vec.size(); ++i){
      for (size_t j=0; j<vec.size(); ++j){
        if (i == j)
          continue;
        ret[ vec[j] ].insert(vec[i]);
      }
    }
  }
  return ret;
}


std::vector< std::set<int> >
openfem::
neighborList2d(std::vector< std::vector<int> > const& links){

  // Construct a list of all edges
  EdgeElements edgeMap;
  for (size_t i=0; i<links.size(); ++i){
    std::vector<int> const& vec = links[i];

    // Map the edges to the current parent
    for (size_t m=0; m<vec.size()-1; ++m)
      edgeMap.insert(vec[m], vec[m+1], i);
    edgeMap.insert(vec.front(), vec.back(), i);
  }

  // Return the list of neighbors
  return neighborList(edgeMap);
}
