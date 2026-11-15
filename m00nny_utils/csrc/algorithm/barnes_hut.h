#ifndef LIBM00NNY_BARNES_HUT_H
#define LIBM00NNY_BARNES_HUT_H

#include <cstdint>
#include <vector>

namespace m00nny {

/*
A space-partitioning tree over N points in d dimensions: each cell is a cube split into 2^d children
(a quadtree for d = 2, an octree for d = 3). Every cell keeps its point count and centre of mass, so a cell far
from a point can stand in for all of its points (Barnes-Hut): a cell of width w whose centre of mass is at
distance r from the point, and which does not contain the point, is summarized when w < angle * r.
angle = 0 never summarizes, which makes the sums exact.
*/
class BarnesHutTree
{
public:
    // points: n_points x dimension, row major; must outlive the tree.
    BarnesHutTree(const double* points, int64_t n_points, int64_t dimension);

    /*
    The t-SNE repulsion at every point i, with w_ij = 1 / (1 + |y_i - y_j|^2) over all j != i:
        Z[i] = sum_j w_ij,      F[i] = sum_j w_ij^2 (y_i - y_j)
    F: n_points x dimension, row major; Z: n_points. Points are processed in parallel.
    */
    void repulsion(double angle, double* F, double* Z) const;

    int64_t size() const { return (int64_t)this->cells.size(); }

private:
    struct Cell
    {
        int64_t first_child = -1;       // the 2^d children are consecutive from here; -1: a leaf
        int64_t count = 0;              // points in the cell
        std::vector<int64_t> points;    // a leaf's points: one, or several when they coincide or depth ran out
    };

    const double* y;
    int64_t N, d, n_children;
    std::vector<Cell> cells;
    std::vector<double> centre;         // cells x d, geometric centre
    std::vector<double> half;           // cells, half width of the cube
    std::vector<double> mass;           // cells x d, centre of mass

    int64_t add_cell(const double* centre, double half);
    int64_t child_of(int64_t cell, const double* p) const;
    void insert(int64_t point);
};

} // namespace m00nny

#endif // LIBM00NNY_BARNES_HUT_H
