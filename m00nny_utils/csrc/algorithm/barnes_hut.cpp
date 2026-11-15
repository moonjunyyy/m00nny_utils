#include "algorithm/barnes_hut.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
constexpr int64_t kMaxDepth = 48;       // below this depth a leaf keeps several points instead of splitting
constexpr int64_t kGrain    = 64;       // points per parallel task
}

m00nny::BarnesHutTree::BarnesHutTree(const double* points, int64_t n_points, int64_t dimension)
    : y(points), N(n_points), d(dimension), n_children(int64_t(1) << dimension)
{
    if (dimension < 1 || dimension > 16) throw std::invalid_argument("BarnesHutTree: dimension must be in [1, 16]");
    if (n_points == 0) return;
    // The root: the bounding cube of the points, widened a little so every point lies strictly inside.
    std::vector<double> lo(d, std::numeric_limits<double>::infinity()), hi(d, -std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < N; ++i)
        for (int64_t k = 0; k < d; ++k) { lo[k] = std::min(lo[k], y[i * d + k]); hi[k] = std::max(hi[k], y[i * d + k]); }
    std::vector<double> c(d);
    double h = 0;
    for (int64_t k = 0; k < d; ++k) { c[k] = 0.5 * (lo[k] + hi[k]); h = std::max(h, 0.5 * (hi[k] - lo[k])); }
    h = h * (1 + 1e-6) + std::numeric_limits<double>::min();
    this->cells.reserve(2 * N);
    this->add_cell(c.data(), h);
    for (int64_t i = 0; i < N; ++i) this->insert(i);
}

int64_t
m00nny::BarnesHutTree::add_cell(const double* c, double h)
{
    this->cells.emplace_back();
    this->centre.insert(this->centre.end(), c, c + this->d);
    this->half.push_back(h);
    this->mass.insert(this->mass.end(), this->d, 0.0);
    return (int64_t)this->cells.size() - 1;
}

int64_t
m00nny::BarnesHutTree::child_of(int64_t cell, const double* p) const
{
    int64_t orthant = 0;
    for (int64_t k = 0; k < this->d; ++k)
        if (p[k] >= this->centre[cell * this->d + k]) orthant |= int64_t(1) << k;
    return this->cells[cell].first_child + orthant;
}

void
m00nny::BarnesHutTree::insert(int64_t point)
{
    const double* p = this->y + point * this->d;
    int64_t cell = 0;
    for (int64_t depth = 0;; ++depth)
    {
        // The point is in this cell: update its count and centre of mass.
        Cell& here = this->cells[cell];
        here.count += 1;
        for (int64_t k = 0; k < this->d; ++k)
            this->mass[cell * this->d + k] += (p[k] - this->mass[cell * this->d + k]) / (double)here.count;

        if (here.first_child < 0)
        {
            if (here.points.empty()) { here.points.push_back(point); return; }
            const double* q = this->y + here.points.front() * this->d;
            if (depth >= kMaxDepth || std::equal(p, p + this->d, q)) { here.points.push_back(point); return; }

            // Split the leaf: its points (all at one location) move down to one child, then this point goes on.
            std::vector<int64_t> moved = std::move(here.points);
            here.points.clear();
            const double h = this->half[cell] / 2;
            const int64_t first = (int64_t)this->cells.size();
            std::vector<double> c(this->d);
            for (int64_t o = 0; o < this->n_children; ++o)
            {
                for (int64_t k = 0; k < this->d; ++k)
                    c[k] = this->centre[cell * this->d + k] + ((o >> k) & 1 ? h : -h);
                this->add_cell(c.data(), h);
            }
            this->cells[cell].first_child = first;      // `here` may be stale after the cells grew
            const int64_t to = this->child_of(cell, q);
            this->cells[to].points = std::move(moved);
            this->cells[to].count = (int64_t)this->cells[to].points.size();
            std::copy(q, q + this->d, this->mass.begin() + to * this->d);
        }
        cell = this->child_of(cell, p);
    }
}

void
m00nny::BarnesHutTree::repulsion(double angle, double* F, double* Z) const
{
    const double angle2 = angle * angle;
    at::parallel_for(0, this->N, kGrain, [&](int64_t begin, int64_t end)
    {
        std::vector<int64_t> stack;
        std::vector<double> diff(this->d);
        for (int64_t i = begin; i < end; ++i)
        {
            const double* yi = this->y + i * this->d;
            double* Fi = F + i * this->d;
            std::fill(Fi, Fi + this->d, 0.0);
            double Zi = 0;
            stack.assign(1, 0);
            while (!stack.empty())
            {
                const int64_t cell = stack.back();
                stack.pop_back();
                const Cell& c = this->cells[cell];
                if (c.count == 0) continue;
                if (c.first_child < 0)
                {
                    for (int64_t j : c.points)
                    {
                        if (j == i) continue;
                        double d2 = 0;
                        for (int64_t k = 0; k < this->d; ++k) { diff[k] = yi[k] - this->y[j * this->d + k]; d2 += diff[k] * diff[k]; }
                        const double w = 1.0 / (1.0 + d2);
                        Zi += w;
                        for (int64_t k = 0; k < this->d; ++k) Fi[k] += w * w * diff[k];
                    }
                    continue;
                }
                bool inside = true;
                double d2 = 0;
                for (int64_t k = 0; k < this->d; ++k)
                {
                    inside = inside && std::abs(yi[k] - this->centre[cell * this->d + k]) <= this->half[cell];
                    diff[k] = yi[k] - this->mass[cell * this->d + k];
                    d2 += diff[k] * diff[k];
                }
                const double width = 2 * this->half[cell];
                if (!inside && width * width < angle2 * d2)
                {
                    const double w = 1.0 / (1.0 + d2);
                    Zi += c.count * w;
                    for (int64_t k = 0; k < this->d; ++k) Fi[k] += c.count * w * w * diff[k];
                    continue;
                }
                for (int64_t o = 0; o < this->n_children; ++o) stack.push_back(c.first_child + o);
            }
            Z[i] = Zi;
        }
    });
}
