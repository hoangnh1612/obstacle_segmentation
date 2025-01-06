#include <iostream>
#include <Eigen/Core>
// #include<Eogen/Dense>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include<g2o/stuff/misc.h>

// Define the vertex: a single variable x
class VertexX : public g2o::BaseVertex<1, Eigen::Vector2d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Reset to zero
    void setToOriginImpl() override {
        _estimate = Eigen::Vector2d (-1,-1);
    }

    // Update rule
    void oplusImpl(const double* update) override {
        Eigen::Vector2d upd(update[0], update[1]);
        _estimate += upd;
        if (_estimate [0] <0) _estimate[0] = 0.0;
        if (_estimate [0] >2.0) _estimate[0] = 2.0;
        if (_estimate [1] <0.0) _estimate[1] = 0.0;
        if (_estimate [1] >3.0) _estimate[1] = 3.0;
    }

    // Read and write functions (not used here)
    bool read(std::istream&) override {
        return false;
    }

    bool write(std::ostream&) const override {
        return false;
    }
};

// Define the edge: the error function (x - target)
class EdgeCostFunction : public g2o::BaseUnaryEdge<1, double, VertexX> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeCostFunction() {
        // Constructor doesn't need to take target value anymore
        // as we'll use setMeasurement() later
    }

    // Compute the error: error = x - target
    void computeError() override {
        const VertexX* v = static_cast<const VertexX*>(_vertices[0]);
        const Eigen::Vector2d xy = v->estimate();
        double x = xy[0];
        double y = xy[1];
        double cost = 2 * x + 3* y - _measurement;

        _error[0] = cost;  // Fixed: Use _error instead of error
    }

    // Read and write functions (not used here)
    bool read(std::istream&) override {
        return false;
    }

    bool write(std::ostream&) const override {
        return false;
    }
};

int main() {
    // Create optimizer
    g2o::SparseOptimizer optimizer;

    // Use a dense linear solver
    std::unique_ptr<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>> linearSolver =
        std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();

    std::unique_ptr<g2o::BlockSolverX> blockSolver =
        std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer.setAlgorithm(solver);

    // Add vertex
    auto* v = new VertexX();
    v->setId(0);
    v->setEstimate(Eigen::Vector2d(0.0, 0.0));  // Initial guess
    optimizer.addVertex(v);

    // Add edge
    auto* edge = new EdgeCostFunction();
    edge->setId(0);
    edge->setVertex(0, v);
    edge->setMeasurement(5.0);  // Target value is 5.0
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());  // Weight of the error
    optimizer.addEdge(edge);

    // Perform optimization
    Eigen::Vector2d initial_estimate = v->estimate();
    std::cout << "Initial estimate: " << initial_estimate[0]<<" "<<initial_estimate[1] << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);  // Number of iterations
    initial_estimate = v->estimate();
    std::cout << "Optimized estimate: " << initial_estimate[0]<<" "<<initial_estimate[1] << std::endl;

    return 0;
}
// int main()
// {
//     double vals[] = {-1, 0, 1,2};
//     for (double v : vals)
//     {
//         int s = g2o::sign(v);
//         std::cout << "sign(" << v << ") = " << s << std::endl;
//     }
//     return 0;
// }
