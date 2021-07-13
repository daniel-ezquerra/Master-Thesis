//
// Created by Daniel on 02/04/2021.
//

#ifndef THESIS_POCOLLOCATIONSOLVER_H
#define THESIS_POCOLLOCATIONSOLVER_H

#include <iostream>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Solver3BP.h"
#include "PeriodicOrbit.h"

// Type definitions
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::map<double, Vector6d, std::less<>,Eigen::aligned_allocator<std::pair<const double, Vector6d>>> stateDict;
typedef Eigen::SparseMatrix<double> SparseMatrix;

class POCollocationSolver: public Solver3BP
{
public:
    enum lagrangePoint { L1, L2, L3, Last };
    enum periodicOrbit { planar, vertical, northHalo, southHalo };
    std::map<lagrangePoint, Eigen::Vector3d> lagrangePosition;

    POCollocationSolver(std::string &bodyPrimary, std::string &bodySecondary, double &distanceBodies);

    void Solve(PeriodicOrbit &orbit, PeriodicOrbit &prevOrbit, double Href, double tol=1E-10, int maxIter=15, bool getManifolds=true, const PeriodicOrbit &tanOrbit=PeriodicOrbit());

    void findManifolds(PeriodicOrbit &orbit, SparseMatrix &stateTransitionMatrices);

    Eigen::VectorXd getCollocationSyst(Eigen::VectorXd &orbitV,
                                       Eigen::VectorXd &prevOrbitV,
                                       Eigen::VectorXd &tanOrbitV,
                                       PeriodicOrbit &orbit,
                                       double Href);

    SparseMatrix getCollocationJacobian(Eigen::VectorXd &orbitV,
                                        Eigen::VectorXd &prevOrbitV,
                                        Eigen::VectorXd &tanOrbitV,
                                        PeriodicOrbit &orbit,
                                        double Href);

    static Eigen::VectorXd orbitToArray(const PeriodicOrbit &orbit);

    void arrayToOrbit(Eigen::VectorXd &array, PeriodicOrbit &orbit);

    /// LINEAR APPROXIMATION CREATION
    Eigen::Vector3d getLagrangePoint(lagrangePoint &point);

    PeriodicOrbit getLinearApproximation(lagrangePoint point, periodicOrbit orbit, int N1, int m, double Href, std::pair<double, double>parSpan=std::make_pair(0.0,1.0));

    PeriodicOrbit threeDLinearApproximation(int N1, int m, double Href, lagrangePoint point, periodicOrbit orbit, std::pair<double, double>AzSpan);

    PeriodicOrbit planarLinearApproximation(int N1, int m, double Href, lagrangePoint point, std::pair<double, double>x0Span);


private:
    Vector6d modelDynamicCR3BP(Vector6d &state, double l1);

    template <class T>
    static void fillSpBlock(std::vector<Eigen::Triplet<float>> &tripletList,
                            const T &block, int row, int col, int height, int width);

    /// LINEAR APPROXIMATION CREATION
    double energyConditionPlanar(const double &x, double Href, lagrangePoint point);

    double energyCondition3D(const double &Az, double Href, lagrangePoint point, periodicOrbit orbit);

    Vector6d planarStateApproximation(double theta, double x0, lagrangePoint point);

    Vector6d threeDStateApproximation(double theta, double Az, lagrangePoint point, periodicOrbit orbit, bool returnPeriod=false);
};


#endif //THESIS_POCOLLOCATIONSOLVER_H
