//
// Created by Daniel on 07/04/2021.
//

#include <chrono>
#include <Eigen/Dense>
#include "src/Torus.h"
#include "src/PeriodicOrbit.h"
#include "src/POCollocationSolver.h"
#include "src/QPCollocationSolver.h"
#include "src/tools3bp.h"

int main() {

    /// PERIODIC ORBIT TESTS
    // Create POSolver object
    int N = 21, m = 6;
    Eigen::VectorXd timeArray = tools3BP::gaussLegendreCollocationArray(N, m);
    double halfPeriod = timeArray[timeArray.size() / 2];
    double quarterPeriod = timeArray[timeArray.size() / 4];
    double distanceEarthMoon = 384402000;
    std::string primaryBody = "Earth";
    std::string secondaryBody = "Moon";
    auto POSolver = POCollocationSolver(primaryBody, secondaryBody, distanceEarthMoon);


    // L1 Lyapunov orbit test
    std::cout << "== L1 Lyapunov orbit test ==" << std::endl;

    Vector6d refStatePlanarL1 = Vector6d(0.8189, 0.0, 0.0, 0.0, 0.1750, 0.0);
    double Href = POSolver.getJacobi(refStatePlanarL1);
    PeriodicOrbit planarOrbit1 = POSolver.getLinearApproximation(POSolver.L1, POSolver.planar, N, m, Href);
    POSolver.Solve(planarOrbit1, planarOrbit1, Href, 1E-10);

    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStatePlanarL1.transpose() << std::endl;
    std::cout << "Time: " << halfPeriod << " | State: " << planarOrbit1.data[halfPeriod].transpose() << std::endl;
    std::cout << "Period ref: 2.7959" << " - Period: " << planarOrbit1.T << std::endl;
    std::cout << "--------------------------" << std::endl;


    // L2 Lyapunov orbit test
    std::cout << "\n\n== L2 Lyapunov orbit test ==" << std::endl;

    Vector6d refStatePlanarL2 = Vector6d(1.1843, 0.0, 0.0, 0.0, -0.1818, 0.0);
    Href = POSolver.getJacobi(refStatePlanarL2);
    PeriodicOrbit planarOrbit2 = POSolver.getLinearApproximation(POSolver.L2, POSolver.planar, N, m, Href);
    POSolver.Solve(planarOrbit2, planarOrbit2, Href, 1E-10);

    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStatePlanarL2.transpose() << std::endl;
    std::cout << "Time: " << 0.0 << " | State: " << planarOrbit2.data[0.0].transpose() << std::endl;
    std::cout << "Period ref: 3.4341" << " - Period: " << planarOrbit2.T << std::endl;
    std::cout << "--------------------------" << std::endl;


    // L1 North Halo orbit test
    std::cout << "\n\n== L1 North Halo orbit test ==" << std::endl;

    Vector6d refStateHaloL1 = Vector6d(0.8234, 0.0, 0.0224, 0.0, 0.1343, 0.0);
    Href = POSolver.getJacobi(refStateHaloL1);
    PeriodicOrbit haloOrbit1 = POSolver.getLinearApproximation(POSolver.L1, POSolver.northHalo, N, m, Href);
    POSolver.Solve(haloOrbit1, haloOrbit1, Href, 1E-10);

    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStateHaloL1.transpose() << std::endl;
    std::cout << "Time: " << 0.0 << " | State: " << haloOrbit1.data[0.0].transpose() << std::endl;
    std::cout << "Period ref: 2.7464" << " - Period: " << haloOrbit1.T << std::endl;
    std::cout << "--------------------------" << std::endl;


    // L2 North Halo orbit test
    std::cout << "\n\n== L2 North Halo orbit test ==" << std::endl;

    Vector6d refStateHaloL2 = Vector6d(1.1807, 0.0, 0.0139, 0.0, -0.1570, 0.0);
    Href = POSolver.getJacobi(refStateHaloL2);
    PeriodicOrbit haloOrbit2 = POSolver.getLinearApproximation(POSolver.L2, POSolver.southHalo, N, m, Href);
    POSolver.Solve(haloOrbit2, haloOrbit2, Href, 1E-10);
    haloOrbit2.writeOrbit("output/perOrbit.dat");

    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStateHaloL2.transpose() << std::endl;
    std::cout << "Time: " << halfPeriod << " | State: " << haloOrbit2.data[halfPeriod].transpose() << std::endl;
    std::cout << "Period ref: 3.4139" << " - Period: " << haloOrbit2.T << std::endl;
    std::cout << "--------------------------" << std::endl;


    // L1 Vertical orbit test
    std::cout << "\n\n== L1 Vertical orbit test ==" << std::endl;

    Vector6d refStateVerticalL1 = Vector6d(1.0118, 0.0, 0.1739, 0.0, -0.0799, 0.0);
    Href = POSolver.getJacobi(refStateVerticalL1);
    PeriodicOrbit verticalOrbit1 = POSolver.getLinearApproximation(POSolver.L1, POSolver.vertical, N, m, Href,
                                                                   std::make_pair(0.0, 2.0));
    POSolver.Solve(verticalOrbit1, verticalOrbit1, Href, 1E-10);
    verticalOrbit1.writeOrbit("output/perOrbit.dat");
    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStateVerticalL1.transpose() << std::endl;
    std::cout << "Time: " << quarterPeriod << " | State: " << verticalOrbit1.data[quarterPeriod].transpose()
              << std::endl;
    std::cout << "Period ref: 5.0950" << " - Period: " << verticalOrbit1.T << std::endl;
    std::cout << "--------------------------" << std::endl;


    // L2 Vertical orbit test
    std::cout << "\n\n== L2 Vertical orbit test ==" << std::endl;

    Vector6d refStateVerticalL2 = Vector6d(1.1119, 0.0, 0.0, 0.0, -0.1812, 0.4358);
    Href = POSolver.getJacobi(refStateVerticalL2);
    PeriodicOrbit verticalOrbit2 = POSolver.getLinearApproximation(POSolver.L2, POSolver.vertical, N, m, Href,
                                                                   std::make_pair(0.0, 2.0));
    POSolver.Solve(verticalOrbit2, verticalOrbit2, Href, 1E-9);

    std::cout << "\n-------Test results-------" << std::endl;
    std::cout << "Reference state: " << refStateVerticalL2.transpose() << std::endl;
    std::cout << "Time: " << quarterPeriod << " | State: " << verticalOrbit2.data[quarterPeriod].transpose()
              << std::endl;
    std::cout << "Period ref: 4.4222" << " - Period: " << verticalOrbit2.T << std::endl;
    std::cout << "--------------------------" << std::endl;

}