//
// Created by Daniel on 29/04/2021.
//

#include <chrono>
#include <Eigen/Dense>
#include "src/Torus.h"
#include "src/PeriodicOrbit.h"
#include "src/POCollocationSolver.h"
#include "src/QPCollocationSolver.h"
#include "src/tools3bp.h"
#include <iomanip>
#include <fstream>
#include <iomanip>

//
// Created by Daniel on 08/04/2021.
//

void saveTorus(const std::string& fileName, Torus &torus, double H, POCollocationSolver::lagrangePoint lPoint, POCollocationSolver::periodicOrbit family)
{
    std::ofstream ofile(fileName, std::ios::out);
    std::string bar = "////////////////////////////////";
    std::string lPointStr, oFamilyStr;
    if (lPoint == POCollocationSolver::lagrangePoint::L1)
        lPointStr = "L1";
    else if (lPoint == POCollocationSolver::lagrangePoint::L2)
        lPointStr = "L2";

    if (family == POCollocationSolver::periodicOrbit::planar)
        oFamilyStr = "Planar Lyapunov";
    else if (family == POCollocationSolver::periodicOrbit::vertical)
        oFamilyStr = "Vertical Lyapunov";
    else if (family == POCollocationSolver::periodicOrbit::northHalo)
        oFamilyStr = "North Halo";
    else if (family == POCollocationSolver::periodicOrbit::southHalo)
        oFamilyStr = "South Halo";

    ofile << bar << '\n';
    ofile << "/    QP ORBIT EXCHANGE FILE    /\n";
    ofile << bar << '\n';
    ofile << "/ METADATA\n";
    ofile << "* Eq. point : " + lPointStr << '\n';
    ofile << "* Family : QP " + oFamilyStr << '\n';
    ofile << "* Parametrization : N1=" << torus.N1 << " N2=" << torus.N2 << ", m=" << torus.m << '\n';
    ofile << "* Jacobi : " << std::setprecision( 16 ) <<  H << '\n';
    ofile << "* Rot Par: " << std::fixed << std::setprecision( 16 ) << torus.rho << '\n';
    ofile << "* Period : " << std::fixed << std::setprecision( 16 ) << torus.T << '\n';
    ofile << "* Stability index : " << std::fixed << std::setprecision( 5 ) << torus.stabilityIdx << '\n';
    ofile << bar << '\n';
    ofile << "/ ORBITAL DATA\n";
    ofile << "/ _th1_ _th2_ | ________ t _______ | ________ x ________ | ________ y ________ | ________ z ________ | _______ dx ________ | _______ dy ________ | _______ dz ________ |\n";

    int countN1 = 0;
    for (auto const& dictTorus : torus.data)
    {
        double t = dictTorus.first;
        for (int n2 = 0; n2 < torus.N2; n2++)
        {
            ofile << "^  " << std::setw(3) << countN1+1 << "   " << std::setw(3) << n2+1 << "    " << std::fixed << std::setprecision( 16 ) << t << "   ";
            for (const auto state: dictTorus.second.row(n2)) {
                if (state < 0.0)
                    ofile << std::fixed << std::setprecision( 16 ) << state << "   ";
                else
                    ofile << " " << std::fixed << std::setprecision( 16 ) << state << "   ";
            }
            ofile << "\n+                                    ";
            for (const auto mani: torus.stableManifoldData[dictTorus.first].row(n2)){
                if (mani < 0.0)
                    ofile << std::fixed << std::setprecision( 16 ) << mani << "   ";
                else
                    ofile << " " << std::fixed << std::setprecision( 16 ) << mani << "   ";
            }
            ofile << "\n-                                    ";
            for (const auto mani: torus.unstableManifoldData[dictTorus.first].row(n2)){
                if (mani < 0.0)
                    ofile << std::fixed << std::setprecision( 16 ) << mani << "   ";
                else
                    ofile << " " << std::fixed << std::setprecision( 16 ) << mani << "   ";
            }
            ofile << '\n';
        }
        countN1++;
    }
    ofile << "/ END" << '\n';
    ofile.close();
}

int main() {
    /// QUASI-PERIODIC SOLVER TEST
    // Test based on the QP North Halo orbit examples in page 67 of 'Characterization of
    // quasi-periodic orbits for applications in the Sun-Earth and Earth-Moon systems"
    // from McCarthy.

    // Create POSolver object
    int N = 11, m = 4;
    int N2 = 20;
    int oneDay = 86400;
    Eigen::VectorXd timeArray = tools3BP::gaussLegendreCollocationArray(N, m);
    double halfPeriod = timeArray[timeArray.size() / 2];
    double quarterPeriod = timeArray[timeArray.size() / 4];
    double distanceEarthMoon = 384402000;
    std::string primaryBody = "Earth";
    std::string secondaryBody = "Moon";
    auto POSolver = POCollocationSolver(primaryBody, secondaryBody, distanceEarthMoon);
    auto lPoint = POSolver.L1;
    auto oFamily = POSolver.northHalo;
    double Href = 3.1389;

    // Create periodic orbit object
    PeriodicOrbit myOrbit = POSolver.getLinearApproximation(lPoint, oFamily, N, m, Href);
    POSolver.Solve(myOrbit, myOrbit, Href, 1E-8);

    // Create torus object
    QPCollocationSolver QPSolver = QPCollocationSolver(primaryBody, secondaryBody, distanceEarthMoon);
    auto orbitPropagated = QPSolver.runSimulation(myOrbit.data[0.0], std::make_pair(0.0, myOrbit.T), 1E-4);
    std::pair<double, double> config(Href, 0.0);

    // Solve torus
    Torus myTorus = Torus(orbitPropagated, N, N2, m, 0.111); // First T = 12.03
    Torus tanTorus = myTorus - orbitPropagated.first;
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-8);

    saveTorus("test_output/mani_test_torus_0.dat", myTorus, Href, lPoint, oFamily);

    // Propagate unstable manifold
    int sectionTh1 = 0;
    double times[4] = {7.79, 9.75, 11.39, 13.02};
    Torus torusRef = myTorus;
    int counter = 0;
    for (const auto &t :times){
        std::cout << "Time: " << t << std::endl;
        auto tSpan = std::make_pair(0.0, POSolver.timeDimensionalToNormalized(86400*t));

        for (const auto &dictStates : torusRef.data)
        {
            std::cout << "Section: " << dictStates.first << std::endl;
            auto initialStateMatrix = torusRef.data[dictStates.first] - 5E-3 * torusRef.unstableManifoldData[dictStates.first];

            for (int n2 = 0; n2 < torusRef.N2; n2++)
            {
                auto initialState = initialStateMatrix.row(n2);
                auto solObj = QPSolver.runSimulation(initialState, tSpan, POSolver.timeDimensionalToNormalized(86400*0.1));
                myTorus.data[dictStates.first].row(n2) = solObj.first.rbegin()->second;
                if (t == 13.02)
                    tools3BP::writeTrajectory(solObj.first, "test_output/mani_test_orbit_" + std::to_string(n2) + ".dat");
            }
        }
        saveTorus("test_output/mani_test_torus_" + std::to_string(counter+1) + ".dat", myTorus, Href, lPoint, oFamily);
        counter++;
    }

    return 0;
}