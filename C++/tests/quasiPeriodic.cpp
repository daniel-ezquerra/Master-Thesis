//
// Created by Daniel on 08/04/2021.
//

double findMaximum(Torus &torus, int component)
{
    double maxEl = 0.0;
    for (const auto &dictTorus :torus.data)
    {
        double maxLocal = dictTorus.second.col(component).maxCoeff();
        if (maxLocal > maxEl)
            maxEl = maxLocal;
    }
    return maxEl;
}


int main() {
    /// QUASI-PERIODIC SOLVER TEST
    // Test based on the QP North Halo orbit examples in page 67 of 'Characterization of
    // quasi-periodic orbits for applications in the Sun-Earth and Earth-Moon systems"
    // from McCarthy.

    // Create POSolver object
    int N = 21, m = 6;
    int oneDay = 86400;
    Eigen::VectorXd timeArray = tools3BP::gaussLegendreCollocationArray(N, m);
    double halfPeriod = timeArray[timeArray.size() / 2];
    double quarterPeriod = timeArray[timeArray.size() / 4];
    double distanceEarthMoon = 384402000;
    std::string primaryBody = "Earth";
    std::string secondaryBody = "Moon";
    auto POSolver = POCollocationSolver(primaryBody, secondaryBody, distanceEarthMoon);

    double Href = 3.1389;
    PeriodicOrbit myOrbit = POSolver.getLinearApproximation(POSolver.L1, POSolver.northHalo, N, m, Href);
    POSolver.Solve(myOrbit, myOrbit, Href, 1E-8);
    double yAmplitude = myOrbit.data[quarterPeriod][1];
    double zAmplitude = myOrbit.data[0.0][2];
    std::cout << yAmplitude * distanceEarthMoon << " | " << zAmplitude * distanceEarthMoon << std::endl;
    std::cout << POSolver.timeNormalizedToDimensional(myOrbit.T) / oneDay << "\n" << std::endl;

    // Create torus object
    QPCollocationSolver QPSolver = QPCollocationSolver(primaryBody, secondaryBody, distanceEarthMoon);
    auto orbitPropagated = QPSolver.runSimulation(myOrbit.data[0.0], std::make_pair(0.0, myOrbit.T), 1E-4);
    std::pair<double, double> config(Href, 0.0);

    // Case 1
    Torus myTorus = Torus(orbitPropagated, 20, 20, 5, 0.019); // First T = 12.03
    Torus tanTorus = myTorus - orbitPropagated.first;
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-8);
    std::cout << "\n== Case 1 ==" << std::endl;
    std::cout << findMaximum(myTorus, 1) * distanceEarthMoon << " | " << findMaximum(myTorus, 2) * distanceEarthMoon
              << std::endl;
    std::cout << QPSolver.timeNormalizedToDimensional(myTorus.T) / oneDay << "\n" << std::endl;
    myTorus.writeTorus("test_output/qp_test_case_1.dat");

    // Case 2
    myTorus = Torus(orbitPropagated, 20, 20, 5, 0.0585); // Second T = 12.09
    tanTorus = myTorus - orbitPropagated.first;
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-8);
    std::cout << "\n== Case 2 ==" << std::endl;
    std::cout << findMaximum(myTorus, 1) * distanceEarthMoon << " | " << findMaximum(myTorus, 2) * distanceEarthMoon
              << std::endl;
    std::cout << QPSolver.timeNormalizedToDimensional(myTorus.T) / oneDay << "\n" << std::endl;
    myTorus.writeTorus("test_output/qp_test_case_2.dat");

    // Case 3
    myTorus = Torus(orbitPropagated, 20, 20, 5, 0.111); //Third T = 12.26
    tanTorus = myTorus - orbitPropagated.first;
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-8);
    std::cout << "\n== Case 3 ==" << std::endl;
    std::cout << findMaximum(myTorus, 1) * distanceEarthMoon << " | " << findMaximum(myTorus, 2) * distanceEarthMoon
              << std::endl;
    std::cout << QPSolver.timeNormalizedToDimensional(myTorus.T) / oneDay << "\n" << std::endl;
    myTorus.writeTorus("test_output/qp_test_case_3.dat");

    // Case 4
    myTorus = Torus(orbitPropagated, 20, 20, 5, 0.1338); // Fourth T = 12.40
    tanTorus = myTorus - orbitPropagated.first;
    myTorus.toCollocationForm();
    tanTorus.toCollocationForm();
    QPSolver.Solve(myTorus, myTorus, tanTorus, config, 1E-8);
    std::cout << "\n== Case 4 ==" << std::endl;
    std::cout << findMaximum(myTorus, 1) * distanceEarthMoon << " | " << findMaximum(myTorus, 2) * distanceEarthMoon
              << std::endl;
    std::cout << QPSolver.timeNormalizedToDimensional(myTorus.T) / oneDay << "\n" << std::endl;
    myTorus.writeTorus("test_output/qp_test_case_4.dat");

}