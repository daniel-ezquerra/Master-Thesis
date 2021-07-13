# cmake_minimum_required(VERSION <specify CMake version here>)
project(Thesis)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)

    cmake_minimum_required(VERSION 3.17)

    set(EIGEN_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\eigen")
    if(NOT EXISTS ${EIGEN_DIR})
        message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
    endif()
    include_directories(${EIGEN_DIR})

    set(BOOST_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\boost_1_75_0")
    if(NOT EXISTS ${BOOST_DIR})
        message(FATAL_ERROR "Could not find boost!")
    endif()
    include_directories(${BOOST_DIR})

    set(EIGEN_UN_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\eigen\\unsupported")
    if(NOT EXISTS ${EIGEN_UN_DIR})
        message(FATAL_ERROR "Could not find fftw!")
    endif()
    include_directories(${EIGEN_UN_DIR})

    add_executable(Thesis main.cpp src/tools3bp.cpp src/Torus.cpp src/Torus.h src/Solver3BP.cpp src/Solver3BP.h
            src/interp.cpp src/interp.h src/QPCollocationSolver.cpp src/QPCollocationSolver.h src/tools3bp.h
            src/POCollocationSolver.cpp src/POCollocationSolver.h src/PeriodicOrbit.cpp src/PeriodicOrbit.h)

    if (MINGW)
     target_compile_options(Thesis PRIVATE "-Wa,-mbig-obj")
    endif()
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen")
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Boost")
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/unsupported")

endif()

## LINUX BUILD CONFIG
#if (CMAKE_BUILD_TYPE MATCHES Debug)
#
#    cmake_minimum_required(VERSION 3.16.3)
#
#    set(EIGEN_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/eigen")
#    if(NOT EXISTS ${EIGEN_DIR})
#        message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
#    endif()
#    include_directories(${EIGEN_DIR})
#
#    set(BOOST_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/boost_1_75_0")
#    if(NOT EXISTS ${BOOST_DIR})
#        message(FATAL_ERROR "Could not find boost!")
#    endif()
#    include_directories(${BOOST_DIR})
#
#    set(EIGEN_UN_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/eigen/unsupported")
#    if(NOT EXISTS ${EIGEN_UN_DIR})
#        message(FATAL_ERROR "Could not find fftw!")
#    endif()
#    include_directories(${EIGEN_UN_DIR})
#
#    add_executable(Thesis main.cpp src/tools3bp.cpp src/Torus.cpp src/Torus.h src/Solver3BP.cpp src/Solver3BP.h
#                   src/interp.cpp src/interp.h src/QPCollocationSolver.cpp src/QPCollocationSolver.h src/tools3bp.h
#                   src/POCollocationSolver.cpp src/POCollocationSolver.h src/PeriodicOrbit.cpp src/PeriodicOrbit.h)
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen")
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Boost")
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/unsupported")
#
#endif()
