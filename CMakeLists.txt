# ============================================================================
#  Nexus-LOB — root build configuration
#  Hybrid C++20 / CUDA / Pybind11 limit-order-book platform.
#
#  Intended dev environment: Linux / WSL-Ubuntu + NVIDIA GPU (see CLAUDE.md §8).
#  Quick start (from repo root):
#      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#      cmake --build build -j
#      ctest --test-dir build --output-on-failure
#
#  Useful toggles:
#      -DNEXUS_BUILD_PYBIND=OFF   # C++ engine + tests only (no Python needed)
#      -DNEXUS_BUILD_TESTS=OFF    # skip abi_check / lob_test
#      -DNEXUS_ENABLE_CUDA=OFF    # skip GPU risk-engine hooks (subsystem 3)
#      -DNEXUS_NATIVE_ARCH=OFF    # portable build (drop -march=native)
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(NexusLOB
    VERSION 0.1.0
    DESCRIPTION "Ultra-low-latency limit-order-book trading & microstructure platform"
    LANGUAGES CXX)                 # CUDA is enabled conditionally, further down.

# ---- global standards ------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)      # -std=c++20, not -std=gnu++20 (portable & strict)

# Position-independent code everywhere: the pybind MODULE links the engine, and a
# shared object must be built from PIC objects.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clangd / IDE tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to an optimized build (the engine's headline metrics are Release-only).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified — defaulting to Release")
endif()

# ---- options ---------------------------------------------------------------
option(NEXUS_BUILD_TESTS  "Build the C++ unit tests (abi_check, lob_test, ring_test, id_map_test)" ON)
option(NEXUS_BUILD_DEMOS  "Build the shmem ring producer/probe demos"        ON)
option(NEXUS_BUILD_PYBIND "Build the nexus_engine Pybind11 module"         ON)
option(NEXUS_ENABLE_CUDA  "Build CUDA risk-engine hooks if a toolkit exists" ON)
option(NEXUS_NATIVE_ARCH  "Optimize for the build machine (-march=native)"  ON)

# ============================================================================
#  1. cpp_engine — the matching engine (Person A)
# ============================================================================
# Header-only TODAY: every symbol lives in cpp_engine/include/nexus/*.hpp.
# The roadmap adds translation units under cpp_engine/src/, so we build a STATIC
# library the moment any .cpp exists there and fall back to a header-only
# INTERFACE target until then. Either way the public name is `nexus::engine`, so
# dependents (tests, the pybind module, the future CUDA lib) never change.
file(GLOB_RECURSE NEXUS_ENGINE_SOURCES CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/cpp_engine/src/*.cpp")

if(NEXUS_ENGINE_SOURCES)
    add_library(cpp_engine STATIC ${NEXUS_ENGINE_SOURCES})
    set(NEXUS_ENGINE_SCOPE PUBLIC)
    message(STATUS "cpp_engine: STATIC library from ${CMAKE_CURRENT_LIST_DIR}/cpp_engine/src")
else()
    add_library(cpp_engine INTERFACE)
    set(NEXUS_ENGINE_SCOPE INTERFACE)
    message(STATUS "cpp_engine: header-only INTERFACE target (no cpp_engine/src/*.cpp yet)")
endif()
add_library(nexus::engine ALIAS cpp_engine)

target_include_directories(cpp_engine ${NEXUS_ENGINE_SCOPE}
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp_engine/include>")
target_compile_features(cpp_engine ${NEXUS_ENGINE_SCOPE} cxx_std_20)

# -O3 (+ optional -march=native) on the optimized configs, GCC/Clang only.
if(NOT MSVC)
    target_compile_options(cpp_engine ${NEXUS_ENGINE_SCOPE}
        "$<$<CONFIG:Release>:-O3>"
        "$<$<CONFIG:RelWithDebInfo>:-O3>")
    if(NEXUS_NATIVE_ARCH)
        target_compile_options(cpp_engine ${NEXUS_ENGINE_SCOPE}
            "$<$<CONFIG:Release>:-march=native>"
            "$<$<CONFIG:RelWithDebInfo>:-march=native>")
    endif()
endif()

# ============================================================================
#  2. C++ tests — standalone, no framework (run via CTest)
# ============================================================================
if(NEXUS_BUILD_TESTS)
    enable_testing()

    # ABI lock: compile-time static_asserts + prints sizeof/offsets (== 448 B).
    add_executable(abi_check cpp_engine/tests/abi_check.cpp)
    target_link_libraries(abi_check PRIVATE nexus::engine)
    add_test(NAME abi_check COMMAND abi_check)

    # Matching-engine correctness: crosses, FIFO priority, IOC/FOK/market,
    # cancel/modify, and every reject path.
    if(EXISTS "${PROJECT_SOURCE_DIR}/cpp_engine/tests/lob_test.cpp")
        add_executable(lob_test cpp_engine/tests/lob_test.cpp)
        target_link_libraries(lob_test PRIVATE nexus::engine)
        if(NOT MSVC)
            target_compile_options(lob_test PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME lob_test COMMAND lob_test)
    endif()

    # Shared-memory ring (subsystem 5): order + integrity across a writer thread,
    # and drop-new-on-full semantics over the real OS shmem segment.
    if(EXISTS "${PROJECT_SOURCE_DIR}/cpp_engine/tests/ring_test.cpp")
        add_executable(ring_test cpp_engine/tests/ring_test.cpp)
        target_link_libraries(ring_test PRIVATE nexus::engine)
        if(NOT MSVC)
            target_compile_options(ring_test PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME ring_test COMMAND ring_test)
    endif()

    # Zero-alloc id -> Order* map: collision / wraparound / backtrack-shift stress.
    if(EXISTS "${PROJECT_SOURCE_DIR}/cpp_engine/tests/id_map_test.cpp")
        add_executable(id_map_test cpp_engine/tests/id_map_test.cpp)
        target_link_libraries(id_map_test PRIVATE nexus::engine)
        if(NOT MSVC)
            target_compile_options(id_map_test PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME id_map_test COMMAND id_map_test)
    endif()

    # Subsystem 3 — Monte-Carlo VaR/CVaR CPU-reference statistical tests
    # (pure C++, no CUDA required; runs in CTest on every build machine).
    if(EXISTS "${PROJECT_SOURCE_DIR}/cpp_engine/tests/risk_test.cpp")
        add_executable(risk_test cpp_engine/tests/risk_test.cpp)
        target_include_directories(risk_test PRIVATE "${PROJECT_SOURCE_DIR}/cuda_risk")
        if(NOT MSVC)
            target_compile_options(risk_test PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME risk_test COMMAND risk_test)
    endif()
endif()

# ============================================================================
#  2b. Subsystem 5 demos — shared-memory book publisher + live probe
# ============================================================================
if(NEXUS_BUILD_DEMOS)
    add_executable(ring_producer cpp_engine/demos/ring_producer.cpp)
    target_link_libraries(ring_producer PRIVATE nexus::engine)
    add_executable(ring_probe cpp_engine/demos/ring_probe.cpp)
    target_link_libraries(ring_probe PRIVATE nexus::engine)
endif()

# ============================================================================
#  3. nexus_engine — the Pybind11 bridge (bindings/, the C++<->Python seam)
# ============================================================================
if(NEXUS_BUILD_PYBIND)
    # Prefer a system / venv pybind11; otherwise fetch a pinned copy. Resolving
    # Python first lets pybind11 target the active interpreter (crucial inside a
    # WSL venv with multiple Pythons on PATH).
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found on system — fetching v2.13.6 via FetchContent")
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(pybind11)
    else()
        message(STATUS "pybind11: using system package ${pybind11_VERSION}")
    endif()

    # MODULE_NAME must match PYBIND11_MODULE(nexus_engine, m) in the source.
    pybind11_add_module(nexus_engine bindings/pybind_wrapper.cpp)
    target_link_libraries(nexus_engine PRIVATE nexus::engine)
    # CPU risk reference (subsystem 3) is exposed as `compute_var_cvar` — the
    # header lives under cuda_risk/ even though it is pure C++ (no toolkit).
    target_include_directories(nexus_engine PRIVATE "${PROJECT_SOURCE_DIR}/cuda_risk")

    # Drop the compiled module into bindings/ so `import nexus_engine` resolves
    # for bindings/tests/test_abi_parity.py (pyproject adds bindings/ to
    # pytest's pythonpath). On Linux/WSL the .so is a LIBRARY artifact.
    set_target_properties(nexus_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bindings")
    # Multi-config generators (MSVC / Visual Studio) otherwise append <config>,
    # dropping the module into bindings/<Config>/ instead of bindings/ — pin every
    # config so `import nexus_engine` resolves on Windows too. Single-config
    # generators (Unix Makefiles / Ninja in WSL) ignore these.
    foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(nexus_engine PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${PROJECT_SOURCE_DIR}/bindings"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${PROJECT_SOURCE_DIR}/bindings")
    endforeach()
    message(STATUS "nexus_engine: Pybind11 module -> ${PROJECT_SOURCE_DIR}/bindings")

    # Harvest the compiled module into the wheel / install root as an importable
    # top-level module (scikit-build-core uses this for `pip install .`). Only
    # affects packaging — the direct `cmake --build` path above is unchanged.
    install(TARGETS nexus_engine
            LIBRARY DESTINATION .
            RUNTIME DESTINATION .)
endif()

# ============================================================================
#  4. CUDA risk engine — subsystem 3 (hooks only; stubbed until .cu lands)
# ============================================================================
# We only *enable the CUDA language and set flags* here — no .cu sources exist
# yet, so no target is created. When cuda_risk/*.cu arrives, uncomment the
# nexus_risk block below; the detection and flags are already in place.
if(NEXUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
    else()
        message(STATUS "CUDA: no nvcc on PATH — GPU risk engine (subsystem 3) skipped")
    endif()
endif()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA: toolkit ${CUDAToolkit_VERSION} found — enabling GPU hooks")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Target real hardware. `native` (CMake >= 3.24) matches the build GPU; older
    # CMake gets a sensible Volta..Ada default list.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
        endif()
    endif()

    # Device-code optimization flags. --use_fast_math is deliberately OFF: VaR/CVaR
    # want reproducible, IEEE-accurate reductions over 100k+ paths.
    set(NEXUS_CUDA_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # --- GPU risk engine (subsystem 3) --------------------------------------
    file(GLOB NEXUS_RISK_SOURCES CONFIGURE_DEPENDS
         "${PROJECT_SOURCE_DIR}/cuda_risk/*.cu")
    add_library(nexus_risk STATIC ${NEXUS_RISK_SOURCES})
    set_target_properties(nexus_risk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(nexus_risk PRIVATE ${NEXUS_CUDA_FLAGS})
    target_link_libraries(nexus_risk PUBLIC CUDA::cudart)
    # ---------------------------------------------------------------------------
endif()

# ============================================================================
#  4b. Risk engine benchmark — CPU always, GPU when a toolkit is present
# ============================================================================
# The CPU reference is header-only, so risk_bench compiles everywhere. When a
# toolkit is found we define NEXUS_HAS_CUDA and link the nexus_risk kernel;
# otherwise it reports the CPU numbers and says the GPU path wasn't built.
add_executable(risk_bench cuda_risk/risk_bench.cpp)
target_include_directories(risk_bench PRIVATE "${PROJECT_SOURCE_DIR}/cuda_risk")
if(NOT MSVC)
    target_compile_options(risk_bench PRIVATE -O3)
endif()
if(CUDAToolkit_FOUND)
    target_compile_definitions(risk_bench PRIVATE NEXUS_HAS_CUDA=1)
    target_link_libraries(risk_bench PRIVATE nexus_risk CUDA::cudart)
endif()

# ---- configuration summary -------------------------------------------------
message(STATUS "")
message(STATUS "Nexus-LOB configured:")
message(STATUS "  build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ tests     : ${NEXUS_BUILD_TESTS}")
message(STATUS "  pybind module : ${NEXUS_BUILD_PYBIND}")
message(STATUS "  CUDA hooks    : ${NEXUS_ENABLE_CUDA} (toolkit found: ${CUDAToolkit_FOUND})")
message(STATUS "  native arch   : ${NEXUS_NATIVE_ARCH}")
message(STATUS "")
