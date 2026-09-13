cmake_minimum_required(VERSION 3.27...4.2)

project(frsr_roaring LANGUAGES C CXX)

set(FRSR_ROARING_BUILD_TESTS_DEFAULT OFF)
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(FRSR_ROARING_BUILD_TESTS_DEFAULT ON)
endif()

option(FRSR_ROARING_BUILD_TESTS "Build frsr::roaring tests" ${FRSR_ROARING_BUILD_TESTS_DEFAULT})
option(FRSR_ROARING_BUILD_BENCHMARKS "Build frsr::roaring benchmark executable" ${FRSR_ROARING_BUILD_TESTS_DEFAULT})
option(FRSR_ROARING_FETCH_GTEST "Fetch GoogleTest when needed" ON)
option(FRSR_ROARING_BUILD_CROARING_CROSSCHECK "Build CRoaring cross-check tests when target 'roaring' is already available" ON)
option(FRSR_ROARING_ADD_LOCAL_CROARING_TARGET "Add local ../croaring target as 'roaring'" OFF)

# psi::vm SBO/heap-vector support
# Auto-detected when ../psiha is present; can be forced OFF to test the std::vector fallback.
if(EXISTS "${PROJECT_SOURCE_DIR}/../psiha/vm/include/psi/vm/containers/small_vector.hpp")
    option(FRSR_ROARING_USE_PSI_VM "Use psi::vm containers (small_vector / heap_vector) in frsr::roaring" ON)
else()
    option(FRSR_ROARING_USE_PSI_VM "Use psi::vm containers (small_vector / heap_vector) in frsr::roaring" OFF)
endif()

if(FRSR_ROARING_ADD_LOCAL_CROARING_TARGET AND NOT TARGET roaring)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/../croaring/CMakeLists.txt")
        message(FATAL_ERROR "FRSR_ROARING_ADD_LOCAL_CROARING_TARGET=ON requires ../croaring with CMakeLists.txt")
    endif()
    add_subdirectory("${PROJECT_SOURCE_DIR}/../croaring" "${PROJECT_BINARY_DIR}/croaring")
endif()

add_library(frsr_roaring STATIC
    src/bitmap.cpp
)
add_library(frsr::roaring ALIAS frsr_roaring)

target_include_directories(frsr_roaring
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(FRSR_ROARING_USE_PSI_VM)
    # Locate Boost headers (required by psi::vm)
    set(_boost_include_dir "")
    if(DEFINED ENV{Boost_ROOT})
        file(TO_CMAKE_PATH "$ENV{Boost_ROOT}" _env_boost_root)
        if(EXISTS "${_env_boost_root}/boost/version.hpp")
            set(_boost_include_dir "${_env_boost_root}")
        endif()
    endif()
    if(NOT _boost_include_dir AND DEFINED Boost_ROOT)
        if(EXISTS "${Boost_ROOT}/boost/version.hpp")
            set(_boost_include_dir "${Boost_ROOT}")
        endif()
    endif()
    # find_package(Boost) in the parent scope populates these as plain paths — the
    # most reliable source when the superproject already located Boost (e.g. CI, where
    # Boost sits under deps/boost and no shell exports Boost_ROOT).
    foreach(_var Boost_INCLUDE_DIR Boost_INCLUDE_DIRS)
        if(NOT _boost_include_dir AND DEFINED ${_var})
            foreach(_dir IN LISTS ${_var})
                if(EXISTS "${_dir}/boost/version.hpp")
                    set(_boost_include_dir "${_dir}")
                    break()
                endif()
            endforeach()
        endif()
    endforeach()
    if(NOT _boost_include_dir AND EXISTS "C:/Projekti/3rdParty/boost/boost/version.hpp")
        set(_boost_include_dir "C:/Projekti/3rdParty/boost")
    endif()
    if(NOT _boost_include_dir AND TARGET Boost::boost)
        # An imported target's INTERFACE_INCLUDE_DIRECTORIES is a *list* that may hold
        # generator expressions (e.g. $<BUILD_INTERFACE:...>). Those DO expand correctly
        # on the compile line — accept them (find_package vouched for the target); only
        # a plain path can be EXISTS-validated here, so validate those and trust genexes.
        get_target_property(_boost_target_dirs Boost::boost INTERFACE_INCLUDE_DIRECTORIES)
        if(_boost_target_dirs)
            foreach(_dir IN LISTS _boost_target_dirs)
                if(_dir MATCHES "\\$<")
                    set(_boost_include_dir "${_dir}")   # genex: trust, can't check now
                    break()
                elseif(EXISTS "${_dir}/boost/version.hpp")
                    set(_boost_include_dir "${_dir}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    # Belt-and-braces for the plain-path branches: a resolved non-genex dir must hold the
    # header (a surviving invalid value would be a probing bug). Genex values are trusted
    # (find_package produced them) and cannot be EXISTS-checked at configure time.
    if(_boost_include_dir AND NOT _boost_include_dir MATCHES "\\$<"
                         AND NOT EXISTS "${_boost_include_dir}/boost/version.hpp")
        message(FATAL_ERROR
            "frsr::roaring: internal probe error — resolved Boost include dir "
            "'${_boost_include_dir}' does not contain boost/version.hpp.")
    endif()

    if(_boost_include_dir)
        message(STATUS "frsr::roaring: psi::vm support enabled (Boost @ ${_boost_include_dir})")

        set(_psiha_root "${PROJECT_SOURCE_DIR}/../psiha")
        target_include_directories(frsr_roaring
            PUBLIC
                ${_psiha_root}/vm/include
                ${_psiha_root}/build/include
                ${_psiha_root}/config_ex/include
                ${_psiha_root}/std_fix/include
        )
        # SYSTEM: a consumer that already reaches Boost through an imported target
        # gets it as -isystem; a plain -I on the same directory from here would
        # override that and surface Boost's own warnings under the consumer's
        # -Werror set (seen with Boost 1.92's sort headers).
        target_include_directories(frsr_roaring SYSTEM PUBLIC ${_boost_include_dir})
        target_compile_definitions(frsr_roaring PUBLIC FRSR_ROARING_HAS_PSI_VM=1)
        target_compile_definitions(frsr_roaring PUBLIC
            BOOST_EXCEPTION_DISABLE
            BOOST_ENABLE_ASSERT_DEBUG_HANDLER
            BOOST_CONTAINER_NO_LIB
        )

        # BOOST_ENABLE_ASSERT_DEBUG_HANDLER above is what makes this necessary. It
        # opts into ::boost::assertion_failed[_msg], and Boost 1.92 moved those
        # declarations ABOVE boost/assert.hpp's own `#include <boost/config.hpp>`
        # (1.91: include at 44, first BOOST_NORETURN at 50; 1.92: BOOST_NORETURN at
        # 36, include at 63), so any translation unit reaching <boost/assert.hpp>
        # first dies with "unknown type name 'BOOST_NORETURN'". psi/vm/align.hpp
        # does exactly that, and it is on the include path of every header here.
        # Boost is consumed as an include directory rather than as a Boost:: target
        # so that this library still configures standalone, which also means it
        # cannot inherit a superproject's fix — hence the same force-include here.
        # Remove once upstream orders the include first.
        target_compile_options(frsr_roaring PUBLIC
            "$<IF:$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>,/FIboost/config.hpp,SHELL:-include boost/config.hpp>"
        )

        # File-mapped serialization + persistent_bitmap (vm_vector map_file):
        # needs the COMPILED psi::vm library, not just its container headers,
        # hence a separate opt-in on top of FRSR_ROARING_USE_PSI_VM. Wired at
        # the END of this file: vm.cmake sets directory-scope compile options
        # that must not leak into the other targets (gtest in particular).
        option(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION "Enable vm_vector serialization + persistent_bitmap (links the psi::vm library)" OFF)
    else()
        # Hard error, not a warning: FRSR_ROARING_USE_PSI_VM=ON is an explicit request for
        # the psi::vm containers, and silently substituting the std::vector fallback
        # produces a binary that differs from the intended one in allocation behaviour —
        # which silently invalidates any A/B measurement taken against it. This bit both a
        # Linux and a macOS host, where a non-login shell simply did not export Boost_ROOT.
        # Building the fallback on purpose is spelled -DFRSR_ROARING_USE_PSI_VM=OFF.
        message(FATAL_ERROR
            "frsr::roaring: FRSR_ROARING_USE_PSI_VM=ON but no Boost headers were found "
            "(looked for boost/version.hpp under $ENV{Boost_ROOT}, the Boost_ROOT cache "
            "variable, and the Boost::boost target). Set Boost_ROOT — note that a "
            "non-login shell may not export it — or pass -DFRSR_ROARING_USE_PSI_VM=OFF "
            "to deliberately build the std::vector fallback.")
    endif()
endif()

set_target_properties(frsr_roaring PROPERTIES
    CXX_STANDARD 26
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Whether the C++ compiler targets x86-64, decided by compiling a check rather than from
# CMAKE_SYSTEM_PROCESSOR, which names the host or system and not the compiler's target.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if !( defined( __x86_64__ ) || defined( _M_X64 ) )
#error not x86-64
#endif
int main() { return 0; }" FRSR_ROARING_X64)

# The headers carry most of the implementation, including kernels whose bodies
# are selected by the instruction-set macros (__SSE4_2__, __AVX2__, ...). Every
# translation unit that includes them instantiates its own copies, and the linker
# keeps one arbitrary copy of each inline function or template. A consumer compiled
# with different instruction-set options than the library therefore sees different
# definitions of the same entities (an ODR violation) and may end up running either
# copy. The options that decide those #if paths are thus a usage requirement: PUBLIC,
# and in every configuration, so the tests and benchmarks built in any configuration
# compile the same kernels as the library. Only additive feature options are used
# (no -march/-mtune), so a consumer's own target selection is not overridden.
if(FRSR_ROARING_X64)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(frsr_roaring PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX>:/clang:-mavx2;/clang:-mpopcnt>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(frsr_roaring PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX>:-mavx2;-mpopcnt>")
    elseif(MSVC)
        target_compile_options(frsr_roaring PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>")
    endif()
endif()

# The optimization level only affects the library's own object code.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(frsr_roaring PRIVATE "$<$<NOT:$<CONFIG:Debug>>:/O2>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(frsr_roaring PRIVATE "$<$<NOT:$<CONFIG:Debug>>:-O3>")
elseif(MSVC)
    target_compile_options(frsr_roaring PRIVATE "$<$<NOT:$<CONFIG:Debug>>:/O2>")
endif()

if(FRSR_ROARING_BUILD_TESTS)
    enable_testing()

    if(NOT TARGET GTest::gtest)
        if(NOT FRSR_ROARING_FETCH_GTEST)
            message(FATAL_ERROR "FRSR_ROARING_BUILD_TESTS requires GTest::gtest or FRSR_ROARING_FETCH_GTEST=ON.")
        endif()

        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.17.0
            GIT_SHALLOW    TRUE
        )
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(frsr_roaring_tests
        tests/bitmap_smoke.cpp
        tests/bitmap_view.cpp
        tests/borrowed_bitmap.cpp
        tests/cow_semantics.cpp
        tests/cow_bitmap.cpp
    )
    target_link_libraries(frsr_roaring_tests
        PRIVATE
            frsr_roaring
            GTest::gtest_main
    )
    set_target_properties(frsr_roaring_tests PROPERTIES
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    include(GoogleTest)
    gtest_discover_tests(frsr_roaring_tests)

    if(FRSR_ROARING_BUILD_CROARING_CROSSCHECK AND TARGET roaring)
        add_executable(frsr_roaring_croaring_crosscheck
            tests/bitmap_crosscheck.cpp
            tests/bitmap_full_crosscheck.cpp
        )
        target_link_libraries(frsr_roaring_croaring_crosscheck
            PRIVATE
                frsr_roaring
                GTest::gtest_main
                roaring
        )
        set_target_properties(frsr_roaring_croaring_crosscheck PROPERTIES
            CXX_STANDARD 26
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        gtest_discover_tests(frsr_roaring_croaring_crosscheck)
    endif()
endif()

if(FRSR_ROARING_BUILD_BENCHMARKS)
    add_executable(frsr_roaring_bench
        bench/bitmap_bench.cpp
    )
    target_link_libraries(frsr_roaring_bench PRIVATE frsr_roaring)
    set_target_properties(frsr_roaring_bench PROPERTIES
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(TARGET roaring)
        target_link_libraries(frsr_roaring_bench PRIVATE roaring)
        target_compile_definitions(frsr_roaring_bench PRIVATE FRSR_ROARING_HAS_CROARING=1)
    endif()
endif()

# Report-only tests: they always pass and print what they find.
#  - header_isa_tokens: instruction-set dependent tokens per public header, i.e. how
#    much of the header code preprocesses differently with different ISA options.
#  - consumer_symbol_census: frsr::roaring symbols each consumer object defines for
#    itself (its own copies of header code), next to the library's own object.
if(FRSR_ROARING_BUILD_TESTS)
    add_test(NAME frsr_roaring_report_header_isa_tokens
        COMMAND ${CMAKE_COMMAND}
            "-DHEADER_DIR=${PROJECT_SOURCE_DIR}/include/frsr/roaring"
            -P "${PROJECT_SOURCE_DIR}/cmake/header_isa_tokens.cmake"
    )

    set(_frsr_roaring_nm "${CMAKE_NM}")
    get_filename_component(_frsr_roaring_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(FRSR_ROARING_LLVM_NM NAMES llvm-nm HINTS "${_frsr_roaring_compiler_dir}")
    if(FRSR_ROARING_LLVM_NM)
        set(_frsr_roaring_nm "${FRSR_ROARING_LLVM_NM}")
    endif()
    set(_frsr_roaring_census_objects
        "$<TARGET_OBJECTS:frsr_roaring>"
        "$<TARGET_OBJECTS:frsr_roaring_tests>"
    )
    if(TARGET frsr_roaring_croaring_crosscheck)
        list(APPEND _frsr_roaring_census_objects "$<TARGET_OBJECTS:frsr_roaring_croaring_crosscheck>")
    endif()
    if(TARGET frsr_roaring_bench)
        list(APPEND _frsr_roaring_census_objects "$<TARGET_OBJECTS:frsr_roaring_bench>")
    endif()
    add_test(NAME frsr_roaring_report_consumer_symbol_census
        COMMAND ${CMAKE_COMMAND}
            "-DNM=${_frsr_roaring_nm}"
            -P "${PROJECT_SOURCE_DIR}/cmake/consumer_symbol_census.cmake"
            ${_frsr_roaring_census_objects}
    )
endif()

# Compiled psi::vm library for file-mapped serialization / persistent_bitmap.
# Kept last: vm.cmake mutates directory-scope compile options (clang-cl
# -std=gnu++2c), which must not apply to the targets declared above.
if(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)
    if(NOT FRSR_ROARING_USE_PSI_VM OR NOT _boost_include_dir)
        message(FATAL_ERROR "FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION requires FRSR_ROARING_USE_PSI_VM with Boost headers available")
    endif()
    if(NOT TARGET psi_vm)
        include(${_psiha_root}/vm/vm.cmake)
    endif()
    target_include_directories(psi_vm PUBLIC
        ${_psiha_root}/build/include
        ${_psiha_root}/err/include
        ${_psiha_root}/config_ex/include
        ${_psiha_root}/std_fix/include
    )
    target_include_directories(psi_vm SYSTEM PUBLIC ${_boost_include_dir})
    # Exceptions are enabled in this build ⇒ the macro expands to nothing
    # (psiha/build/build_options.cmake is the reference for these).
    target_compile_definitions(psi_vm PUBLIC
        PSI_NOEXCEPT_EXCEPT_BADALLOC=
        PSI_OVERCOMMIT_Disabled=0
        PSI_OVERCOMMIT_Partial=1
        PSI_OVERCOMMIT_Full=2
        PSI_MALLOC_OVERCOMMIT=PSI_OVERCOMMIT_Disabled
    )
    if(WIN32)
        target_compile_definitions(psi_vm PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    target_link_libraries(frsr_roaring PUBLIC psi_vm)
    target_compile_definitions(frsr_roaring PUBLIC FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION=1)
endif()
