#!/usr/bin/env bash
# Builds and runs the tests in every supported configuration of the table sharing knobs
# (include/frsr/roaring/table_sharing_policy.hpp): the defaults, sharing off, and the defaults with each knob set
# otherwise, each also in the build that counts every table as shared (FRSR_ROARING_TEST_SHARE_EVERY_TABLE). The
# builds keep assertions on.
#
#   tests/table_sharing_matrix.sh <build root> [extra cmake arguments...]
#
# MATRIX_CONFIGS=<names> limits the run to those configurations; MATRIX_SHARE_EVERY_TABLE=0 skips the builds that
# count every table as shared; MATRIX_JOBS sets the build parallelism. Extra arguments go to every configure (a
# compiler, a sanitizer: -DCMAKE_CXX_FLAGS=-fsanitize=thread, ...). With psi::vm the assertions call
# boost::assertion_failed_msg (BOOST_ENABLE_ASSERT_DEBUG_HANDLER), which the program defines: pass an object that does
# (-DCMAKE_CXX_STANDARD_LIBRARIES=<object>). Exits non-zero when any build or test fails.
set -u

source_dir=$( cd "$( dirname "$0" )/.." && pwd )
build_root=${1:?usage: table_sharing_matrix.sh <build root> [cmake arguments...]}
shift
mkdir -p "${build_root}"

# name and knob settings of each supported configuration
configs=(
    "default|"
    "sharing-off|-DFRSR_ROARING_TABLE_SHARING=0"
    "small-tables-shared|-DFRSR_ROARING_TABLE_SHARING_SMALL_CAPACITY=0"
    "small-tables-to-16|-DFRSR_ROARING_TABLE_SHARING_SMALL_CAPACITY=16"
    "union-checks-first|-DFRSR_ROARING_TABLE_SHARING_UNION_CHECKS_FIRST=1"
    "subtract-builds-aside|-DFRSR_ROARING_TABLE_SHARING_SUBTRACT_CHECKS_FIRST=0"
    "key-list-unshares|-DFRSR_ROARING_TABLE_SHARING_KEY_LIST_CHECKS_FIRST=0"
    "optimize-in-place|-DFRSR_ROARING_TABLE_SHARING_OPTIMIZE_ASIDE=0"
    "prefetch-8|-DFRSR_ROARING_TABLE_SHARING_PREFETCH_DISTANCE=8"
)

failed=()
for entry in "${configs[@]}"; do
    name=${entry%%|*}
    knobs=${entry#*|}
    if [ -n "${MATRIX_CONFIGS:-}" ] && [[ " ${MATRIX_CONFIGS} " != *" ${name} "* ]]; then
        continue
    fi
    for every in 0 1; do
        if [ "${every}" = 1 ] && { [ "${MATRIX_SHARE_EVERY_TABLE:-1}" = 0 ] || [ "${name}" = sharing-off ]; }; then
            continue
        fi
        label=${name}; [ "${every}" = 1 ] && label=${name}-share-every-table
        dir=${build_root}/${label}
        echo "== ${label} ${knobs}"
        # shellcheck disable=SC2086
        if ! cmake --fresh -S "${source_dir}" -B "${dir}" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 \
                -DFRSR_ROARING_BUILD_TESTS=ON -DFRSR_ROARING_BUILD_BENCHMARKS=OFF \
                -DFRSR_ROARING_TEST_SHARE_EVERY_TABLE=$( [ "${every}" = 1 ] && echo ON || echo OFF ) ${knobs} "$@" > "${dir}.configure.log" 2>&1; then
            echo "   configure failed: ${dir}.configure.log"; failed+=( "${label}" ); continue
        fi
        if ! cmake --build "${dir}" --target frsr_roaring_tests -j "${MATRIX_JOBS:-8}" > "${dir}.build.log" 2>&1; then
            echo "   build failed: ${dir}.build.log"; failed+=( "${label}" ); continue
        fi
        tests=$( find "${dir}" \( -name 'frsr_roaring_tests' -o -name 'frsr_roaring_tests.exe' \) -type f | head -n 1 )
        if ! "${tests}" --gtest_brief=1 > "${dir}.tests.log" 2>&1; then
            echo "   tests failed: ${dir}.tests.log"; failed+=( "${label}" ); continue
        fi
        tail -n 2 "${dir}.tests.log" | sed 's/^/   /'
    done
done

if [ ${#failed[@]} -ne 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all configurations passed"
