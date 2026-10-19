# Report: frsr::roaring symbols defined by each object file.
#
#   cmake -DNM=<nm or llvm-nm> -P consumer_symbol_census.cmake <object>...
#
# Header code is instantiated by every translation unit that uses it, so a consumer
# object defines its own copies of library functions and templates. This prints,
# per object, how many such copies it defines, and how many of them are specific
# array/bitset/run kernels. Counting rules (identical for every object):
#   - symbol lines come from `nm <object>` with raw (mangled) names;
#   - "frsr::roaring" = the name contains both "frsr" and "roaring", which covers
#     Itanium (..4frsr7roaring..) and MSVC (..@roaring@frsr@@..) mangling, including
#     other templates instantiated over frsr::roaring types;
#   - defined  = the line carries an address, or dashes for an LTO bitcode object
#                (undefined symbols have neither);
#   - code     = defined, type T/t/W/w, and not a $-prefixed COFF metadata symbol
#                ($pdata$, $unwind$, ...);
#   - external = code with an upper-case type: the copies of which the linker keeps
#                one across all objects (lower-case code is local to the object,
#                e.g. instantiations over a lambda type or exception-handling
#                funclets, and is never shared);
#   - kernel columns = code whose name contains that kernel name (substring match, so
#                e.g. union_array_array also counts union_array_array_inplace).
# Always succeeds: a missing tool or object is reported, not failed.

set(_kernels combine_array_array_into extract_setbits intersect_run_bitset difference_array_array union_array_array)

# Objects are the arguments after the script path.
set(_objects)
set(_script_index -1)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 1 ${_last})
    if(_script_index GREATER 0 AND _i GREATER _script_index)
        list(APPEND _objects "${CMAKE_ARGV${_i}}")
    elseif(_script_index LESS 0 AND "${CMAKE_ARGV${_i}}" STREQUAL "-P")
        math(EXPR _script_index "${_i} + 1")
    endif()
endforeach()

if(NOT NM OR NOT EXISTS "${NM}")
    find_program(_nm_found NAMES llvm-nm nm)
    set(NM "${_nm_found}")
endif()
if(NOT NM)
    message("frsr::roaring consumer symbol census: no nm tool available, nothing to report")
    return()
endif()

function(_pad out text width)
    string(LENGTH "${text}" _len)
    set(_result "${text}")
    while(_len LESS width)
        string(APPEND _result " ")
        math(EXPR _len "${_len} + 1")
    endwhile()
    set(${out} "${_result}" PARENT_SCOPE)
endfunction()

message("frsr::roaring consumer symbol census (nm: ${NM})")
message("('defined' also counts data and metadata symbols, whose number depends on the object format; compare 'code', 'external' and the kernel columns across platforms)")
set(_measured 0)
list(LENGTH _objects _requested)
set(_width 10)
set(_line "")
foreach(_h defined code external ${_kernels})
    string(LENGTH "${_h}" _hl)
    math(EXPR _w "${_hl} + 2")
    if(_w LESS _width)
        set(_w ${_width})
    endif()
    set(_w_${_h} ${_w})
    _pad(_cell "${_h}" ${_w})
    string(APPEND _line "${_cell}")
endforeach()
string(APPEND _line "object")
message("${_line}")

foreach(_object IN LISTS _objects)
    if(NOT EXISTS "${_object}")
        message("(not built) ${_object}")
        continue()
    endif()
    execute_process(COMMAND "${NM}" "${_object}"
        OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        string(REGEX MATCH "[^\n]*" _first_error "${_err}")
        message("(nm failed: ${_first_error}) ${_object}")
        continue()
    endif()
    # One output line per list element: neutralise the list separator and the
    # square brackets that would otherwise group elements.
    string(REPLACE ";" "," _out "${_out}")
    string(REPLACE "[" "<" _out "${_out}")
    string(REPLACE "]" ">" _out "${_out}")
    string(REPLACE "\n" ";" _lines "${_out}")
    list(FILTER _lines INCLUDE REGEX "frsr")
    list(FILTER _lines INCLUDE REGEX "roaring")
    list(FILTER _lines INCLUDE REGEX "^[-0-9a-fA-F]+ [A-Za-z?] ")
    list(LENGTH _lines _count_defined)
    set(_code ${_lines})
    list(FILTER _code INCLUDE REGEX "^[-0-9a-fA-F]+ [TtWw] [^$]")
    list(LENGTH _code _count_code)
    set(_external ${_code})
    list(FILTER _external INCLUDE REGEX "^[-0-9a-fA-F]+ [TW] ")
    list(LENGTH _external _count_external)
    foreach(_k IN LISTS _kernels)
        set(_hits ${_code})
        list(FILTER _hits INCLUDE REGEX "${_k}")
        list(LENGTH _hits _count_${_k})
    endforeach()

    set(_line "")
    foreach(_h defined code external ${_kernels})
        _pad(_cell "${_count_${_h}}" ${_w_${_h}})
        string(APPEND _line "${_cell}")
    endforeach()
    string(APPEND _line "${_object}")
    message("${_line}")
    math(EXPR _measured "${_measured} + 1")
endforeach()
if(_measured EQUAL 0)
    message("measured 0 of ${_requested} requested objects: NOTHING WAS MEASURED")
else()
    message("measured ${_measured} of ${_requested} requested objects")
endif()
