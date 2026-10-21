#pragma once

// Explicit instantiation definitions of the bitmaps that bitmap.hpp declares extern. Each translation unit of
// the library instantiates one of them, so that the library builds in parallel and a program links only the
// instantiations it uses.

#define FRSR_ROARING_DEFINES_LIBRARY_CONFIGURATION 1
#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/bitmap_definitions.hpp>

#define FRSR_ROARING_INSTANTIATE( KEY, COW, RUN_SELECTION ) \
    template class frsr::roaring::bitmap< \
        KEY, \
        frsr::roaring::default_container_set<KEY>, \
        frsr::roaring::detail::COW, \
        frsr::roaring::detail::RUN_SELECTION \
    >; \
    template auto frsr::roaring::bitmap< \
        KEY, \
        frsr::roaring::default_container_set<KEY>, \
        frsr::roaring::detail::COW, \
        frsr::roaring::detail::RUN_SELECTION \
    >::combine<frsr::roaring::detail::set_operation::bit_and, false>( bitmap const &, bitmap * ) -> bitmap &
