#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if FRSR_ROARING_HAS_CROARING
#include <roaring/roaring.h>
#include <roaring/roaring64.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using TestBitmap32 = frsr::roaring::bitmap<std::uint32_t>;
using TestBitmap64 = frsr::roaring::bitmap<std::uint64_t>;

static inline std::uint64_t rand_u64( std::mt19937_64 & rng ) {
    return std::uniform_int_distribution<std::uint64_t>(
        std::numeric_limits<std::uint64_t>::min(),
        std::numeric_limits<std::uint64_t>::max()
    )( rng );
}

#if FRSR_ROARING_HAS_CROARING
// CRoaring builds in this repo expose the C 64-bit API but not the C++
// Roaring64Map wrapper class. Provide a minimal local shim for benchmark parity.
class Roaring64Map {
public:
    Roaring64Map() : bitmap_{ roaring64_bitmap_create() } {}
    explicit Roaring64Map( roaring64_bitmap_t * bitmap ) : bitmap_{ bitmap } {}

    ~Roaring64Map() {
        if ( bitmap_ != nullptr ) {
            roaring64_bitmap_free( bitmap_ );
        }
    }

    Roaring64Map( Roaring64Map const & other )
        : bitmap_{ other.bitmap_ != nullptr ? roaring64_bitmap_copy( other.bitmap_ ) : nullptr } {}

    Roaring64Map & operator=( Roaring64Map const & other ) {
        if ( this == &other ) {
            return *this;
        }
        roaring64_bitmap_t * copy{ other.bitmap_ != nullptr ? roaring64_bitmap_copy( other.bitmap_ ) : nullptr };
        if ( bitmap_ != nullptr ) {
            roaring64_bitmap_free( bitmap_ );
        }
        bitmap_ = copy;
        return *this;
    }

    Roaring64Map( Roaring64Map && other ) noexcept : bitmap_{ other.bitmap_ } { other.bitmap_ = nullptr; }

    Roaring64Map & operator=( Roaring64Map && other ) noexcept {
        if ( this != &other ) {
            if ( bitmap_ != nullptr ) {
                roaring64_bitmap_free( bitmap_ );
            }
            bitmap_ = other.bitmap_;
            other.bitmap_ = nullptr;
        }
        return *this;
    }

    void add( std::uint64_t const value ) { roaring64_bitmap_add( bitmap_, value ); }
    void remove( std::uint64_t const value ) { roaring64_bitmap_remove( bitmap_, value ); }
    [[nodiscard]] bool contains( std::uint64_t const value ) const { return roaring64_bitmap_contains( bitmap_, value ); }
    [[nodiscard]] std::uint64_t cardinality() const { return roaring64_bitmap_get_cardinality( bitmap_ ); }

    [[nodiscard]] std::size_t getSizeInBytes( bool const portable ) const {
        if ( portable ) {
            return roaring64_bitmap_portable_size_in_bytes( bitmap_ );
        }
        auto * copy{ roaring64_bitmap_copy( bitmap_ ) };
        roaring64_bitmap_shrink_to_fit( copy );
        auto const size{ roaring64_bitmap_frozen_size_in_bytes( copy ) };
        roaring64_bitmap_free( copy );
        return size;
    }

    [[nodiscard]] std::size_t getFrozenSizeInBytes() const { return getSizeInBytes( false ); }

    [[nodiscard]] std::size_t write( char * dst, bool const portable ) const {
        if ( portable ) {
            return roaring64_bitmap_portable_serialize( bitmap_, dst );
        }
        auto * copy{ roaring64_bitmap_copy( bitmap_ ) };
        roaring64_bitmap_shrink_to_fit( copy );
        auto const written{ roaring64_bitmap_frozen_serialize( copy, dst ) };
        roaring64_bitmap_free( copy );
        return written;
    }

    void writeFrozen( char * dst ) const {
        auto * copy{ roaring64_bitmap_copy( bitmap_ ) };
        roaring64_bitmap_shrink_to_fit( copy );
        static_cast<void>( roaring64_bitmap_frozen_serialize( copy, dst ) );
        roaring64_bitmap_free( copy );
    }

    [[nodiscard]] static Roaring64Map read( char const * src, bool const portable ) {
        if ( portable ) {
            // Portable read in this benchmark is always called on trusted buffers.
            // Use a large max size bound; deserializer validates format internally.
            auto * bitmap{ roaring64_bitmap_portable_deserialize_safe( src, std::numeric_limits<std::size_t>::max() ) };
            return Roaring64Map{ bitmap };
        }
        return Roaring64Map{ nullptr };
    }

    [[nodiscard]] static Roaring64Map frozenView( char const * src, std::size_t const maxbytes ) {
        return Roaring64Map{ roaring64_bitmap_frozen_view( src, maxbytes ) };
    }

private:
    roaring64_bitmap_t * bitmap_{};
};
#endif

// ========================================================================
// Entry-based Benchmark System
// ========================================================================

struct Entry {
    std::string name;
    std::string description;
    std::function<void*()> setup;           // Returns opaque state pointer
    std::function<int64_t(void*)> run;      // Runs benchmark, returns checksum
    std::function<void(void*)> teardown;    // Cleans up state
    int64_t ops_per_run = 1;                // Operations per run for rate calculation
    int64_t inner_reps = 1;                 // How many times to repeat measurement
    bool reusable_state = false;            // If true, state is reused across iterations
};

// Global benchmark registry
static std::vector<Entry> g_benchmarks;

// ========================================================================
// Synthetic Benchmark Constants (ported from CRoaring benchmark.cpp)
// ========================================================================

namespace synthetic {

// google-benchmark's CreateRange(1000, 1000000, 10) yields these four.
static constexpr size_t kCounts[] = {1000, 10000, 100000, 1000000};

// google-benchmark's CreateRange(1, 1<<48, 256) yields these seven.
static const uint64_t kSteps[] = {
    1ULL,
    256ULL,
    65536ULL,
    16777216ULL,
    4294967296ULL,       // 2^32
    1099511627776ULL,    // 2^40
    281474976710656ULL,  // 2^48
};

// The 10 bitmasks from synthetic_bench.cpp used for "random" variants.
static constexpr uint64_t kBitmasks[] = {
    0x00000000000FFFFFULL, 0x0000000FFFFF0000ULL, 0x000FFFFF00000000ULL,
    0xFFFFF00000000000ULL, 0x000000005DBFC83EULL, 0x00005DBFC83E0000ULL,
    0x5DBFC83E00000000ULL, 0x0000493B189604B6ULL, 0x493B189604B60000ULL,
    0x420C684950A2D088ULL};

// CRoaring microbench uses 2^20 preload. Keep a lower default here so the full
// cross-backend suite remains runnable in one session while preserving behavior.
static constexpr std::size_t kRandomPreload = (1U << 16);

// Thread-local RNG for the random-bitmask variants. Seeded deterministically
// so runs are reproducible.
static inline uint64_t rand_u64(std::mt19937_64 &rng) {
    return std::uniform_int_distribution<uint64_t>(
        std::numeric_limits<uint64_t>::min(),
        std::numeric_limits<uint64_t>::max())(rng);
}

// ---- helper: print "count=1000000/step=16777216" into a filter-friendly
// name fragment --------------------------------------------------------
static std::string fmt_params(size_t count, uint64_t step) {
    char buf[64];
    snprintf(buf, sizeof(buf), "count=%zu/step=%" PRIu64, count, step);
    return buf;
}

// ====== FAMILY 1: ContainsHit / ContainsMiss (r64, cpp, set) ===================
// ported from deps/croaring/benchmarks/benchmark.cpp:3233-3423

#if FRSR_ROARING_HAS_CROARING

struct r64HitState {
    roaring64_bitmap_t *r;
    size_t count;
    uint64_t step;
    size_t i;
};

struct cppHitState {
    Roaring64Map *r;
    size_t count;
    uint64_t step;
    size_t i;
};

struct setHitState {
    std::set<uint64_t> *s;
    size_t count;
    uint64_t step;
    size_t i;
};

struct frsrHitState {
    TestBitmap64 r;
    size_t count;
    uint64_t step;
    size_t i;
};

static void register_contains_variants() {
    for (size_t count : kCounts) {
        for (uint64_t step : kSteps) {
            std::string ptag = fmt_params(count, step);

            // r64ContainsHit
            // ported from deps/croaring/benchmarks/benchmark.cpp:3239-3271
            {
                Entry e;
                e.name = "synthetic/r64ContainsHit/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64ContainsHit: preload a "
                    "roaring64_bitmap_t with `count` values at positions "
                    "{0, step, 2*step, ..., (count-1)*step}; each "
                    "measured call does one roaring64_bitmap_contains() "
                    "for a value guaranteed to be present, cycling "
                    "through the stored values.";
                e.setup = [count, step]() -> void * {
                    auto *s = new r64HitState{roaring64_bitmap_create(), count,
                                              step, 0};
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<r64HitState *>(sv);
                    uint64_t v = s->i * s->step;
                    s->i = (s->i + 1) % s->count;
                    return roaring64_bitmap_contains(s->r, v) ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<r64HitState *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppContainsHit
            // ported from deps/croaring/benchmarks/benchmark.cpp:3272-3301
            {
                Entry e;
                e.name = "synthetic/frsrContainsHit/" + ptag;
                e.description =
                    "frsr variant of r64ContainsHit using "
                    "frsr::roaring::bitmap<uint64_t>::contains().";
                e.setup = [count, step]() -> void * {
                    auto *s = new frsrHitState{ TestBitmap64{}, count, step, 0 };
                    for (size_t i = 0; i < count; ++i) {
                        s->r.add(i * step);
                    }
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<frsrHitState *>(sv);
                    uint64_t v = s->i * s->step;
                    s->i = (s->i + 1) % s->count;
                    return s->r.contains(v) ? 1 : 0;
                };
                e.teardown = [](void *sv) { delete static_cast<frsrHitState *>(sv); };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppContainsHit
            // ported from deps/croaring/benchmarks/benchmark.cpp:3272-3301
            {
                Entry e;
                e.name = "synthetic/cppContainsHit/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppContainsHit: same as "
                    "r64ContainsHit but through the Roaring64Map C++ "
                    "wrapper (one .contains(v) per measured call).";
                e.setup = [count, step]() -> void * {
                    auto *s =
                        new cppHitState{new Roaring64Map(), count, step, 0};
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<cppHitState *>(sv);
                    uint64_t v = s->i * s->step;
                    s->i = (s->i + 1) % s->count;
                    return s->r->contains(v) ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<cppHitState *>(sv);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // setContainsHit
            // ported from deps/croaring/benchmarks/benchmark.cpp:3302-3331
            {
                Entry e;
                e.name = "synthetic/setContainsHit/" + ptag;
                e.description =
                    "synthetic_bench.cpp setContainsHit: std::set<uint64_t> "
                    "baseline for the same count×step cycling-hit query "
                    "pattern. Each call is `set.find(v) != set.end()`.";
                e.setup = [count, step]() -> void * {
                    auto *s = new setHitState{new std::set<uint64_t>(), count,
                                              step, 0};
                    for (size_t i = 0; i < count; ++i) s->s->insert(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<setHitState *>(sv);
                    uint64_t v = s->i * s->step;
                    s->i = (s->i + 1) % s->count;
                    return s->s->find(v) != s->s->end() ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<setHitState *>(sv);
                    delete s->s;
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // r64ContainsMiss
            // ported from deps/croaring/benchmarks/benchmark.cpp:3333-3364
            {
                Entry e;
                e.name = "synthetic/r64ContainsMiss/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64ContainsMiss: same preload as "
                    "r64ContainsHit, but each query is at (i+1)*step-1 — "
                    "a value that is not in the bitmap. Measures the "
                    "negative-lookup path.";
                e.setup = [count, step]() -> void * {
                    auto *s = new r64HitState{roaring64_bitmap_create(), count,
                                              step, 0};
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<r64HitState *>(sv);
                    uint64_t v = (s->i + 1) * s->step - 1;
                    s->i = (s->i + 1) % s->count;
                    return roaring64_bitmap_contains(s->r, v) ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<r64HitState *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppContainsMiss
            // ported from deps/croaring/benchmarks/benchmark.cpp:3365-3392
            {
                Entry e;
                e.name = "synthetic/frsrContainsMiss/" + ptag;
                e.description =
                    "frsr variant of r64ContainsMiss using "
                    "frsr::roaring::bitmap<uint64_t>::contains() miss path.";
                e.setup = [count, step]() -> void * {
                    auto *s = new frsrHitState{ TestBitmap64{}, count, step, 0 };
                    for (size_t i = 0; i < count; ++i) {
                        s->r.add(i * step);
                    }
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<frsrHitState *>(sv);
                    uint64_t v = ( ( s->i + 1 ) * s->step ) - 1;
                    s->i = (s->i + 1) % s->count;
                    return s->r.contains(v) ? 1 : 0;
                };
                e.teardown = [](void *sv) { delete static_cast<frsrHitState *>(sv); };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppContainsMiss
            // ported from deps/croaring/benchmarks/benchmark.cpp:3365-3392
            {
                Entry e;
                e.name = "synthetic/cppContainsMiss/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppContainsMiss: miss pattern "
                    "via Roaring64Map::contains().";
                e.setup = [count, step]() -> void * {
                    auto *s =
                        new cppHitState{new Roaring64Map(), count, step, 0};
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<cppHitState *>(sv);
                    uint64_t v = (s->i + 1) * s->step - 1;
                    s->i = (s->i + 1) % s->count;
                    return s->r->contains(v) ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<cppHitState *>(sv);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // setContainsMiss
            // ported from deps/croaring/benchmarks/benchmark.cpp:3393-3420
            {
                Entry e;
                e.name = "synthetic/setContainsMiss/" + ptag;
                e.description =
                    "synthetic_bench.cpp setContainsMiss: miss pattern "
                    "against std::set.";
                e.setup = [count, step]() -> void * {
                    auto *s = new setHitState{new std::set<uint64_t>(), count,
                                              step, 0};
                    for (size_t i = 0; i < count; ++i) s->s->insert(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<setHitState *>(sv);
                    uint64_t v = (s->i + 1) * s->step - 1;
                    s->i = (s->i + 1) % s->count;
                    return s->s->find(v) != s->s->end() ? 1 : 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<setHitState *>(sv);
                    delete s->s;
                    delete s;
                };
                e.ops_per_run = 1;
                e.inner_reps = 10000;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }
        }
    }
}

#endif // FRSR_ROARING_HAS_CROARING

// ====== FAMILY 2: ContainsRandom / InsertRemoveRandom (per bitmask) ============
// ported from deps/croaring/benchmarks/benchmark.cpp:3425-3643

#if FRSR_ROARING_HAS_CROARING

struct r64RandState {
    roaring64_bitmap_t *r;
    uint64_t bitmask;
    std::mt19937_64 rng;
};

struct cppRandState {
    Roaring64Map *r;
    uint64_t bitmask;
    std::mt19937_64 rng;
};

struct setRandState {
    std::set<uint64_t> *s;
    uint64_t bitmask;
    std::mt19937_64 rng;
};

struct frsrRandState {
    TestBitmap64 r;
    uint64_t bitmask;
    std::mt19937_64 rng;
};

static void register_random_variants() {
    for (size_t bi = 0; bi < sizeof(kBitmasks) / sizeof(kBitmasks[0]); ++bi) {
        uint64_t mask = kBitmasks[bi];
        char tag[32];
        snprintf(tag, sizeof(tag), "bitmask=%zu", bi);

        // r64ContainsRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3449-3481
        {
            Entry e;
            e.name = std::string("synthetic/r64ContainsRandom/") + tag;
            e.description =
                "synthetic_bench.cpp r64ContainsRandom: preload a "
                "roaring64_bitmap_t with 2^20 values randomly chosen via "
                "rand() & bitmask, then time one contains() per call on "
                "fresh random (& bitmask) values. The 10 bitmasks cover "
                "20, 32, 48, and 64 bit spreads.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new r64RandState{roaring64_bitmap_create(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    roaring64_bitmap_add(s->r, rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<r64RandState *>(sv);
                uint64_t v = rand_u64(s->rng) & s->bitmask;
                return roaring64_bitmap_contains(s->r, v) ? 1 : 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<r64RandState *>(sv);
                roaring64_bitmap_free(s->r);
                delete s;
            };
            e.ops_per_run = 1;
            e.inner_reps = 10000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // cppContainsRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3482-3511
        {
            Entry e;
            e.name = std::string("synthetic/frsrContainsRandom/") + tag;
            e.description =
                "frsr variant of r64ContainsRandom using "
                "frsr::roaring::bitmap<uint64_t>::contains().";
            e.setup = [mask]() -> void * {
                auto *s =
                    new frsrRandState{TestBitmap64{}, mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i) {
                    s->r.add(rand_u64(s->rng) & mask);
                }
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<frsrRandState *>(sv);
                uint64_t v = rand_u64(s->rng) & s->bitmask;
                return s->r.contains(v) ? 1 : 0;
            };
            e.teardown = [](void *sv) { delete static_cast<frsrRandState *>(sv); };
            e.ops_per_run = 1;
            e.inner_reps = 10000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // cppContainsRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3482-3511
        {
            Entry e;
            e.name = std::string("synthetic/cppContainsRandom/") + tag;
            e.description =
                "synthetic_bench.cpp cppContainsRandom: random-bitmask "
                "contains through the Roaring64Map C++ wrapper.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new cppRandState{new Roaring64Map(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    s->r->add(rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<cppRandState *>(sv);
                uint64_t v = rand_u64(s->rng) & s->bitmask;
                return s->r->contains(v) ? 1 : 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<cppRandState *>(sv);
                delete s->r;
                delete s;
            };
            e.ops_per_run = 1;
            e.inner_reps = 10000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // setContainsRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3512-3541
        {
            Entry e;
            e.name = std::string("synthetic/setContainsRandom/") + tag;
            e.description =
                "synthetic_bench.cpp setContainsRandom: random-bitmask "
                "contains against std::set<uint64_t>.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new setRandState{new std::set<uint64_t>(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    s->s->insert(rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<setRandState *>(sv);
                uint64_t v = rand_u64(s->rng) & s->bitmask;
                return s->s->find(v) != s->s->end() ? 1 : 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<setRandState *>(sv);
                delete s->s;
                delete s;
            };
            e.ops_per_run = 1;
            e.inner_reps = 10000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // r64InsertRemoveRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3543-3577
        {
            Entry e;
            e.name = std::string("synthetic/r64InsertRemoveRandom/") + tag;
            e.description =
                "synthetic_bench.cpp r64InsertRemoveRandom: preload a "
                "roaring64_bitmap_t with 2^20 random (& bitmask) values, "
                "then each call adds one fresh random value and removes "
                "another. Reports per-pair cost (insert + remove).";
            e.setup = [mask]() -> void * {
                auto *s =
                    new r64RandState{roaring64_bitmap_create(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    roaring64_bitmap_add(s->r, rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<r64RandState *>(sv);
                uint64_t a = rand_u64(s->rng) & s->bitmask;
                uint64_t r = rand_u64(s->rng) & s->bitmask;
                roaring64_bitmap_add(s->r, a);
                roaring64_bitmap_remove(s->r, r);
                return 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<r64RandState *>(sv);
                roaring64_bitmap_free(s->r);
                delete s;
            };
            e.ops_per_run = 2;
            e.inner_reps = 5000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // cppInsertRemoveRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3578-3609
        {
            Entry e;
            e.name = std::string("synthetic/frsrInsertRemoveRandom/") + tag;
            e.description =
                "frsr variant of r64InsertRemoveRandom using paired add/remove.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new frsrRandState{TestBitmap64{}, mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i) {
                    s->r.add(rand_u64(s->rng) & mask);
                }
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<frsrRandState *>(sv);
                uint64_t a = rand_u64(s->rng) & s->bitmask;
                uint64_t r = rand_u64(s->rng) & s->bitmask;
                s->r.add(a);
                s->r.remove(r);
                return 0;
            };
            e.teardown = [](void *sv) { delete static_cast<frsrRandState *>(sv); };
            e.ops_per_run = 2;
            e.inner_reps = 5000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // cppInsertRemoveRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3578-3609
        {
            Entry e;
            e.name = std::string("synthetic/cppInsertRemoveRandom/") + tag;
            e.description =
                "synthetic_bench.cpp cppInsertRemoveRandom: paired "
                "add/remove through Roaring64Map.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new cppRandState{new Roaring64Map(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    s->r->add(rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<cppRandState *>(sv);
                uint64_t a = rand_u64(s->rng) & s->bitmask;
                uint64_t r = rand_u64(s->rng) & s->bitmask;
                s->r->add(a);
                s->r->remove(r);
                return 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<cppRandState *>(sv);
                delete s->r;
                delete s;
            };
            e.ops_per_run = 2;
            e.inner_reps = 5000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }

        // setInsertRemoveRandom
        // ported from deps/croaring/benchmarks/benchmark.cpp:3610-3641
        {
            Entry e;
            e.name = std::string("synthetic/setInsertRemoveRandom/") + tag;
            e.description =
                "synthetic_bench.cpp setInsertRemoveRandom: paired "
                "insert/erase against std::set.";
            e.setup = [mask]() -> void * {
                auto *s =
                    new setRandState{new std::set<uint64_t>(), mask,
                                     std::mt19937_64(0xdeadbeefULL + mask)};
                for (size_t i = 0; i < kRandomPreload; ++i)
                    s->s->insert(rand_u64(s->rng) & mask);
                return s;
            };
            e.run = [](void *sv) -> int64_t {
                auto *s = static_cast<setRandState *>(sv);
                uint64_t a = rand_u64(s->rng) & s->bitmask;
                uint64_t r = rand_u64(s->rng) & s->bitmask;
                s->s->insert(a);
                s->s->erase(r);
                return 0;
            };
            e.teardown = [](void *sv) {
                auto *s = static_cast<setRandState *>(sv);
                delete s->s;
                delete s;
            };
            e.ops_per_run = 2;
            e.inner_reps = 5000;
            e.reusable_state = true;
            g_benchmarks.push_back(std::move(e));
        }
    }
}

#endif // FRSR_ROARING_HAS_CROARING

// ====== FAMILY 3: Insert / Remove (build-from-scratch timed) ===================
// ported from deps/croaring/benchmarks/benchmark.cpp:3645-3837

#if FRSR_ROARING_HAS_CROARING

struct insertParams {
    size_t count;
    uint64_t step;
};

static void register_insert_remove() {
    for (size_t count : kCounts) {
        for (uint64_t step : kSteps) {
            std::string ptag = fmt_params(count, step);

            // r64Insert — each timed iteration builds a fresh bitmap.
            // ported from deps/croaring/benchmarks/benchmark.cpp:3657-3686
            {
                Entry e;
                e.name = "synthetic/r64Insert/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64Insert: each timed call "
                    "allocates a fresh roaring64_bitmap_t, inserts "
                    "`count` values at positions {0, step, 2*step, ...}, "
                    "and frees it. Measures build-from-scratch cost, "
                    "including allocation/free.";
                e.setup = [count, step]() -> void * {
                    return new insertParams{count, step};
                };
                e.run = [](void *sv) -> int64_t {
                    auto *p = static_cast<insertParams *>(sv);
                    roaring64_bitmap_t *r = roaring64_bitmap_create();
                    for (size_t i = 0; i < p->count; ++i)
                        roaring64_bitmap_add(r, i * p->step);
                    int64_t c = static_cast<int64_t>(
                        roaring64_bitmap_get_cardinality(r));
                    roaring64_bitmap_free(r);
                    return c;
                };
                e.teardown = [](void *sv) {
                    delete static_cast<insertParams *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppInsert
            // ported from deps/croaring/benchmarks/benchmark.cpp:3687-3708
            {
                Entry e;
                e.name = "synthetic/frsrInsert/" + ptag;
                e.description =
                    "frsr variant of r64Insert: build-from-scratch "
                    "with bitmap<uint64_t> and sequential add().";
                e.setup = [count, step]() -> void * {
                    return new insertParams{count, step};
                };
                e.run = [](void *sv) -> int64_t {
                    auto *p = static_cast<insertParams *>(sv);
                    TestBitmap64 r;
                    for (size_t i = 0; i < p->count; ++i) {
                        r.add(i * p->step);
                    }
                    return static_cast<int64_t>(r.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<insertParams *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppInsert
            // ported from deps/croaring/benchmarks/benchmark.cpp:3687-3708
            {
                Entry e;
                e.name = "synthetic/cppInsert/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppInsert: build-from-scratch "
                    "cost through the Roaring64Map C++ wrapper.";
                e.setup = [count, step]() -> void * {
                    return new insertParams{count, step};
                };
                e.run = [](void *sv) -> int64_t {
                    auto *p = static_cast<insertParams *>(sv);
                    Roaring64Map r;
                    for (size_t i = 0; i < p->count; ++i) r.add(i * p->step);
                    return static_cast<int64_t>(r.cardinality());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<insertParams *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // setInsert
            // ported from deps/croaring/benchmarks/benchmark.cpp:3709-3730
            {
                Entry e;
                e.name = "synthetic/setInsert/" + ptag;
                e.description =
                    "synthetic_bench.cpp setInsert: build-from-scratch "
                    "cost for std::set<uint64_t>.";
                e.setup = [count, step]() -> void * {
                    return new insertParams{count, step};
                };
                e.run = [](void *sv) -> int64_t {
                    auto *p = static_cast<insertParams *>(sv);
                    std::set<uint64_t> s;
                    for (size_t i = 0; i < p->count; ++i) s.insert(i * p->step);
                    return static_cast<int64_t>(s.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<insertParams *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // r64Remove — setup rebuilds the populated container each iter
            // ported from deps/croaring/benchmarks/benchmark.cpp:3735-3772
            {
                Entry e;
                e.name = "synthetic/r64Remove/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64Remove: each measured call "
                    "removes `count` previously-inserted values. Setup "
                    "rebuilds the populated bitmap before every iteration "
                    "so the timed region only covers the removes, "
                    "matching bench.cpp's PauseTiming/ResumeTiming "
                    "pattern.";
                struct S {
                    size_t count;
                    uint64_t step;
                    roaring64_bitmap_t *r;
                };
                e.setup = [count, step]() -> void * {
                    auto *s = new S{count, step, roaring64_bitmap_create()};
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<S *>(sv);
                    for (size_t i = 0; i < s->count; ++i)
                        roaring64_bitmap_remove(s->r, i * s->step);
                    return static_cast<int64_t>(
                        roaring64_bitmap_get_cardinality(s->r));
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<S *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                // Not reusable: setup rebuilds the populated state
                g_benchmarks.push_back(std::move(e));
            }

            // cppRemove
            // ported from deps/croaring/benchmarks/benchmark.cpp:3773-3803
            {
                Entry e;
                e.name = "synthetic/frsrRemove/" + ptag;
                e.description =
                    "frsr variant of r64Remove: remove-all with setup "
                    "rebuilding populated bitmap before each iteration.";
                struct S {
                    size_t count;
                    uint64_t step;
                    TestBitmap64 r;
                };
                e.setup = [count, step]() -> void * {
                    auto *s = new S{count, step, TestBitmap64{}};
                    for (size_t i = 0; i < count; ++i) {
                        s->r.add(i * step);
                    }
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<S *>(sv);
                    for (size_t i = 0; i < s->count; ++i) {
                        s->r.remove(i * s->step);
                    }
                    return static_cast<int64_t>(s->r.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<S *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                g_benchmarks.push_back(std::move(e));
            }

            // cppRemove
            // ported from deps/croaring/benchmarks/benchmark.cpp:3773-3803
            {
                Entry e;
                e.name = "synthetic/cppRemove/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppRemove: remove-all through "
                    "Roaring64Map, with setup rebuilding the populated "
                    "map before each measured iteration.";
                struct S {
                    size_t count;
                    uint64_t step;
                    Roaring64Map *r;
                };
                e.setup = [count, step]() -> void * {
                    auto *s = new S{count, step, new Roaring64Map()};
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<S *>(sv);
                    for (size_t i = 0; i < s->count; ++i)
                        s->r->remove(i * s->step);
                    return static_cast<int64_t>(s->r->cardinality());
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<S *>(sv);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                g_benchmarks.push_back(std::move(e));
            }

            // setRemove
            // ported from deps/croaring/benchmarks/benchmark.cpp:3804-3834
            {
                Entry e;
                e.name = "synthetic/setRemove/" + ptag;
                e.description =
                    "synthetic_bench.cpp setRemove: remove-all through "
                    "std::set<uint64_t>::erase, setup rebuilding before "
                    "each iteration.";
                struct S {
                    size_t count;
                    uint64_t step;
                    std::set<uint64_t> *s;
                };
                e.setup = [count, step]() -> void * {
                    auto *s = new S{count, step, new std::set<uint64_t>()};
                    for (size_t i = 0; i < count; ++i) s->s->insert(i * step);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<S *>(sv);
                    for (size_t i = 0; i < s->count; ++i)
                        s->s->erase(i * s->step);
                    return static_cast<int64_t>(s->s->size());
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<S *>(sv);
                    delete s->s;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                g_benchmarks.push_back(std::move(e));
            }
        }
    }
}

#endif // FRSR_ROARING_HAS_CROARING
// ====== FAMILY 4: Serialize / Deserialize (r64 + cpp; portable + frozen) =======
// ported from deps/croaring/benchmarks/benchmark.cpp:3839-4142

#if FRSR_ROARING_HAS_CROARING

struct serState {
    roaring64_bitmap_t *r;
    std::vector<char> buf;
};

struct serStateCpp {
    Roaring64Map *r;
    std::vector<char> buf;
    size_t size;
};

struct serStateFrozen {
    roaring64_bitmap_t *r;
    char *buf;
    size_t size;
};

struct serStateCppFrozen {
    Roaring64Map *r;
    char *buf;
    size_t size;
};

struct serStateFrsr {
    TestBitmap64 r;
#ifdef FRSR_ROARING_HAS_PSI_VM
    TestBitmap64::serialized_byte_vector serialized;
#else
    std::vector<std::byte> serialized;
#endif
};

static void register_ser_deser() {
    for (size_t count : kCounts) {
        for (uint64_t step : kSteps) {
            std::string ptag = fmt_params(count, step);

            // r64PortableSerialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:3866-3899
            {
                Entry e;
                e.name = "synthetic/r64PortableSerialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64PortableSerialize: preload a "
                    "bitmap of `count` values at stride `step`, then time "
                    "one roaring64_bitmap_portable_serialize() call per "
                    "measured iteration into a preallocated buffer.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serState;
                    s->r = roaring64_bitmap_create();
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    s->buf.resize(
                        roaring64_bitmap_portable_size_in_bytes(s->r));
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serState *>(sv);
                    return static_cast<int64_t>(
                        roaring64_bitmap_portable_serialize(s->r,
                                                            s->buf.data()));
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serState *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // r64FrozenSerialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:3900-3932
            {
                Entry e;
                e.name = "synthetic/r64FrozenSerialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64FrozenSerialize: preload a "
                    "bitmap (run-optimised + shrunk), then time one "
                    "roaring64_bitmap_frozen_serialize() call per "
                    "iteration.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serState;
                    s->r = roaring64_bitmap_create();
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    roaring64_bitmap_shrink_to_fit(s->r);
                    s->buf.resize(roaring64_bitmap_frozen_size_in_bytes(s->r));
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serState *>(sv);
                    roaring64_bitmap_frozen_serialize(s->r, s->buf.data());
                    return 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serState *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // frsrPortableSerialize
            {
                Entry e;
                e.name = "synthetic/frsrPortableSerialize/" + ptag;
                e.description =
                    "frsr portable serialize: materialize native vm-backed "
                    "wire format into a vm_vector<byte>.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateFrsr;
                    for (size_t i = 0; i < count; ++i) {
                        auto const v = static_cast<std::uint64_t>(i) * step;
                        s->r.add(v);
                    }
                    s->r.optimize_for_storage();
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateFrsr *>(sv);
                    s->r.serialize_to_vm_vector(s->serialized);
                    return static_cast<int64_t>(s->serialized.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<serStateFrsr *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // frsrFrozenSerialize
            {
                Entry e;
                e.name = "synthetic/frsrFrozenSerialize/" + ptag;
                e.description =
                    "frsr frozen serialize into the native indexed frozen "
                    "wire format.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateFrsr;
                    for (size_t i = 0; i < count; ++i) {
                        auto const v = static_cast<std::uint64_t>(i) * step;
                        s->r.add(v);
                    }
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateFrsr *>(sv);
                    s->r.serialize_frozen_to_vm_vector(s->serialized);
                    return static_cast<int64_t>(s->serialized.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<serStateFrsr *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppPortableSerialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:3933-3963
            {
                Entry e;
                e.name = "synthetic/cppPortableSerialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppPortableSerialize: "
                    "Roaring64Map.write(portable=true) into a preallocated "
                    "buffer.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateCpp;
                    s->r = new Roaring64Map();
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    s->size = s->r->getSizeInBytes(/*portable=*/true);
                    s->buf.resize(s->size);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateCpp *>(sv);
                    return static_cast<int64_t>(
                        s->r->write(s->buf.data(), /*portable=*/true));
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serStateCpp *>(sv);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppFrozenSerialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:3964-3994
            {
                Entry e;
                e.name = "synthetic/cppFrozenSerialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppFrozenSerialize: "
                    "Roaring64Map.writeFrozen() into a 32-byte aligned "
                    "(overallocated) buffer.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateCppFrozen;
                    s->r = new Roaring64Map();
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    s->size = s->r->getFrozenSizeInBytes();
                    s->buf = static_cast<char *>(roaring_aligned_malloc(64, s->size));
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateCppFrozen *>(sv);
                    s->r->writeFrozen(s->buf);
                    return 0;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serStateCppFrozen *>(sv);
                    roaring_aligned_free(s->buf);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 5;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // r64PortableDeserialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:3996-4033
            {
                Entry e;
                e.name = "synthetic/r64PortableDeserialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64PortableDeserialize: preload "
                    "+ serialize to a buffer once in setup, then time "
                    "roaring64_bitmap_portable_deserialize_safe(buf, size) "
                    "+ free per iteration.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serState;
                    s->r = roaring64_bitmap_create();
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    s->buf.resize(
                        roaring64_bitmap_portable_size_in_bytes(s->r));
                    roaring64_bitmap_portable_serialize(s->r, s->buf.data());
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serState *>(sv);
                    roaring64_bitmap_t *r2 =
                        roaring64_bitmap_portable_deserialize_safe(
                            s->buf.data(), s->buf.size());
                    int64_t ok = r2 ? 1 : 0;
                    if (r2) roaring64_bitmap_free(r2);
                    return ok;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serState *>(sv);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 3;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // r64FrozenDeserialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:4034-4074
            {
                Entry e;
                e.name = "synthetic/r64FrozenDeserialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp r64FrozenDeserialize: setup "
                    "produces a 64-byte aligned frozen buffer; timed "
                    "region calls roaring64_bitmap_frozen_view() + free "
                    "per iteration (view construction + teardown, no "
                    "container copy).";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateFrozen;
                    s->r = roaring64_bitmap_create();
                    for (size_t i = 0; i < count; ++i)
                        roaring64_bitmap_add(s->r, i * step);
                    roaring64_bitmap_shrink_to_fit(s->r);
                    s->size = roaring64_bitmap_frozen_size_in_bytes(s->r);
                    s->buf = static_cast<char *>(
                        roaring_aligned_malloc(64, s->size));
                    roaring64_bitmap_frozen_serialize(s->r, s->buf);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateFrozen *>(sv);
                    roaring64_bitmap_t *r2 =
                        roaring64_bitmap_frozen_view(s->buf, s->size);
                    int64_t ok = r2 ? 1 : 0;
                    if (r2) roaring64_bitmap_free(r2);
                    return ok;
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serStateFrozen *>(sv);
                    roaring_aligned_free(s->buf);
                    roaring64_bitmap_free(s->r);
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 20;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // frsrPortableDeserialize
            {
                Entry e;
                e.name = "synthetic/frsrPortableDeserialize/" + ptag;
                e.description =
                    "frsr portable deserialize from native vm-backed wire "
                    "format prepared once in setup.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateFrsr;
                    for (size_t i = 0; i < count; ++i) {
                        s->r.add(static_cast<std::uint64_t>(i) * step);
                    }
                    s->r.serialize_to_vm_vector(s->serialized);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateFrsr *>(sv);
                    TestBitmap64 roundtrip =
                        TestBitmap64::deserialize_from_vm_vector(s->serialized);
                    return static_cast<int64_t>(roundtrip.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<serStateFrsr *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 3;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // frsrFrozenDeserialize
            {
                Entry e;
                e.name = "synthetic/frsrFrozenDeserialize/" + ptag;
                e.description =
                    "frsr frozen view construction over the native indexed "
                    "frozen wire format.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateFrsr;
                    for (size_t i = 0; i < count; ++i) {
                        s->r.add(static_cast<std::uint64_t>(i) * step);
                    }
                    s->r.optimize_for_storage();
                    s->r.serialize_frozen_to_vm_vector(s->serialized);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateFrsr *>(sv);
                    auto view = TestBitmap64::frozen_view_from_vm_vector(s->serialized);
                    return static_cast<int64_t>(view.size());
                };
                e.teardown = [](void *sv) {
                    delete static_cast<serStateFrsr *>(sv);
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 3;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppPortableDeserialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:4075-4107
            {
                Entry e;
                e.name = "synthetic/cppPortableDeserialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppPortableDeserialize: "
                    "Roaring64Map::read() on a portable buffer prepared "
                    "once in setup.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateCpp;
                    s->r = new Roaring64Map();
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    s->size = s->r->getSizeInBytes(/*portable=*/true);
                    s->buf.resize(s->size);
                    s->r->write(s->buf.data(), /*portable=*/true);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateCpp *>(sv);
                    Roaring64Map r2 =
                        Roaring64Map::read(s->buf.data(), /*portable=*/true);
                    return static_cast<int64_t>(r2.cardinality());
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serStateCpp *>(sv);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 3;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }

            // cppFrozenDeserialize
            // ported from deps/croaring/benchmarks/benchmark.cpp:4108-4139
            {
                Entry e;
                e.name = "synthetic/cppFrozenDeserialize/" + ptag;
                e.description =
                    "synthetic_bench.cpp cppFrozenDeserialize: "
                    "Roaring64Map::frozenView() on a frozen buffer "
                    "prepared once in setup.";
                e.setup = [count, step]() -> void * {
                    auto *s = new serStateCppFrozen;
                    s->r = new Roaring64Map();
                    for (size_t i = 0; i < count; ++i) s->r->add(i * step);
                    s->size = s->r->getFrozenSizeInBytes();
                    s->buf = static_cast<char *>(roaring_aligned_malloc(64, s->size));
                    s->r->writeFrozen(s->buf);
                    return s;
                };
                e.run = [](void *sv) -> int64_t {
                    auto *s = static_cast<serStateCppFrozen *>(sv);
                    Roaring64Map r2 = Roaring64Map::frozenView(s->buf, s->size);
                    return static_cast<int64_t>(r2.cardinality());
                };
                e.teardown = [](void *sv) {
                    auto *s = static_cast<serStateCppFrozen *>(sv);
                    roaring_aligned_free(s->buf);
                    delete s->r;
                    delete s;
                };
                e.ops_per_run = static_cast<int64_t>(count);
                e.inner_reps = 20;
                e.reusable_state = true;
                g_benchmarks.push_back(std::move(e));
            }
        }
    }
}

#endif // FRSR_ROARING_HAS_CROARING

} // namespace synthetic
// ported from deps/croaring/benchmarks/benchmark.cpp:4144-4297

#if FRSR_ROARING_HAS_CROARING

namespace density_contains {

constexpr size_t kSyntheticCount = 10000;
constexpr uint32_t kSyntheticUniverse = 1u << 18;
constexpr uint32_t kSyntheticBlockSize = 1u << 16;
constexpr uint32_t kSyntheticBlockCount =
    kSyntheticUniverse / kSyntheticBlockSize;
constexpr size_t kWarmRepeats = 1000;
constexpr size_t kWarmBitmaps = kSyntheticCount / kWarmRepeats;

struct Data {
    std::vector<roaring_bitmap_t *> low;
    std::vector<roaring_bitmap_t *> mod;
    std::vector<roaring_bitmap_t *> high;
    std::vector<TestBitmap32> low_frsr;
    std::vector<TestBitmap32> mod_frsr;
    std::vector<TestBitmap32> high_frsr;
    std::vector<uint32_t> cold_queries;
    std::vector<uint32_t> warm_queries;

    ~Data() {
        for (auto *b : low)  { roaring_bitmap_free(b); }
        for (auto *b : mod)  { roaring_bitmap_free(b); }
        for (auto *b : high) { roaring_bitmap_free(b); }
    }
};

// ported from deps/croaring/benchmarks/benchmark.cpp:4174-4196
static std::vector<roaring_bitmap_t *> build_bitmaps(double density,
                                                      uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::poisson_distribution<int> poisson(density * kSyntheticBlockSize);
    std::uniform_int_distribution<uint32_t> within_block(
        0, kSyntheticBlockSize - 1);
    std::vector<roaring_bitmap_t *> out(kSyntheticCount);
    for (size_t i = 0; i < kSyntheticCount; ++i) {
        out[i] = roaring_bitmap_create();
        for (uint32_t b = 0; b < kSyntheticBlockCount; ++b) {
            int n = poisson(rng);
            if (n < 0) n = 0;
            if (n > (int)kSyntheticBlockSize) n = (int)kSyntheticBlockSize;
            uint32_t base = b * kSyntheticBlockSize;
            for (int k = 0; k < n; ++k) {
                roaring_bitmap_add(out[i], base + within_block(rng));
            }

        }
        roaring_bitmap_run_optimize(out[i]);
        roaring_bitmap_shrink_to_fit(out[i]);
    }
    return out;
}

static std::vector<TestBitmap32> clone_to_frsr(std::vector<roaring_bitmap_t *> const & src) {
    std::vector<TestBitmap32> out;
    out.reserve(src.size());
    for (auto const * bm : src) {
        auto const card = static_cast<std::size_t>(roaring_bitmap_get_cardinality(bm));
        std::vector<std::uint32_t> values(card);
        roaring_bitmap_to_uint32_array(bm, values.data());
        out.emplace_back(values);
    }
    return out;
}

// Single shared dataset used by all entries in this namespace; built lazily
// on first setup() call, freed by release_data() after all scenarios run.
// ported from deps/croaring/benchmarks/benchmark.cpp:4198-4214
static Data *g_data = nullptr;

static Data *get_data() {
    if (g_data != nullptr) { return g_data; }
    g_data = new Data;
    g_data->low = build_bitmaps(0.001, 0xC0FFEE0001ULL);
    g_data->mod = build_bitmaps(0.01, 0xC0FFEE0002ULL);
    g_data->high = build_bitmaps(0.1, 0xC0FFEE0003ULL);
    g_data->low_frsr = clone_to_frsr(g_data->low);
    g_data->mod_frsr = clone_to_frsr(g_data->mod);
    g_data->high_frsr = clone_to_frsr(g_data->high);
    std::mt19937_64 rng(0xDEADBEEFULL);
    std::uniform_int_distribution<uint32_t> dist(0, kSyntheticUniverse - 1);
    g_data->cold_queries.resize(kSyntheticCount);
    for (size_t i = 0; i < kSyntheticCount; ++i) { g_data->cold_queries[i] = dist(rng); }
    g_data->warm_queries.resize(kWarmRepeats);
    for (size_t i = 0; i < kWarmRepeats; ++i) { g_data->warm_queries[i] = dist(rng); }
    return g_data;
}

static void release_data() {
    delete g_data;
    g_data = nullptr;
}

using Pick = const std::vector<roaring_bitmap_t *> &(*)(Data *);
using PickFrsr = const std::vector<TestBitmap32> &(*)(Data *);

// ported from deps/croaring/benchmarks/benchmark.cpp:4217-4225
static const std::vector<roaring_bitmap_t *> &pick_low(Data *d) {
    return d->low;
}
static const std::vector<roaring_bitmap_t *> &pick_mod(Data *d) {
    return d->mod;
}
static const std::vector<roaring_bitmap_t *> &pick_high(Data *d) {
    return d->high;
}
static const std::vector<TestBitmap32> &pick_low_frsr(Data *d) {
    return d->low_frsr;
}
static const std::vector<TestBitmap32> &pick_mod_frsr(Data *d) {
    return d->mod_frsr;
}
static const std::vector<TestBitmap32> &pick_high_frsr(Data *d) {
    return d->high_frsr;
}

// ported from deps/croaring/benchmarks/benchmark.cpp:4227-4254
static void add_cold(const char *name, const char *density_label, Pick pick) {
    Entry e;
    e.name = name;
    e.description =
        std::string("10,000 bitmaps over [0, 2^18) at ") + density_label +
        " density (per-block cardinality drawn from Poisson(density*2^16), "
        "values placed uniformly within each 2^16 block). Cold variant: "
        "issues one uniformly-random membership query against each of the "
        "10,000 bitmaps in turn, so every probe touches a fresh bitmap and "
        "the previous one is evicted from cache before its next access. "
        "Reported cost is per query.";
    e.setup = []() -> void * { return get_data(); };
    e.run = [pick](void *sv) -> int64_t {
        auto *d = static_cast<Data *>(sv);
        const auto &bms = pick(d);
        int64_t marker = 0;
        for (size_t i = 0; i < kSyntheticCount; ++i) {
            marker += roaring_bitmap_contains(bms[i], d->cold_queries[i]);
        }
        return marker;
    };
    e.teardown = nullptr;
    e.ops_per_run = static_cast<int64_t>(kSyntheticCount);
    e.inner_reps = 1;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// ported from deps/croaring/benchmarks/benchmark.cpp:4256-4286
static void add_warm(const char *name, const char *density_label, Pick pick) {
    Entry e;
    e.name = name;
    e.description =
        std::string("10,000 bitmaps over [0, 2^18) at ") + density_label +
        " density (per-block cardinality drawn from Poisson(density*2^16), "
        "values placed uniformly within each 2^16 block). Warm variant: "
        "probes the first 10 bitmaps 1,000 random times each, so every "
        "bitmap is fully cache-resident for all but the first probe. The "
        "total probe count (10,000) matches the cold variant, so per-query "
        "times are directly comparable.";
    e.setup = []() -> void * { return get_data(); };
    e.run = [pick](void *sv) -> int64_t {
        auto *d = static_cast<Data *>(sv);
        const auto &bms = pick(d);
        int64_t marker = 0;
        for (size_t i = 0; i < kWarmBitmaps; ++i) {
            roaring_bitmap_t *b = bms[i];
            for (size_t r = 0; r < kWarmRepeats; ++r) {
                marker += roaring_bitmap_contains(b, d->warm_queries[r]);
            }
        }
        return marker;
    };
    e.teardown = nullptr;
    e.ops_per_run = static_cast<int64_t>(kWarmBitmaps * kWarmRepeats);
    e.inner_reps = 1;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void add_cold_frsr(const char *name, const char *density_label, PickFrsr pick) {
    Entry e;
    e.name = name;
    e.description =
        std::string("frsr variant of ContainsCold at ") + density_label +
        " density over identical pre-generated bitmaps and query stream.";
    e.setup = []() -> void * { return get_data(); };
    e.run = [pick](void *sv) -> int64_t {
        auto *d = static_cast<Data *>(sv);
        auto const &bms = pick(d);
        int64_t marker = 0;
        for (size_t i = 0; i < kSyntheticCount; ++i) {
            marker += bms[i].contains(d->cold_queries[i]) ? 1 : 0;
        }
        return marker;
    };
    e.teardown = nullptr;
    e.ops_per_run = static_cast<int64_t>(kSyntheticCount);
    e.inner_reps = 1;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void add_warm_frsr(const char *name, const char *density_label, PickFrsr pick) {
    Entry e;
    e.name = name;
    e.description =
        std::string("frsr variant of ContainsWarm at ") + density_label +
        " density over identical pre-generated bitmaps and query stream.";
    e.setup = []() -> void * { return get_data(); };
    e.run = [pick](void *sv) -> int64_t {
        auto *d = static_cast<Data *>(sv);
        auto const &bms = pick(d);
        int64_t marker = 0;
        for (size_t i = 0; i < kWarmBitmaps; ++i) {
            auto const &b = bms[i];
            for (size_t r = 0; r < kWarmRepeats; ++r) {
                marker += b.contains(d->warm_queries[r]) ? 1 : 0;
            }
        }
        return marker;
    };
    e.teardown = nullptr;
    e.ops_per_run = static_cast<int64_t>(kWarmBitmaps * kWarmRepeats);
    e.inner_reps = 1;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// ported from deps/croaring/benchmarks/benchmark.cpp:4288-4297
static void register_benchmarks() {
    add_cold("synthetic/ContainsColdLow", "low (0.001)", pick_low);
    add_cold("synthetic/ContainsColdMod", "moderate (0.01)", pick_mod);
    add_cold("synthetic/ContainsColdHigh", "high (0.1)", pick_high);
    add_warm("synthetic/ContainsWarmLow", "low (0.001)", pick_low);
    add_warm("synthetic/ContainsWarmMod", "moderate (0.01)", pick_mod);
    add_warm("synthetic/ContainsWarmHigh", "high (0.1)", pick_high);
    add_cold_frsr("synthetic/frsrContainsColdLow", "low (0.001)", pick_low_frsr);
    add_cold_frsr("synthetic/frsrContainsColdMod", "moderate (0.01)", pick_mod_frsr);
    add_cold_frsr("synthetic/frsrContainsColdHigh", "high (0.1)", pick_high_frsr);
    add_warm_frsr("synthetic/frsrContainsWarmLow", "low (0.001)", pick_low_frsr);
    add_warm_frsr("synthetic/frsrContainsWarmMod", "moderate (0.01)", pick_mod_frsr);
    add_warm_frsr("synthetic/frsrContainsWarmHigh", "high (0.1)", pick_high_frsr);
    // Patch the last entry with a teardown that frees the shared dataset so its
    // ~96 MB of CRoaring bitmaps are returned to the allocator before the
    // set_ops scenarios (which allocate large working sets) begin.
    g_benchmarks.back().teardown = [](void *) { release_data(); };
}

} // namespace density_contains

// ========================================================================
// set_ops benchmark family
// Apples-to-apples comparison of set operations across all four backends:
//   frsr  = frsr::roaring::bitmap<uint32_t>
//   cpp   = CRoaring 32-bit C API (roaring_bitmap_t)
//   r64   = CRoaring 64-bit C API (roaring64_bitmap_t) with widened values
//   set   = std::set<uint64_t>
//
// [croaring-ref] deps/croaring/benchmarks/benchmark.cpp:SuccessiveOrBenchmark
// [croaring-ref] deps/croaring/benchmarks/benchmark.cpp:TotalUnion
// ========================================================================

namespace set_ops {

// Scenario dimensions:
//   count   ∈ {1'000, 100'000}
//   overlap ∈ {high (~75%), mid (~50%), low (~10%)}
//
// Bitmap A = [0, count)
// Bitmap B = [offset, offset + count)   where offset controls overlap:
//   high  → offset = count / 4          overlap ≈ 75 %
//   mid   → offset = count / 2          overlap ≈ 50 %
//   low   → offset = 9 * count / 10     overlap ≈ 10 %
//
// N-way union uses K=64 bitmaps, each shifted by count/K from the previous.

static constexpr std::size_t kCounts[]        = {1'000, 100'000};
static constexpr std::size_t kNWayFanout      = 64;
static constexpr int         kBinaryInnerReps = 20;
static constexpr int         kNWayInnerReps   = 5;

struct OverlapSpec {
    const char *label;   // "high"/"mid"/"low"
    // offset = count * num / den  (integer arithmetic, no FP)
    std::size_t num;
    std::size_t den;
};

static constexpr OverlapSpec kOverlaps[] = {
    { "high", 1, 4  },
    { "mid",  1, 2  },
    { "low",  9, 10 },
};

// ---- helper: bench name fragment ----
static std::string fmt_so(const char *lib, const char *op,
                           std::size_t count, const char *overlap) {
    char buf[128];
    snprintf(buf, sizeof(buf),
             "set_ops/%s%s/count=%zu/overlap=%s",
             lib, op, count, overlap);
    return buf;
}

// ========== frsr (32-bit) binary ops ==========

struct FrsrBinaryState {
    TestBitmap32 a;
    TestBitmap32 b;
    TestBitmap32 scratch;
};

static void register_frsr_binary(std::size_t count, std::size_t offset,
                                  const char *overlap) {
    // setup shared across the six frsr binary variants for this (count,overlap)
    auto make_state = [count, offset]() -> void * {
        auto *s = new FrsrBinaryState;
        for (std::size_t i = 0; i < count; ++i) { std::ignore = s->a.add(static_cast<std::uint32_t>(i)); }
        for (std::size_t i = 0; i < count; ++i) { std::ignore = s->b.add(static_cast<std::uint32_t>(i + offset)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrBinaryState *>(sv); };

    // frsr Union  (operator|)
    {
        Entry e;
        e.name        = fmt_so("frsr", "Union", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator| (materialised union). "
                        "Checksum = result cardinality.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a | s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr Intersection  (operator&)
    {
        Entry e;
        e.name        = fmt_so("frsr", "Intersection", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator& (materialised intersection).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a & s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr Difference  (operator-)
    {
        Entry e;
        e.name        = fmt_so("frsr", "Difference", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator- (materialised difference a\\b).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a - s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr UnionInplace  (operator|=)
    {
        Entry e;
        e.name        = fmt_so("frsr", "UnionInplace", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator|= (in-place union, copies a first).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 tmp = s->a;
            tmp |= s->b;
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr IntersectionInplace  (operator&=)
    {
        Entry e;
        e.name        = fmt_so("frsr", "IntersectionInplace", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator&= (in-place intersection, copies a first).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 tmp = s->a;
            tmp &= s->b;
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr DifferenceInplace  (operator-=)
    {
        Entry e;
        e.name        = fmt_so("frsr", "DifferenceInplace", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> operator-= (in-place difference, copies a first).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 tmp = s->a;
            tmp -= s->b;
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // frsr UnionScratch  (union_into — allocation-free path)
    {
        Entry e;
        e.name        = fmt_so("frsr", "UnionScratch", count, overlap);
        e.description = "frsr::roaring::bitmap<uint32_t> union_into(b, scratch) — "
                        "reuses scratch storage to avoid allocations per call. "
                        "frsr-only: no backend twin. "
                        "Checksum = result cardinality.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            s->a.union_into(s->b, s->scratch);
            return static_cast<int64_t>(s->scratch.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========== cpp (CRoaring 32-bit C API) binary ops ==========

struct CppBinaryState {
    roaring_bitmap_t *a{};
    roaring_bitmap_t *b{};

    ~CppBinaryState() {
        roaring_bitmap_free(a);
        roaring_bitmap_free(b);
    }
};

static void register_cpp_binary(std::size_t count, std::size_t offset,
                                  const char *overlap) {
    auto make_state = [count, offset]() -> void * {
        auto *s = new CppBinaryState;
        s->a = roaring_bitmap_create();
        s->b = roaring_bitmap_create();
        for (std::size_t i = 0; i < count; ++i) { roaring_bitmap_add(s->a, static_cast<uint32_t>(i)); }
        for (std::size_t i = 0; i < count; ++i) { roaring_bitmap_add(s->b, static_cast<uint32_t>(i + offset)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppBinaryState *>(sv); };

    // cpp Union
    {
        Entry e;
        e.name        = fmt_so("cpp", "Union", count, overlap);
        e.description = "CRoaring roaring_bitmap_or() (materialised union). Checksum = cardinality.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_or(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp Intersection
    {
        Entry e;
        e.name        = fmt_so("cpp", "Intersection", count, overlap);
        e.description = "CRoaring roaring_bitmap_and() (materialised intersection).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_and(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp Difference
    {
        Entry e;
        e.name        = fmt_so("cpp", "Difference", count, overlap);
        e.description = "CRoaring roaring_bitmap_andnot() (materialised difference a\\b).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_andnot(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp UnionInplace
    {
        Entry e;
        e.name        = fmt_so("cpp", "UnionInplace", count, overlap);
        e.description = "CRoaring roaring_bitmap_or_inplace() (copies a first, then |= b).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<CppBinaryState *>(sv);
            auto *tmp = roaring_bitmap_copy(s->a);
            roaring_bitmap_or_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(tmp));
            roaring_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp IntersectionInplace
    {
        Entry e;
        e.name        = fmt_so("cpp", "IntersectionInplace", count, overlap);
        e.description = "CRoaring roaring_bitmap_and_inplace() (copies a first, then &= b).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<CppBinaryState *>(sv);
            auto *tmp = roaring_bitmap_copy(s->a);
            roaring_bitmap_and_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(tmp));
            roaring_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp DifferenceInplace
    {
        Entry e;
        e.name        = fmt_so("cpp", "DifferenceInplace", count, overlap);
        e.description = "CRoaring roaring_bitmap_andnot_inplace() (copies a first, then -= b).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<CppBinaryState *>(sv);
            auto *tmp = roaring_bitmap_copy(s->a);
            roaring_bitmap_andnot_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(tmp));
            roaring_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========== r64 (CRoaring 64-bit C API) binary ops ==========
// Values widened to uint64_t (same numeric range, just stored 64-bit).

struct R64BinaryState {
    roaring64_bitmap_t *a{};
    roaring64_bitmap_t *b{};

    ~R64BinaryState() {
        roaring64_bitmap_free(a);
        roaring64_bitmap_free(b);
    }
};

static void register_r64_binary(std::size_t count, std::size_t offset,
                                  const char *overlap) {
    auto make_state = [count, offset]() -> void * {
        auto *s = new R64BinaryState;
        s->a = roaring64_bitmap_create();
        s->b = roaring64_bitmap_create();
        for (std::size_t i = 0; i < count; ++i) { roaring64_bitmap_add(s->a, static_cast<uint64_t>(i)); }
        for (std::size_t i = 0; i < count; ++i) { roaring64_bitmap_add(s->b, static_cast<uint64_t>(i + offset)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<R64BinaryState *>(sv); };

    // r64 Union
    {
        Entry e;
        e.name        = fmt_so("r64", "Union", count, overlap);
        e.description = "CRoaring roaring64_bitmap_or() (materialised union, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *r   = roaring64_bitmap_or(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(r));
            roaring64_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 Intersection
    {
        Entry e;
        e.name        = fmt_so("r64", "Intersection", count, overlap);
        e.description = "CRoaring roaring64_bitmap_and() (materialised intersection, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *r   = roaring64_bitmap_and(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(r));
            roaring64_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 Difference
    {
        Entry e;
        e.name        = fmt_so("r64", "Difference", count, overlap);
        e.description = "CRoaring roaring64_bitmap_andnot() (materialised difference, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *r   = roaring64_bitmap_andnot(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(r));
            roaring64_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 UnionInplace
    {
        Entry e;
        e.name        = fmt_so("r64", "UnionInplace", count, overlap);
        e.description = "CRoaring roaring64_bitmap_or_inplace() (copies a first, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *tmp = roaring64_bitmap_copy(s->a);
            roaring64_bitmap_or_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(tmp));
            roaring64_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 IntersectionInplace
    {
        Entry e;
        e.name        = fmt_so("r64", "IntersectionInplace", count, overlap);
        e.description = "CRoaring roaring64_bitmap_and_inplace() (copies a first, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *tmp = roaring64_bitmap_copy(s->a);
            roaring64_bitmap_and_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(tmp));
            roaring64_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 DifferenceInplace
    {
        Entry e;
        e.name        = fmt_so("r64", "DifferenceInplace", count, overlap);
        e.description = "CRoaring roaring64_bitmap_andnot_inplace() (copies a first, 64-bit).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64BinaryState *>(sv);
            auto *tmp = roaring64_bitmap_copy(s->a);
            roaring64_bitmap_andnot_inplace(tmp, s->b);
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(tmp));
            roaring64_bitmap_free(tmp);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========== set (std::set<uint64_t>) binary ops ==========

struct SetBinaryState {
    std::set<uint64_t> a;
    std::set<uint64_t> b;
};

static void register_set_binary(std::size_t count, std::size_t offset,
                                  const char *overlap) {
    auto make_state = [count, offset]() -> void * {
        auto *s = new SetBinaryState;
        for (std::size_t i = 0; i < count; ++i) { s->a.insert(static_cast<uint64_t>(i)); }
        for (std::size_t i = 0; i < count; ++i) { s->b.insert(static_cast<uint64_t>(i + offset)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<SetBinaryState *>(sv); };

    // set Union
    {
        Entry e;
        e.name        = fmt_so("set", "Union", count, overlap);
        e.description = "std::set<uint64_t> std::set_union into a temporary vector; checksum = result size.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::vector<uint64_t> out;
            out.reserve(s->a.size() + s->b.size());
            std::set_union(s->a.begin(), s->a.end(),
                           s->b.begin(), s->b.end(),
                           std::back_inserter(out));
            return static_cast<int64_t>(out.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set Intersection
    {
        Entry e;
        e.name        = fmt_so("set", "Intersection", count, overlap);
        e.description = "std::set<uint64_t> std::set_intersection into a temporary vector.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::vector<uint64_t> out;
            out.reserve(std::min(s->a.size(), s->b.size()));
            std::set_intersection(s->a.begin(), s->a.end(),
                                  s->b.begin(), s->b.end(),
                                  std::back_inserter(out));
            return static_cast<int64_t>(out.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set Difference
    {
        Entry e;
        e.name        = fmt_so("set", "Difference", count, overlap);
        e.description = "std::set<uint64_t> std::set_difference (a\\b) into a temporary vector.";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::vector<uint64_t> out;
            out.reserve(s->a.size());
            std::set_difference(s->a.begin(), s->a.end(),
                                s->b.begin(), s->b.end(),
                                std::back_inserter(out));
            return static_cast<int64_t>(out.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set UnionInplace
    {
        Entry e;
        e.name        = fmt_so("set", "UnionInplace", count, overlap);
        e.description = "std::set<uint64_t> copy a, then insert all of b (in-place union analogue).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::set<uint64_t> tmp = s->a;
            tmp.insert(s->b.begin(), s->b.end());
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set IntersectionInplace
    {
        Entry e;
        e.name        = fmt_so("set", "IntersectionInplace", count, overlap);
        e.description = "std::set<uint64_t> copy a, erase elements not in b (in-place intersection analogue).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::set<uint64_t> tmp = s->a;
            for (auto it = tmp.begin(); it != tmp.end(); ) {
                if (s->b.find(*it) == s->b.end()) {
                    it = tmp.erase(it);
                } else {
                    ++it;
                }
            }
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set DifferenceInplace
    {
        Entry e;
        e.name        = fmt_so("set", "DifferenceInplace", count, overlap);
        e.description = "std::set<uint64_t> copy a, erase all elements of b (in-place difference analogue).";
        e.setup    = make_state;
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetBinaryState *>(sv);
            std::set<uint64_t> tmp = s->a;
            for (auto const &v : s->b) { tmp.erase(v); }
            return static_cast<int64_t>(tmp.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========== N-way UnionMany ==========

struct FrsrNWayState {
    std::vector<TestBitmap32>       bitmaps;
    std::vector<TestBitmap32 const *> ptrs;
    TestBitmap32                    accumulator;
};

struct CppNWayState {
    std::vector<roaring_bitmap_t *> bitmaps;
    std::vector<roaring_bitmap_t const *> ptrs;

    ~CppNWayState() {
        for (auto *b : bitmaps) { roaring_bitmap_free(b); }
    }
};

struct R64NWayState {
    std::vector<roaring64_bitmap_t *> bitmaps;

    ~R64NWayState() {
        for (auto *b : bitmaps) { roaring64_bitmap_free(b); }
    }
};

struct SetNWayState {
    std::vector<std::set<uint64_t>> bitmaps;
};

static void register_nway_union(std::size_t count) {
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "count=%zu", count);
    std::string ptag(name_buf);

    // Each bitmap k covers [k*shift, k*shift + count) where shift = count/K.
    // That gives moderate pairwise overlap (~(K-1)/K unique total = ~(1-1/K)*count*K).
    std::size_t const shift = std::max(std::size_t{1}, count / kNWayFanout);

    // frsr UnionMany  (or_many_in_place)
    {
        Entry e;
        e.name        = "set_ops/frsr" "UnionMany/" + ptag;
        e.description = "frsr::roaring::bitmap<uint32_t> or_many_in_place() over K=64 operands. "
                        "[croaring-ref] deps/croaring/benchmarks/benchmark.cpp:TotalUnion";
        e.setup    = [count, shift]() -> void * {
            auto *s = new FrsrNWayState;
            s->bitmaps.resize(kNWayFanout);
            s->ptrs.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                for (std::size_t i = 0; i < count; ++i) {
                    std::ignore = s->bitmaps[k].add(static_cast<std::uint32_t>(k * shift + i));
                }
                s->ptrs[k] = &s->bitmaps[k];
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrNWayState *>(sv);
            s->accumulator = TestBitmap32{};
            s->accumulator.or_many_in_place({ s->ptrs.data(), s->ptrs.size() });
            return static_cast<int64_t>(s->accumulator.size());
        };
        e.teardown      = [](void *sv) { delete static_cast<FrsrNWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // cpp UnionMany  (roaring_bitmap_or_many)
    {
        Entry e;
        e.name        = "set_ops/cpp" "UnionMany/" + ptag;
        e.description = "CRoaring roaring_bitmap_or_many() over K=64 operands. "
                        "[croaring-ref] deps/croaring/benchmarks/benchmark.cpp:TotalUnion";
        e.setup    = [count, shift]() -> void * {
            auto *s = new CppNWayState;
            s->bitmaps.resize(kNWayFanout);
            s->ptrs.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                s->bitmaps[k] = roaring_bitmap_create();
                for (std::size_t i = 0; i < count; ++i) {
                    roaring_bitmap_add(s->bitmaps[k], static_cast<uint32_t>(k * shift + i));
                }
                s->ptrs[k] = s->bitmaps[k];
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppNWayState *>(sv);
            auto *r   = roaring_bitmap_or_many(kNWayFanout, s->ptrs.data());
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = [](void *sv) { delete static_cast<CppNWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // r64 UnionMany  (successive roaring64_bitmap_or_inplace — no or_many in r64 API)
    {
        Entry e;
        e.name        = "set_ops/r64" "UnionMany/" + ptag;
        e.description = "CRoaring 64-bit: successive roaring64_bitmap_or_inplace() over K=64 operands "
                        "(r64 has no or_many equivalent). "
                        "[croaring-ref] deps/croaring/benchmarks/benchmark.cpp:SuccessiveOrBenchmark";
        e.setup    = [count, shift]() -> void * {
            auto *s = new R64NWayState;
            s->bitmaps.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                s->bitmaps[k] = roaring64_bitmap_create();
                for (std::size_t i = 0; i < count; ++i) {
                    roaring64_bitmap_add(s->bitmaps[k], static_cast<uint64_t>(k * shift + i));
                }
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<R64NWayState *>(sv);
            auto *acc = roaring64_bitmap_copy(s->bitmaps[0]);
            for (std::size_t k = 1; k < kNWayFanout; ++k) {
                roaring64_bitmap_or_inplace(acc, s->bitmaps[k]);
            }
            int64_t c = static_cast<int64_t>(roaring64_bitmap_get_cardinality(acc));
            roaring64_bitmap_free(acc);
            return c;
        };
        e.teardown      = [](void *sv) { delete static_cast<R64NWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }

    // set UnionMany  (std::set fold)
    {
        Entry e;
        e.name        = "set_ops/set" "UnionMany/" + ptag;
        e.description = "std::set<uint64_t>: fold K=64 sets via insert-range (union-all). "
                        "[croaring-ref] deps/croaring/benchmarks/benchmark.cpp:SuccessiveOrBenchmark";
        e.setup    = [count, shift]() -> void * {
            auto *s = new SetNWayState;
            s->bitmaps.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                for (std::size_t i = 0; i < count; ++i) {
                    s->bitmaps[k].insert(static_cast<uint64_t>(k * shift + i));
                }
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<SetNWayState *>(sv);
            std::set<uint64_t> acc = s->bitmaps[0];
            for (std::size_t k = 1; k < kNWayFanout; ++k) {
                acc.insert(s->bitmaps[k].begin(), s->bitmaps[k].end());
            }
            return static_cast<int64_t>(acc.size());
        };
        e.teardown      = [](void *sv) { delete static_cast<SetNWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========== sparse (many-chunk) variants ==========
//
// The dense scenarios above use contiguous value ranges, which collapse to a handful of
// bitset chunks — so they never exercise the many-chunk key-merge walk (the AoS-vs-SoA
// lever) nor the per-chunk dispatch over many small containers. These sparse variants
// spread each operand's `count` values kSparsePerChunk-per-chunk across many chunks (each
// a small ARRAY container), which is the regime that matches a downstream engine's real
// (sparse) index data and the only regime where the SoA `keys[]` walk can show a win.

static constexpr std::size_t kSparsePerChunk = 4;

// value at logical index `i` for an operand whose chunk range starts at `chunk_base`.
static inline std::uint32_t sparse_value(std::size_t chunk_base, std::size_t i) {
    std::size_t const chunk = chunk_base + i / kSparsePerChunk;
    std::size_t const low   = i % kSparsePerChunk;
    return static_cast<std::uint32_t>(chunk * std::size_t{ 65536 } + low);
}

static void register_frsr_binary_sparse(std::size_t count, std::size_t b_chunk_offset) {
    auto make_state = [count, b_chunk_offset]() -> void * {
        auto *s = new FrsrBinaryState;
        for (std::size_t i = 0; i < count; ++i) { std::ignore = s->a.add(sparse_value(0,             i)); }
        for (std::size_t i = 0; i < count; ++i) { std::ignore = s->b.add(sparse_value(b_chunk_offset, i)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrBinaryState *>(sv); };

    {
        Entry e;
        e.name        = fmt_so("frsr", "Union", count, "sparse");
        e.description  = "frsr::roaring::bitmap<uint32_t> operator| over many small array chunks.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a | s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
    {
        Entry e;
        e.name        = fmt_so("frsr", "Intersection", count, "sparse");
        e.description  = "frsr::roaring::bitmap<uint32_t> operator& over many small array chunks.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a & s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
    {
        Entry e;
        e.name        = fmt_so("frsr", "Difference", count, "sparse");
        e.description  = "frsr::roaring::bitmap<uint32_t> operator- over many small array chunks.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrBinaryState *>(sv);
            TestBitmap32 r = s->a - s->b;
            return static_cast<int64_t>(r.size());
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

static void register_cpp_binary_sparse(std::size_t count, std::size_t b_chunk_offset) {
    auto make_state = [count, b_chunk_offset]() -> void * {
        auto *s = new CppBinaryState;
        s->a = roaring_bitmap_create();
        s->b = roaring_bitmap_create();
        for (std::size_t i = 0; i < count; ++i) { roaring_bitmap_add(s->a, sparse_value(0,             i)); }
        for (std::size_t i = 0; i < count; ++i) { roaring_bitmap_add(s->b, sparse_value(b_chunk_offset, i)); }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppBinaryState *>(sv); };

    {
        Entry e;
        e.name        = fmt_so("cpp", "Union", count, "sparse");
        e.description  = "CRoaring roaring_bitmap_or() over many small array containers.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_or(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
    {
        Entry e;
        e.name        = fmt_so("cpp", "Intersection", count, "sparse");
        e.description  = "CRoaring roaring_bitmap_and() over many small array containers.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_and(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
    {
        Entry e;
        e.name        = fmt_so("cpp", "Difference", count, "sparse");
        e.description  = "CRoaring roaring_bitmap_andnot() over many small array containers.";
        e.setup        = make_state;
        e.run          = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppBinaryState *>(sv);
            auto *r   = roaring_bitmap_andnot(s->a, s->b);
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = free_state;
        e.ops_per_run   = 1;
        e.inner_reps    = kBinaryInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

static void register_nway_union_sparse(std::size_t count) {
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "count=%zu", count);
    std::string ptag(name_buf);

    // Each operand k covers nchunks chunks starting at chunk k*chunk_shift, kSparsePerChunk
    // values per chunk — so the K=64 folds interleave across thousands of small array chunks.
    std::size_t const nchunks     = std::max(std::size_t{ 1 }, count / kSparsePerChunk);
    std::size_t const chunk_shift = std::max(std::size_t{ 1 }, nchunks / kNWayFanout);

    {
        Entry e;
        e.name        = "set_ops/frsr" "UnionManySparse/" + ptag;
        e.description = "frsr::roaring::bitmap<uint32_t> or_many_in_place() over K=64 sparse "
                        "(many small array chunk) operands.";
        e.setup    = [count, chunk_shift]() -> void * {
            auto *s = new FrsrNWayState;
            s->bitmaps.resize(kNWayFanout);
            s->ptrs.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                for (std::size_t i = 0; i < count; ++i) {
                    std::ignore = s->bitmaps[k].add(sparse_value(k * chunk_shift, i));
                }
                s->ptrs[k] = &s->bitmaps[k];
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s = static_cast<FrsrNWayState *>(sv);
            s->accumulator = TestBitmap32{};
            s->accumulator.or_many_in_place({ s->ptrs.data(), s->ptrs.size() });
            return static_cast<int64_t>(s->accumulator.size());
        };
        e.teardown      = [](void *sv) { delete static_cast<FrsrNWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
    {
        Entry e;
        e.name        = "set_ops/cpp" "UnionManySparse/" + ptag;
        e.description = "CRoaring roaring_bitmap_or_many() over K=64 sparse operands.";
        e.setup    = [count, chunk_shift]() -> void * {
            auto *s = new CppNWayState;
            s->bitmaps.resize(kNWayFanout);
            s->ptrs.resize(kNWayFanout);
            for (std::size_t k = 0; k < kNWayFanout; ++k) {
                s->bitmaps[k] = roaring_bitmap_create();
                for (std::size_t i = 0; i < count; ++i) {
                    roaring_bitmap_add(s->bitmaps[k], sparse_value(k * chunk_shift, i));
                }
                s->ptrs[k] = s->bitmaps[k];
            }
            return s;
        };
        e.run      = [](void *sv) -> int64_t {
            auto *s   = static_cast<CppNWayState *>(sv);
            auto *r   = roaring_bitmap_or_many(kNWayFanout, s->ptrs.data());
            int64_t c = static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
            return c;
        };
        e.teardown      = [](void *sv) { delete static_cast<CppNWayState *>(sv); };
        e.ops_per_run   = static_cast<int64_t>(kNWayFanout);
        e.inner_reps    = kNWayInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========================================================================
// Run-heavy repeated intersection
//
// The Union/Intersection/Difference families above build each operand from a
// SINGLE contiguous add() range, which collapses to exactly one run per
// operand — never exercising the run-container merge (intersect_run_run)
// under realistic call volume. That gap is precisely what let a regression
// (container_handle::payload_data_raw() re-deriving the spilled payload
// pointer through std::launder on every `.runs[i]` index, instead of once
// before the loop) slip past this suite while still causing a large
// regression on multiple downstream workloads — real workloads intersect
// run-encoded, interval-like sets at high call volume, not once per
// benchmark iteration.
//
// This family rebuilds each operand from kRunHeavyNumRuns DISJOINT closed
// ranges (moderate run count, matching the finding), forces run-container
// encoding via optimize()/run_optimize(), and calls intersect() `count` times
// in a tight loop inside a single timed run() — i.e. `count` is the repeated
// small-intersect call volume, not an operand cardinality. B's ranges are
// offset by half the run LENGTH so each of B's runs overlaps the
// corresponding run of A by half its length (~50% overlap): the merge walk
// does real work instead of a trivial all-hit/all-miss short-circuit.
// ========================================================================

static constexpr std::size_t kRunHeavyNumRuns   = 1000;
static constexpr std::size_t kRunHeavyRunLength = 30;
static constexpr std::size_t kRunHeavyStride    = 60;
static constexpr int         kRunHeavyInnerReps = 5;

static void add_run_heavy_ranges_frsr(TestBitmap32 &bitmap, std::size_t start_offset) {
    for (std::size_t run = 0; run < kRunHeavyNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(start_offset + run * kRunHeavyStride);
        auto const end   = static_cast<std::uint32_t>(begin + kRunHeavyRunLength - 1);
        bitmap.add_closed_range(begin, end);
    }
}

static void add_run_heavy_ranges_cpp(roaring_bitmap_t *bitmap, std::size_t start_offset) {
    for (std::size_t run = 0; run < kRunHeavyNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(start_offset + run * kRunHeavyStride);
        auto const end   = static_cast<std::uint32_t>(begin + kRunHeavyRunLength - 1);
        roaring_bitmap_add_range_closed(bitmap, begin, end);
    }
}

struct FrsrRunHeavyState {
    TestBitmap32 a;
    TestBitmap32 b;
};

static void register_frsr_run_heavy(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrRunHeavyState;
        add_run_heavy_ranges_frsr(s->a, 0);
        // Offset by half the RUN LENGTH (not half the stride — the gap between
        // consecutive runs is stride - run_length, so a half-stride shift lands
        // in that gap and produces zero overlap): this way each of B's runs
        // overlaps the corresponding run of A by exactly half its length.
        add_run_heavy_ranges_frsr(s->b, kRunHeavyRunLength / 2);
        s->a.optimize();
        s->b.optimize();
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrRunHeavyState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "RunHeavyIntersect", repeat, "runs");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& over run-encoded, disjoint-range "
                     "operands (" + std::to_string(kRunHeavyNumRuns) + " runs/operand), called " +
                     std::to_string(repeat) + " times per timed run — reproduces the "
                     "intersect_run_run repeated-payload-pointer-derivation regression (2026-07-12).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrRunHeavyState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->a & s->b;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kRunHeavyInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

struct CppRunHeavyState {
    roaring_bitmap_t *a{};
    roaring_bitmap_t *b{};

    ~CppRunHeavyState() {
        roaring_bitmap_free(a);
        roaring_bitmap_free(b);
    }
};

static void register_cpp_run_heavy(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppRunHeavyState;
        s->a = roaring_bitmap_create();
        s->b = roaring_bitmap_create();
        add_run_heavy_ranges_cpp(s->a, 0);
        add_run_heavy_ranges_cpp(s->b, kRunHeavyRunLength / 2);
        roaring_bitmap_run_optimize(s->a);
        roaring_bitmap_run_optimize(s->b);
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppRunHeavyState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "RunHeavyIntersect", repeat, "runs");
    e.description = "CRoaring roaring_bitmap_and() over run-encoded, disjoint-range operands — "
                     "apples-to-apples counterpart of the frsr RunHeavyIntersect scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppRunHeavyState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->a, s->b);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kRunHeavyInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// ========================================================================
// Run-heavy N-way AND fold (array accumulator ∩ run operands)
//
// Reproduces the shape a downstream N-way AND fold hits that the plain
// binary-op families miss: a fold where the accumulator holds
// ARRAY containers and the operands hold RUN containers over clustered ID
// ranges — the encoding optimize()/run_optimize() produces at storage time
// for contiguous member-ID ranges. A differential A/B on a downstream
// workload put the whole remaining gap under this call site: the reference
// implementation walks each run once and block-copies the covered array
// slice, while a per-VALUE intersect_array_run steps every accumulator
// element past the run list.
//
// Two variants:
//  - ArrayRunAndFold: the first AND materializes a fresh (sole-owned)
//    accumulator, the remaining K-1 fold in place at rc==1 — the steady
//    state of a fold over unique bitmaps.
//  - ArrayRunAndFoldShared: the accumulator is seeded as a shallow
//    copy-on-write copy of a persistent bitmap (rc > 1 on first touch), so
//    the write-barrier/clone cost of folding over SHARED containers is
//    measured too, exactly as a downstream engine's fold starts from a
//    copied first operand. The CRoaring arm enables copy_on_write for parity.
//
// Operand k's runs are offset by a few values so every fold step trims the
// accumulator (real work, no all-hit short circuit), while the accumulator
// stays array-encoded end to end.
// ========================================================================

static constexpr std::size_t kArrayRunFoldNumRuns   = 100;
static constexpr std::size_t kArrayRunFoldRunLength = 60;
static constexpr std::size_t kArrayRunFoldStride    = 120;
static constexpr std::size_t kArrayRunFoldSeedStep  = 3;    // 4000 values < 4096 => stays array
static constexpr std::size_t kArrayRunFoldOperands  = 4;
static constexpr std::size_t kArrayRunFoldOpShift   = 7;    // per-operand run offset
static constexpr int         kArrayRunFoldInnerReps = 5;

struct FrsrArrayRunFoldState {
    TestBitmap32 seed;
    TestBitmap32 ops[kArrayRunFoldOperands];
};

struct CppArrayRunFoldState {
    roaring_bitmap_t *seed{};
    roaring_bitmap_t *ops[kArrayRunFoldOperands]{};

    ~CppArrayRunFoldState() {
        roaring_bitmap_free(seed);
        for (auto *op : ops) {
            roaring_bitmap_free(op);
        }
    }
};

static void *make_frsr_array_run_fold_state() {
    auto *s = new FrsrArrayRunFoldState;
    auto const domain = kArrayRunFoldNumRuns * kArrayRunFoldStride;
    for (std::size_t v = 0; v < domain; v += kArrayRunFoldSeedStep) {
        (void)s->seed.add(static_cast<std::uint32_t>(v));   // individual adds keep array encoding
    }
    for (std::size_t k = 0; k < kArrayRunFoldOperands; ++k) {
        for (std::size_t run = 0; run < kArrayRunFoldNumRuns; ++run) {
            auto const begin = static_cast<std::uint32_t>(k * kArrayRunFoldOpShift + run * kArrayRunFoldStride);
            s->ops[k].add_closed_range(begin, static_cast<std::uint32_t>(begin + kArrayRunFoldRunLength - 1));
        }
        s->ops[k].optimize();   // force run encoding, as storage-time optimize does
    }
    return s;
}

static void *make_cpp_array_run_fold_state() {
    auto *s = new CppArrayRunFoldState;
    s->seed = roaring_bitmap_create();
    auto const domain = kArrayRunFoldNumRuns * kArrayRunFoldStride;
    for (std::size_t v = 0; v < domain; v += kArrayRunFoldSeedStep) {
        roaring_bitmap_add(s->seed, static_cast<std::uint32_t>(v));
    }
    for (std::size_t k = 0; k < kArrayRunFoldOperands; ++k) {
        s->ops[k] = roaring_bitmap_create();
        for (std::size_t run = 0; run < kArrayRunFoldNumRuns; ++run) {
            auto const begin = static_cast<std::uint32_t>(k * kArrayRunFoldOpShift + run * kArrayRunFoldStride);
            roaring_bitmap_add_range_closed(s->ops[k], begin, static_cast<std::uint32_t>(begin + kArrayRunFoldRunLength - 1));
        }
        roaring_bitmap_run_optimize(s->ops[k]);
    }
    return s;
}

// ========================================================================
// Lazy-union promotion band — the array/bitset form decision and what it
// costs DOWNSTREAM.
//
// CRoaring promotes a lazy array∪array union result to a bitset once the
// union exceeds ARRAY_LAZY_LOWERBOUND (1024), a quarter of its array/bitset
// threshold (containers/perfparameters.h, mixed_union.c). The rationale
// (CRoaring's own comment) is not about the union: it is that an accumulated
// union is subsequently INTERSECTED against many operands, and array∩bitset
// (one direct bit test per element of the small side) is markedly cheaper
// than array∩array (a two-sided merge).
//
// A union-only benchmark therefore prefers arrays and picks the wrong
// threshold — the whole payoff is in the consumers. These benchmarks exist
// so that trap is measured rather than reasoned about:
//
//   * AndSmallVsBand — the cost model in isolation: small array ∩ a
//     container of cardinality `band`, held as an array vs as a bitset.
//   * LazyUnionFold  — the end-to-end shape: lazily accumulate a union,
//     then fold many small operands against it.
//
// Both are parameterised by cardinality band, with the interesting window
// (1024–4096, where the two libraries' policies disagree) bracketed by two
// bands where they agree — below 1024 both hold arrays, above 4096 both
// hold bitsets. Those two are controls: a change that moves them is a
// change to something other than the promotion policy.
// ========================================================================

namespace band_fold {

static constexpr std::size_t kChunks    { 64 };   // distinct 2^16 chunks touched
static constexpr std::size_t kOperands  { 16 };   // bitmaps lazily OR-ed into the accumulator
static constexpr std::size_t kProbes    { 32 };   // small operands folded against it
static constexpr std::size_t kProbeCard { 64 };   // elements per chunk in each probe (the small side)
static const std::int64_t kInnerReps{ []{
    if (char const *s = std::getenv("FRSR_BENCH_BAND_REPS")) { return static_cast<std::int64_t>(std::strtoll(s, nullptr, 10)); }
    return std::int64_t{ 200 };
}() };

// Values are strided across each chunk so the operands stay array/bitset
// shaped and never collapse into run containers (which would measure a
// different kernel entirely).
static std::uint32_t band_value(std::size_t chunk, std::size_t band, std::size_t index) {
    auto const stride = 65536U / static_cast<std::uint32_t>(band);
    return static_cast<std::uint32_t>(chunk) * 65536U + static_cast<std::uint32_t>(index) * stride;
}

struct FrsrState {
    std::vector<TestBitmap32> sources;
    std::vector<TestBitmap32> probes;
    TestBitmap32              target;   // AndSmallVsBand only: the pre-built band container
};

// Source k owns every kOperands-th element of the band, so the K-way union
// covers the band exactly and each individual operand stays sparse.
static void fill_frsr_sources(FrsrState &s, std::size_t band) {
    s.sources.resize(kOperands);
    for (std::size_t k = 0; k < kOperands; ++k) {
        for (std::size_t c = 0; c < kChunks; ++c) {
            for (std::size_t i = k; i < band; i += kOperands) {
                std::ignore = s.sources[k].add(band_value(c, band, i));
            }
        }
    }
}

// Probes hit elements that are present in the union, so the intersection is
// non-degenerate; each probe picks a different offset into the band.
static void fill_frsr_probes(FrsrState &s, std::size_t band) {
    s.probes.resize(kProbes);
    auto const step = std::max<std::size_t>(1, band / kProbeCard);
    for (std::size_t p = 0; p < kProbes; ++p) {
        for (std::size_t c = 0; c < kChunks; ++c) {
            for (std::size_t j = 0; j < kProbeCard; ++j) {
                std::ignore = s.probes[p].add(band_value(c, band, (j * step + p) % band));
            }
        }
    }
}

} // namespace band_fold

// --- A: the cost model in isolation -------------------------------------
// Identical contents, identical operation; the ONLY difference is whether the
// right-hand operand is held as an array or as a bitset. `as_bitset` forces the
// promotion explicitly via promote_large_arrays(), isolating the form's cost from
// any question of which build path would choose it.
static void register_frsr_and_small_vs_band(std::size_t band, bool as_bitset) {
    using namespace band_fold;
    Entry e;
    char buf[128];
    snprintf(buf, sizeof(buf), "set_ops/frsrAndSmallVsBand/band=%zu/form=%s",
             band, as_bitset ? "bitset" : "array");
    e.name        = buf;
    e.description = std::string("frsr small-array ∩ cardinality-") + std::to_string(band) +
                    " container held as " + (as_bitset ? "a bitset" : "an array") +
                    ". Isolates the downstream cost that sets the lazy-union promotion point "
                    "([croaring-ref] deps/croaring/src/containers/mixed_union.c:"
                    "array_array_container_lazy_union, ARRAY_LAZY_LOWERBOUND).";
    e.setup       = [band, as_bitset]() -> void * {
        auto *s = new FrsrState;
        for (std::size_t c = 0; c < kChunks; ++c) {
            for (std::size_t i = 0; i < band; ++i) {
                std::ignore = s->target.add(band_value(c, band, i));
            }
        }
        if (as_bitset) {
            s->target.promote_large_arrays(1);   // force every chunk to bitset form
        }
        fill_frsr_probes(*s, band);
        return s;
    };
    e.run         = [](void *sv) -> int64_t {
        auto *s = static_cast<FrsrState *>(sv);
        int64_t checksum = 0;
        for (std::size_t p = 0; p < kProbes; ++p) {
            TestBitmap32 probe = s->probes[p];
            probe &= s->target;
            checksum += static_cast<int64_t>(probe.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(kProbes * kChunks);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// --- B: the end-to-end shape --------------------------------------------
// Lazily accumulate a K-way union, then fold small operands against it — the
// shape whose form decision the promotion threshold governs. Measured
// end-to-end on purpose: the union phase pays for the promotion and the fold
// phase collects on it, and only their sum says whether it was worth it.
static void register_frsr_lazy_union_fold(std::size_t band) {
    using namespace band_fold;
    Entry e;
    char buf[128];
    snprintf(buf, sizeof(buf), "set_ops/frsrLazyUnionFold/band=%zu", band);
    e.name        = buf;
    e.description = "frsr lazy K-way union accumulate (bulk_or_intermediate + "
                    "bulk_or_finish_keep_bitsets) then AND-fold of " + std::to_string(kProbes) +
                    " small operands, per-chunk union cardinality ~" + std::to_string(band) + ".";
    e.setup       = [band]() -> void * {
        auto *s = new FrsrState;
        fill_frsr_sources(*s, band);
        fill_frsr_probes (*s, band);
        return s;
    };
    e.run         = [](void *sv) -> int64_t {
        auto *s = static_cast<FrsrState *>(sv);
        TestBitmap32 acc;
        for (auto const &src : s->sources) {
            acc.bulk_or_intermediate(src);
        }
        acc.bulk_or_finish_keep_bitsets();
        int64_t checksum = static_cast<int64_t>(acc.size());
        for (std::size_t p = 0; p < kProbes; ++p) {
            TestBitmap32 probe = s->probes[p];
            probe &= acc;
            checksum += static_cast<int64_t>(probe.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>((kOperands + kProbes) * kChunks);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// Same shape, but finishing through optimize() — the finish an index-building
// caller actually performs. optimize() re-decides array-vs-bitset at
// array_to_bitset_threshold and so discards a lazy union's promotion below it,
// where optimize_keep_bitsets() (CRoaring's run_optimize semantics) does not.
// This variant, not the one above, is the regression guard: it should track
// the keep-bitsets variant, and any divergence at band=2048 while the 512 and
// 8192 controls stay level is that down-conversion coming back.
static void register_frsr_lazy_union_fold_optimized(std::size_t band, bool keep_bitsets) {
    using namespace band_fold;
    Entry e;
    char buf[128];
    snprintf(buf, sizeof(buf), "set_ops/frsrLazyUnionFoldOptimize/band=%zu/finish=%s",
             band, keep_bitsets ? "keep_bitsets" : "optimize");
    e.name        = buf;
    e.description = std::string("frsr lazy K-way union accumulate then ") +
                    (keep_bitsets ? "optimize_keep_bitsets()" : "optimize()") +
                    ", then AND-fold of " + std::to_string(kProbes) +
                    " small operands, per-chunk union cardinality ~" + std::to_string(band) +
                    " — the index-build finish path.";
    e.setup       = [band]() -> void * {
        auto *s = new FrsrState;
        fill_frsr_sources(*s, band);
        fill_frsr_probes (*s, band);
        return s;
    };
    e.run         = [keep_bitsets](void *sv) -> int64_t {
        auto *s = static_cast<FrsrState *>(sv);
        TestBitmap32 acc;
        for (auto const &src : s->sources) {
            acc.bulk_or_intermediate(src);
        }
        acc.bulk_or_finish_keep_bitsets();
        if (keep_bitsets) { acc.optimize_keep_bitsets(); } else { acc.optimize(); }
        int64_t checksum = static_cast<int64_t>(acc.size());
        for (std::size_t p = 0; p < kProbes; ++p) {
            TestBitmap32 probe = s->probes[p];
            probe &= acc;
            checksum += static_cast<int64_t>(probe.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>((kOperands + kProbes) * kChunks);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
struct CppBandState {
    std::vector<roaring_bitmap_t *> sources;
    std::vector<roaring_bitmap_t *> probes;
    ~CppBandState() {
        for (auto *b : sources) { roaring_bitmap_free(b); }
        for (auto *b : probes ) { roaring_bitmap_free(b); }
    }
};

// CRoaring counterpart. Note the finish differs by necessity:
// roaring_bitmap_repair_after_lazy() down-converts any bitset at or below
// DEFAULT_MAX_SIZE back to an array, so CRoaring's public API undoes its own
// lazy promotion at finish time; frsr's bulk_or_finish_keep_bitsets() does
// not. The comparison is therefore between each library's natural bulk-union
// idiom, which is what a caller actually gets.
static void register_cpp_lazy_union_fold(std::size_t band) {
    using namespace band_fold;
    Entry e;
    char buf[128];
    snprintf(buf, sizeof(buf), "set_ops/cppLazyUnionFold/band=%zu", band);
    e.name        = buf;
    e.description = "CRoaring lazy K-way union accumulate (roaring_bitmap_lazy_or_inplace + "
                    "roaring_bitmap_repair_after_lazy) then AND-fold of " + std::to_string(kProbes) +
                    " small operands, per-chunk union cardinality ~" + std::to_string(band) + ".";
    e.setup       = [band]() -> void * {
        auto *s = new CppBandState;
        s->sources.resize(kOperands);
        for (std::size_t k = 0; k < kOperands; ++k) {
            s->sources[k] = roaring_bitmap_create();
            for (std::size_t c = 0; c < kChunks; ++c) {
                for (std::size_t i = k; i < band; i += kOperands) {
                    roaring_bitmap_add(s->sources[k], band_value(c, band, i));
                }
            }
        }
        s->probes.resize(kProbes);
        auto const step = std::max<std::size_t>(1, band / kProbeCard);
        for (std::size_t p = 0; p < kProbes; ++p) {
            s->probes[p] = roaring_bitmap_create();
            for (std::size_t c = 0; c < kChunks; ++c) {
                for (std::size_t j = 0; j < kProbeCard; ++j) {
                    roaring_bitmap_add(s->probes[p], band_value(c, band, (j * step + p) % band));
                }
            }
        }
        return s;
    };
    e.run         = [](void *sv) -> int64_t {
        auto *s   = static_cast<CppBandState *>(sv);
        auto *acc = roaring_bitmap_create();
        for (auto *src : s->sources) {
            roaring_bitmap_lazy_or_inplace(acc, src, true /* bitset conversion */);
        }
        roaring_bitmap_repair_after_lazy(acc);
        int64_t checksum = static_cast<int64_t>(roaring_bitmap_get_cardinality(acc));
        for (std::size_t p = 0; p < kProbes; ++p) {
            auto *probe = roaring_bitmap_copy(s->probes[p]);
            roaring_bitmap_and_inplace(probe, acc);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(probe));
            roaring_bitmap_free(probe);
        }
        roaring_bitmap_free(acc);
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<CppBandState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>((kOperands + kProbes) * kChunks);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

static void register_frsr_array_run_and_fold(std::size_t repeat) {
    Entry e;
    e.name        = fmt_so("frsr", "ArrayRunAndFold", repeat, "arrayXruns");
    e.description = "frsr N-way AND fold: array-encoded accumulator &= " +
                     std::to_string(kArrayRunFoldOperands) + " run-encoded operands "
                     "(clustered ranges), sole-owned accumulator — a downstream N-way AND fold shape.";
    e.setup       = make_frsr_array_run_fold_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrArrayRunFoldState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 acc = s->seed & s->ops[0];
            for (std::size_t k = 1; k < kArrayRunFoldOperands; ++k) {
                acc &= s->ops[k];
            }
            checksum += static_cast<int64_t>(acc.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrArrayRunFoldState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(repeat * kArrayRunFoldOperands);
    e.inner_reps     = kArrayRunFoldInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_frsr_array_run_and_fold_shared(std::size_t repeat) {
    Entry e;
    e.name        = fmt_so("frsr", "ArrayRunAndFoldShared", repeat, "arrayXruns");
    e.description = "frsr N-way AND fold over a SHARED accumulator: shallow copy-on-write "
                     "copy of a persistent bitmap (rc > 1), first &= pays the write "
                     "barrier/clone — a downstream engine's fold-from-copied-first-operand shape.";
    e.setup       = make_frsr_array_run_fold_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrArrayRunFoldState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 acc{ s->seed };   // shallow COW copy, containers shared
            for (std::size_t k = 0; k < kArrayRunFoldOperands; ++k) {
                acc &= s->ops[k];
            }
            checksum += static_cast<int64_t>(acc.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrArrayRunFoldState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(repeat * kArrayRunFoldOperands);
    e.inner_reps     = kArrayRunFoldInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_array_run_and_fold(std::size_t repeat) {
    Entry e;
    e.name        = fmt_so("cpp", "ArrayRunAndFold", repeat, "arrayXruns");
    e.description = "CRoaring apples-to-apples counterpart of the frsr ArrayRunAndFold "
                     "scenario (roaring_bitmap_and + and_inplace fold).";
    e.setup       = make_cpp_array_run_fold_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppArrayRunFoldState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *acc = roaring_bitmap_and(s->seed, s->ops[0]);
            for (std::size_t k = 1; k < kArrayRunFoldOperands; ++k) {
                roaring_bitmap_and_inplace(acc, s->ops[k]);
            }
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(acc));
            roaring_bitmap_free(acc);
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<CppArrayRunFoldState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(repeat * kArrayRunFoldOperands);
    e.inner_reps     = kArrayRunFoldInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_array_run_and_fold_shared(std::size_t repeat) {
    Entry e;
    e.name        = fmt_so("cpp", "ArrayRunAndFoldShared", repeat, "arrayXruns");
    e.description = "CRoaring counterpart of ArrayRunAndFoldShared: copy_on_write enabled, "
                     "accumulator = roaring_bitmap_copy (shallow), and_inplace fold.";
    e.setup       = []() -> void * {
        auto *s = static_cast<CppArrayRunFoldState *>(make_cpp_array_run_fold_state());
        roaring_bitmap_set_copy_on_write(s->seed, true);
        return s;
    };
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppArrayRunFoldState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *acc = roaring_bitmap_copy(s->seed);   // COW-shallow
            for (std::size_t k = 0; k < kArrayRunFoldOperands; ++k) {
                roaring_bitmap_and_inplace(acc, s->ops[k]);
            }
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(acc));
            roaring_bitmap_free(acc);
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<CppArrayRunFoldState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(repeat * kArrayRunFoldOperands);
    e.inner_reps     = kArrayRunFoldInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// ========================================================================
// Shared run-encoded accumulator, single AND (run∩array / run∩bitset)
//
// The complement of ArrayRunAndFold above: a downstream engine's N-way AND
// fold seeds its accumulator as a shallow COW copy of a STORAGE-OPTIMIZE()D
// bitmap, so the hot pair has the RUN container on the LEFT (accumulator)
// side — probe counts on the run-heavy witness model put run×array and
// run×bitset AND at ~30x the array×run volume. Each repeat copies the shared seed (rc > 1) and
// performs ONE in-place AND, so the run-left pair is exercised every
// iteration (a fold would convert the accumulator to array after step one).
// ========================================================================

struct FrsrRunAccumState {
    TestBitmap32 seed;      // run-encoded after optimize()
    TestBitmap32 array_op;  // sparse -> array-encoded
    TestBitmap32 bitset_op; // dense  -> bitset-encoded
};

struct CppRunAccumState {
    roaring_bitmap_t *seed{};
    roaring_bitmap_t *array_op{};
    roaring_bitmap_t *bitset_op{};

    ~CppRunAccumState() {
        roaring_bitmap_free(seed);
        roaring_bitmap_free(array_op);
        roaring_bitmap_free(bitset_op);
    }
};

static void *make_frsr_run_accum_state() {
    auto *s = new FrsrRunAccumState;
    auto const domain = kArrayRunFoldNumRuns * kArrayRunFoldStride;
    for (std::size_t run = 0; run < kArrayRunFoldNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(run * kArrayRunFoldStride + 7);
        s->seed.add_closed_range(begin, static_cast<std::uint32_t>(begin + kArrayRunFoldRunLength - 1));
    }
    s->seed.optimize();
    for (std::size_t v = 0; v < domain; v += 3) {
        (void)s->array_op.add(static_cast<std::uint32_t>(v));
    }
    for (std::size_t v = 0; v < domain; v += 2) {
        (void)s->bitset_op.add(static_cast<std::uint32_t>(v));
    }
    return s;
}

static void *make_cpp_run_accum_state() {
    auto *s = new CppRunAccumState;
    s->seed      = roaring_bitmap_create();
    s->array_op  = roaring_bitmap_create();
    s->bitset_op = roaring_bitmap_create();
    auto const domain = kArrayRunFoldNumRuns * kArrayRunFoldStride;
    for (std::size_t run = 0; run < kArrayRunFoldNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(run * kArrayRunFoldStride + 7);
        roaring_bitmap_add_range_closed(s->seed, begin, static_cast<std::uint32_t>(begin + kArrayRunFoldRunLength - 1));
    }
    roaring_bitmap_run_optimize(s->seed);
    roaring_bitmap_set_copy_on_write(s->seed, true);
    for (std::size_t v = 0; v < domain; v += 3) {
        roaring_bitmap_add(s->array_op, static_cast<std::uint32_t>(v));
    }
    for (std::size_t v = 0; v < domain; v += 2) {
        roaring_bitmap_add(s->bitset_op, static_cast<std::uint32_t>(v));
    }
    return s;
}

template <TestBitmap32 FrsrRunAccumState::*Operand>
static int64_t run_frsr_run_accum(void *sv, std::size_t const repeat) {
    auto *s = static_cast<FrsrRunAccumState *>(sv);
    int64_t checksum = 0;
    for (std::size_t i = 0; i < repeat; ++i) {
        TestBitmap32 acc{ s->seed };   // shallow COW copy: run-left, rc > 1
        acc &= s->*Operand;
        checksum += static_cast<int64_t>(acc.size());
    }
    return checksum;
}

template <roaring_bitmap_t *CppRunAccumState::*Operand>
static int64_t run_cpp_run_accum(void *sv, std::size_t const repeat) {
    auto *s = static_cast<CppRunAccumState *>(sv);
    int64_t checksum = 0;
    for (std::size_t i = 0; i < repeat; ++i) {
        auto *acc = roaring_bitmap_copy(s->seed);   // COW-shallow
        roaring_bitmap_and_inplace(acc, s->*Operand);
        checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(acc));
        roaring_bitmap_free(acc);
    }
    return checksum;
}

static void register_run_accum_and(std::size_t repeat) {
    struct Variant {
        const char *lib;
        const char *op;
        const char *overlap;
        void *(*setup)();
        int64_t (*run)(void *, std::size_t);
        void (*teardown)(void *);
    };
    static constexpr auto free_frsr = [](void *sv) { delete static_cast<FrsrRunAccumState *>(sv); };
    static constexpr auto free_cpp  = [](void *sv) { delete static_cast<CppRunAccumState *>(sv); };
    Variant const variants[]{
        { "frsr", "RunArrayAndShared",  "runXarray",  make_frsr_run_accum_state, run_frsr_run_accum<&FrsrRunAccumState::array_op>,  free_frsr },
        { "frsr", "RunBitsetAndShared", "runXbitset", make_frsr_run_accum_state, run_frsr_run_accum<&FrsrRunAccumState::bitset_op>, free_frsr },
        { "cpp",  "RunArrayAndShared",  "runXarray",  make_cpp_run_accum_state,  run_cpp_run_accum<&CppRunAccumState::array_op>,   free_cpp  },
        { "cpp",  "RunBitsetAndShared", "runXbitset", make_cpp_run_accum_state,  run_cpp_run_accum<&CppRunAccumState::bitset_op>,  free_cpp  },
    };
    for (auto const &v : variants) {
        Entry e;
        e.name        = fmt_so(v.lib, v.op, repeat, v.overlap);
        e.description = std::string(v.lib) + " single in-place AND per iteration from a shared "
                         "(COW, rc > 1) run-encoded accumulator — a downstream N-way AND fold seed shape.";
        e.setup       = v.setup;
        e.run         = [run = v.run, repeat](void *sv) -> int64_t { return run(sv, repeat); };
        e.teardown       = v.teardown;
        e.ops_per_run    = static_cast<int64_t>(repeat);
        e.inner_reps     = kArrayRunFoldInnerReps;
        e.reusable_state = true;
        g_benchmarks.push_back(std::move(e));
    }
}

// ========================================================================
// Skewed-size repeated intersection
//
// The Union/Intersection/Difference families above always build BOTH operands
// at the same `count` — array∩array is always size-balanced there. Real
// downstream workloads (a differential profiling investigation) intersect a
// small selection/filter array (tens of elements) against a large near-
// array_to_bitset_threshold-sized array container (thousands of elements) at
// very high call volume — e.g. one downstream workload calls
// combine_array_array_into a very large number of times for bit_and alone, with a
// meaningful fraction of those calls at a small:large size ratio >= 64 (CRoaring's own array-vs-array
// skewed-intersection gate, `array_container_intersection`,
// deps/croaring/src/containers/array.c). A balanced-size bench never exercises
// that shape, so it never caught frsr lacking CRoaring's
// intersect_skewed_uint16 binary-search fast path (O(small*log(large)) vs the
// O(small+large) linear merge frsr always ran) — this family closes that gap.
//
// Large operand = a single contiguous run of kSkewedLargeSize values (kept as
// an ARRAY container, not run-encoded: TestBitmap32's default
// run_selection_eager only auto-run-encodes via the smallest-serialized-size
// estimate at container-build time, and a plain array under the
// array_to_bitset_threshold beats a 1-run encoding on that estimate here).
// Small operand = kSkewedSmallSize values strided across TWICE the large
// operand's range, so roughly half fall inside (real matches, non-trivial
// merge work) and half fall outside (misses) — not a degenerate all-hit
// membership test.
// ========================================================================

static constexpr std::size_t kSkewedLargeSize  = 4'000;   // just under the 4096 array/bitset threshold
static constexpr std::size_t kSkewedSmallSize  = 40;      // ratio = 100, well above the 64 skew gate
static constexpr int         kSkewedInnerReps  = 5;

struct FrsrSkewedState {
    TestBitmap32 large;
    TestBitmap32 small;
};

static void register_frsr_skewed_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrSkewedState;
        for (std::size_t i = 0; i < kSkewedLargeSize; ++i) {
            std::ignore = s->large.add(static_cast<std::uint32_t>(i));
        }
        std::size_t const stride = (2 * kSkewedLargeSize) / kSkewedSmallSize;
        for (std::size_t i = 0; i < kSkewedSmallSize; ++i) {
            std::ignore = s->small.add(static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrSkewedState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "SkewedIntersect", repeat, "skew100x");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a " +
                     std::to_string(kSkewedLargeSize) + "-element array and a " +
                     std::to_string(kSkewedSmallSize) + "-element array (ratio " +
                     std::to_string(kSkewedLargeSize / kSkewedSmallSize) +
                     "x), called " + std::to_string(repeat) + " times per timed run — "
                     "reproduces a downstream real-workload array-intersect size shape "
                     "(differential-VTune finding).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrSkewedState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->small & s->large;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kSkewedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
struct CppSkewedState {
    roaring_bitmap_t *large{};
    roaring_bitmap_t *small{};

    ~CppSkewedState() {
        roaring_bitmap_free(large);
        roaring_bitmap_free(small);
    }
};

static void register_cpp_skewed_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppSkewedState;
        s->large = roaring_bitmap_create();
        s->small = roaring_bitmap_create();
        // Individual adds, NOT add_range_closed: add_range materializes a RUN
        // container, which routes roaring_bitmap_and through the trivial
        // run∩array path instead of the skewed array∩array kernel this bench
        // exists to compare (found 2026-07: the run-container cpp arm looked
        // ~3.7x faster on M1 than frsr's array∩array — a representation
        // mismatch, not a kernel gap).
        for (std::size_t i = 0; i < kSkewedLargeSize; ++i) {
            roaring_bitmap_add(s->large, static_cast<std::uint32_t>(i));
        }
        std::size_t const stride = (2 * kSkewedLargeSize) / kSkewedSmallSize;
        for (std::size_t i = 0; i < kSkewedSmallSize; ++i) {
            roaring_bitmap_add(s->small, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppSkewedState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "SkewedIntersect", repeat, "skew100x");
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the "
                     "frsr SkewedIntersect scenario (exercises CRoaring's own "
                     "array_container_intersection skewed/binary-search gate).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppSkewedState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->small, s->large);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kSkewedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

// ========================================================================
// Mid-skew array∩array repeated intersection (the kSimdArrayIntersect shape)
//
// The SkewedIntersect family above covers the >=64x ratio that resolves via the
// binary-search fast path. A shape probe on a downstream engine's live workload
// showed the dominant array∩array population is different: a large volume of
// bit_and calls with a mid-size, moderately-skewed operand profile (few-hundred
// vs few-thousand elements — below the skew gate), i.e. it rides the linear
// merge / SSE4.2 kernel path. The scalar merge loop alone was a large fraction
// of that workload's self time; this family pins that shape so
// kSimdArrayIntersect decisions are measured on it (enabling the kernel gave a
// large win on x86).
// ========================================================================

static constexpr std::size_t kMidSkewLargeSize = 1'600;
static constexpr std::size_t kMidSkewSmallSize = 600;    // ratio ~2.7, well below the 64 skew gate
static constexpr int         kMidSkewInnerReps = 5;

static void register_frsr_midskew_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrSkewedState;
        for (std::size_t i = 0; i < kMidSkewLargeSize; ++i) {
            std::ignore = s->large.add(static_cast<std::uint32_t>(i * 2));  // strided: array-encoded, not run-friendly
        }
        std::size_t const stride = (2 * 2 * kMidSkewLargeSize) / kMidSkewSmallSize;
        for (std::size_t i = 0; i < kMidSkewSmallSize; ++i) {
            std::ignore = s->small.add(static_cast<std::uint32_t>(i * stride));  // ~half land inside the large range
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrSkewedState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MidSkewIntersect", repeat, "skew3x");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a " +
                     std::to_string(kMidSkewLargeSize) + "-element array and a " +
                     std::to_string(kMidSkewSmallSize) + "-element array (ratio ~3x, below the "
                     "skew gate — rides the linear/SSE4.2 merge), called " +
                     std::to_string(repeat) + " times per timed run — reproduces the dominant "
                     "downstream array-intersect shape.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrSkewedState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->small & s->large;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMidSkewInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
static void register_cpp_midskew_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppSkewedState;
        s->large = roaring_bitmap_create();
        s->small = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMidSkewLargeSize; ++i) {
            roaring_bitmap_add(s->large, static_cast<std::uint32_t>(i * 2));
        }
        std::size_t const stride = (2 * 2 * kMidSkewLargeSize) / kMidSkewSmallSize;
        for (std::size_t i = 0; i < kMidSkewSmallSize; ++i) {
            roaring_bitmap_add(s->small, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppSkewedState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MidSkewIntersect", repeat, "skew3x");
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the "
                     "frsr MidSkewIntersect scenario (CRoaring routes it through its "
                     "intersect_vector16 SIMD kernel).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppSkewedState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->small, s->large);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMidSkewInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

// ========================================================================
// Mixed array/bitset repeated intersection (filter_array_bitset kernel)
//
// The Union/Intersection/Difference families above never force a bitset
// container on one side while keeping an array container on the other — both
// operands always cross the array_to_bitset_threshold together. Real
// downstream workloads (a differential-VTune investigation) hit
// exactly this shape at high call volume: a dense dimension/date range
// (bitset-encoded) intersected against a small selection/filter array. This
// family pins the array∩bitset membership-test kernel (filter_array_bitset /
// CRoaring's array_bitset_container_intersection) directly, catching the
// mispredicting-branch regression that investigation found and fixed
// (frsr gated the output write behind an `if`; CRoaring's own kernel
// deliberately avoids that branch with an unconditional store + arithmetic
// advance — see mixed_intersection.c's inline comment on the same kernel).
// ========================================================================

static constexpr std::size_t kMixedBitsetDenseSize = 32'768;  // half of a 65536 chunk: well past the 4096 array/bitset threshold
static constexpr std::size_t kMixedArraySmallSize  = 64;      // stays array-encoded; strided across the dense range for an unpredictable hit/miss pattern
static constexpr int         kMixedInnerReps       = 5;

struct FrsrMixedArrayBitsetState {
    TestBitmap32 dense;
    TestBitmap32 sparse;
};

static void register_frsr_mixed_array_bitset_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(2 * i));  // every other value -> bitset container, ~50% hit rate for the array side
        }
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetIntersect", repeat, "arrayXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a " +
                     std::to_string(kMixedArraySmallSize) + "-element array and a " +
                     std::to_string(kMixedBitsetDenseSize) + "-element bitset-encoded operand, "
                     "called " + std::to_string(repeat) + " times per timed run — reproduces a "
                     "downstream real-workload array-vs-bitset intersect shape "
                     "(differential-VTune finding, filter_array_bitset kernel).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->sparse & s->dense;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// Clustered counterpart of the intersect scenario above: the strided 64-key shape
// has exactly one key per 64-bit bitset word (zero word reuse — worst case for the
// word-cached filter kernel), while real downstream arrays are hundreds-to-thousands
// of keys over a chunk with dense bitset regions. This shape pins the win case:
// 4096 keys at stride 16 (4 keys share each word) against a bitset whose lower
// half is a solid range (all-ones words -> bulk-copy shortcut) and upper half is
// empty (zero words -> bulk-skip shortcut).
static constexpr std::size_t kMixedClusteredArraySize = 4'096;

static void register_frsr_mixed_array_bitset_intersect_clustered(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(i));  // solid [0, 32768) -> all-ones words below, zero words above
        }
        for (std::size_t i = 0; i < kMixedClusteredArraySize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * 16));  // 4 keys per word, spans [0, 65536)
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetIntersectClustered", repeat, "wordreuse");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a " +
                     std::to_string(kMixedClusteredArraySize) + "-element stride-16 array and a "
                     "solid-range bitset-encoded operand — word-reuse + all-ones/zero-word "
                     "shortcut shape of the word-cached array-vs-bitset filter kernel.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->sparse & s->dense;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// Small-run ∩ bitset: a run-encoded operand of modest cardinality against a
// bitset-encoded operand — pins the extract_small_run_bitset ctz-walk arm of
// combine_containers (result emitted as an array container) vs the previous
// full-word_array bitset materialization, and vs CRoaring's run×bitset path.
static constexpr std::size_t kRunBitsetNumRuns   = 64;
static constexpr std::size_t kRunBitsetRunLength = 32;   // 64 runs x 32 = 2048 values, well under the 4096 array threshold
static constexpr std::size_t kRunBitsetStride    = 1'024; // spans [0, 65536) — one chunk

static void register_frsr_run_bitset_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(i));  // solid [0, 32768) -> bitset container
        }
        for (std::size_t run = 0; run < kRunBitsetNumRuns; ++run) {
            auto const begin = static_cast<std::uint32_t>(run * kRunBitsetStride);
            s->sparse.add_closed_range(begin, static_cast<std::uint32_t>(begin + kRunBitsetRunLength - 1));
        }
        s->sparse.optimize();  // force run encoding
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "RunBitsetIntersect", repeat, "runXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a run-encoded operand "
                     "(" + std::to_string(kRunBitsetNumRuns) + " runs x " +
                     std::to_string(kRunBitsetRunLength) + " values) and a " +
                     std::to_string(kMixedBitsetDenseSize) + "-element bitset-encoded operand — "
                     "small-run-vs-bitset combine arm (array-result ctz extraction).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->sparse & s->dense;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// array\bitset (andnot) counterpart of the intersect scenario above: same operands,
// same combine_containers array∩bitset arm, but keep_matches == false — pins the
// difference route of the D2a direct-fill path (filter_array_bitset_into into the
// result payload, no scratch + no scratch→payload copy).
static void register_frsr_mixed_array_bitset_andnot(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(2 * i));
        }
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetAndnot", repeat, "arrayXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator- (array \\ bitset) between a " +
                     std::to_string(kMixedArraySmallSize) + "-element array and a " +
                     std::to_string(kMixedBitsetDenseSize) + "-element bitset-encoded operand, "
                     "called " + std::to_string(repeat) + " times per timed run — the andnot route "
                     "of the D2a array∩bitset direct-fill combine arm.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->sparse - s->dense;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// In-place operator&= counterpart: exercises the same array∩bitset combine arm via
// the bitmap's in-place intersect fallthrough (combine_containers_for_policy), the
// other live caller of the D2a path in real downstream filtering.
static void register_frsr_mixed_array_bitset_intersect_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(2 * i));
        }
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetIntersectInplace", repeat, "arrayXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator&= (in-place array∩bitset) — copies "
                     "the sparse array operand then intersects the dense bitset in place, " +
                     std::to_string(repeat) + " times per timed run; pins the operator&= "
                     "fallthrough route of the D2a array∩bitset combine arm.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            TestBitmap32 r = s->sparse;
            r &= s->dense;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// Sole-referent (rc==1) in-place array\bitset — the operator-= twin of the D1a
// fast path below. The sparse array sits on ODD values while the dense bitset
// holds EVEN ones, so `sparse -= dense` removes nothing: idempotent, repeatable,
// and alloc-free on the rc==1 branch while still walking the full filter kernel
// with keep_matches == false.
static void register_frsr_mixed_array_bitset_subtract_unique_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(2 * i));
        }
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * stride + 1));  // odd ⇒ disjoint from dense
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetSubtractUniqueInplace", repeat, "arrayXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator-= on a UNIQUELY-OWNED (rc==1) " +
                     std::to_string(kMixedArraySmallSize) + "-element array minus a " +
                     std::to_string(kMixedBitsetDenseSize) + "-element bitset (disjoint, idempotent), " +
                     std::to_string(repeat) + " times per timed run — isolates the sole-referent "
                     "in-place array\\bitset filter (zero allocation, no fresh container_handle).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            s->sparse -= s->dense;  // sole-owned, disjoint → rc==1 in-place, alloc-free
            checksum += static_cast<int64_t>(s->sparse.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

// Sole-referent (rc==1) in-place array∩bitset — the D1a fast path. The array is
// uniquely owned (never copied ⇒ no CoW clone) and sparse ⊂ dense, so `sparse &=
// dense` is idempotent and compacts in place with ZERO allocation, repeatable
// across reps. Contrast MixedArrayBitsetIntersectInplace above, whose `r = sparse`
// shares the payload (rc==2) and forces the one-time clone-on-write path.
static void register_frsr_mixed_array_bitset_filter_unique_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new FrsrMixedArrayBitsetState;
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            std::ignore = s->dense.add(static_cast<std::uint32_t>(2 * i));
        }
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            std::ignore = s->sparse.add(static_cast<std::uint32_t>(i * stride));  // sparse ⊂ dense ⇒ &= is idempotent
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<FrsrMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("frsr", "MixedArrayBitsetFilterUniqueInplace", repeat, "arrayXbitset");
    e.description = "frsr::roaring::bitmap<uint32_t> operator&= on a UNIQUELY-OWNED (rc==1) " +
                     std::to_string(kMixedArraySmallSize) + "-element array intersected with a " +
                     std::to_string(kMixedBitsetDenseSize) + "-element bitset (sparse ⊂ dense, "
                     "idempotent), " + std::to_string(repeat) + " times per timed run — isolates the "
                     "D1a sole-referent in-place filter (zero allocation, no fresh container_handle).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<FrsrMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            s->sparse &= s->dense;  // sole-owned, idempotent → rc==1 in-place, alloc-free
            checksum += static_cast<int64_t>(s->sparse.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
struct CppMixedArrayBitsetState {
    roaring_bitmap_t *dense{};
    roaring_bitmap_t *sparse{};

    ~CppMixedArrayBitsetState() {
        roaring_bitmap_free(dense);
        roaring_bitmap_free(sparse);
    }
};

static void register_cpp_mixed_array_bitset_intersect_clustered(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(i));
        }
        for (std::size_t i = 0; i < kMixedClusteredArraySize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * 16));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetIntersectClustered", repeat, "wordreuse");
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the "
                     "frsr MixedArrayBitsetIntersectClustered scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_run_bitset_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(i));
        }
        for (std::size_t run = 0; run < kRunBitsetNumRuns; ++run) {
            auto const begin = static_cast<std::uint32_t>(run * kRunBitsetStride);
            roaring_bitmap_add_range_closed(s->sparse, begin, static_cast<std::uint32_t>(begin + kRunBitsetRunLength - 1));
        }
        roaring_bitmap_run_optimize(s->sparse);
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "RunBitsetIntersect", repeat, "runXbitset");
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the "
                     "frsr RunBitsetIntersect scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_mixed_array_bitset_intersect(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(2 * i));
        }
        roaring_bitmap_run_optimize(s->dense);  // no-op here (not run-shaped), kept for parity with real usage patterns
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetIntersect", repeat, "arrayXbitset");
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the "
                     "frsr MixedArrayBitsetIntersect scenario (exercises CRoaring's own "
                     "array_bitset_container_intersection kernel).";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_and(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_mixed_array_bitset_andnot(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(2 * i));
        }
        roaring_bitmap_run_optimize(s->dense);
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetAndnot", repeat, "arrayXbitset");
    e.description = "CRoaring roaring_bitmap_andnot() — apples-to-apples counterpart of the "
                     "frsr MixedArrayBitsetAndnot scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_andnot(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_mixed_array_bitset_intersect_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(2 * i));
        }
        roaring_bitmap_run_optimize(s->dense);
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetIntersectInplace", repeat, "arrayXbitset");
    e.description = "CRoaring roaring_bitmap_and_inplace() on a copy of the sparse array — "
                     "apples-to-apples counterpart of the frsr MixedArrayBitsetIntersectInplace scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            auto *r = roaring_bitmap_copy(s->sparse);
            roaring_bitmap_and_inplace(r, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_mixed_array_bitset_subtract_unique_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(2 * i));
        }
        roaring_bitmap_run_optimize(s->dense);
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride + 1));  // odd ⇒ disjoint from dense
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetSubtractUniqueInplace", repeat, "arrayXbitset");
    e.description = "CRoaring roaring_bitmap_andnot_inplace() on a sole-owned sparse array (idempotent, "
                     "sparse disjoint from dense) — apples-to-apples counterpart of the frsr "
                     "MixedArrayBitsetSubtractUniqueInplace scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            roaring_bitmap_andnot_inplace(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(s->sparse));
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

static void register_cpp_mixed_array_bitset_filter_unique_inplace(std::size_t repeat) {
    auto make_state = []() -> void * {
        auto *s = new CppMixedArrayBitsetState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        for (std::size_t i = 0; i < kMixedBitsetDenseSize; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(2 * i));
        }
        roaring_bitmap_run_optimize(s->dense);
        std::size_t const stride = (2 * kMixedBitsetDenseSize) / kMixedArraySmallSize;
        for (std::size_t i = 0; i < kMixedArraySmallSize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    auto free_state = [](void *sv) { delete static_cast<CppMixedArrayBitsetState *>(sv); };

    Entry e;
    e.name        = fmt_so("cpp", "MixedArrayBitsetFilterUniqueInplace", repeat, "arrayXbitset");
    e.description = "CRoaring roaring_bitmap_and_inplace() on a sole-owned sparse array (idempotent, "
                     "sparse ⊂ dense) — apples-to-apples counterpart of the frsr "
                     "MixedArrayBitsetFilterUniqueInplace scenario.";
    e.setup       = make_state;
    e.run         = [repeat](void *sv) -> int64_t {
        auto *s = static_cast<CppMixedArrayBitsetState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < repeat; ++i) {
            roaring_bitmap_and_inplace(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(s->sparse));
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(repeat);
    e.inner_reps     = kMixedInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

// ========================================================================
// Cold/streaming heap-scale working set (array x bitset AND / ANDNOT)
//
// Every family above hot-loops the SAME 1-64 operand pairs (reusable_state
// keeps `state` fixed across every timed call), so the whole working set
// stays L1/L2-resident for the life of the run. A downstream workload's real
// access pattern is the opposite: set-operation calls stream a very large
// number of small, DISTINCT operand pairs scattered across a multi-gigabyte
// live heap, touching most containers once (or a handful of times) rather
// than looping the same few forever — mostly COLD accesses, not hot ones.
//
// This facility builds a working set of N individually heap-allocated
// array/bitset operand pairs sized off a target byte budget, then drives a
// SHUFFLED (never sequential) traversal over them. Sequential traversal of a
// big contiguous array is bandwidth-friendly and defeats the point — real
// hardware prefetchers happily stream it, understating the real cost of
// scattered first-touch access. A shuffled index order over individually-
// `new`-allocated pair objects gives the prefetcher nothing to lock onto.
// ========================================================================
namespace cold_heap {

// Weighted cardinality bucket: draw a count in [lo, hi] with relative weight
// `weight`. Lets a band mix several size classes instead of one fixed size.
struct CardinalityBucket {
    std::size_t lo;
    std::size_t hi;
    double      weight;
};

static std::size_t sample_bucket(std::mt19937_64 &rng, std::span<CardinalityBucket const> buckets) {
    double total = 0.0;
    for (auto const &b : buckets) total += b.weight;
    double const r = std::uniform_real_distribution<double>(0.0, total)(rng);
    double acc = 0.0;
    for (auto const &b : buckets) {
        acc += b.weight;
        if (r <= acc) {
            return b.lo + (b.hi > b.lo ? static_cast<std::size_t>(rand_u64(rng) % (b.hi - b.lo + 1)) : 0);
        }
    }
    return buckets.back().lo;
}

// Shape distributions for the array x bitset AND / ANDNOT band: the small
// operand skews heavily toward tiny cardinalities; the dense operand is
// large and roughly fixed-cost once it is bitset-encoded regardless of its
// exact count. These weights encode a generic "small filter array against a
// dense range/dimension operand" shape, not any specific dataset.
static constexpr CardinalityBucket kArrayBuckets[] = {
    { 1,     16,    0.33 },
    { 17,    64,    0.14 },
    { 65,    256,   0.18 },
    { 257,   1024,  0.22 },
    { 1025,  4096,  0.13 },
};
static constexpr CardinalityBucket kBitsetBuckets[] = {
    { 4097,  16384, 0.61 },
    { 16385, 65536, 0.33 },
};

// Nominal average pair footprint, used only to size N pairs from a byte
// budget: ~1 KiB average array payload (weighted mean of kArrayBuckets) plus
// a ~8 KiB fixed-size bitset container plus small wrapper/object overhead.
static constexpr std::size_t kNominalPairBytes = 10 * 1024;

// Sweep points: fits-in-L2, near/above a typical LLC, and two heap-scale
// tiers well past any cache. The point of the sweep is to see WHERE behavior
// changes as the working set crosses cache boundaries, not any one number.
static constexpr std::size_t kWorkingSetSizes[] = {
    std::size_t{1}   << 20,               // ~1 MiB
    std::size_t{30}  << 20,               // ~30 MiB
    std::size_t{500} << 20,               // ~500 MiB
    std::size_t{2}   << 30,               // ~2 GiB
};

static constexpr std::size_t kOpsPerRun  = 20'000;  // pair-touches per timed run()
static constexpr int         kInnerReps  = 1;       // repetition is driven externally (fresh process per rep)

// Build a shuffled traversal order over [0, n).
static std::vector<std::uint32_t> make_shuffled_order(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

struct FrsrArrayBitsetPair {
    TestBitmap32 arr;   // small, skewed-cardinality operand (stays array-encoded)
    TestBitmap32 bmp;   // dense operand (bitset-encoded)
};

struct FrsrColdHeapState {
    std::vector<FrsrArrayBitsetPair *> pairs;   // individually-allocated: NOT one contiguous vector<Pair>
    std::vector<std::uint32_t>         order;
};

static FrsrColdHeapState *make_frsr_cold_heap(std::size_t working_set_bytes, std::uint64_t seed) {
    auto *s = new FrsrColdHeapState;
    std::size_t const n = std::max<std::size_t>(2, working_set_bytes / kNominalPairBytes);
    s->pairs.reserve(n);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        auto *pair = new FrsrArrayBitsetPair;   // scattered heap allocation, not a slab
        std::size_t const bcount = sample_bucket(rng, kBitsetBuckets);
        for (std::size_t v = 0; v < bcount; ++v) {
            std::ignore = pair->bmp.add(static_cast<std::uint32_t>(v));
        }
        std::size_t const acount = sample_bucket(rng, kArrayBuckets);
        // Strided across TWICE the dense range so roughly half the small
        // operand's values land inside the dense one (real hits) and half
        // miss — not a degenerate all-hit/all-miss membership test.
        std::size_t const span   = 2 * bcount;
        std::size_t const stride = std::max<std::size_t>(1, span / std::max<std::size_t>(acount, 1));
        for (std::size_t v = 0; v < acount; ++v) {
            std::ignore = pair->arr.add(static_cast<std::uint32_t>(v * stride));
        }
        s->pairs.push_back(pair);
    }
    s->order = make_shuffled_order(n, seed ^ 0x5eed5eedULL);
    return s;
}

static void free_frsr_cold_heap(FrsrColdHeapState *s) {
    for (auto *p : s->pairs) delete p;
    delete s;
}

#if FRSR_ROARING_HAS_CROARING
struct CppArrayBitsetPair {
    roaring_bitmap_t *arr{};
    roaring_bitmap_t *bmp{};

    ~CppArrayBitsetPair() {
        roaring_bitmap_free(arr);
        roaring_bitmap_free(bmp);
    }
};

struct CppColdHeapState {
    std::vector<CppArrayBitsetPair *> pairs;
    std::vector<std::uint32_t>        order;
};

static CppColdHeapState *make_cpp_cold_heap(std::size_t working_set_bytes, std::uint64_t seed) {
    auto *s = new CppColdHeapState;
    std::size_t const n = std::max<std::size_t>(2, working_set_bytes / kNominalPairBytes);
    s->pairs.reserve(n);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        auto *pair = new CppArrayBitsetPair;
        pair->arr = roaring_bitmap_create();
        pair->bmp = roaring_bitmap_create();
        std::size_t const bcount = sample_bucket(rng, kBitsetBuckets);
        for (std::size_t v = 0; v < bcount; ++v) {
            roaring_bitmap_add(pair->bmp, static_cast<std::uint32_t>(v));
        }
        std::size_t const acount = sample_bucket(rng, kArrayBuckets);
        std::size_t const span   = 2 * bcount;
        std::size_t const stride = std::max<std::size_t>(1, span / std::max<std::size_t>(acount, 1));
        for (std::size_t v = 0; v < acount; ++v) {
            roaring_bitmap_add(pair->arr, static_cast<std::uint32_t>(v * stride));
        }
        s->pairs.push_back(pair);
    }
    s->order = make_shuffled_order(n, seed ^ 0x5eed5eedULL);
    return s;
}

static void free_cpp_cold_heap(CppColdHeapState *s) {
    for (auto *p : s->pairs) delete p;
    delete s;
}
#endif // FRSR_ROARING_HAS_CROARING

static std::string fmt_cold(const char *lib, const char *op, std::size_t ws_bytes) {
    char label[32];
    if (ws_bytes >= (std::size_t{1} << 30)) {
        snprintf(label, sizeof(label), "%zug", ws_bytes >> 30);
    } else {
        snprintf(label, sizeof(label), "%zum", ws_bytes >> 20);
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "cold_heap/%s%s/ws=%s", lib, op, label);
    return buf;
}

static void register_frsr_cold_array_bitset(std::size_t ws_bytes, bool andnot) {
    auto make_state = [ws_bytes]() -> void * {
        return make_frsr_cold_heap(ws_bytes, 0x9e3779b97f4a7c15ULL ^ ws_bytes);
    };
    auto free_state = [](void *sv) { free_frsr_cold_heap(static_cast<FrsrColdHeapState *>(sv)); };

    Entry e;
    e.name        = fmt_cold("frsr", andnot ? "ColdArrayBitsetAndnot" : "ColdArrayBitsetIntersect", ws_bytes);
    e.description = std::string("frsr::roaring::bitmap<uint32_t> array ") + (andnot ? "\\" : "&") +
                     " bitset over a working set of individually heap-allocated operand pairs "
                     "spread across ~" + std::to_string(ws_bytes >> 20) + " MiB total, visited in a "
                     "shuffled (non-sequential) order each timed run — models a downstream "
                     "workload's access pattern of many small, scattered, mostly-cold set "
                     "operations across a large live heap, in contrast to the cache-resident "
                     "hot-loop bands above.";
    e.setup       = make_state;
    e.run         = [andnot](void *sv) -> int64_t {
        auto *s = static_cast<FrsrColdHeapState *>(sv);
        std::size_t const n = s->pairs.size();
        int64_t checksum = 0;
        for (std::size_t k = 0; k < kOpsPerRun; ++k) {
            auto *p = s->pairs[s->order[k % n]];
            TestBitmap32 r = andnot ? (p->arr - p->bmp) : (p->arr & p->bmp);
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(kOpsPerRun);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
static void register_cpp_cold_array_bitset(std::size_t ws_bytes, bool andnot) {
    auto make_state = [ws_bytes]() -> void * {
        return make_cpp_cold_heap(ws_bytes, 0x9e3779b97f4a7c15ULL ^ ws_bytes);
    };
    auto free_state = [](void *sv) { free_cpp_cold_heap(static_cast<CppColdHeapState *>(sv)); };

    Entry e;
    e.name        = fmt_cold("cpp", andnot ? "ColdArrayBitsetAndnot" : "ColdArrayBitsetIntersect", ws_bytes);
    e.description = "CRoaring roaring_bitmap_and()/andnot() — apples-to-apples counterpart of the "
                     "frsr ColdArrayBitset scenario over the same scattered, heap-scale working set.";
    e.setup       = make_state;
    e.run         = [andnot](void *sv) -> int64_t {
        auto *s = static_cast<CppColdHeapState *>(sv);
        std::size_t const n = s->pairs.size();
        int64_t checksum = 0;
        for (std::size_t k = 0; k < kOpsPerRun; ++k) {
            auto *p = s->pairs[s->order[k % n]];
            auto *r = andnot ? roaring_bitmap_andnot(p->arr, p->bmp) : roaring_bitmap_and(p->arr, p->bmp);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = free_state;
    e.ops_per_run    = static_cast<int64_t>(kOpsPerRun);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

} // namespace cold_heap

// ========================================================================
// Saturated-bitset band — bitset operand at/near the 65536-per-chunk domain
// maximum, against a small array probe. Every AND/ANDNOT band above tops out
// at a 50%-full bitset operand; a downstream workload's bitset operands are
// occasionally saturated (all-ones) or near-saturated, a shape this suite
// never exercised. Held as a plain bitset (no run_optimize()) so both arms
// pay the same array/bitset combine kernel — the "form" question (whether a
// full container should collapse to a run) is Part B's concern, not this
// band's.
// ========================================================================
namespace saturated_bitset {

static constexpr std::size_t kChunkDomain   = 65'536;
static constexpr std::size_t kArraySize     = 64;      // stays array-encoded
static constexpr std::size_t kRepeat        = 100'000;
static constexpr int         kInnerReps     = 5;

struct FrsrState {
    TestBitmap32 dense;    // saturated / near-saturated bitset operand
    TestBitmap32 sparse;   // small array probe
};

// fill_fraction in (0, 1]: 1.0 == fully saturated (all 65536 bits).
static void fill_dense_frsr(TestBitmap32 &dense, double fill_fraction) {
    auto const count = static_cast<std::size_t>(static_cast<double>(kChunkDomain) * fill_fraction);
    for (std::size_t i = 0; i < count; ++i) {
        std::ignore = dense.add(static_cast<std::uint32_t>(i));
    }
}

static void fill_sparse_frsr(TestBitmap32 &sparse) {
    // Strided across TWICE the domain so roughly half the probe hits.
    std::size_t const stride = std::max<std::size_t>(1, (2 * kChunkDomain) / kArraySize);
    for (std::size_t i = 0; i < kArraySize; ++i) {
        std::ignore = sparse.add(static_cast<std::uint32_t>(i * stride));
    }
}

static void register_frsr_saturated(double fill_fraction, const char *tag, bool andnot) {
    Entry e;
    char buf[160];
    snprintf(buf, sizeof(buf), "set_ops/frsrSaturatedBitset%s/fill=%s",
             andnot ? "Andnot" : "Intersect", tag);
    e.name        = buf;
    e.description = std::string("frsr::roaring::bitmap<uint32_t> ") + (andnot ? "operator-" : "operator&") +
                     " between a " + std::to_string(kArraySize) + "-element array probe and a bitset "
                     "operand filled to ~" + std::to_string(static_cast<int>(fill_fraction * 100)) +
                     "% of the 65536-per-chunk domain (no run_optimize — plain bitset form), " +
                     std::to_string(kRepeat) + " times per timed run. A downstream workload's "
                     "operands are occasionally saturated or near-saturated, a density this suite's "
                     "other bands (which top out at 50% full) never reach.";
    e.setup       = [fill_fraction]() -> void * {
        auto *s = new FrsrState;
        fill_dense_frsr(s->dense, fill_fraction);
        fill_sparse_frsr(s->sparse);
        return s;
    };
    e.run         = [andnot](void *sv) -> int64_t {
        auto *s = static_cast<FrsrState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < kRepeat; ++i) {
            TestBitmap32 r = andnot ? (s->sparse - s->dense) : (s->sparse & s->dense);
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(kRepeat);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
struct CppState {
    roaring_bitmap_t *dense{};
    roaring_bitmap_t *sparse{};

    ~CppState() {
        roaring_bitmap_free(dense);
        roaring_bitmap_free(sparse);
    }
};

static void register_cpp_saturated(double fill_fraction, const char *tag, bool andnot) {
    Entry e;
    char buf[160];
    snprintf(buf, sizeof(buf), "set_ops/cppSaturatedBitset%s/fill=%s",
             andnot ? "Andnot" : "Intersect", tag);
    e.name        = buf;
    e.description = "CRoaring roaring_bitmap_and()/andnot() — apples-to-apples counterpart of the "
                     "frsr SaturatedBitset scenario (plain bitset operand, no run_optimize).";
    e.setup       = [fill_fraction]() -> void * {
        auto *s = new CppState;
        s->dense  = roaring_bitmap_create();
        s->sparse = roaring_bitmap_create();
        auto const count = static_cast<std::size_t>(static_cast<double>(kChunkDomain) * fill_fraction);
        for (std::size_t i = 0; i < count; ++i) {
            roaring_bitmap_add(s->dense, static_cast<std::uint32_t>(i));
        }
        std::size_t const stride = std::max<std::size_t>(1, (2 * kChunkDomain) / kArraySize);
        for (std::size_t i = 0; i < kArraySize; ++i) {
            roaring_bitmap_add(s->sparse, static_cast<std::uint32_t>(i * stride));
        }
        return s;
    };
    e.run         = [andnot](void *sv) -> int64_t {
        auto *s = static_cast<CppState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < kRepeat; ++i) {
            auto *r = andnot ? roaring_bitmap_andnot(s->sparse, s->dense) : roaring_bitmap_and(s->sparse, s->dense);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<CppState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(kRepeat);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

static void register_benchmarks() {
    // (fill_fraction, tag): 100% full, and ~95-99% ("near-full", the other
    // half of the real-workload's ">=95%" saturated-bitset bucket).
    static constexpr std::pair<double, const char *> kFills[] = {
        { 1.00, "100pct" },
        { 0.97, "97pct"  },
    };
    for (auto const &[fraction, tag] : kFills) {
        for (bool const andnot : { false, true }) {
            register_frsr_saturated(fraction, tag, andnot);
#if FRSR_ROARING_HAS_CROARING
            register_cpp_saturated(fraction, tag, andnot);
#endif
        }
    }
}

} // namespace saturated_bitset

// ========================================================================
// Large-run band — run operand cardinality swept well above the existing
// RunBitsetIntersect band's fixed 2048 (64 runs x 32 values), against BOTH
// an array and a bitset operand. A downstream workload's run operands skew
// heavily larger: the majority sit above cardinality 4096, and a sizeable
// minority sit near the 65536-per-chunk domain maximum — a shape this suite
// never swept.
//
// Fixed 64 runs, run length = card/64, stride = 1024 (domain/64). The top
// swept point (card=65024, runLength=1016) is intentionally kept STRICTLY
// below runLength==stride==1024: at that equality every run would merge
// into one [0,65535] run, i.e. a FULL container — a different, degenerate
// shape (CRoaring short-circuits via run_container_is_full; frsr does not)
// that would silently skew a "large but not full" band. The genuinely-full
// case is its own separately-named band below (full=true), not folded into
// this sweep.
// ========================================================================
namespace large_run {

static constexpr std::size_t kNumRuns    = 64;
static constexpr std::size_t kStride     = 1'024;   // domain (65536) / kNumRuns
static constexpr std::size_t kArraySize  = 64;
static constexpr std::size_t kBitsetSize = 32'768;
static constexpr std::size_t kRepeat     = 100'000;
static constexpr int         kInnerReps  = 5;

static void add_runs_frsr(TestBitmap32 &b, std::size_t run_length) {
    for (std::size_t run = 0; run < kNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(run * kStride);
        auto const end   = static_cast<std::uint32_t>(begin + run_length - 1);
        b.add_closed_range(begin, end);
    }
    b.optimize();   // force run encoding
}

static void add_runs_cpp(roaring_bitmap_t *b, std::size_t run_length) {
    for (std::size_t run = 0; run < kNumRuns; ++run) {
        auto const begin = static_cast<std::uint32_t>(run * kStride);
        auto const end   = static_cast<std::uint32_t>(begin + run_length - 1);
        roaring_bitmap_add_range_closed(b, begin, end);
    }
    roaring_bitmap_run_optimize(b);
}

struct FrsrState {
    TestBitmap32 run_op;
    TestBitmap32 other;   // array or bitset operand
};

static void register_frsr_run_vs(std::size_t card, bool run_length_full, bool vs_bitset) {
    std::size_t const run_length = run_length_full ? kStride : (card / kNumRuns);
    Entry e;
    char buf[160];
    snprintf(buf, sizeof(buf), "set_ops/frsrRunVs%s/card=%zu%s",
             vs_bitset ? "Bitset" : "Array", card, run_length_full ? "/full=true" : "");
    e.name        = buf;
    e.description = "frsr::roaring::bitmap<uint32_t> operator& between a run-encoded operand "
                     "(" + std::to_string(kNumRuns) + " runs, cardinality ~" + std::to_string(card) +
                     ") and " + (vs_bitset ? "a bitset" : "an array") + " operand, " +
                     std::to_string(kRepeat) + " times per timed run — sweeps run-operand "
                     "cardinality well past the existing fixed-2048 RunBitsetIntersect band.";
    e.setup       = [run_length, vs_bitset]() -> void * {
        auto *s = new FrsrState;
        add_runs_frsr(s->run_op, run_length);
        if (vs_bitset) {
            for (std::size_t i = 0; i < kBitsetSize; ++i) {
                std::ignore = s->other.add(static_cast<std::uint32_t>(i));
            }
        } else {
            std::size_t const stride = std::max<std::size_t>(1, (2 * kStride * kNumRuns) / kArraySize);
            for (std::size_t i = 0; i < kArraySize; ++i) {
                std::ignore = s->other.add(static_cast<std::uint32_t>(i * stride));
            }
        }
        return s;
    };
    e.run         = [](void *sv) -> int64_t {
        auto *s = static_cast<FrsrState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < kRepeat; ++i) {
            TestBitmap32 r = s->run_op & s->other;
            checksum += static_cast<int64_t>(r.size());
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<FrsrState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(kRepeat);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}

#if FRSR_ROARING_HAS_CROARING
struct CppState {
    roaring_bitmap_t *run_op{};
    roaring_bitmap_t *other{};

    ~CppState() {
        roaring_bitmap_free(run_op);
        roaring_bitmap_free(other);
    }
};

static void register_cpp_run_vs(std::size_t card, bool run_length_full, bool vs_bitset) {
    std::size_t const run_length = run_length_full ? kStride : (card / kNumRuns);
    Entry e;
    char buf[160];
    snprintf(buf, sizeof(buf), "set_ops/cppRunVs%s/card=%zu%s",
             vs_bitset ? "Bitset" : "Array", card, run_length_full ? "/full=true" : "");
    e.name        = buf;
    e.description = "CRoaring roaring_bitmap_and() — apples-to-apples counterpart of the frsr "
                     "RunVs" + std::string(vs_bitset ? "Bitset" : "Array") + " scenario.";
    e.setup       = [run_length, vs_bitset]() -> void * {
        auto *s = new CppState;
        s->run_op = roaring_bitmap_create();
        s->other  = roaring_bitmap_create();
        add_runs_cpp(s->run_op, run_length);
        if (vs_bitset) {
            for (std::size_t i = 0; i < kBitsetSize; ++i) {
                roaring_bitmap_add(s->other, static_cast<std::uint32_t>(i));
            }
        } else {
            std::size_t const stride = std::max<std::size_t>(1, (2 * kStride * kNumRuns) / kArraySize);
            for (std::size_t i = 0; i < kArraySize; ++i) {
                roaring_bitmap_add(s->other, static_cast<std::uint32_t>(i * stride));
            }
        }
        return s;
    };
    e.run         = [](void *sv) -> int64_t {
        auto *s = static_cast<CppState *>(sv);
        int64_t checksum = 0;
        for (std::size_t i = 0; i < kRepeat; ++i) {
            auto *r = roaring_bitmap_and(s->run_op, s->other);
            checksum += static_cast<int64_t>(roaring_bitmap_get_cardinality(r));
            roaring_bitmap_free(r);
        }
        return checksum;
    };
    e.teardown       = [](void *sv) { delete static_cast<CppState *>(sv); };
    e.ops_per_run    = static_cast<int64_t>(kRepeat);
    e.inner_reps     = kInnerReps;
    e.reusable_state = true;
    g_benchmarks.push_back(std::move(e));
}
#endif // FRSR_ROARING_HAS_CROARING

static void register_benchmarks() {
    static constexpr std::size_t kCards[] = { 4'096, 8'192, 16'384, 32'768, 65'024 };
    for (std::size_t const card : kCards) {
        for (bool const vs_bitset : { false, true }) {
            register_frsr_run_vs(card, /*run_length_full=*/false, vs_bitset);
#if FRSR_ROARING_HAS_CROARING
            register_cpp_run_vs(card, /*run_length_full=*/false, vs_bitset);
#endif
        }
    }
    // Separately-named, genuinely-full-domain point (single [0,65535] run) —
    // NOT part of the sweep above; see the namespace comment.
    for (bool const vs_bitset : { false, true }) {
        register_frsr_run_vs(65'536, /*run_length_full=*/true, vs_bitset);
#if FRSR_ROARING_HAS_CROARING
        register_cpp_run_vs(65'536, /*run_length_full=*/true, vs_bitset);
#endif
    }
}

} // namespace large_run

// ========== top-level registration ==========

void register_benchmarks() {
    for (std::size_t count : kCounts) {
        for (auto const &ov : kOverlaps) {
            std::size_t const offset = count * ov.num / ov.den;
            register_frsr_binary(count, offset, ov.label);
            register_cpp_binary(count, offset, ov.label);
            register_r64_binary(count, offset, ov.label);
            register_set_binary(count, offset, ov.label);
        }
        // sparse many-chunk variant: b shifted by ~half the chunk span for ~50% chunk overlap.
        std::size_t const b_chunk_offset = std::max(std::size_t{ 1 }, count / kSparsePerChunk / 2 );
        register_frsr_binary_sparse(count, b_chunk_offset);
        register_cpp_binary_sparse (count, b_chunk_offset);
    }
    for (std::size_t count : kCounts) {
        register_nway_union(count);
        register_nway_union_sparse(count);
        register_frsr_run_heavy(count);
        register_cpp_run_heavy(count);
    }
    // Promotion-band sweep: 512 and 8192 are controls (both libraries agree on
    // the form there); 2048 is the window where the policies diverge.
    for (std::size_t band : { std::size_t{512}, std::size_t{2048}, std::size_t{8192} }) {
        register_frsr_and_small_vs_band(band, /*as_bitset=*/false);
        register_frsr_and_small_vs_band(band, /*as_bitset=*/true );
        register_frsr_lazy_union_fold(band);
        register_frsr_lazy_union_fold_optimized(band, /*keep_bitsets=*/false);
        register_frsr_lazy_union_fold_optimized(band, /*keep_bitsets=*/true );
#if FRSR_ROARING_HAS_CROARING
        register_cpp_lazy_union_fold(band);
#endif
    }
    // High-cardinality extension of the same fold shape: a downstream
    // workload's dominant OR shape is bitset x array with the bitset side
    // mostly in 16385-65536, well past the 512/2048/8192 sweep above.
    for (std::size_t band : { std::size_t{16384}, std::size_t{32768}, std::size_t{65536} }) {
        register_frsr_lazy_union_fold(band);
#if FRSR_ROARING_HAS_CROARING
        register_cpp_lazy_union_fold(band);
#endif
    }

    register_run_accum_and(20'000);
    register_frsr_array_run_and_fold(20'000);
    register_frsr_array_run_and_fold_shared(20'000);
    register_cpp_array_run_and_fold(20'000);
    register_cpp_array_run_and_fold_shared(20'000);

    register_frsr_skewed_intersect(100'000);
#if FRSR_ROARING_HAS_CROARING
    register_cpp_skewed_intersect(100'000);
#endif
    register_frsr_midskew_intersect(100'000);
#if FRSR_ROARING_HAS_CROARING
    register_cpp_midskew_intersect(100'000);
#endif
    register_frsr_mixed_array_bitset_intersect(100'000);
    register_frsr_mixed_array_bitset_intersect_clustered(100'000);
    register_frsr_run_bitset_intersect(100'000);
    register_frsr_mixed_array_bitset_andnot(100'000);
    register_frsr_mixed_array_bitset_intersect_inplace(100'000);
    register_frsr_mixed_array_bitset_filter_unique_inplace(100'000);
    register_frsr_mixed_array_bitset_subtract_unique_inplace(100'000);
#if FRSR_ROARING_HAS_CROARING
    register_cpp_mixed_array_bitset_intersect(100'000);
    register_cpp_mixed_array_bitset_intersect_clustered(100'000);
    register_cpp_run_bitset_intersect(100'000);
    register_cpp_mixed_array_bitset_andnot(100'000);
    register_cpp_mixed_array_bitset_intersect_inplace(100'000);
    register_cpp_mixed_array_bitset_filter_unique_inplace(100'000);
    register_cpp_mixed_array_bitset_subtract_unique_inplace(100'000);
#endif

    for (std::size_t const ws : cold_heap::kWorkingSetSizes) {
        for (bool const andnot : { false, true }) {
            cold_heap::register_frsr_cold_array_bitset(ws, andnot);
#if FRSR_ROARING_HAS_CROARING
            cold_heap::register_cpp_cold_array_bitset(ws, andnot);
#endif
        }
    }

    saturated_bitset::register_benchmarks();
    large_run::register_benchmarks();
}

} // namespace set_ops

#endif // FRSR_ROARING_HAS_CROARING

// Benchmark Execution and Main
// ========================================================================

// Execute a single benchmark entry
static void run_benchmark(const Entry &e) {
    if (e.setup == nullptr || e.run == nullptr) {
        fprintf(stderr, "ERROR: Benchmark %s missing setup or run\n", e.name.c_str());
        return;
    }

    // Setup once if not reusable; otherwise setup per iteration
    void *state = nullptr;
    int64_t total_checksum = 0;
    int64_t total_time_ns = 0;

    if (e.reusable_state) {
        state = e.setup();
    }

    for (int64_t rep = 0; rep < e.inner_reps; ++rep) {
        if (!e.reusable_state) {
            state = e.setup();
        }

        auto start = Clock::now();
        int64_t checksum = e.run(state);
        auto end = Clock::now();

        total_checksum += checksum;
        total_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (!e.reusable_state && e.teardown) {
            e.teardown(state);
            state = nullptr;
        }
    }

    if (e.reusable_state && e.teardown) {
        e.teardown(state);
    }

    // Report results
    double time_per_op_us = static_cast<double>(total_time_ns) / (e.inner_reps * e.ops_per_run * 1000.0);
    // 4 decimals: point-lookup cases land around 0.005-0.03 us/op, where %.2f
    // quantises every result to 0.01 and hides the effect being measured.
    printf("%s\t%.4f us/op\tchecksum=%ld\n", e.name.c_str(), time_per_op_us, (long)total_checksum);
}

// ---------------------------------------------------------------------------
// chunk_probe: isolate the cost/benefit of an IMPLICIT hot-chunk cache inside
// contains(), versus CRoaring's shape (pure read + opt-in bulk context).
//
// Every case here drives a PRECOMPUTED key stream held in a vector, so the
// timed region contains only the lookup — unlike the older *ContainsRandom
// family, which times rand_u64() alongside contains() and so dilutes the very
// difference we are trying to resolve.
//
// The three streams map onto measured reality:
//   Scatter   - chunk key jumps every probe. A downstream engine's dominant
//               pattern: its row streams are ordered by PK coords, so the
//               high-16 chunk key is NOT monotonic (most audited contains()
//               call sites, per an internal audit).
//   Ascending - monotonically increasing keys: the pattern an implicit hot
//               cache is built for. This is the REGRESSION GUARD for removing it.
//   Bulk      - the same ascending stream through the opt-in caller-owned
//               context, i.e. how CRoaring exposes this amortization.
// ---------------------------------------------------------------------------
namespace chunk_probe {

inline constexpr std::uint32_t kChunkSpan{ 65536 };   // keys per chunk for uint32
inline constexpr std::size_t   kKeysPerChunk{ 512 };  // set density inside a chunk
inline constexpr std::size_t   kProbesPerRun{ 1024 }; // batch per timed run()

// Values are strided inside each chunk so that roughly half the probes hit.
static std::vector<std::uint32_t> make_values( std::size_t const chunks ) {
    std::vector<std::uint32_t> values;
    values.reserve( chunks * kKeysPerChunk );
    for ( std::size_t c = 0; c < chunks; ++c ) {
        auto const base{ static_cast<std::uint32_t>( c ) * kChunkSpan };
        for ( std::size_t k = 0; k < kKeysPerChunk; ++k ) {
            values.push_back( base + static_cast<std::uint32_t>( k * 64 ) );
        }
    }
    return values;
}

// Probe stream with a chunk key that jumps on every probe.
static std::vector<std::uint32_t> make_scatter_probes( std::size_t const chunks, std::size_t const n ) {
    std::mt19937_64 rng{ 0x5eed5eedULL + chunks };
    std::vector<std::uint32_t> probes;
    probes.reserve( n );
    for ( std::size_t i = 0; i < n; ++i ) {
        auto const c{ static_cast<std::uint32_t>( rand_u64( rng ) % chunks ) };
        auto const low{ static_cast<std::uint32_t>( rand_u64( rng ) % kChunkSpan ) };
        probes.push_back( c * kChunkSpan + low );
    }
    return probes;
}

// Probe stream that sweeps the whole domain in increasing order, wrapping.
static std::vector<std::uint32_t> make_ascending_probes( std::size_t const chunks, std::size_t const n ) {
    std::vector<std::uint32_t> probes;
    probes.reserve( n );
    auto const domain{ static_cast<std::uint64_t>( chunks ) * kChunkSpan };
    auto const stride{ domain / n == 0 ? 1U : static_cast<std::uint32_t>( domain / n ) };
    std::uint64_t v{ 0 };
    for ( std::size_t i = 0; i < n; ++i ) {
        probes.push_back( static_cast<std::uint32_t>( v % domain ) );
        v += stride;
    }
    return probes;
}

struct FrsrState {
    TestBitmap32 r;
    std::vector<std::uint32_t> probes;
    std::size_t pos{ 0 };
};

struct R64State {
    roaring_bitmap_t * r;
    std::vector<std::uint32_t> probes;
    std::size_t pos{ 0 };
    ~R64State() { roaring_bitmap_free( r ); }
};

struct SharedState {
    TestBitmap32 r;                       // probed by every thread at once
    std::vector<TestBitmap32> copies;     // one per thread (the control arm)
    std::vector<std::uint32_t> probes;
    std::size_t threads{ 1 };
};

static FrsrState * make_frsr_state( std::size_t chunks, bool ascending ) {
    auto * s{ new FrsrState{} };
    for ( auto const v : make_values( chunks ) ) { (void)s->r.add( v ); }
    s->probes = ascending ? make_ascending_probes( chunks, 1U << 16 )
                          : make_scatter_probes  ( chunks, 1U << 16 );
    return s;
}

static R64State * make_r64_state( std::size_t chunks, bool ascending ) {
    auto * s{ new R64State{ roaring_bitmap_create(), {}, 0 } };
    for ( auto const v : make_values( chunks ) ) { roaring_bitmap_add( s->r, v ); }
    s->probes = ascending ? make_ascending_probes( chunks, 1U << 16 )
                          : make_scatter_probes  ( chunks, 1U << 16 );
    return s;
}

static void register_benchmarks() {
    // 64 and 256 chunks sit under the 512-chunk adaptive-index threshold (a
    // downstream workload's bitmaps typically stay well under it too). 1024
    // crosses it, which is what arms the lazily-built chunk hash map on the
    // read path.
    for ( std::size_t const chunks : { std::size_t{ 64 }, std::size_t{ 256 }, std::size_t{ 1024 } } ) {
        char tag[ 32 ];
        snprintf( tag, sizeof( tag ), "chunks=%zu", chunks );

        struct Variant {
            char const * name;
            bool ascending;
            char const * what;
        };
        for ( auto const & variant : {
            Variant{ "Scatter",   false, "chunk key jumps every probe (a downstream engine's dominant access pattern)" },
            Variant{ "Ascending", true,  "monotonically increasing probes (favours an implicit hot-chunk cache)" },
        } ) {
            {
                Entry e;
                e.name = std::string( "chunk_probe/frsr" ) + variant.name + "/" + tag;
                e.description = std::string( "frsr bitmap<uint32_t>::contains() over a precomputed key stream: " ) + variant.what;
                e.setup = [chunks, asc = variant.ascending]() -> void * { return make_frsr_state( chunks, asc ); };
                e.run = []( void * sv ) -> int64_t {
                    auto * s{ static_cast<FrsrState *>( sv ) };
                    int64_t hits{ 0 };
                    for ( std::size_t i = 0; i < kProbesPerRun; ++i ) {
                        auto const v{ s->probes[ s->pos ] };
                        s->pos = ( s->pos + 1 ) & ( s->probes.size() - 1 );
                        hits += s->r.contains( v ) ? 1 : 0;
                    }
                    return hits;
                };
                e.teardown = []( void * sv ) { delete static_cast<FrsrState *>( sv ); };
                e.ops_per_run = kProbesPerRun;
                e.inner_reps = 200;
                e.reusable_state = true;
                g_benchmarks.push_back( std::move( e ) );
            }
            {
                Entry e;
                e.name = std::string( "chunk_probe/r64" ) + variant.name + "/" + tag;
                e.description = std::string( "CRoaring roaring_bitmap_contains() reference for the same stream: " ) + variant.what;
                e.setup = [chunks, asc = variant.ascending]() -> void * { return make_r64_state( chunks, asc ); };
                e.run = []( void * sv ) -> int64_t {
                    auto * s{ static_cast<R64State *>( sv ) };
                    int64_t hits{ 0 };
                    for ( std::size_t i = 0; i < kProbesPerRun; ++i ) {
                        auto const v{ s->probes[ s->pos ] };
                        s->pos = ( s->pos + 1 ) & ( s->probes.size() - 1 );
                        hits += roaring_bitmap_contains( s->r, v ) ? 1 : 0;
                    }
                    return hits;
                };
                e.teardown = []( void * sv ) { delete static_cast<R64State *>( sv ); };
                e.ops_per_run = kProbesPerRun;
                e.inner_reps = 200;
                e.reusable_state = true;
                g_benchmarks.push_back( std::move( e ) );
            }
        }

        // Opt-in amortization: the caller owns the cursor. This is the arm that
        // must recover the ascending-stream win once contains() goes pure.
        {
            Entry e;
            e.name = std::string( "chunk_probe/frsrBulkAscending/" ) + tag;
            e.description = "frsr contains_bulk() with a caller-owned bulk_context over an ascending stream.";
            e.setup = [chunks]() -> void * { return make_frsr_state( chunks, true ); };
            e.run = []( void * sv ) -> int64_t {
                auto * s{ static_cast<FrsrState *>( sv ) };
                TestBitmap32::bulk_context ctx{};
                int64_t hits{ 0 };
                for ( std::size_t i = 0; i < kProbesPerRun; ++i ) {
                    auto const v{ s->probes[ s->pos ] };
                    s->pos = ( s->pos + 1 ) & ( s->probes.size() - 1 );
                    hits += s->r.contains_bulk( ctx, v ) ? 1 : 0;
                }
                return hits;
            };
            e.teardown = []( void * sv ) { delete static_cast<FrsrState *>( sv ); };
            e.ops_per_run = kProbesPerRun;
            e.inner_reps = 200;
            e.reusable_state = true;
            g_benchmarks.push_back( std::move( e ) );
        }
        {
            Entry e;
            e.name = std::string( "chunk_probe/r64BulkAscending/" ) + tag;
            e.description = "CRoaring roaring_bitmap_contains_bulk() reference over an ascending stream.";
            e.setup = [chunks]() -> void * { return make_r64_state( chunks, true ); };
            e.run = []( void * sv ) -> int64_t {
                auto * s{ static_cast<R64State *>( sv ) };
                roaring_bulk_context_t ctx{};
                int64_t hits{ 0 };
                for ( std::size_t i = 0; i < kProbesPerRun; ++i ) {
                    auto const v{ s->probes[ s->pos ] };
                    s->pos = ( s->pos + 1 ) & ( s->probes.size() - 1 );
                    hits += roaring_bitmap_contains_bulk( s->r, &ctx, v ) ? 1 : 0;
                }
                return hits;
            };
            e.teardown = []( void * sv ) { delete static_cast<R64State *>( sv ); };
            e.ops_per_run = kProbesPerRun;
            e.inner_reps = 200;
            e.reusable_state = true;
            g_benchmarks.push_back( std::move( e ) );
        }
    }

    // Concurrent readers of ONE bitmap. CRoaring's README guarantees this is
    // safe on an unmodified bitmap; an implicit cache written by contains()
    // breaks that guarantee and turns shared reads into cacheline write-sharing.
    // SharedConst vs PerThreadCopy isolates exactly that: identical work, the
    // only difference is whether the readers share one object.
    for ( std::size_t const threads : { std::size_t{ 2 }, std::size_t{ 4 }, std::size_t{ 8 } } ) {
        char tag[ 32 ];
        snprintf( tag, sizeof( tag ), "threads=%zu", threads );
        constexpr std::size_t kChunks{ 256 };
        constexpr std::size_t kProbesPerThread{ 1U << 14 };

        auto make_shared_state = [threads]() -> void * {
            auto * s{ new SharedState{} };
            s->threads = threads;
            for ( auto const v : make_values( kChunks ) ) { (void)s->r.add( v ); }
            s->probes = make_scatter_probes( kChunks, 1U << 16 );
            s->copies.assign( threads, s->r ); // copies start with cold caches
            return s;
        };

        {
            Entry e;
            e.name = std::string( "chunk_probe/frsrSharedConst/" ) + tag;
            e.description =
                "N threads calling contains() concurrently on ONE shared const bitmap. "
                "Pure read in the parity shape; write-sharing on the bitmap header if "
                "contains() maintains an implicit hot-chunk cache.";
            e.setup = make_shared_state;
            e.run = []( void * sv ) -> int64_t {
                auto * s{ static_cast<SharedState *>( sv ) };
                TestBitmap32 const & shared{ s->r };
                std::vector<std::thread> pool;
                std::vector<int64_t> hits( s->threads, 0 );
                pool.reserve( s->threads );
                for ( std::size_t t = 0; t < s->threads; ++t ) {
                    pool.emplace_back( [&shared, s, t, &hits]() {
                        std::size_t pos{ ( t * 7919 ) & ( s->probes.size() - 1 ) };
                        int64_t local{ 0 };
                        for ( std::size_t i = 0; i < kProbesPerThread; ++i ) {
                            local += shared.contains( s->probes[ pos ] ) ? 1 : 0;
                            pos = ( pos + 1 ) & ( s->probes.size() - 1 );
                        }
                        hits[ t ] = local;
                    } );
                }
                for ( auto & th : pool ) { th.join(); }
                return std::accumulate( hits.begin(), hits.end(), int64_t{ 0 } );
            };
            e.teardown = []( void * sv ) { delete static_cast<SharedState *>( sv ); };
            e.ops_per_run = static_cast<int64_t>( threads * kProbesPerThread );
            e.inner_reps = 20;
            e.reusable_state = true;
            g_benchmarks.push_back( std::move( e ) );
        }
        {
            Entry e;
            e.name = std::string( "chunk_probe/frsrPerThreadCopy/" ) + tag;
            e.description =
                "Control for frsrSharedConst: identical work, but each thread probes its "
                "own copy, so no header is shared. The SharedConst/PerThreadCopy ratio is "
                "the write-sharing penalty.";
            e.setup = make_shared_state;
            e.run = []( void * sv ) -> int64_t {
                auto * s{ static_cast<SharedState *>( sv ) };
                std::vector<std::thread> pool;
                std::vector<int64_t> hits( s->threads, 0 );
                pool.reserve( s->threads );
                for ( std::size_t t = 0; t < s->threads; ++t ) {
                    pool.emplace_back( [s, t, &hits]() {
                        auto const & own{ s->copies[ t ] };
                        std::size_t pos{ ( t * 7919 ) & ( s->probes.size() - 1 ) };
                        int64_t local{ 0 };
                        for ( std::size_t i = 0; i < kProbesPerThread; ++i ) {
                            local += own.contains( s->probes[ pos ] ) ? 1 : 0;
                            pos = ( pos + 1 ) & ( s->probes.size() - 1 );
                        }
                        hits[ t ] = local;
                    } );
                }
                for ( auto & th : pool ) { th.join(); }
                return std::accumulate( hits.begin(), hits.end(), int64_t{ 0 } );
            };
            e.teardown = []( void * sv ) { delete static_cast<SharedState *>( sv ); };
            e.ops_per_run = static_cast<int64_t>( threads * kProbesPerThread );
            e.inner_reps = 20;
            e.reusable_state = true;
            g_benchmarks.push_back( std::move( e ) );
        }
    }
}

} // namespace chunk_probe

} // namespace

int main(int argc, char *argv[]) {
#if FRSR_ROARING_HAS_CROARING
    // Register all benchmarks
    synthetic::register_contains_variants();
    synthetic::register_random_variants();
    synthetic::register_insert_remove();
    synthetic::register_ser_deser();
    density_contains::register_benchmarks();
    set_ops::register_benchmarks();
    chunk_probe::register_benchmarks();
#endif

    std::string filter;
    bool list_only = false;
    std::optional<std::uint64_t> max_count;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        } else if (arg.substr(0, 9) == "--filter=") {
            filter = arg.substr(9);
        } else if (arg == "--max-count" && i + 1 < argc) {
            max_count = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg.substr(0, 12) == "--max-count=") {
            max_count = std::strtoull(arg.c_str() + 12, nullptr, 10);
        }
    }

    // List or run benchmarks
    if (list_only) {
        printf("Available benchmarks (%zu total):\n", g_benchmarks.size());
        for (const auto &e : g_benchmarks) {
            printf("  %s\n", e.name.c_str());
        }
        return 0;
    }

    // Run benchmarks, optionally filtered
    size_t run_count = 0;
    size_t skipped_count = 0;
    auto const parse_count = [](std::string const & name) -> std::optional<std::uint64_t> {
        auto const pos = name.find("count=");
        if (pos == std::string::npos) {
            return std::nullopt;
        }
        auto const start = pos + 6;
        auto end = name.find('/', start);
        if (end == std::string::npos) {
            end = name.size();
        }
        auto const slice = name.substr(start, end - start);
        if (slice.empty()) {
            return std::nullopt;
        }
        return std::strtoull(slice.c_str(), nullptr, 10);
    };
    for (const auto &e : g_benchmarks) {
        // Check if benchmark matches filter
        if (!filter.empty()) {
            if (e.name.find(filter) == std::string::npos) {
                continue;
            }
        }
        if (max_count.has_value()) {
            if (auto const scenario_count = parse_count(e.name); scenario_count.has_value() && *scenario_count > *max_count) {
                ++skipped_count;
                continue;
            }
        }
        run_benchmark(e);
        run_count++;
    }

    if (run_count == 0) {
        if (!filter.empty()) {
            printf("No benchmarks matched filter: %s\n", filter.c_str());
        } else {
            printf("ERROR: No benchmarks registered (FRSR_ROARING_HAS_CROARING may not be defined)\n");
        }
        return 1;
    }

    printf("\nRan %zu benchmarks", run_count);
    if (max_count.has_value()) {
        printf(" (skipped %zu over max-count=%llu)", skipped_count, static_cast<unsigned long long>(*max_count));
    }
    printf("\n");
    return 0;
}
