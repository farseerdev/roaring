// A bitmap in the middle of a lazy fold (bulk_or_intermediate / bulk_xor_intermediate) holds bitsets whose
// counts, and a size, that the fold did not maintain. This drives every operation the library lets run in that
// state against a bit-per-key oracle, in random order and on random dense, sparse and run containers, and
// then checks the finished bitmap through every read. Built without NDEBUG, a count that is read while it is
// stale trips an assertion at the reader; under AddressSanitizer a wrong count that sizes a buffer is a
// heap overflow.
//
// What may run between the folds and the finisher is documented on bulk_or_intermediate: further folds,
// every in-place update, and contains(). The other reads need the finished bitmap.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace frsr::roaring;

constexpr std::uint32_t chunk_span   { 65'536 };
constexpr std::uint32_t chunk_count  { 4 };
constexpr std::uint32_t universe_size{ chunk_span * chunk_count };

// The model: one byte per key of the universe. It shares no code with the library.
class oracle {
public:
    oracle() : bits_( universe_size, 0 ) {}

    [[nodiscard]] bool test( std::uint32_t const key ) const { return bits_[ key ] != 0; }
    void set  ( std::uint32_t const key ) { bits_[ key ] = 1; }
    void reset( std::uint32_t const key ) { bits_[ key ] = 0; }
    void flip ( std::uint32_t const key ) { bits_[ key ] ^= 1; }
    void clear() { std::ranges::fill( bits_, 0 ); }

    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>( std::ranges::count( bits_, 1 ) ); }
    [[nodiscard]] std::vector<std::uint32_t> keys() const {
        std::vector<std::uint32_t> result;
        for ( std::uint32_t key{ 0 }; key < universe_size; ++key ) {
            if ( bits_[ key ] ) { result.push_back( key ); }
        }
        return result;
    }
    [[nodiscard]] std::size_t count_in( std::uint32_t const lo, std::uint32_t const hi ) const {
        return lo >= hi ? 0 : static_cast<std::size_t>( std::count( bits_.begin() + lo, bits_.begin() + hi, 1 ) );
    }
    void apply_range( std::uint32_t const lo, std::uint32_t const hi, char const what ) {   // [lo, hi)
        for ( auto key{ lo }; key < hi; ++key ) {
            what == 'a' ? set( key ) : what == 'r' ? reset( key ) : flip( key );
        }
    }
    friend oracle operator|( oracle lhs, oracle const & rhs ) { for ( std::size_t i{ 0 }; i < lhs.bits_.size(); ++i ) { lhs.bits_[ i ] |= rhs.bits_[ i ]; } return lhs; }
    friend oracle operator&( oracle lhs, oracle const & rhs ) { for ( std::size_t i{ 0 }; i < lhs.bits_.size(); ++i ) { lhs.bits_[ i ] &= rhs.bits_[ i ]; } return lhs; }
    friend oracle operator-( oracle lhs, oracle const & rhs ) { for ( std::size_t i{ 0 }; i < lhs.bits_.size(); ++i ) { lhs.bits_[ i ] &= static_cast<std::uint8_t>( ~rhs.bits_[ i ] ); } return lhs; }
    friend oracle operator^( oracle lhs, oracle const & rhs ) { for ( std::size_t i{ 0 }; i < lhs.bits_.size(); ++i ) { lhs.bits_[ i ] ^= rhs.bits_[ i ]; } return lhs; }
    [[nodiscard]] bool intersects( oracle const & other ) const { return ( *this & other ).size() != 0; }

private:
    std::vector<std::uint8_t> bits_;
};

// The container kinds a chunk of a generated bitmap can start as.
enum class chunk_shape { empty, tiny, sparse, boundary, dense, nearly_full, full, runs };

template <typename Bitmap>
struct generated {
    Bitmap bitmap;
    oracle model;
};

[[nodiscard]] std::vector<std::uint32_t> chunk_keys( std::mt19937_64 & rng, std::uint32_t const chunk, chunk_shape const shape ) {
    auto const base{ chunk * chunk_span };
    std::vector<std::uint32_t> keys;
    auto const random_keys = [ & ]( std::size_t const n ) {
        std::vector<bool> taken( chunk_span );
        while ( keys.size() < n ) {
            auto const low{ static_cast<std::uint32_t>( rng() % chunk_span ) };
            if ( !taken[ low ] ) { taken[ low ] = true; keys.push_back( base + low ); }
        }
    };
    switch ( shape ) {
        case chunk_shape::empty:    break;
        case chunk_shape::tiny:     random_keys( 1 + rng() % 3 ); break;
        case chunk_shape::sparse:   random_keys( 4 + rng() % 300 ); break;
        case chunk_shape::boundary: random_keys( 4090 + rng() % 12 ); break;   // around the array limit
        case chunk_shape::dense:    random_keys( 6000 + rng() % 50'000 ); break;
        case chunk_shape::nearly_full: {
            std::vector<bool> missing( chunk_span );
            for ( auto n{ 1 + rng() % 40 }; n != 0; --n ) { missing[ rng() % chunk_span ] = true; }
            for ( std::uint32_t low{ 0 }; low < chunk_span; ++low ) { if ( !missing[ low ] ) { keys.push_back( base + low ); } }
            return keys;
        }
        case chunk_shape::full:
            keys.resize( chunk_span );
            std::iota( keys.begin(), keys.end(), base );
            return keys;
        case chunk_shape::runs: {
            std::uint32_t low{ static_cast<std::uint32_t>( rng() % 500 ) };
            while ( low < chunk_span ) {
                auto const length{ static_cast<std::uint32_t>( 1 + rng() % 3000 ) };
                for ( auto i{ low }; i < std::min<std::uint32_t>( low + length, chunk_span ); ++i ) { keys.push_back( base + i ); }
                low += length + 1 + static_cast<std::uint32_t>( rng() % 3000 );
            }
            return keys;
        }
    }
    std::ranges::sort( keys );
    return keys;
}

template <typename Bitmap>
[[nodiscard]] generated<Bitmap> generate( std::mt19937_64 & rng ) {
    generated<Bitmap> result;
    std::vector<std::uint32_t> keys;
    for ( std::uint32_t chunk{ 0 }; chunk < chunk_count; ++chunk ) {
        auto const shape{ static_cast<chunk_shape>( rng() % 8 ) };
        auto const chunk_values{ chunk_keys( rng, chunk, shape ) };
        keys.insert( keys.end(), chunk_values.begin(), chunk_values.end() );
    }
    for ( auto const key : keys ) { result.model.set( key ); }
    result.bitmap.add_many( keys );
    if ( rng() % 3 == 0 ) { result.bitmap.optimize(); }   // run containers
    return result;
}

[[nodiscard]] std::vector<std::uint32_t> random_keys( std::mt19937_64 & rng, std::size_t const n, oracle const & near ) {
    std::vector<std::uint32_t> keys;
    auto const members{ rng() % 2 == 0 ? near.keys() : std::vector<std::uint32_t>{} };
    for ( std::size_t i{ 0 }; i < n; ++i ) {
        keys.push_back( !members.empty() && rng() % 2 == 0 ? members[ rng() % members.size() ] : static_cast<std::uint32_t>( rng() % universe_size ) );
    }
    return keys;
}

template <typename Bitmap>
class fold_session {
public:
    explicit fold_session( std::uint64_t const seed ) : rng_{ seed } {
        auto start{ generate<Bitmap>( rng_ ) };
        if ( rng_() % 4 == 0 ) { start.bitmap.clear(); start.model.clear(); }
        subject_ = std::move( start.bitmap );
        model_   = std::move( start.model );
    }

    void run() {
        auto const steps{ 3 + rng_() % 14 };
        for ( std::uint64_t step{ 0 }; step != steps; ++step ) {
            apply_random_step();
            probe_contains();
            probe_finished_copy();
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
        finish_and_verify();
    }

private:
    void probe_contains() {
        for ( int i{ 0 }; i < 24; ++i ) {
            auto const key{ static_cast<std::uint32_t>( rng_() % universe_size ) };
            ASSERT_EQ( subject_.contains( key ), model_.test( key ) ) << "key " << key << " step " << last_step_;
        }
        auto const members{ model_.keys() };
        for ( auto const key : members | std::views::take( 8 ) ) {
            ASSERT_TRUE( subject_.contains( key ) ) << "member " << key << " step " << last_step_;
        }
    }

    // A copy of the deferred bitmap, finished, holds exactly the model's keys.
    void probe_finished_copy() {
        Bitmap copy{ subject_ };
        copy.bulk_xor_repair_after_lazy();
        ASSERT_EQ( copy.validate(), nullptr ) << copy.validate() << " after steps " << trace_;
        ASSERT_EQ( copy.size(), model_.size() ) << "after steps " << trace_;
        auto const listed{ copy.to_vector() };
        auto const expected{ model_.keys() };
        ASSERT_TRUE( listed.size() == expected.size() && std::equal( listed.begin(), listed.end(), expected.begin() ) ) << "after steps " << trace_;
    }

    void apply_random_step() {
        last_step_ = rng_() % 31;
        trace_ += std::to_string( last_step_ ) + ' ';
        switch ( last_step_ ) {
            case 0: case 1: case 2: case 3: {
                auto operand{ generate<Bitmap>( rng_ ) };
                subject_.bulk_or_intermediate( operand.bitmap );
                model_ = model_ | operand.model;
                break;
            }
            case 4: {
                auto operand{ generate<Bitmap>( rng_ ) };
                subject_.bulk_xor_intermediate( operand.bitmap );
                model_ = model_ ^ operand.model;
                any_xor_fold_ = true;
                break;
            }
            case 5: for ( int i{ 0 }; i < 40; ++i ) { auto const k{ static_cast<std::uint32_t>( rng_() % universe_size ) }; ASSERT_EQ( subject_.add( k ), !model_.test( k ) ); model_.set( k ); } break;
            case 6: {
                auto const members{ model_.keys() };
                for ( int i{ 0 }; i < 40; ++i ) {
                    auto const k{ !members.empty() && rng_() % 2 == 0 ? members[ rng_() % members.size() ] : static_cast<std::uint32_t>( rng_() % universe_size ) };
                    ASSERT_EQ( subject_.remove( k ), model_.test( k ) );
                    model_.reset( k );
                }
                break;
            }
            case 7: { auto keys{ random_keys( rng_, rng_() % 6000, model_ ) }; subject_.add_many( keys ); for ( auto const k : keys ) { model_.set( k ); } break; }
            case 8: { auto keys{ random_keys( rng_, rng_() % 6000, model_ ) }; subject_.remove_many( keys ); for ( auto const k : keys ) { model_.reset( k ); } break; }
            case 9: {
                auto keys{ random_keys( rng_, rng_() % 3000, model_ ) };
                std::deque<std::uint32_t> as_range( keys.begin(), keys.end() );
                subject_.add_many( as_range );
                for ( auto const k : keys ) { model_.set( k ); }
                break;
            }
            case 10: {
                auto keys{ random_keys( rng_, rng_() % 3000, model_ ) };
                std::ranges::sort( keys );
                keys.erase( std::unique( keys.begin(), keys.end() ), keys.end() );
                if ( rng_() % 2 ) { subject_.add_many_sorted( keys ); for ( auto const k : keys ) { model_.set( k ); } }
                else              { subject_.remove_many_sorted( keys ); for ( auto const k : keys ) { model_.reset( k ); } }
                break;
            }
            case 11: case 12: case 13: {
                auto const lo{ static_cast<std::uint32_t>( rng_() % universe_size ) };
                auto const hi{ std::min<std::uint32_t>( universe_size - 1, lo + static_cast<std::uint32_t>( rng_() % 90'000 ) ) };
                auto const what{ "arf"[ last_step_ - 11 ] };
                what == 'a' ? subject_.add_closed_range( lo, hi ) : what == 'r' ? subject_.remove_closed_range( lo, hi ) : subject_.flip_closed_range( lo, hi );
                model_.apply_range( lo, hi + 1, what );
                break;
            }
            case 14: { auto operand{ generate<Bitmap>( rng_ ) }; subject_ |= operand.bitmap; model_ = model_ | operand.model; break; }
            case 15: { auto operand{ generate<Bitmap>( rng_ ) }; subject_ -= operand.bitmap; model_ = model_ - operand.model; break; }
            case 16: { auto operand{ generate<Bitmap>( rng_ ) }; subject_ ^= operand.bitmap; model_ = model_ ^ operand.model; break; }
            case 17: { if ( rng_() % 3 == 0 ) { auto operand{ generate<Bitmap>( rng_ ) }; subject_ &= operand.bitmap; model_ = model_ & operand.model; } break; }
            case 18: {
                auto const lo{ static_cast<std::uint64_t>( rng_() % universe_size ) };
                auto const hi{ std::min<std::uint64_t>( universe_size, lo + rng_() % 100'000 ) };
                oracle range;
                range.apply_range( static_cast<std::uint32_t>( lo ), static_cast<std::uint32_t>( hi ), 'a' );
                auto const which{ rng_() % 4 };
                trace_ += "(" + std::to_string( which ) + " " + std::to_string( lo ) + "-" + std::to_string( hi ) + ") ";
                switch ( which ) {
                    case 0: subject_.or_with    ( key_range::half_open( lo, hi ) ); model_ = model_ | range; break;
                    case 1: subject_.andnot_with( key_range::half_open( lo, hi ) ); model_ = model_ - range; break;
                    case 2: subject_.xor_with   ( key_range::half_open( lo, hi ) ); model_ = model_ ^ range; break;
                    default: if ( rng_() % 3 == 0 ) { subject_.and_with( key_range::half_open( lo, hi ) ); model_ = model_ & range; }
                }
                break;
            }
            case 19: {
                auto keys{ random_keys( rng_, rng_() % 3000, model_ ) };
                std::ranges::sort( keys );
                oracle operand;
                for ( auto const k : keys ) { operand.set( k ); }
                switch ( rng_() % 3 ) {
                    case 0: subject_.or_with ( std::span<std::uint32_t const>{ keys } ); model_ = model_ | operand; break;
                    case 1: subject_.xor_with( std::span<std::uint32_t const>{ keys } ); model_ = model_ ^ operand; break;
                    default: subject_.andnot_with( std::span<std::uint32_t const>{ keys } ); model_ = model_ - operand;
                }
                break;
            }
            case 20: subject_.optimize(); break;
            case 21: subject_.optimize_keep_bitsets(); break;
            case 22: subject_.shrink_to_fit(); break;
            case 23: {
                typename Bitmap::bulk_context context;
                auto const base{ static_cast<std::uint32_t>( rng_() % ( universe_size - 5000 ) ) };
                for ( std::uint32_t i{ 0 }; i < 2000; ++i ) {
                    auto const k{ base + static_cast<std::uint32_t>( rng_() % 5000 ) };
                    ASSERT_EQ( subject_.add_bulk( context, k ), !model_.test( k ) );
                    model_.set( k );
                }
                break;
            }
            case 24: if ( rng_() % 6 == 0 ) { subject_.clear(); model_.clear(); } break;
            case 25: {   // empty one chunk key by key: the count has to tell when the last one goes
                auto const chunk{ static_cast<std::uint32_t>( rng_() % chunk_count ) };
                for ( auto const key : model_.keys() ) {
                    if ( key / chunk_span == chunk ) { ASSERT_TRUE( subject_.remove( key ) ); model_.reset( key ); }
                }
                break;
            }
            case 26: {
                auto first{ generate<Bitmap>( rng_ ) }, second{ generate<Bitmap>( rng_ ) };
                std::array<Bitmap const *, 2> operands{ &first.bitmap, &second.bitmap };
                subject_.or_many_in_place( operands );
                model_ = model_ | first.model | second.model;
                break;
            }
            case 27: {
                auto first{ generate<Bitmap>( rng_ ) }, second{ generate<Bitmap>( rng_ ) };
                std::array<Bitmap const *, 2> operands{ &first.bitmap, &second.bitmap };
                subject_.xor_many_in_place( operands );
                model_ = model_ ^ first.model ^ second.model;
                break;
            }
            case 28: case 29: case 30: {   // a fold of a bitmap that is dense where the accumulator is
                auto operand{ generate<Bitmap>( rng_ ) };
                subject_.bulk_or_intermediate( operand.bitmap );
                model_ = model_ | operand.model;
                break;
            }
        }
    }

    void finish_and_verify() {
        if ( any_xor_fold_ ) {
            subject_.bulk_xor_repair_after_lazy();
        } else {
            switch ( rng_() % 3 ) {
                case 0:  subject_.bulk_or_finish(); break;
                case 1:  subject_.bulk_or_finish_keep_bitsets(); break;
                default: subject_.bulk_or_repair_after_lazy();
            }
        }
        verify();
    }

    void verify() {
        SCOPED_TRACE( "steps: " + trace_ );
        ASSERT_EQ( subject_.validate(), nullptr ) << subject_.validate();
        ASSERT_EQ( subject_.size(), model_.size() );
        ASSERT_EQ( subject_.empty(), model_.size() == 0 );
        auto const expected{ model_.keys() };
        auto const listed{ subject_.to_vector() };
        ASSERT_EQ( listed.size(), expected.size() );
        ASSERT_TRUE( std::equal( listed.begin(), listed.end(), expected.begin() ) );
        std::vector<std::uint32_t> iterated( subject_.begin(), subject_.end() );
        ASSERT_EQ( iterated, expected );
        if ( !expected.empty() ) {
            ASSERT_EQ( subject_.front(), expected.front() );
            ASSERT_EQ( subject_.back (), expected.back () );
        }
        for ( auto const key : std::span{ expected }.first( std::min<std::size_t>( expected.size(), 64 ) ) ) { ASSERT_TRUE( subject_.contains( key ) ); }

        auto other{ generate<Bitmap>( rng_ ) };
        ASSERT_EQ( subject_.and_cardinality( other.bitmap ), ( model_ & other.model ).size() );
        ASSERT_EQ( subject_.xor_cardinality( other.bitmap ), ( model_ ^ other.model ).size() );
        ASSERT_EQ( subject_.intersects( other.bitmap ), model_.intersects( other.model ) );
        ASSERT_EQ( subject_.is_subset_of( other.bitmap ), ( model_ - other.model ).size() == 0 );
        ASSERT_EQ( subject_.equals( other.bitmap ), ( model_ ^ other.model ).size() == 0 );
        auto const lo{ static_cast<std::uint32_t>( rng_() % universe_size ) };
        auto const hi{ std::min<std::uint32_t>( universe_size, lo + static_cast<std::uint32_t>( rng_() % 120'000 ) ) };
        ASSERT_EQ( subject_.cardinality_in_range( lo, hi ), model_.count_in( lo, hi ) );
        ASSERT_EQ( subject_.contains_range( lo, hi ), model_.count_in( lo, hi ) == hi - lo );
        ASSERT_EQ( subject_.intersects_range( lo, hi ), model_.count_in( lo, hi ) != 0 );

        Bitmap const copy{ subject_ };
        ASSERT_TRUE( copy.equals( subject_ ) );
        ASSERT_EQ( ( subject_ | other.bitmap ).size(), ( model_ | other.model ).size() );
        ASSERT_EQ( ( subject_ & other.bitmap ).size(), ( model_ & other.model ).size() );
        ASSERT_EQ( ( subject_ - other.bitmap ).size(), ( model_ - other.model ).size() );
        ASSERT_EQ( ( subject_ ^ other.bitmap ).size(), ( model_ ^ other.model ).size() );

        if constexpr ( requires { subject_.portable_size_in_bytes(); } ) {
            std::vector<std::byte> bytes( subject_.portable_size_in_bytes() );
            ASSERT_EQ( subject_.portable_serialize( bytes ), bytes.size() );
            auto const restored{ Bitmap::portable_deserialize( bytes ) };
            ASSERT_TRUE( restored.has_value() );
            ASSERT_TRUE( restored->equals( subject_ ) );
        }
        std::vector<std::byte> frozen( subject_.frozen_size_bytes() );
        ASSERT_EQ( subject_.frozen_serialize( frozen ), frozen.size() );

        // The finished bitmap goes on being updated, and finishes again.
        subject_.add_many( std::span<std::uint32_t const>{ listed }.first( std::min<std::size_t>( listed.size(), 50 ) ) );
        subject_.optimize();
        ASSERT_EQ( subject_.validate(), nullptr );
        ASSERT_EQ( subject_.size(), model_.size() );
    }

    std::mt19937_64 rng_;
    Bitmap          subject_;
    oracle          model_;
    bool            any_xor_fold_{ false };
    std::uint64_t   last_step_{};
    std::string     trace_;
};

using eager_bitmap = bitmap<std::uint32_t>;
using lazy_runs_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_value_semantics, detail::run_selection_lazy>;

template <typename Bitmap>
void run_sessions( std::uint64_t const first_seed, std::uint64_t const count ) {
    for ( auto seed{ first_seed }; seed != first_seed + count; ++seed ) {
        fold_session<Bitmap> session{ seed };
        session.run();
        if ( ::testing::Test::HasFatalFailure() ) {
            ADD_FAILURE() << "session seed " << seed;
            return;
        }
    }
}

#ifndef FRSR_ROARING_LAZY_FOLD_SESSIONS
#   define FRSR_ROARING_LAZY_FOLD_SESSIONS 150
#endif

TEST( lazy_fold_state, operations_between_folds_and_the_finisher ) {
    run_sessions<eager_bitmap>( 1, FRSR_ROARING_LAZY_FOLD_SESSIONS );
}

// A view is taken of a finished bitmap: of one in the middle of a fold a debug build asserts; after the finisher it is fine.
TEST( lazy_fold_state, a_view_is_taken_of_a_finished_bitmap ) {
    bitmap<std::uint32_t> a, b;
    for ( std::uint32_t key{ 0 }; key < 40'000; key += 2 ) { a.add( key ); }
    for ( std::uint32_t key{ 1 }; key < 40'000; key += 3 ) { b.add( key ); }
    a.bulk_or_intermediate( b );
    EXPECT_DEBUG_DEATH( { auto const deferred_view{ view_of( a ) }; (void)deferred_view; }, "finished" );
    a.bulk_or_finish();
    auto const view{ view_of( a ) };
    EXPECT_EQ( view.size(), a.size() );
    EXPECT_EQ( a.and_cardinality( view ), a.size() );
}

TEST( lazy_fold_state, operations_between_folds_under_the_lazy_run_policy ) {
    run_sessions<lazy_runs_bitmap>( 100'000, FRSR_ROARING_LAZY_FOLD_SESSIONS / 2 );
}

}  // namespace
