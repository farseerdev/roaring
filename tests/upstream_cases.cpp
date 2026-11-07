// The cases of the reference's tests/toplevel_unit.c and tests/cpp_unit.cpp that pin its regressions (copy-on-write
// sharing, range ends past the key space, shrink_to_fit of shared containers, frozen images of shared and empty
// bitmaps, the per-bitmap end iterator, a bitset's maximum in its first word) and the scenarios it shares with the
// Java and Go libraries, replayed against bitmap with the upstream data and expected values. Each test carries the
// name of the upstream one. Where the reference has a function this library spells differently the semantic is
// ported: rank through and_cardinality( key_range::closed( 0, x ) ) and ranked_view, seek through
// ranked_view::for_each_indexed_from, the range functions through key_range operands (whose ends are 64-bit, so an
// end past 2^32 is expressible), lazy unions through bulk_or_intermediate.
//
// The typed tests run with copy-on-write off and on (the value-semantics and the atomic-refcount policies), and with
// either run-selection policy: the lazy one picks containers as the reference does, so the assertions on a
// container's kind are made with it only.

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/ranked_view.hpp>

#include "allocation_fault.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring;

template <typename Cow, typename Runs>
using bitmap_of = bitmap<std::uint32_t, default_container_set<std::uint32_t>, Cow, Runs>;

using atomic_lazy_bitmap = bitmap_of<detail::cow_atomic_refcount, detail::run_selection_lazy>;

template <typename Bitmap>
class UpstreamCases : public ::testing::Test {};

using upstream_bitmaps = ::testing::Types<
    bitmap_of<detail::cow_value_semantics, detail::run_selection_eager>,
    bitmap_of<detail::cow_atomic_refcount, detail::run_selection_eager>,
    bitmap_of<detail::cow_value_semantics, detail::run_selection_lazy>,
    bitmap_of<detail::cow_atomic_refcount, detail::run_selection_lazy>
>;
TYPED_TEST_SUITE( UpstreamCases, upstream_bitmaps );

// Whether the bitmap picks its containers as the reference does (run containers only when asked to).
template <typename Bitmap>
constexpr bool reference_kinds{ std::same_as<typename Bitmap::run_selection_policy, detail::run_selection_lazy> };

constexpr std::uint32_t chunk{ 65'536U };
constexpr std::uint64_t universe{ std::uint64_t{ 1 } << 32 };
constexpr std::uint32_t max32{ std::numeric_limits<std::uint32_t>::max() };
constexpr std::uint64_t max64{ std::numeric_limits<std::uint64_t>::max() };

using values = std::vector<std::uint32_t>;

template <typename Bitmap>
[[nodiscard]] Bitmap of( std::initializer_list<std::uint64_t> const keys ) {
    Bitmap result;
    for ( auto const key : keys ) {
        std::ignore = result.add( static_cast<std::uint32_t>( key ) );
    }
    return result;
}

template <typename Bitmap>
[[nodiscard]] Bitmap of_values( std::span<std::uint32_t const> const keys ) {
    Bitmap result;
    for ( auto const key : keys ) {
        std::ignore = result.add( key );
    }
    return result;
}

// roaring_bitmap_from_range: the values of [min, max) `step` apart
template <typename Bitmap>
[[nodiscard]] Bitmap from_range( std::uint64_t const min, std::uint64_t const max, std::uint32_t const step ) {
    Bitmap result;
    if ( step == 1 ) {
        if ( min < max ) {
            result.add_closed_range( static_cast<std::uint32_t>( min ), static_cast<std::uint32_t>( max - 1 ) );
        }
        return result;
    }
    for ( auto value{ min }; value < max; value += step ) {
        std::ignore = result.add( static_cast<std::uint32_t>( value ) );
    }
    return result;
}

// The values of [lo, hi) added one at a time (an array or a bitset where the run-selection policy is the reference's).
template <typename Bitmap>
[[nodiscard]] Bitmap one_by_one( std::uint32_t const lo, std::uint32_t const hi ) {
    Bitmap result;
    for ( auto value{ lo }; value < hi; ++value ) {
        std::ignore = result.add( value );
    }
    return result;
}

template <typename Bitmap>
[[nodiscard]] values values_of( Bitmap const & bitmap ) {
    auto const held{ bitmap.to_vector() };
    return values( held.begin(), held.end() );
}

// The naive count of the sorted `held` in [lo, hi), the interval clipped to the 32-bit keys.
[[nodiscard]] std::uint64_t count_in( values const & held, std::uint64_t const lo, std::uint64_t const hi ) {
    auto const first{ std::ranges::lower_bound( held, std::min( lo, universe ) ) };
    auto const last{ std::ranges::lower_bound( held, std::min( hi, universe ) ) };
    return last > first ? static_cast<std::uint64_t>( last - first ) : 0U;
}

template <typename Bitmap>
void expect_valid( Bitmap const & bitmap ) {
    EXPECT_EQ( bitmap.validate(), nullptr );
}

// A frozen image in 8-byte aligned storage of its own.
class frozen_image {
public:
    template <typename Bitmap>
    frozen_image( Bitmap const & source, frozen_options const options )
        : storage_( ( source.frozen_size_bytes( options ) + 7 ) / 8 ), size_{ source.frozen_size_bytes( options ) } {
        EXPECT_EQ( source.frozen_serialize( std::as_writable_bytes( std::span{ storage_ } ).first( size_ ), options ), size_ );
    }

    [[nodiscard]] std::span<std::byte const> bytes() const noexcept { return std::as_bytes( std::span{ storage_ } ).first( size_ ); }

private:
    std::vector<std::uint64_t> storage_;
    std::size_t size_;
};

// Runs `f`, which must not allocate, with every allocation of this thread failing where the test binary can make
// allocations fail.
template <typename F>
decltype( auto ) without_allocating( F && f ) {
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    return f();
#else
    frsr::roaring::kernel_test::alloc_fault::armed const armed{ 1 };
    return f();
#endif
}

// The data of the reference's tests, shared with the Java and Go libraries.
constexpr std::uint32_t issue330_values0[]{
    1448147, 1331557, 1166809, 1404655, 1404657, 1448993, 1448994, 1555026,
    1568981, 1166795, 1578735, 1456755, 1581128, 1166754, 1357064, 1166799,
    1581142, 1166774, 1549034, 1090425, 1061936, 1581118, 1568668, 1456470,
    1396063, 1597976, 1458021, 1344102, 1428259, 1166742, 1332378, 1456750,
    1313881, 1371860, 1166770, 1513470, 1456761, 1520695, 1322567, 1456765,
    1457788, 1166816, 1432713, 1581004, 1451025, 1166729, 1587500, 1581022,
    1166707, 1489623, 1581108, 1547596, 1166727, 1345858, 1166741, 1473887,
    1581152, 1581114, 1259737, 1434713, 1456740, 1492705, 1316224, 1448997,
    1481940, 1456767, 1467183, 1576718, 1458286, 1475626, 1166785, 1428366,
    1303084, 1061926, 1388553, 1453950, 1400529, 1581133, 1166713, 1581166,
    1279364, 1322319, 1581027, 1166708, 1442325, 1510314, 1166761, 1404658,
    1062733, 1166764, 1431819, 1568982, 1322271, 1427065, 1374050, 1166721,
    1166714, 1321056, 1303185, 1329366, 1531398, 1071494, 1476413, 1373526,
    1166793, 1404659, 1525886, 1166735, 1593361, 1496990, 1166748, 1366912,
    1541272, 1166697, 1432913, 1559279, 1456736, 1315451, 1365178, 1068808,
    1166768, 1581107, 1345349, 1166792, 1316413, 1449633, 1456758, 1567180,
    1448998, 1423954, 1458607, 1480406, 1493217, 1469065, 1581164, 1581024,
    1486803, 1550949, 1166803, 1166783, 1072253, 1499822, 1166724, 1559280,
    1166732, 1488319, 1166796, 1062666, 1581165, 1507483, 1544041, 1483167,
    1315400, 1166762, 1404660, 1581156, 1166786, 1499824, 1340819, 1166775,
    1166710, 1473917, 1525946, 1166722, 1457787, 1166752, 1581158, 1500566,
    1166736, 1581005, 1525788, 1166700, 1166720, 1341004, 1356523, 1259716,
    1166815, 1166734, 1456734, 1393359, 1315351, 1166808, 1589134, 1166749,
    1405717, 1386258, 1166733, 1166738, 1316508, 1510168, 1434714, 1341261,
    1499823, 1166728, 1089511, 1166790, 1564869, 1316551, 1356413, 1401448,
    1166703, 1456762, 1473571, 1405729, 1340780, 1263511, 1464349, 1166810,
    1314421, 1519020, 1581006, 1514759, 1468171, 1320091, 1522288, 1456753,
    1568393, 1581110, 1530461, 1456744, 1166750, 1518612, 1448325, 1314799,
    1166696, 1166813, 1316622, 1489636, 1456741, 1166699, 1529489, 1481939,
    1347707, 1448995, 1320330, 1466667, 1166739, 1166787, 1581009, 1323304,
    1428380, 1456752, 1450469, 1496415, 1438461, 1450731, 1529496, 1166731,
    1581141, 1581103, 1166706, 1064607, 1587650, 1474681, 1064608, 1166756,
    1581015, 1573170, 1166801, 1581130, 1549033, 1166694, 1166800, 1593359,
    1581155, 1540878, 1599952, 1538583, 1060736, 1166771, 1166711, 1166791,
    1475206, 1166755, 1166798, 1455255, 1456751, 1464345, 1370294, 1401160,
    1530295, 1573169, 1166804, 1061935, 1090681, 1349316, 1447675, 1449634,
    1558271, 1581113, 1587248, 1540108, 1460873, 1562278, 1166788, 1530457,
    1493594, 1456754, 1166753, 1320442, 1581153, 1166758, 1166737, 1322566,
    1588195, 1166746, 1166777, 1492640, 1322682, 1166698, 1528367, 1445599,
    1581026, 1166751, 1445781, 1319119, 1354380, 1581143, 1460447, 1166797,
    1500720, 1369579, 1166806, 1581147, 1464393, 1492632, 1166780, 1448996,
    1166712, 1593362, 1581151, 1166807, 1478408, 1322672, 1581137, 1581002,
    1456764, 1405048, 1166766, 1063491, 1166811, 1296758, 1384972, 1314177,
    1166730, 1066109, 1514764, 1567450, 1581119, 1573634, 1415662, 1563521,
    1059518, 1456738, 1581157, 1550678, 1166726, 1166719, 1581013, 1495998,
    1371312, 1584782, 1456471, 1166778, 1458759, 1060733, 1166784, 1581028,
    1166781, 1583260, 1522601, 1491470, 1166717, 1319114, 1541228, 1456756,
    1448992, 1166744, 1549031, 1570164, 1489634, 1370005, 1475624, 1514141,
    1475625, 1525883, 1166817, 1166759, 1488332, 1313675, 1259957, 1581011,
    1166725, 1483313, 1417102, 1166740, 1260680, 1166779, 1581134, 1315627,
    1430811, 1525947, 1166763, 1435903, 1389401, 1351767, 1343593, 1537920,
    1480377, 1520699, 1319223, 1581148, 1589060, 1060735, 1468871, 1087817,
    1259068, 1581146, 1456759, 1404656, 1166776, 1166695, 1324844, 1589473,
    1374167, 1166769, 1166789, 1456871, 1477673, 1166805, 1581106, 1379727,
    1166723, 1166767, 1166745, 1313278, 1581124, 1451652, 1405710, 1408256,
    1360274, 1325001, 1581136, 1166718, 1072246, 1410198, 1456746, 1557149,
    1166760, 1308054, 1581159, 1581131, 1467536, 1166757, 1434322, 1541831,
    1418436, 1574603, 1437326, 1166814, 1584856, 1388750, 1166743, 1372695,
    1593360, 1166765, 1166794, 1474005, 1166782, 1456748, 1539639, 1056239,
    1166802, 1166747, 1210989};

constexpr std::uint32_t issue330_values1[]{
    1332290, 1447737, 1549291, 1187244, 1598129, 1579851, 1424171, 1538999,
    1445358, 1586739, 1575050, 1437657, 1366343, 1062799, 1421550, 1460317,
    1474875, 1060737, 1330773, 1447797, 1348633, 1559437, 1556214, 1187305,
    1187234, 1187240, 1464660, 1567794, 1187260, 1260646, 1311938, 1573195,
    1318525, 1484524, 1456152, 1087954, 1556007, 1187265, 1460920, 1485316,
    1447849, 1447744, 1474001, 1537891, 1478211, 1313292, 1488405, 1187239,
    1378814, 1343620, 1500498, 1567809, 1435838, 1506575, 1368282, 1447441,
    1598101, 1067076, 1572997, 1598102, 1332697, 1324653, 1561437, 1187290,
    1059945, 1187278, 1457187, 1430003, 1450643, 1447436, 1260650, 1473393,
    1187247, 1087323, 1324967, 1187291, 1480771, 1472729, 1555881, 1187253,
    1456481, 1452672, 1447435, 1378603, 1574771, 1187235, 1417857, 1568706,
    1576739, 1534662, 1410189, 1587745, 1473791, 1187308, 1447730, 1328158,
    1409164, 1506591, 1500147, 1433961, 1483709, 1187227, 1456479, 1562595,
    1314333, 1187281, 1598012, 1415200, 1447791, 1371379, 1598019, 1435836,
    1457188, 1457351, 1187248, 1417111, 1187289, 1187252, 1187257, 1316665,
    1473464, 1187263, 1447732, 1520371, 1525651, 1598177, 1406947, 1465787,
    1524659, 1324213, 1418439, 1575816, 1522696, 1187286, 1497821, 1333179,
    1187282, 1335988, 1548952, 1556066, 1314993, 1187276, 1420503, 1187301,
    1456468, 1423939, 1598089, 1504357, 1343247, 1437659, 1525768, 1539279,
    1307385, 1187275, 1524305, 1332938, 1516498, 1303247, 1304237, 1187238,
    1385283, 1595495, 1187300, 1187241, 1061740, 1316383, 1187307, 1062037,
    1538693, 1454292, 1447731, 1187272, 1561442, 1187268, 1567150, 1597966,
    1447745, 1598178, 1187262, 1464067, 1325394, 1537893, 1332693, 1479200,
    1522335, 1589378, 1450573, 1399161, 1421274, 1561501, 1187232, 1187302,
    1258469, 1331600, 1447740, 1187242, 1328147, 1264069, 1187294, 1299943,
    1598013, 1526975, 1260604, 1487518, 1187229, 1487617, 1354087, 1456595,
    1462047, 1561438, 1598363, 1332691, 1424655, 1567105, 1574774, 1598035,
    1526981, 1384038, 1475987, 1343587, 1447437, 1454912, 1382215, 1447739,
    1456512, 1447779, 1187283, 1440988, 1187293, 1187298, 1574754, 1354772,
    1598018, 1447429, 1598181, 1447738, 1598273, 1312197, 1574752, 1572995,
    1526127, 1473908, 1437660, 1447743, 1362262, 1456513, 1539280, 1348625,
    1415878, 1332694, 1471020, 1432462, 1058088, 1526710, 1371788, 1187288,
    1537984, 1316874, 1187270, 1333565, 1187292, 1447796, 1187311, 1187237,
    1187231, 1574755, 1553822, 1522019, 1447418, 1187269, 1332692, 1447735,
    1529638, 1468154, 1328031, 1447733, 1447402, 1593884, 1332696, 1560622,
    1564819, 1538967, 1315756, 1328338, 1598113, 1324212, 1449895, 1567793,
    1260629, 1430010, 1187266, 1187256, 1312754, 1449417, 1595494, 1529054,
    1187261, 1187306, 1526976, 1425490, 1366922, 1527390, 1187299, 1561510,
    1319222, 1187250, 1057262, 1457999, 1332937, 1187243, 1556213, 1278602,
    1546839, 1187296, 1548950, 1580141, 1187303, 1187255, 1525650, 1572998,
    1576740, 1187267, 1464664, 1440427, 1456467, 1187271, 1187258, 1585428,
    1548760, 1342254, 1447793, 1406348, 1500177, 1260644, 1416954, 1323722,
    1412713, 1187280, 1187310, 1538015, 1537285, 1187285, 1456482, 1260611,
    1490508, 1187274, 1585641, 1416648, 1484655, 1421520, 1347485, 1525652,
    1568987, 1526974, 1314375, 1187246, 1455623, 1488117, 1445025, 1447401,
    1478237, 1561440, 1187287, 1561434, 1509337, 1451859, 1599630, 1348639,
    1449436, 1361844, 1464661, 1263064, 1526973, 1187279, 1562080, 1354770,
    1454521, 1520719, 1478236, 1526972, 1423948, 1334866, 1325026, 1438275,
    1422582, 1437646, 1315530, 1458323, 1447795, 1528218, 1187254, 1598056,
    1417853, 1423514, 1187297, 1187245, 1187264, 1524662, 1187251, 1524660,
    1328113, 1187304, 1374767, 1474057, 1187284, 1331601, 1598180, 1062814,
    1488818, 1187309, 1087494, 1063499, 1458325, 1187295, 1432336, 1260001,
    1597982, 1537147, 1445355, 1595491, 1396111, 1546848, 1474048, 1495251,
    1447734, 1464071, 1526978, 1187236, 1526977, 1566267, 1187277, 1421549,
    1430015, 1316024, 1332695, 1561433, 1435837, 1087250, 1574753, 1476183,
    1325395, 1561432, 1447736, 1500181, 1424164, 1456483, 1187228, 1573384,
    1273769, 1598085, 1437661, 1306415, 1407257, 1187249, 1338215, 1458047,
    1520791, 1447741, 1537263, 1472490, 1524661, 1061729, 1187273, 1417861,
    1470196, 1485881, 1260595, 1538846, 1568762, 1315170, 1500469, 1372455,
    1558140, 1425202, 1432702, 1472734, 1187230, 1187312, 1598191, 1569680,
    1187233, 1263091, 1447417, 1504429, 1430016, 1435839, 1458324, 1546845,
    1575027, 1187259, 1464610};

constexpr std::uint32_t issue330_values2[]{
    1578810, 1166701, 1335901, 1063397, 1578812, 1526471, 1166702, 1067008,
    1412862, 1059750, 1060729, 1166812, 1493768, 1335772, 1336194, 1166772,
    1465775, 1166704, 1355314, 1314138, 1060727, 1388260, 1465786, 1565378,
    1522096, 1312490, 1319284, 1418573, 1319169, 1060730, 1307995, 1465780,
    1060728, 1404061, 1407842, 1326256, 1578754, 1410577, 1060732, 1461338,
    1264209, 1166705, 1581273, 1414196, 1565398, 1355106, 1060731, 1412641,
    1306618, 1060726, 1356302, 1310896, 1457885, 1497721, 1166709, 1067009,
    1310812, 1578800, 1422203, 1484409, 1485278, 1322057, 1369956, 1311089,
    1576614, 1355711, 1355798, 1564832, 1304166, 1166773, 1319071, 1578805,
    1575869, 1403066};

constexpr std::uint32_t issue178_values0[]{
    3585028,   65901253,  143441994, 211160474, 286511937,
    356744840, 434332509, 502812785, 576097614, 646557334,
    714794241, 775083485, 833704249, 889329147, 941367043};

constexpr std::uint32_t issue178_values1[]{
    17883,     54494426,  113908938, 174519827, 235465665, 296685741,
    357644666, 420192495, 476104304, 523046142, 577855081, 634889665,
    692460635, 751350463, 809989192, 863494316, 919127240};

constexpr std::uint32_t karpenske_values[]{
    343798, 343799, 343800, 343801, 343803, 343804, 343805, 343807, 343809,
    343811, 343812, 343815, 343816, 343817, 343818, 343819, 343821, 343825,
    343827, 343828, 343830, 343831, 343832, 343833, 343835, 343836, 343837,
    343838, 343839, 343840, 343841, 343842, 343843, 343844, 343845, 343847,
    343848, 343849, 343850, 343851, 343853, 343854, 343855, 343856, 343858,
    343859, 343860, 343861, 343862, 343863, 343864, 343865, 343866, 343868,
    343869, 343874, 343875, 343877, 343879, 343880, 343881, 343882, 343883,
    343887, 343889, 343890, 343891, 343894, 343895, 343898, 343899, 343900,
    343901, 343902, 343904, 343906, 343907, 343908, 343909, 343910, 343911,
    343912, 343913, 343914, 343915, 343916, 343917, 343918, 343919, 343921,
    343922, 343923, 343924, 343927, 343928, 343929, 343930, 343931, 343932,
    343933, 343934, 343935, 343938, 343939, 343941, 343942, 343943, 343944,
    343945, 343946, 343949, 343951, 343953, 343954, 343955, 343956, 343958,
    343959, 343961, 343962, 343964, 343965, 343966, 343967, 343968, 343969,
    343971, 343972, 343973, 343974, 343976, 343978, 343979, 343981, 343982,
    343983, 343985, 343987, 343988, 343989, 343992, 343993, 343994, 343995,
    343996, 343997, 343998, 344000, 344001, 344002, 344003, 344004, 344006,
    344008, 344009, 344011, 344012, 344013, 344015, 344017, 344019, 344020,
    344021, 344023, 344025, 344026, 344027, 344028, 344029, 344030, 344031,
    344034, 344035, 344036, 344037, 344038, 344039, 344040, 344042, 344043,
    344046, 344047};

constexpr std::uint32_t issue370_a[]{
    239, 240, 241, 242, 243, 244, 259, 260, 261, 262, 263, 264, 265, 266,
    267, 268, 269, 270, 273, 274, 275, 276, 277, 278, 398, 399, 400, 401,
    402, 403, 404, 405, 406, 408, 409, 410, 411, 412, 413, 420, 421, 422,
    509, 510, 511, 512, 513, 514, 539, 540, 541, 542, 543, 544, 547, 548,
    549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 578, 579, 580, 581,
    582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
    624, 625, 634, 635, 636, 649, 650, 651, 652, 653, 654, 714, 715, 716,
    718, 719, 720, 721, 722, 723, 724, 725, 726, 728, 729, 730, 731, 732,
    733, 734, 735, 736, 739, 740, 741, 742, 743, 744, 771, 772, 773};

constexpr std::uint32_t issue370_b[]{
    239, 240, 241, 242, 243, 244, 259, 260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 273, 274, 275, 276, 277, 278, 398, 399, 400, 401, 402, 403,
    404, 405, 406, 408, 409, 410, 411, 412, 413, 420, 421, 422, 509, 510, 511,
    512, 513, 514, 539, 540, 541, 542, 543, 544, 547, 548, 549, 550, 551, 552,
    553, 554, 555, 556, 557, 558, 578, 579, 580, 581, 582, 583, 584, 585, 586,
    607, 608, 634, 635, 636, 649, 650, 651, 652, 653, 654, 714, 715, 716, 718,
    719, 720, 721, 722, 723, 724, 725, 726, 728, 729, 730, 731, 732, 733, 734,
    735, 736, 739, 740, 741, 742, 743, 744, 771, 772, 773};

constexpr std::uint32_t issue467_large_ops[][ 3 ]{
    {0, 0, 16385},       {0, 16385, 20482},   {0, 20482, 27862},
    {1, 0, 16385},       {1, 16385, 20482},   {1, 20482, 27862},
    {0, 27862, 44247},   {0, 44247, 45576},   {0, 45576, 61961},
    {0, 61961, 66058},   {0, 66058, 67247},   {1, 27862, 44247},
    {1, 45576, 61961},   {1, 44247, 45576},   {1, 61961, 66058},
    {1, 66058, 67247},   {0, 67247, 68819},   {0, 68819, 73028},
    {0, 73028, 89413},   {0, 89413, 92266},   {0, 92266, 108651},
    {0, 108651, 113772}, {0, 113772, 118757}, {1, 68819, 73028},
    {1, 73028, 89413},   {1, 92266, 108651},  {1, 89413, 92266},
    {1, 108651, 113772}, {1, 113772, 118757}, {0, 118757, 132098},
    {1, 118757, 132098}, {0, 132098, 137544}, {1, 132098, 137544},
    {0, 137544, 153929}, {0, 153929, 155151}, {0, 155151, 162078},
    {0, 162078, 167119}, {0, 167119, 181012}, {0, 181012, 197397},
    {0, 197397, 201244}, {0, 201244, 217629}, {0, 217629, 222750},
    {0, 222750, 227708}, {0, 227708, 235777}, {0, 235777, 252162},
    {0, 252162, 256259}, {1, 252162, 256259}, {1, 227708, 235777},
    {1, 235777, 252162}};

// tests/toplevel_unit.c: overwrite_copy_on_write_with_plain: overwriting a bitmap that shares its containers
// with another one releases them and leaves the other one intact
TYPED_TEST( UpstreamCases, overwrite_copy_on_write_with_plain ) {
    using Bitmap = TypeParam;
    auto const r{ from_range<Bitmap>( 0, 3 * chunk, 3 ) };
    auto const src{ of<Bitmap>( { 1, 2, 3 } ) };
    Bitmap dest{ r };
    dest = src;
    EXPECT_EQ( values_of( dest ), values_of( src ) );
    expect_valid( dest );
    EXPECT_EQ( r.size(), 65'536U );
    expect_valid( r );

    // a move releases the shared containers without copying any of them
    Bitmap moved_into{ r };
    auto plain{ src };
    without_allocating( [ & ] { moved_into = std::move( plain ); } );
    EXPECT_EQ( values_of( moved_into ), values_of( src ) );
    EXPECT_EQ( values_of( r ), values_of( from_range<Bitmap>( 0, 3 * chunk, 3 ) ) );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_shrink_to_fit_copy_left_sole_holder_shrinks_container: a container is not shrunk while
// another bitmap shares it, and is once the copy is its only holder
TEST( UpstreamCasesShared, test_shrink_to_fit_copy_left_sole_holder_shrinks_container ) {
    using Bitmap = atomic_lazy_bitmap;
    values odd;
    for ( std::uint32_t i{ 1 }; i < 1000; i += 2 ) {
        odd.emplace_back( i );
    }
    Bitmap copy;
    {
        Bitmap source;
        for ( std::uint32_t i{ 0 }; i < 1000; ++i ) {
            std::ignore = source.add( i );
        }
        for ( std::uint32_t i{ 0 }; i < 1000; i += 2 ) {
            std::ignore = source.remove( i );   // room for 1000 values, 500 left
        }
        copy = source;
        EXPECT_EQ( copy.shrink_to_fit(), 0U );   // the source still reads the container
        EXPECT_EQ( values_of( source ), odd );
    }
    EXPECT_GT( copy.shrink_to_fit(), 0U );
    EXPECT_EQ( copy.size(), 500U );
    EXPECT_EQ( values_of( copy ), odd );
    expect_valid( copy );
}

// tests/toplevel_unit.c: test_shrink_to_fit_fitted_bitmap_allocates_nothing
TYPED_TEST( UpstreamCases, test_shrink_to_fit_fitted_bitmap_allocates_nothing ) {
    using Bitmap = TypeParam;
    auto r{ from_range<Bitmap>( 0, 200'000, 3 ) };
    for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
        std::ignore = r.add( ( 1U << 20 ) + i * 7 );   // an array container
    }
    r.add_closed_range( 1U << 24, ( 1U << 24 ) + 4999 );   // a run
    r.optimize_keep_bitsets();
    std::ignore = r.shrink_to_fit();
    auto const before{ values_of( r ) };
    EXPECT_EQ( without_allocating( [ & ] { return r.shrink_to_fit(); } ), 0U );
    EXPECT_EQ( values_of( r ), before );
}

// tests/cpp_unit.cpp: test_cpp_shrink_to_fit_erases_and_counts_emptied_bitmap_64: a chunk emptied by removals is
// erased, so nothing of it is left to shrink
TEST( UpstreamCasesWideKeys, test_cpp_shrink_to_fit_erases_and_counts_emptied_bitmap_64 ) {
    using Bitmap = bitmap<std::uint64_t>;
    Bitmap emptied;
    for ( std::uint64_t i{ 0 }; i < 1000; ++i ) {
        std::ignore = emptied.add( i << 16 );
    }
    for ( std::uint64_t i{ 0 }; i < 1000; ++i ) {
        std::ignore = emptied.remove( i << 16 );
    }
    EXPECT_TRUE( emptied.empty() );
    EXPECT_EQ( emptied.container_count(), 0U );
    EXPECT_EQ( emptied.shrink_to_fit(), 0U );
    EXPECT_EQ( emptied.byte_size(), Bitmap{}.byte_size() );
}

// tests/toplevel_unit.c: test_range_end_beyond_universe: an interval end past 2^32 is 2^32, never wrapped
TYPED_TEST( UpstreamCases, test_range_end_beyond_universe ) {
    using Bitmap = TypeParam;

    // remove_range (Go TestRoaringRangeEnd)
    auto r{ of<Bitmap>( { max32 } ) };
    r.andnot_with( key_range::half_open( 0, max32 ) );
    EXPECT_EQ( r.size(), 1U );
    r.andnot_with( key_range::half_open( 0, max64 ) );
    EXPECT_EQ( r.size(), 0U );
    std::ignore = r.add( max32 );
    r.andnot_with( key_range::half_open( 0, universe + 1 ) );
    EXPECT_EQ( r.size(), 0U );
    std::ignore = r.add( max32 );
    r.andnot_with( key_range::half_open( 0, universe ) );
    EXPECT_EQ( r.size(), 0U );
    // a range starting at 2^32 is empty: it must not remove 0
    std::ignore = r.add( 0 );
    r.andnot_with( key_range::half_open( universe, universe + 1 ) );
    EXPECT_EQ( r.size(), 1U );

    // add_range
    r = Bitmap{};
    r.or_with( key_range::half_open( universe, universe + 1 ) );
    r.or_with( key_range::half_open( universe + 5, universe + 10 ) );
    EXPECT_TRUE( r.empty() );
    r.or_with( key_range::half_open( max32, max64 ) );
    EXPECT_EQ( r.size(), 1U );
    EXPECT_TRUE( r.contains( max32 ) );
    r.or_with( key_range::half_open( 0, universe + 1 ) );
    EXPECT_EQ( r.size(), universe );

    // range_cardinality
    r = of<Bitmap>( { 5, max32 } );
    EXPECT_EQ( r.and_cardinality( key_range::half_open( 0, universe + 1 ) ), 2U );
    EXPECT_EQ( r.and_cardinality( key_range::half_open( 0, max64 ) ), 2U );
    EXPECT_EQ( r.and_cardinality( key_range::half_open( max32, max64 ) ), 1U );
    EXPECT_EQ( r.and_cardinality( key_range::half_open( universe, max64 ) ), 0U );

    // contains_range
    r = of<Bitmap>( { 0 } );
    EXPECT_FALSE( r.is_superset_of( key_range::half_open( 0, universe + 1 ) ) );
    EXPECT_FALSE( r.is_superset_of( key_range::half_open( 0, universe + 6 ) ) );
    std::ignore = r.add( max32 );
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( max32, max64 ) ) );
    r.or_with( key_range::half_open( 0, universe ) );
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( 0, universe + 1 ) ) );
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( 0, max64 ) ) );

    // flip and flip_inplace
    Bitmap const empty;
    auto f{ empty };
    f.xor_with( key_range::half_open( universe, universe + 1 ) );
    EXPECT_TRUE( f.empty() );
    f = empty;
    f.xor_with( key_range::half_open( 0, universe + 1 ) );
    EXPECT_EQ( f.size(), universe );
    f = empty;
    f.xor_with( key_range::half_open( max32, max64 ) );
    EXPECT_EQ( f.size(), 1U );
    EXPECT_TRUE( f.contains( max32 ) );
    f.xor_with( key_range::half_open( universe, universe + 1 ) );
    EXPECT_EQ( f.size(), 1U );
    f.xor_with( key_range::half_open( max32, universe + 5 ) );
    EXPECT_TRUE( f.empty() );
    f.xor_with( key_range::half_open( 0, universe + 1 ) );
    EXPECT_EQ( f.size(), universe );

    // intersect_with_range
    r = of<Bitmap>( { 5 } );
    EXPECT_FALSE( r.intersects( key_range::half_open( universe + 5, universe + 10 ) ) );
    EXPECT_FALSE( r.intersects( key_range::half_open( universe, max64 ) ) );
    EXPECT_TRUE( r.intersects( key_range::half_open( 0, max64 ) ) );
    EXPECT_TRUE( r.intersects( key_range::half_open( 5, universe + 1 ) ) );
    std::ignore = r.add( max32 );
    EXPECT_TRUE( r.intersects( key_range::half_open( max32, max64 ) ) );
    EXPECT_FALSE( r.intersects( key_range::half_open( universe + max32, max64 ) ) );
}

// tests/cpp_unit.cpp: test_cpp_range_end_beyond_universe
TYPED_TEST( UpstreamCases, test_cpp_range_end_beyond_universe ) {
    using Bitmap = TypeParam;
    {
        // addRange with an end past 2^32 adds up to the largest key
        Bitmap r1;
        r1.or_with( key_range::half_open( max32 - 1, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ max32 - 1, max32 } ) );
        // a range starting at 2^32 is empty: 0 must not be added
        r1.or_with( key_range::half_open( universe, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ max32 - 1, max32 } ) );
    }
    {
        // removeRange with an end past 2^32 removes the largest key
        auto r1{ of<Bitmap>( { 0, 7, max32 } ) };
        r1.andnot_with( key_range::half_open( 5, max64 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ 0 } ) );
        // a range starting at 2^32 is empty: 0 must not be removed
        r1.andnot_with( key_range::half_open( universe, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ 0 } ) );
    }
    {
        // containsRange: {0} does not contain [0, 2^32 + 1)
        auto r1{ of<Bitmap>( { 0 } ) };
        EXPECT_FALSE( r1.is_superset_of( key_range::half_open( 0, universe + 1 ) ) );
        std::ignore = r1.add( max32 );
        EXPECT_TRUE( r1.is_superset_of( key_range::half_open( max32, max64 ) ) );
    }
    {
        // flip with an end past 2^32 flips up to the largest key
        Bitmap r1;
        r1.xor_with( key_range::half_open( max32 - 1, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ max32 - 1, max32 } ) );
        // a range starting at 2^32 is empty: 0 must not be flipped
        r1.xor_with( key_range::half_open( universe, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ max32 - 1, max32 } ) );
    }
    {
        // mask with an end past 2^32 keeps the largest key
        auto r1{ of<Bitmap>( { 1, 5, max32 } ) };
        r1.and_with( key_range::half_open( 2, universe + 1 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ 5, max32 } ) );
    }
}

// tests/toplevel_unit.c: test_frozen_serialization_empty, with this library's frozen image
TYPED_TEST( UpstreamCases, test_frozen_serialization_empty ) {
    using Bitmap = TypeParam;
    Bitmap const r;
    for ( auto const options : { frozen_options{}, frozen_options{ .ranked = true } } ) {
        frozen_image const image{ r, options };
        auto const view{ Bitmap::frozen_view_from( image.bytes() ) };
        ASSERT_TRUE( static_cast<bool>( view ) );
        EXPECT_TRUE( view.empty() );
        EXPECT_FALSE( view.contains( 0 ) );
        auto const back{ view.materialize() };
        EXPECT_TRUE( back.empty() );
        expect_valid( back );
    }
}

// tests/toplevel_unit.c: test_frozen_serialization_shared_containers: a bitmap whose containers are shared
// with a copy writes the image it wrote before the copy
TYPED_TEST( UpstreamCases, test_frozen_serialization_shared_containers ) {
    using Bitmap = TypeParam;
    Bitmap r;
    std::ignore = r.add( 1 );   // array container
    r.add_closed_range( chunk, chunk + 99 );   // run container
    for ( std::uint32_t i{ 0 }; i < chunk; i += 2 ) {   // bitset container
        std::ignore = r.add( 2 * chunk + i );
    }
    r.optimize_keep_bitsets();
    auto const statistics{ r.statistics() };
    EXPECT_EQ( statistics.array_containers, 1U );
    EXPECT_EQ( statistics.run_containers, 1U );
    EXPECT_EQ( statistics.bitset_containers, 1U );
    for ( auto const options : { frozen_options{}, frozen_options{ .ranked = true } } ) {
        frozen_image const before{ r, options };
        Bitmap const copy{ r };
        EXPECT_EQ( r.frozen_size_bytes( options ), before.bytes().size() );
        frozen_image const shared{ r, options };
        frozen_image const of_copy{ copy, options };
        EXPECT_TRUE( std::ranges::equal( shared.bytes(), before.bytes() ) );
        EXPECT_TRUE( std::ranges::equal( of_copy.bytes(), before.bytes() ) );
        auto const view{ Bitmap::frozen_view_from( shared.bytes() ) };
        ASSERT_TRUE( static_cast<bool>( view ) );
        EXPECT_EQ( values_of( view.materialize() ), values_of( r ) );
    }
}

// tests/toplevel_unit.c: test_portable_deserialize_frozen_unaligned: the portable format read from a buffer at any
// alignment (this library's frozen view requires 8-byte alignment by contract, so the view itself does not apply)
template <typename Bitmap>
void check_portable_queries( Bitmap const & r, std::size_t const offset ) {
    auto const size{ r.portable_size_in_bytes() };
    std::vector<std::byte> storage( size + offset );
    auto const buffer{ std::span{ storage }.subspan( offset ) };
    ASSERT_EQ( r.portable_serialize( buffer ), size );
    auto const v{ Bitmap::portable_deserialize( buffer ) };
    ASSERT_TRUE( v.has_value() );
    expect_valid( *v );
    EXPECT_TRUE( v->equals( r ) );
    EXPECT_TRUE( r.equals( *v ) );
    EXPECT_TRUE( v->is_subset_of( r ) );
    EXPECT_FALSE( v->is_strict_subset_of( r ) );
    EXPECT_EQ( v->size(), r.size() );
    EXPECT_EQ( v->front(), r.front() );
    EXPECT_EQ( v->back(), r.back() );
    EXPECT_EQ( v->and_cardinality( key_range::half_open( 100, 200'000 ) ), r.and_cardinality( key_range::half_open( 100, 200'000 ) ) );
    EXPECT_EQ( v->contains_range( 140'000, 150'000 ), r.contains_range( 140'000, 150'000 ) );
    EXPECT_EQ( v->intersects_range( 70'000, 70'100 ), r.intersects_range( 70'000, 70'100 ) );
    EXPECT_EQ( ( *v ^ r ).size(), 0U );
    EXPECT_EQ( ( *v - r ).size(), 0U );
    EXPECT_EQ( values_of( *v | r ), values_of( r ) );
    std::vector<std::byte> again( size );
    ASSERT_EQ( v->portable_serialize( again ), size );
    EXPECT_TRUE( std::ranges::equal( again, buffer ) );
}

TYPED_TEST( UpstreamCases, test_portable_deserialize_frozen_unaligned ) {
    using Bitmap = TypeParam;
    Bitmap r;
    for ( std::uint32_t i{ 1 }; i < 3000; i += 3 ) {
        std::ignore = r.add( i );   // array
    }
    for ( std::uint32_t i{ chunk }; i < chunk + 20'000; i += 2 ) {
        std::ignore = r.add( i );   // bitset
    }
    r.add_closed_range( 140'000, 149'999 );   // run
    r.optimize_keep_bitsets();
    // three containers: no offset headers, and a single run-flag byte puts every payload at an odd offset
    for ( std::size_t offset{ 0 }; offset < 8; ++offset ) {
        check_portable_queries( r, offset );
    }
    // enough containers for offset headers
    for ( std::uint32_t k{ 3 }; k < 12; ++k ) {
        std::ignore = r.add( ( k << 16 ) + 7 * k );
    }
    for ( std::size_t offset{ 0 }; offset < 8; ++offset ) {
        check_portable_queries( r, offset );
    }
    // the bitset container first
    r.andnot_with( key_range::half_open( 0, chunk ) );
    for ( std::size_t offset{ 0 }; offset < 8; ++offset ) {
        check_portable_queries( r, offset );
    }
}

// tests/toplevel_unit.c: andnot_run_containers_tail: the first operand keeps several runs past the second's last one
TYPED_TEST( UpstreamCases, andnot_run_containers_tail ) {
    using Bitmap = TypeParam;
    Bitmap r1, r2;
    for ( std::uint32_t x{ 0 }; x < 4000; x += 20 ) {
        r1.add_closed_range( x, x + 4 );
        if ( x < 1000 ) {
            r2.add_closed_range( x + 1, x + 3 );
        }
    }
    r1.optimize_keep_bitsets();
    r2.optimize_keep_bitsets();
    values expected;
    std::ranges::set_difference( values_of( r1 ), values_of( r2 ), std::back_inserter( expected ) );
    EXPECT_EQ( values_of( r1 - r2 ), expected );
    auto in_place{ r1 };
    in_place -= r2;
    EXPECT_EQ( values_of( in_place ), expected );
    expect_valid( in_place );
}

// gen_bitmap of tests/toplevel_unit.c: runs of `run` values whose chance to be present starts at `start` and changes
// by `gradient` per value, none in [blank_begin, blank_end)
template <typename Bitmap>
[[nodiscard]] Bitmap synthetic_bitmap( std::mt19937_64 & rng, double const start, double const gradient, std::uint32_t const run, std::uint32_t const blank_begin, std::uint32_t const blank_end, std::uint32_t const universe_end ) {
    Bitmap result;
    std::uniform_real_distribution<double> draw{ 0.0, 1.0 };
    for ( std::uint32_t i{ 0 }; i < universe_end; i += run ) {
        if ( draw( rng ) < start + i * gradient && !( i >= blank_begin && i < blank_end ) ) {
            result.add_closed_range( i, i + run - 1 );
        }
    }
    result.optimize_keep_bitsets();
    return result;
}

// tests/toplevel_unit.c: test_andnot_inplace_false: the in-place difference of two bitmaps, with copy-on-write off in
// the reference
TYPED_TEST( UpstreamCases, test_andnot_inplace_false ) {
    using Bitmap = TypeParam;
    constexpr std::uint64_t c{ chunk };
    auto const rb1{ of<Bitmap>( { 1, 2, c * 2 + 1, c * 2 + 2, c * 3 + 1, c * 3 + 2, c * 10 + 1, c * 10 + 2, c * 16 + 1, c * 16 + 2, c * 20 + 1, c * 21 + 1 } ) };
    auto const rb2{ of<Bitmap>( { 2, 3, c * 10 + 2, c * 10 + 3, c * 12 + 2, c * 12 + 3, c * 14 + 2, c * 14 + 3, c * 16 + 2, c * 16 + 3 } ) };
    auto const rb3{ of<Bitmap>( { 2, 3, c * 10 + 1, c * 10 + 2, c * 12 + 2, c * 12 + 3, c * 14 + 2, c * 14 + 3, c * 16 + 2, c * 16 + 3 } ) };
    auto const in_place_difference{ []( Bitmap const & lhs, Bitmap const & rhs ) {
        auto copy{ lhs };
        copy -= rhs;
        EXPECT_EQ( copy.validate(), nullptr );
        return values_of( copy );
    } };
    EXPECT_EQ( in_place_difference( rb1, rb2 ), values_of( of<Bitmap>( { 1, c * 2 + 1, c * 2 + 2, c * 3 + 1, c * 3 + 2, c * 10 + 1, c * 16 + 1, c * 20 + 1, c * 21 + 1 } ) ) );
    EXPECT_EQ( in_place_difference( rb1, rb3 ), values_of( of<Bitmap>( { 1, c * 2 + 1, c * 2 + 2, c * 3 + 1, c * 3 + 2, c * 16 + 1, c * 20 + 1, c * 21 + 1 } ) ) );
    EXPECT_EQ( in_place_difference( rb2, rb1 ), values_of( of<Bitmap>( { 3, c * 10 + 3, c * 12 + 2, c * 12 + 3, c * 14 + 2, c * 14 + 3, c * 16 + 3 } ) ) );
    EXPECT_EQ( in_place_difference( rb3, rb1 ), values_of( of<Bitmap>( { 3, c * 12 + 2, c * 12 + 3, c * 14 + 2, c * 14 + 3, c * 16 + 3 } ) ) );
    EXPECT_EQ( in_place_difference( rb3, rb2 ), ( values{ chunk * 10 + 1 } ) );
    auto const large_run{ from_range<Bitmap>( 2, 11 * c + 27, 1 ) };
    EXPECT_EQ( in_place_difference( rb1, large_run ), values_of( of<Bitmap>( { 1, c * 16 + 1, c * 16 + 2, c * 20 + 1, c * 21 + 1 } ) ) );

    Bitmap r1, r2;
    std::size_t diff_cardinality{ 0 };
    for ( std::uint32_t i{ 0 }; i < 300; ++i ) {
        if ( i % 2 == 0 ) {
            std::ignore = r1.add( i );
        }
        if ( i % 3 == 0 ) {
            std::ignore = r2.add( i );
        }
        if ( i % 2 == 0 && i % 3 != 0 ) {
            ++diff_cardinality;
        }
    }
    r1 -= r2;
    EXPECT_EQ( r1.size(), diff_cardinality );

    std::mt19937_64 rng{ 2688 };
    std::vector<Bitmap> r;
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 0.0, 1e-6, 1, 0, 0, 1'000'000 ) );   // ascending density, last containers might be runs
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 1.0, -1e-6, 1, 0, 0, 1'000'000 ) );   // descending density, first containers might be runs
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 1e-5, 0.0, 1, 0, 0, 2'000'000 ) );   // uniformly rather sparse
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 1e-5, 0.0, 3, 0, 0, 2'000'000 ) );   // uniformly rather sparse with runs
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 1e-1, 0.0, 1, 0, 0, 2'000'000 ) );   // uniformly rather dense
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 0.001, 1e-7, 1, 0, 0, 1'000'000 ) );   // ascending density but never too dense
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 0.0, 1e-10, 1, 0, 0, 1'000'000 ) );   // ascending density but very sparse
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 0.5, -1e-6, 1, 600'000, 800'000, 1'000'000 ) );   // descending with a gap
    r.emplace_back( synthetic_bitmap<Bitmap>( rng, 1, -1e-6, 1, 300'000, 500'000, 1'000'000 ) );   // gap elsewhere
    for ( std::size_t i{ 0 }; i < r.size(); ++i ) {
        for ( auto j{ i + 1 }; j < r.size(); ++j ) {
            values expected;
            std::ranges::set_difference( values_of( r[ i ] ), values_of( r[ j ] ), std::back_inserter( expected ) );
            EXPECT_EQ( in_place_difference( r[ i ], r[ j ] ), expected ) << i << ' ' << j;
        }
    }
}

// tests/toplevel_unit.c: test_intersect_with_range_edges (Java intersects(long, long), Go TestIntersectsWithInterval)
TYPED_TEST( UpstreamCases, test_intersect_with_range_edges ) {
    using Bitmap = TypeParam;
    auto const intersects{ []( Bitmap const & r, std::uint64_t const lo, std::uint64_t const hi ) { return r.intersects( key_range::half_open( lo, hi ) ); } };
    auto r{ of<Bitmap>( { 5 } ) };
    EXPECT_TRUE( intersects( r, 5, max64 ) );
    EXPECT_FALSE( intersects( r, 5, 5 ) );

    r = of<Bitmap>( { max32 } );
    EXPECT_TRUE( intersects( r, max32, universe ) );
    EXPECT_FALSE( intersects( r, max32 - 1, max32 ) );
    EXPECT_TRUE( intersects( r, 0, universe ) );
    EXPECT_TRUE( intersects( r, 0, max64 ) );

    r = from_range<Bitmap>( 21, 26, 1 );
    EXPECT_FALSE( intersects( r, 22, 22 ) );
    EXPECT_FALSE( intersects( r, 27, 27 ) );
    EXPECT_TRUE( intersects( r, 22, 23 ) );
    EXPECT_TRUE( intersects( r, 19, 23 ) );
    EXPECT_TRUE( intersects( r, 23, 30 ) );
    EXPECT_TRUE( intersects( r, 19, 30 ) );
    EXPECT_FALSE( intersects( r, 19, 20 ) );
    EXPECT_FALSE( intersects( r, 19, 21 ) );
    EXPECT_TRUE( intersects( r, 19, 22 ) );
    EXPECT_FALSE( intersects( r, 26, 30 ) );
    EXPECT_FALSE( intersects( r, 28, 30 ) );
    r.add_closed_range( 30, 39 );
    EXPECT_FALSE( intersects( r, 28, 29 ) );
    EXPECT_TRUE( intersects( r, 28, 31 ) );

    // a bitset container (key 0) with values only in the lower half
    r = from_range<Bitmap>( 1, 32'768, 2 );
    r.promote_large_arrays( 1 );
    ASSERT_EQ( r.statistics().bitset_containers, 1U );
    EXPECT_FALSE( intersects( r, 32'768, 65'535 ) );
    EXPECT_FALSE( intersects( r, 32'768, 65'536 ) );
    EXPECT_TRUE( intersects( r, 32'767, 32'769 ) );

    // bitset containers at key 0x8000 (values of 2^31 and above)
    for ( auto const word : { std::uint64_t{ 1789303257167203747U }, std::uint64_t{ 2697219678014362575U } } ) {
        r = Bitmap{};
        for ( std::uint32_t i{ 0 }; i < chunk; ++i ) {
            if ( ( word >> ( i % 64 ) ) & 1 ) {
                std::ignore = r.add( 0x8000'0000U + i );
            }
        }
        r.promote_large_arrays( 1 );
        ASSERT_EQ( r.statistics().bitset_containers, 1U );
        std::uint64_t const first{ r.front() };
        std::uint64_t const last{ r.back() };
        EXPECT_FALSE( intersects( r, 0, first ) );
        EXPECT_FALSE( intersects( r, first - 1, first ) );
        EXPECT_TRUE( intersects( r, first, first + 1 ) );
        EXPECT_TRUE( intersects( r, first - 1, first + 1 ) );
        EXPECT_TRUE( intersects( r, last, last + 1 ) );
        EXPECT_TRUE( intersects( r, last - 1, last + 1 ) );
        EXPECT_FALSE( intersects( r, last + 1, universe ) );
    }
}

// The bitmaps of the reference's interval tests: arrays, bitsets and runs at the given chunks.
template <typename Bitmap>
[[nodiscard]] Bitmap mixed_containers_bitmap( std::initializer_list<std::uint32_t> const array_keys, std::initializer_list<std::uint32_t> const bitset_keys, std::initializer_list<std::uint32_t> const run_keys ) {
    Bitmap r;
    for ( auto const key : array_keys ) {
        for ( std::uint32_t i{ 0 }; i < 1000; i += 17 ) {
            std::ignore = r.add( ( key << 16 ) + i );
        }
    }
    for ( auto const key : bitset_keys ) {
        for ( std::uint32_t i{ 0 }; i < chunk; i += 2 ) {
            std::ignore = r.add( ( key << 16 ) + i );
        }
    }
    for ( auto const key : run_keys ) {
        r.add_closed_range( key << 16, ( key << 16 ) + 49'999 );
    }
    r.optimize_keep_bitsets();
    return r;
}

template <typename Bitmap>
[[nodiscard]] std::vector<Bitmap> interval_bitmaps() {
    std::vector<Bitmap> bitmaps;
    bitmaps.emplace_back( of<Bitmap>( { 1, 2, 3 } ) );
    bitmaps.emplace_back( of<Bitmap>( { 0xC000'0000U } ) );
    bitmaps.emplace_back( of<Bitmap>( { 1, 0xC000'0000U } ) );
    bitmaps.emplace_back( of<Bitmap>( { 1, chunk, 0xC000'0000U } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 10 }, { 20 }, { 30 } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 0 }, { 1 }, { 20 } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 1 }, { 20 }, { 0 } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 20 }, { 0 }, { 1 } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 0, 1, 2 }, { 200 }, { 205 } ) );
    bitmaps.emplace_back( mixed_containers_bitmap<Bitmap>( { 2 }, { 0 }, { 1 } ) );
    bitmaps[ 4 ].add_closed_range( 70'000, 149'999 );
    return bitmaps;
}

constexpr std::uint64_t interval_points[]{
    0, 1, 2, 3, 4, 256, 999, 1000, 65'535, 65'536, 65'537, 67'000, 70'000, 71'000, 140'000, 150'000,
    std::uint64_t{ 10 } << 16, ( std::uint64_t{ 10 } << 16 ) + 1, std::uint64_t{ 20 } << 16, ( std::uint64_t{ 20 } << 16 ) + 1,
    std::uint64_t{ 30 } << 16, ( std::uint64_t{ 30 } << 16 ) + 50'000, std::uint64_t{ 199 } << 16, ( std::uint64_t{ 200 } << 16 ) + ( 1 << 14 ),
    std::uint64_t{ 205 } << 16, 0xC000'0000U, 0xC000'0001U, max32, universe
};

// tests/toplevel_unit.c: test_intersect_with_range_matches_cardinality (Java RoaringBitmapIntervalIntersectionTest),
// here against the values counted one by one
TYPED_TEST( UpstreamCases, test_intersect_with_range_matches_cardinality ) {
    using Bitmap = TypeParam;
    for ( auto const & r : interval_bitmaps<Bitmap>() ) {
        expect_valid( r );
        auto const held{ values_of( r ) };
        for ( std::size_t i{ 0 }; i < std::size( interval_points ); ++i ) {
            for ( auto j{ i }; j < std::size( interval_points ); ++j ) {
                auto const range{ key_range::half_open( interval_points[ i ], interval_points[ j ] ) };
                auto const expected{ count_in( held, interval_points[ i ], interval_points[ j ] ) };
                ASSERT_EQ( r.intersects( range ), expected > 0 ) << interval_points[ i ] << ' ' << interval_points[ j ];
                ASSERT_EQ( r.and_cardinality( range ), expected ) << interval_points[ i ] << ' ' << interval_points[ j ];
            }
        }
        std::uint64_t const first{ r.front() };
        std::uint64_t const last{ r.back() };
        EXPECT_TRUE( r.intersects( key_range::half_open( first > 0 ? first - 1 : 0, first + 1 ) ) );
        EXPECT_TRUE( r.intersects( key_range::half_open( last - 1, last + 1 ) ) );
        EXPECT_FALSE( r.intersects( key_range::half_open( 0, first ) ) );
        EXPECT_FALSE( r.intersects( key_range::half_open( last + 1, universe ) ) );
    }
}

// tests/toplevel_unit.c: test_contains_range_agrees_with_range_cardinality (Java testContainsRange_*, issue 623)
TYPED_TEST( UpstreamCases, test_contains_range_agrees_with_range_cardinality ) {
    using Bitmap = TypeParam;
    constexpr std::uint64_t points[]{
        0, 1, 0xFFFF, 0x10000, 0x10001, 0x1FFFF, 0x20000, 0x30000, 0x30001, 0xFFFE'FFFF, 0xFFFF'0000, 0xFFFF'0001,
        0xFFFF'FFFE, 0xFFFF'FFFF, universe
    };
    std::vector<Bitmap> bitmaps;
    bitmaps.emplace_back( from_range<Bitmap>( 0, chunk, 1 ) );
    bitmaps.emplace_back( from_range<Bitmap>( 0, chunk, 1 ) );
    std::ignore = bitmaps[ 1 ].add( 3 << 16 );
    bitmaps.emplace_back( from_range<Bitmap>( 0, 2 * chunk, 1 ) );
    bitmaps[ 2 ].or_with( key_range::half_open( 0xFFFF'0000U, universe ) );
    bitmaps.emplace_back( from_range<Bitmap>( 0xFFFE'0000U, universe, 1 ) );
    bitmaps.emplace_back( of<Bitmap>( { 0, 1 << 16, 3 << 16, 0xFFFF'FFFEU, 0xFFFF'FFFFU } ) );
    bitmaps.emplace_back( from_range<Bitmap>( 0, universe, 1 ) );
    // the spans a bitmap covers, for the bitmaps too large to list
    std::vector<std::vector<key_range>> const covered{
        { key_range::closed( 0, 0xFFFF ) },
        { key_range::closed( 0, 0xFFFF ), key_range::closed( 0x30000, 0x30000 ) },
        { key_range::closed( 0, 0x1FFFF ), key_range::closed( 0xFFFF'0000U, max32 ) },
        { key_range::closed( 0xFFFE'0000U, max32 ) },
        { key_range::closed( 0, 0 ), key_range::closed( 0x10000, 0x10000 ), key_range::closed( 0x30000, 0x30000 ), key_range::closed( 0xFFFF'FFFEU, max32 ) },
        { key_range::closed( 0, max32 ) },
    };
    for ( std::size_t b{ 0 }; b < bitmaps.size(); ++b ) {
        for ( std::size_t i{ 0 }; i < std::size( points ); ++i ) {
            for ( auto j{ i + 1 }; j < std::size( points ); ++j ) {
                std::uint64_t expected_card{ 0 };
                for ( auto const span : covered[ b ] ) {
                    auto const lo{ std::max( span.first, points[ i ] ) };
                    auto const hi{ std::min( span.last + 1, points[ j ] ) };
                    expected_card += hi > lo ? hi - lo : 0;
                }
                auto const range{ key_range::half_open( points[ i ], points[ j ] ) };
                ASSERT_EQ( bitmaps[ b ].and_cardinality( range ), expected_card ) << b << ' ' << points[ i ] << ' ' << points[ j ];
                ASSERT_EQ( bitmaps[ b ].is_superset_of( range ), expected_card == points[ j ] - points[ i ] ) << b << ' ' << points[ i ] << ' ' << points[ j ];
            }
        }
    }

    auto r{ of<Bitmap>( { max32 } ) };
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( max32, universe ) ) );
    EXPECT_FALSE( r.is_superset_of( key_range::half_open( max32 - 1, universe ) ) );
    r = from_range<Bitmap>( 0xFFFF'0000U, universe, 1 );
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( 0xFFFF'8000U, universe ) ) );

    r = from_range<Bitmap>( 1, 7, 1 );
    EXPECT_TRUE( r.contains_range( 1, 7 ) );
    EXPECT_FALSE( r.contains_range( 2, 8 ) );
    EXPECT_FALSE( r.contains_range( 0, 1 ) );
    EXPECT_FALSE( r.contains_range( 1 << 17, 1 << 19 ) );

    // issue 623
    r = of<Bitmap>( { 65'535, 65'536 } );
    EXPECT_TRUE( r.contains_range( 65'535, 65'537 ) );
    for ( std::uint32_t i{ 1 }; i <= 100'000; ++i ) {
        r.add_closed_range( i, i );
    }
    for ( std::uint32_t i{ 1 }; i <= 100'000; ++i ) {
        ASSERT_TRUE( r.contains_range( i, i + 1 ) ) << i;
    }
    EXPECT_TRUE( r.contains_range( 1, 100'001 ) );
    EXPECT_FALSE( r.contains_range( 0, 100'001 ) );
}

// tests/toplevel_unit.c: test_range_cardinality_layouts (Go TestCardinalityInRange*, Java rangeCardinalityFullContainer
// and testRangeCardinalityAtBoundary, issue 285)
TYPED_TEST( UpstreamCases, test_range_cardinality_layouts ) {
    using Bitmap = TypeParam;
    auto const cardinality{ []( Bitmap const & r, std::uint64_t const lo, std::uint64_t const hi ) { return r.and_cardinality( key_range::half_open( lo, hi ) ); } };
    auto r{ from_range<Bitmap>( 0, 300, 3 ) };
    EXPECT_EQ( cardinality( r, 0, 300 ), 100U );
    EXPECT_EQ( cardinality( r, 0, 1 ), 1U );
    EXPECT_EQ( cardinality( r, 0, 4 ), 2U );
    EXPECT_EQ( cardinality( r, 3, 7 ), 2U );

    r = from_range<Bitmap>( 0, 16'384, 2 );
    if constexpr ( reference_kinds<Bitmap> ) {
        EXPECT_EQ( r.statistics().bitset_containers, 1U );
    }
    EXPECT_EQ( cardinality( r, 0, 16'384 ), 8192U );
    EXPECT_EQ( cardinality( r, 1, 2 ), 0U );
    EXPECT_EQ( cardinality( r, 1, 3 ), 1U );
    EXPECT_EQ( cardinality( r, 100, 200 ), 50U );

    r = Bitmap{};
    for ( std::uint32_t const key : { 0U, 1U, 3U } ) {
        for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
            std::ignore = r.add( ( key << 16 ) + i );
        }
    }
    EXPECT_EQ( cardinality( r, 50, 196'658 ), 200U );
    EXPECT_EQ( cardinality( r, 131'072, 196'608 ), 0U );

    r = from_range<Bitmap>( 0, 1024, 1 );
    r.add_closed_range( 2048, 3071 );
    r.optimize_keep_bitsets();
    EXPECT_EQ( cardinality( r, 1023, 1025 ), 1U );
    EXPECT_EQ( cardinality( r, 512, 2560 ), 1024U );

    r = of<Bitmap>( { 0, 0x8000'0000U, max32 } );
    EXPECT_EQ( cardinality( r, 0, universe ), 3U );
    EXPECT_EQ( cardinality( r, max32, universe ), 1U );

    r = of<Bitmap>( { 66'236 } );
    EXPECT_EQ( cardinality( r, 60'000, 70'000 ), 1U );

    // whole containers of every kind, and agreement with the values counted one by one
    r = from_range<Bitmap>( 0, chunk, 37 );   // array
    r |= from_range<Bitmap>( chunk, 2 * chunk, 3 );   // bitset
    r.add_closed_range( 2 * chunk, 3 * chunk - 1 );   // full run
    r.or_with( key_range::half_open( 0xFFFF'0000U, universe ) );   // full run
    r.optimize_keep_bitsets();
    auto const held{ values_of( r ) };
    for ( std::uint64_t const key : { 0U, 1U, 2U, 0xFFFFU } ) {
        EXPECT_EQ( cardinality( r, key << 16, ( key + 1 ) << 16 ), count_in( held, key << 16, ( key + 1 ) << 16 ) ) << key;
    }
    constexpr std::uint64_t probes[]{
        0, 1, 36, 37, 65'535, 65'536, 65'537, 100'000, 131'071, 131'072, 150'000, 196'607, 196'608, 1 << 20, 0xFFFF'0000U, 0xFFFF'FFFFU, universe
    };
    for ( std::size_t i{ 0 }; i < std::size( probes ); ++i ) {
        for ( auto j{ i }; j < std::size( probes ); ++j ) {
            ASSERT_EQ( cardinality( r, probes[ i ], probes[ j ] ), count_in( held, probes[ i ], probes[ j ] ) ) << probes[ i ] << ' ' << probes[ j ];
        }
    }
}

// tests/toplevel_unit.c: test_run_optimize_recovers_space (Go TestRunOptimizeRecoversSpaceAfterRemoveRange)
TYPED_TEST( UpstreamCases, test_run_optimize_recovers_space ) {
    using Bitmap = TypeParam;
    auto r{ from_range<Bitmap>( 0, chunk, 1 ) };
    for ( std::uint32_t i{ 0 }; i < chunk; i += 2 ) {
        r.remove_closed_range( i, i );
    }
    auto const expected{ from_range<Bitmap>( 1, chunk, 2 ) };
    auto const before{ r.portable_size_in_bytes() };
    // the container may already have changed kind while values were removed; if it is still a degenerate run
    // container, optimizing must shrink it
    auto const was_run{ r.statistics().run_containers == 1 };
    r.optimize_keep_bitsets();
    auto const after{ r.portable_size_in_bytes() };
    EXPECT_LE( after, before );
    if ( was_run ) {
        EXPECT_LT( after, before / 10 );
    }
    EXPECT_EQ( values_of( r ), values_of( expected ) );
    expect_valid( r );
    EXPECT_EQ( r.statistics().run_containers, 0U );

    r = from_range<Bitmap>( 0, 100, 1 );
    r.optimize_keep_bitsets();
    EXPECT_EQ( r.statistics().run_containers, 1U );
    for ( std::uint32_t i{ 200 }; i < chunk; i += 2 ) {
        std::ignore = r.add( i );
    }
    auto const copy{ r };
    auto const size_before{ r.portable_size_in_bytes() };
    r.optimize_keep_bitsets();
    EXPECT_LE( r.portable_size_in_bytes(), size_before );
    EXPECT_EQ( values_of( r ), values_of( copy ) );
    expect_valid( r );
}

// tests/cpp_unit.cpp: test_cpp_run_optimize_recovers_space_64 (the return value of runOptimize has no counterpart:
// optimize() returns nothing)
TEST( UpstreamCasesWideKeys, test_cpp_run_optimize_recovers_space_64 ) {
    using Bitmap = bitmap<std::uint64_t, default_container_set<std::uint64_t>, detail::cow_value_semantics, detail::run_selection_lazy>;
    Bitmap r;
    r.add_closed_range( 0, 99 );
    r.optimize_keep_bitsets();
    for ( std::uint64_t i{ 200 }; i < chunk; i += 2 ) {
        std::ignore = r.add( i );   // each value becomes a run of length one
    }
    auto const expected{ r };
    auto const before{ r.byte_size() };
    auto const was_run{ r.statistics().run_containers == 1 };
    r.optimize_keep_bitsets();
    EXPECT_LE( r.byte_size(), before );
    if ( was_run ) {
        EXPECT_LT( r.byte_size(), before / 10 );
    }
    EXPECT_TRUE( r == expected );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_issue467 (Go TestIssue467CaseSmall / TestIssue467CaseLarge)
TYPED_TEST( UpstreamCases, test_issue467 ) {
    using Bitmap = TypeParam;
    constexpr std::uint32_t small_ranges[][ 2 ]{
        { 0, 16385 }, { 16385, 20482 }, { 20482, 27862 }, { 27862, 44247 }, { 45576, 61961 }, { 61961, 66058 },
        { 66058, 67247 }, { 68819, 73028 }, { 73028, 89413 }, { 92266, 108651 }, { 108651, 113772 },
        { 113772, 118757 }, { 118757, 132098 }
    };
    Bitmap r;
    std::uint64_t total{ 0 };
    for ( auto const & range : small_ranges ) {
        r.or_with( key_range::half_open( range[ 0 ], range[ 1 ] ) );
        total += range[ 1 ] - range[ 0 ];
    }
    r.optimize_keep_bitsets();
    expect_valid( r );
    EXPECT_EQ( r.size(), total );

    r = Bitmap{};
    std::set<std::uint32_t> expected;
    for ( auto const & op : issue467_large_ops ) {
        if ( op[ 0 ] != 0 ) {
            r.or_with( key_range::half_open( op[ 1 ], op[ 2 ] ) );
            for ( auto v{ op[ 1 ] }; v < op[ 2 ]; ++v ) {
                expected.insert( v );
            }
        } else {
            r.andnot_with( key_range::half_open( op[ 1 ], op[ 2 ] ) );
            for ( auto v{ op[ 1 ] }; v < op[ 2 ]; ++v ) {
                expected.erase( v );
            }
        }
    }
    r.optimize_keep_bitsets();
    expect_valid( r );
    EXPECT_EQ( values_of( r ), values( expected.begin(), expected.end() ) );
}

// tests/toplevel_unit.c: test_checked_add_remove_transitions (Java testCheckedAdd / testCheckedRemove): add and remove
// report whether they changed the bitmap, across the array/bitset transitions
TYPED_TEST( UpstreamCases, test_checked_add_remove_transitions ) {
    using Bitmap = TypeParam;
    Bitmap r;
    for ( std::uint32_t i{ 0 }; i < 131'072; i += 2 ) {
        ASSERT_TRUE( r.add( i ) ) << i;
    }
    for ( std::uint32_t i{ 1 }; i < 131'072; i += 2 ) {
        ASSERT_TRUE( r.add( i ) ) << i;
    }
    EXPECT_TRUE( r.add( 0xFFFF'FFF7U ) );
    EXPECT_TRUE( r.add( 0xFFFF'FFFFU ) );
    for ( std::uint32_t i{ 0 }; i < 131'072; ++i ) {
        ASSERT_FALSE( r.add( i ) ) << i;
    }
    EXPECT_FALSE( r.add( 0xFFFF'FFF7U ) );
    EXPECT_FALSE( r.add( 0xFFFF'FFFFU ) );
    expect_valid( r );
    for ( std::uint32_t i{ 0 }; i < 131'072; i += 2 ) {
        ASSERT_TRUE( r.remove( i ) ) << i;
        ASSERT_FALSE( r.remove( i ) ) << i;
    }
    expect_valid( r );
    for ( std::uint32_t i{ 1 }; i < 131'072; i += 2 ) {
        ASSERT_TRUE( r.remove( i ) ) << i;
    }
    EXPECT_TRUE( r.remove( 0xFFFF'FFF7U ) );
    EXPECT_TRUE( r.remove( 0xFFFF'FFFFU ) );
    EXPECT_FALSE( r.remove( 0xFFFF'FFFFU ) );
    EXPECT_TRUE( r.empty() );
    expect_valid( r );

    // bitset -> array while removing
    for ( std::uint32_t i{ 0 }; i < 20'000; i += 2 ) {
        std::ignore = r.add( i );
    }
    if constexpr ( reference_kinds<Bitmap> ) {
        ASSERT_EQ( r.statistics().bitset_containers, 1U );
    }
    for ( std::uint32_t i{ 0 }; i < 20'000; i += 2 ) {
        ASSERT_TRUE( r.remove( i ) ) << i;
        ASSERT_EQ( r.validate(), nullptr ) << i;
    }
    EXPECT_TRUE( r.empty() );
}

// tests/toplevel_unit.c: test_flip_karpenske (Java fliptest_Karpenske)
TYPED_TEST( UpstreamCases, test_flip_karpenske ) {
    using Bitmap = TypeParam;
    constexpr auto n{ std::size( karpenske_values ) };
    Bitmap r;
    std::size_t start{ 0 };
    for ( std::size_t end{ 1 }; end <= n; ++end ) {
        if ( end == n || karpenske_values[ end - 1 ] + 1 != karpenske_values[ end ] ) {
            if ( start == end - 1 ) {
                std::ignore = r.add( karpenske_values[ start ] );
            } else {
                r.flip_closed_range( karpenske_values[ start ], karpenske_values[ end - 1 ] );
            }
            start = end;
        }
    }
    EXPECT_EQ( r.size(), n );
    EXPECT_EQ( values_of( r ), values( std::begin( karpenske_values ), std::end( karpenske_values ) ) );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_remove_range_issue377 (Java regressionTestRemove377)
TYPED_TEST( UpstreamCases, test_remove_range_issue377 ) {
    using Bitmap = TypeParam;
    auto r{ from_range<Bitmap>( 0, 64, 1 ) };
    r.optimize_keep_bitsets();
    for ( std::uint32_t i{ 0 }; i < 64; ++i ) {
        if ( i != 30 && i != 32 ) {
            std::ignore = r.remove( i );
        }
    }
    r.remove_closed_range( 0, 30 );
    EXPECT_FALSE( r.contains( 30 ) );
    EXPECT_TRUE( r.contains( 32 ) );
    EXPECT_EQ( r.size(), 1U );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_equals_issue370 (Java regressionTestEquals370)
TYPED_TEST( UpstreamCases, test_equals_issue370 ) {
    using Bitmap = TypeParam;
    auto a{ of_values<Bitmap>( issue370_a ) };
    auto b{ of_values<Bitmap>( issue370_b ) };
    auto const differ{ [ & ] {
        EXPECT_FALSE( a == b );
        EXPECT_FALSE( b == a );
        EXPECT_FALSE( a.equals( b ) );
        EXPECT_FALSE( b.equals( a ) );
    } };
    differ();
    a.optimize_keep_bitsets();
    differ();
    b.optimize_keep_bitsets();
    differ();
}

// tests/toplevel_unit.c: test_remove_run_compression_top_of_universe (removing the run compression has no counterpart:
// the run container at the top of the key space is checked)
TYPED_TEST( UpstreamCases, test_remove_run_compression_top_of_universe ) {
    using Bitmap = TypeParam;
    Bitmap r;
    r.optimize_keep_bitsets();
    EXPECT_TRUE( r.empty() );
    r.or_with( key_range::half_open( 0xFFFF'FFF5U, universe ) );
    r.optimize_keep_bitsets();
    EXPECT_EQ( r.statistics().run_containers, 1U );
    EXPECT_EQ( r.size(), 11U );
    EXPECT_TRUE( r.contains( max32 ) );
    EXPECT_TRUE( r.is_superset_of( key_range::half_open( 0xFFFF'FFF5U, universe ) ) );
    EXPECT_EQ( r.back(), max32 );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_rank_high_values (Java testRankOverflow / testRankBigInts): the rank of a value (how many
// values are at most it), as the cardinality of [0, x] and from a ranked image
TEST( UpstreamCasesRanked, test_rank_high_values ) {
    using Bitmap = bitmap<std::uint32_t>;
    auto const check{ []( Bitmap const & r, std::uint32_t const x, std::uint64_t const expected ) {
        frozen_image const image{ r, { .ranked = true } };
        auto const view{ ranked_view<std::uint32_t>::from( image.bytes() ) };
        ASSERT_TRUE( static_cast<bool>( view ) );
        EXPECT_EQ( r.and_cardinality( key_range::closed( 0, x ) ), expected ) << x;
        EXPECT_EQ( view.count_below( x ) + ( view.contains( x ) ? 1U : 0U ), expected ) << x;
    } };
    auto r{ of<Bitmap>( { 65'537 } ) };
    check( r, 1, 0 );
    check( r, 65'537, 1 );
    check( r, 65'538, 1 );
    check( r, max32, 1 );

    r = Bitmap{};
    for ( std::uint32_t k{ 0 }; k < 100'000; k += 7 ) {
        std::ignore = r.add( 0x8000'0000U + k );
    }
    frozen_image const image{ r, { .ranked = true } };
    auto const view{ ranked_view<std::uint32_t>::from( image.bytes() ) };
    ASSERT_TRUE( static_cast<bool>( view ) );
    EXPECT_EQ( r.and_cardinality( key_range::closed( 0, 0x7FFF'FFFFU ) ), 0U );
    EXPECT_EQ( view.count_below( 0x8000'0000U ), 0U );
    for ( std::uint32_t k{ 0 }; k < 100'000; k += 7 ) {
        ASSERT_EQ( r.and_cardinality( key_range::closed( 0, 0x8000'0000U + k ) ), 1 + k / 7 ) << k;
        ASSERT_EQ( view.index_of( 0x8000'0000U + k ) + 1, 1 + k / 7 ) << k;
    }
    EXPECT_EQ( r.and_cardinality( key_range::closed( 0, max32 ) ), r.size() );
    EXPECT_EQ( view.count_below( max32 ), r.size() );
}

// tests/toplevel_unit.c: test_minimum_maximum_extremes (Java testFirstLastSigned; the minimum and maximum of an empty
// bitmap are not defined here)
TYPED_TEST( UpstreamCases, test_minimum_maximum_extremes ) {
    using Bitmap = TypeParam;
    auto r{ of<Bitmap>( { 777, 0xFFFF'FFF9U } ) };
    EXPECT_EQ( r.front(), 777U );
    EXPECT_EQ( r.back(), 0xFFFF'FFF9U );
    r = of<Bitmap>( { 0x7FFF'FFFFU, 0x8000'0000U } );
    EXPECT_EQ( r.front(), 0x7FFF'FFFFU );
    EXPECT_EQ( r.back(), 0x8000'0000U );
    r = of<Bitmap>( { 0x8000'0000U, max32 } );
    EXPECT_EQ( r.front(), 0x8000'0000U );
    EXPECT_EQ( r.back(), max32 );
}

// bitset_container_unit.c: min_max_test (CRoaring issue 924): the maximum of a bitset container whose values all sit
// in its first word
TYPED_TEST( UpstreamCases, bitset_container_maximum_in_the_first_word ) {
    using Bitmap = TypeParam;
    auto r{ of<Bitmap>( { 3, 5 } ) };
    r.promote_large_arrays( 1 );
    ASSERT_EQ( r.statistics().bitset_containers, 1U );
    EXPECT_EQ( r.front(), 3U );
    EXPECT_EQ( r.back(), 5U );
    EXPECT_EQ( values_of( r ), ( values{ 3, 5 } ) );
    frozen_image const image{ r, { .ranked = true } };
    auto const view{ ranked_view<std::uint32_t>::from( image.bytes() ) };
    ASSERT_TRUE( static_cast<bool>( view ) );
    EXPECT_EQ( view.front(), 3U );
    EXPECT_EQ( view.back(), 5U );
    EXPECT_EQ( Bitmap::frozen_view_from( image.bytes() ).materialize().back(), 5U );

    r = of<Bitmap>( { 0 } );
    r.promote_large_arrays( 1 );
    ASSERT_EQ( r.statistics().bitset_containers, 1U );
    EXPECT_EQ( r.front(), 0U );
    EXPECT_EQ( r.back(), 0U );
    for ( std::uint32_t const added : { 63U, 64U, 65'535U } ) {
        std::ignore = r.add( added );
        ASSERT_EQ( r.statistics().bitset_containers, 1U );
        EXPECT_EQ( r.back(), added );
    }
}

// tests/toplevel_unit.c: test_minimal_run_point_mutations (Go TestFuzzerRepro_*)
TYPED_TEST( UpstreamCases, test_minimal_run_point_mutations ) {
    using Bitmap = TypeParam;
    auto r{ from_range<Bitmap>( 2, 6, 1 ) };
    r.optimize_keep_bitsets();
    std::ignore = r.remove( 3 );
    expect_valid( r );
    EXPECT_EQ( values_of( r ), ( values{ 2, 4, 5 } ) );

    r = from_range<Bitmap>( 54, 58, 1 );
    r.optimize_keep_bitsets();
    std::ignore = r.add( 112 );
    expect_valid( r );
    EXPECT_EQ( values_of( r ), ( values{ 54, 55, 56, 57, 112 } ) );

    r = from_range<Bitmap>( 1'048'552, 1'048'556, 1 );
    r.optimize_keep_bitsets();
    ASSERT_EQ( r.statistics().run_containers, 1U );
    r.flip_closed_range( 1'048'555, 1'048'555 );
    expect_valid( r );
    EXPECT_EQ( values_of( r ), ( values{ 1'048'552, 1'048'553, 1'048'554 } ) );
}

// tests/toplevel_unit.c: test_flip_to_empty_is_valid
TYPED_TEST( UpstreamCases, test_flip_to_empty_is_valid ) {
    using Bitmap = TypeParam;
    std::vector<Bitmap> inputs;
    inputs.emplace_back( one_by_one<Bitmap>( 0, 100 ) );
    inputs.emplace_back( one_by_one<Bitmap>( 0, 10'000 ) );
    inputs.emplace_back( from_range<Bitmap>( 0, 10'000, 1 ) );
    constexpr std::uint32_t ends[]{ 100, 10'000, 10'000 };
    for ( std::size_t k{ 0 }; k < inputs.size(); ++k ) {
        auto f{ inputs[ k ] };
        f.xor_with( key_range::half_open( 0, ends[ k ] ) );
        EXPECT_TRUE( f.empty() ) << k;
        EXPECT_EQ( f.container_count(), 0U ) << k;
        expect_valid( f );
        inputs[ k ].flip_closed_range( 0, ends[ k ] - 1 );
        EXPECT_TRUE( inputs[ k ].empty() ) << k;
        EXPECT_EQ( inputs[ k ].container_count(), 0U ) << k;
        expect_valid( inputs[ k ] );
    }
    auto r{ from_range<Bitmap>( 0, 66'002, 1 ) };
    r.flip_closed_range( 0, 987'653'575 );
    EXPECT_EQ( r.size(), 987'653'576U - 66'002U );
    EXPECT_EQ( r.front(), 66'002U );
    expect_valid( r );
}

// tests/toplevel_unit.c: test_inplace_ops_do_not_share_mutated_containers (Java containerSharingWithXor, testAstesana)
TYPED_TEST( UpstreamCases, test_inplace_ops_do_not_share_mutated_containers ) {
    using Bitmap = TypeParam;
    for ( auto const use_xor : { true, false } ) {
        Bitmap r1;
        r1.flip_closed_range( 131'000, 131'000 );
        auto r2{ of<Bitmap>( { 220'000 } ) };
        auto const r3{ of<Bitmap>( { 66'000 } ) };
        if ( use_xor ) {
            r2 ^= r1;
        } else {
            r2 |= r1;
        }
        r2 |= r3;
        EXPECT_TRUE( r2.contains( 131'000 ) );
        EXPECT_TRUE( r2.contains( 66'000 ) );
        EXPECT_TRUE( r2.contains( 220'000 ) );
        EXPECT_FALSE( r1.contains( 66'000 ) );
        EXPECT_EQ( values_of( r1 ), ( values{ 131'000 } ) );
        // mutate the result: the inputs must not change
        std::ignore = r2.add( 131'001 );
        std::ignore = r2.remove( 66'000 );
        EXPECT_EQ( values_of( r1 ), ( values{ 131'000 } ) );
        EXPECT_EQ( values_of( r3 ), ( values{ 66'000 } ) );
        expect_valid( r1 );
        expect_valid( r2 );
    }
}

// tests/toplevel_unit.c: test_results_do_not_share_containers_with_inputs (Java xorDoesNotShareContainersWithItsInputs)
TYPED_TEST( UpstreamCases, test_results_do_not_share_containers_with_inputs ) {
    using Bitmap = TypeParam;
    for ( auto const from_buffer : { false, true } ) {
        auto const a{ from_range<Bitmap>( 1, 20, 2 ) };
        auto runs{ from_range<Bitmap>( 100, 105, 1 ) };
        runs.optimize_keep_bitsets();
        if ( from_buffer ) {
            std::vector<std::byte> buffer( runs.portable_size_in_bytes() );
            ASSERT_EQ( runs.portable_serialize( buffer ), buffer.size() );
            auto read{ Bitmap::portable_deserialize( buffer ) };
            ASSERT_TRUE( read.has_value() );
            runs = std::move( *read );
        }
        auto const expected_runs{ values_of( from_range<Bitmap>( 100, 105, 1 ) ) };
        auto const a_copy{ a };
        Bitmap const * const inputs[]{ &a, &a_copy, &runs };
        auto result{ Bitmap::xor_many( inputs ) };
        EXPECT_EQ( values_of( result ), expected_runs );
        result |= runs;
        std::ignore = result.add( 5000 );
        std::ignore = result.remove( 102 );
        EXPECT_EQ( result.size(), 5U );
        EXPECT_EQ( values_of( runs ), expected_runs );

        Bitmap const empty;
        std::vector<Bitmap> results;
        results.emplace_back( runs ^ empty );
        results.emplace_back( runs - empty );
        results.emplace_back( runs | empty );
        results.emplace_back( empty | runs );
        Bitmap const * const single[]{ &runs };
        results.emplace_back().or_many_in_place( single );
        results.emplace_back( Bitmap::xor_many( single ) );
        for ( auto & each : results ) {
            EXPECT_EQ( values_of( each ), expected_runs );
            std::ignore = each.remove( 102 );
            std::ignore = each.add( 106 );
            EXPECT_EQ( values_of( runs ), expected_runs );
            expect_valid( each );
        }
    }
}

// tests/toplevel_unit.c: test_xor_many_drops_cancelled_containers (Java horizontal_xorDropsBlocksThatCancelOut)
TYPED_TEST( UpstreamCases, test_xor_many_drops_cancelled_containers ) {
    using Bitmap = TypeParam;
    auto const a{ of<Bitmap>( { 1, 3, 5, 70'000 } ) };
    auto const b{ of<Bitmap>( { 1, 3, 5 } ) };
    Bitmap const * const ab[]{ &a, &b };
    auto x{ Bitmap::xor_many( ab ) };
    EXPECT_EQ( values_of( x ), ( values{ 70'000 } ) );
    EXPECT_EQ( x.container_count(), 1U );
    expect_valid( x );

    x = a ^ b;
    EXPECT_EQ( x.container_count(), 1U );

    auto const a_copy{ a };
    Bitmap const * const aa[]{ &a, &a_copy };
    x = Bitmap::xor_many( aa );
    EXPECT_TRUE( x.empty() );
    EXPECT_EQ( x.container_count(), 0U );
    expect_valid( x );

    // the lazy exclusive-or, then the repair
    x = a;
    x.bulk_xor_intermediate( b );
    x.bulk_xor_repair_after_lazy();
    EXPECT_EQ( values_of( x ), ( values{ 70'000 } ) );
    EXPECT_EQ( x.container_count(), 1U );
    expect_valid( x );

    x = a;
    x.bulk_xor_intermediate( a_copy );
    x.bulk_xor_repair_after_lazy();
    EXPECT_TRUE( x.empty() );
    EXPECT_EQ( x.container_count(), 0U );
    expect_valid( x );
}

// tests/toplevel_unit.c: test_lazy_or_inplace_then_repair (Java testLazyOr / testFirstLast_AfterLazyMutation1/2), with
// this library's lazy union: bulk_or_intermediate, finished by either finisher
TYPED_TEST( UpstreamCases, test_lazy_or_inplace_then_repair ) {
    using Bitmap = TypeParam;
    for ( auto const repair : { true, false } ) {
        auto const finish{ [ & ]( Bitmap & r ) {
            if ( repair ) {
                r.bulk_or_repair_after_lazy();
            } else {
                r.bulk_or_finish();
            }
        } };
        auto r{ of<Bitmap>( { 65'536, 262'144, 524'288 } ) };
        auto const r2{ of<Bitmap>( { 4, 7, 8, 9 } ) };
        auto const r3{ of<Bitmap>( { 1, 2, 3, 4, 5, 65'536, 131'072, 1'048'576 } ) };
        r.bulk_or_intermediate( r2 );
        r.bulk_or_intermediate( r3 );
        finish( r );
        EXPECT_EQ( values_of( r ), ( values{ 1, 2, 3, 4, 5, 7, 8, 9, 65'536, 131'072, 262'144, 524'288, 1'048'576 } ) );
        expect_valid( r );

        // dense: a bitset container
        r = from_range<Bitmap>( 3, 65'537, 3 );
        auto const dense_other{ of<Bitmap>( { 0, 2, 4, 6, 8 } ) };
        auto const eager{ r | dense_other };
        r.bulk_or_intermediate( dense_other );
        finish( r );
        EXPECT_EQ( r.front(), 0U );
        EXPECT_EQ( r.back(), 65'535U );
        EXPECT_EQ( r.size(), eager.size() );
        EXPECT_EQ( values_of( r ), values_of( eager ) );
        expect_valid( r );
    }
}

// tests/toplevel_unit.c: test_issue330_wide_or_with_runs (Go TestIssue330)
TYPED_TEST( UpstreamCases, test_issue330_wide_or_with_runs ) {
    using Bitmap = TypeParam;
    std::span<std::uint32_t const> const inputs[]{ issue330_values0, issue330_values1, issue330_values2 };
    std::vector<Bitmap> b;
    std::set<std::uint32_t> expected;
    for ( auto const input : inputs ) {
        auto & each{ b.emplace_back( of_values<Bitmap>( input ) ) };
        each.optimize_keep_bitsets();
        EXPECT_EQ( each.size(), input.size() );
        expect_valid( each );
        expected.insert( input.begin(), input.end() );
    }
    EXPECT_EQ( b[ 0 ].and_cardinality( b[ 1 ] ), 0U );
    EXPECT_EQ( b[ 0 ].and_cardinality( b[ 2 ] ), 0U );
    EXPECT_EQ( b[ 1 ].and_cardinality( b[ 2 ] ), 0U );
    Bitmap const * const order012[]{ &b[ 0 ], &b[ 1 ], &b[ 2 ] };
    Bitmap const * const order210[]{ &b[ 2 ], &b[ 1 ], &b[ 0 ] };
    std::vector<Bitmap> results;
    results.emplace_back().or_many_in_place( order012 );
    results.emplace_back().or_many_in_place( order210 );
    for ( auto const order : { std::span<Bitmap const * const>{ order012 }, std::span<Bitmap const * const>{ order210 } } ) {
        std::vector<Bitmap const *> operands( order.begin(), order.end() );
        results.emplace_back().or_many_heap_in_place( operands );
    }
    results.emplace_back( b[ 0 ] | b[ 1 ] ) |= b[ 2 ];
    results.emplace_back( b[ 2 ] | b[ 1 ] ) |= b[ 0 ];
    for ( auto const & result : results ) {
        EXPECT_EQ( result.size(), 1040U );
        EXPECT_EQ( values_of( result ), values( expected.begin(), expected.end() ) );
        expect_valid( result );
    }
}

// tests/toplevel_unit.c: test_issue178_sparse_or_many (Go aggregation_test.go "issue 178")
TYPED_TEST( UpstreamCases, test_issue178_sparse_or_many ) {
    using Bitmap = TypeParam;
    auto const b1{ of_values<Bitmap>( issue178_values0 ) };
    auto const b2{ of_values<Bitmap>( issue178_values1 ) };
    Bitmap const * const inputs[]{ &b1, &b2 };
    Bitmap u1;
    u1.or_many_in_place( inputs );
    Bitmap u2;
    std::vector<Bitmap const *> heap_inputs( std::begin( inputs ), std::end( inputs ) );
    u2.or_many_heap_in_place( heap_inputs );
    EXPECT_EQ( u1.size(), 32U );
    EXPECT_EQ( u2.size(), 32U );
    EXPECT_EQ( ( b1 | b2 ).size(), 32U );
    EXPECT_EQ( b1.or_cardinality( b2 ), 32U );
}

// tests/toplevel_unit.c: test_subset_run_container_regressions (Java issues 887, 721 and 723; the container kinds the
// reference forces are checked where this library builds them)
TYPED_TEST( UpstreamCases, test_subset_run_container_regressions ) {
    using Bitmap = TypeParam;
    // issue 887
    Bitmap big;
    for ( std::uint32_t i{ 0 }; i < 10'000; ++i ) {
        if ( i != 100 ) {
            std::ignore = big.add( i );
        }
    }
    auto sub{ from_range<Bitmap>( 90, 101, 1 ) };
    EXPECT_FALSE( sub.is_subset_of( big ) );
    std::ignore = big.add( 100 );
    EXPECT_TRUE( sub.is_subset_of( big ) );

    big = one_by_one<Bitmap>( 0, 65'535 );
    big.promote_large_arrays( 1 );
    sub = from_range<Bitmap>( 65'530, 65'536, 1 );
    EXPECT_FALSE( sub.is_subset_of( big ) );
    sub = from_range<Bitmap>( 65'530, 65'535, 1 );
    EXPECT_TRUE( sub.is_subset_of( big ) );
    std::ignore = big.add( 65'535 );
    sub = from_range<Bitmap>( 65'530, 65'536, 1 );
    EXPECT_TRUE( sub.is_subset_of( big ) );

    auto const as_bitset{ []( Bitmap r ) {
        r.promote_large_arrays( 1 );
        if constexpr ( reference_kinds<Bitmap> ) {
            EXPECT_EQ( r.statistics().bitset_containers, 1U );
        }
        return r;
    } };
    auto const as_run{ []( Bitmap r ) {
        r.optimize_keep_bitsets();
        EXPECT_EQ( r.statistics().run_containers, 1U );
        return r;
    } };
    // issue 721, case 1: run {[0,60),[63,64),[66,67)} contains bitset {63}
    auto run{ from_range<Bitmap>( 0, 60, 1 ) };
    std::ignore = run.add( 63 );
    std::ignore = run.add( 66 );
    run = as_run( run );
    auto bits{ as_bitset( of<Bitmap>( { 63 } ) ) };
    EXPECT_TRUE( bits.is_subset_of( run ) );
    // issue 721, case 2: run {[0,10),[12,13)} does not contain bitset {11}
    run = from_range<Bitmap>( 0, 10, 1 );
    std::ignore = run.add( 12 );
    run = as_run( run );
    bits = as_bitset( of<Bitmap>( { 11 } ) );
    EXPECT_FALSE( bits.is_subset_of( run ) );
    // run [128,512) contains bitset [256,320) (empty leading words)
    run = as_run( from_range<Bitmap>( 128, 512, 1 ) );
    bits = as_bitset( one_by_one<Bitmap>( 256, 320 ) );
    EXPECT_TRUE( bits.is_subset_of( run ) );
    std::ignore = bits.add( 100 );
    EXPECT_FALSE( bits.is_subset_of( run ) );
    // issue 723: array [0,10) contains {5} but not run [5,11)
    auto const array{ one_by_one<Bitmap>( 0, 10 ) };
    EXPECT_TRUE( of<Bitmap>( { 5 } ).is_subset_of( array ) );
    run = as_run( from_range<Bitmap>( 5, 11, 1 ) );
    EXPECT_FALSE( run.is_subset_of( array ) );
}

// tests/toplevel_unit.c: test_ops_emptying_or_splitting_containers (Go TestFuzzerRepro_* / TestFuzzerPanicRepro_*)
TYPED_TEST( UpstreamCases, test_ops_emptying_or_splitting_containers ) {
    using Bitmap = TypeParam;
    auto a{ of<Bitmap>( { 647'253, 1'048'572 } ) };
    auto b{ of<Bitmap>( { 647'253 } ) };
    a -= b;
    EXPECT_EQ( values_of( a ), ( values{ 1'048'572 } ) );
    EXPECT_EQ( a.container_count(), 1U );
    expect_valid( a );

    a = from_range<Bitmap>( 3, 7, 1 );
    a.optimize_keep_bitsets();
    a |= of<Bitmap>( { 1 } );
    EXPECT_EQ( values_of( a ), ( values{ 1, 3, 4, 5, 6 } ) );
    expect_valid( a );

    a = of<Bitmap>( { 0, 1'048'575 } );
    b = a;
    auto c{ a };
    a ^= b;
    EXPECT_TRUE( a.empty() );
    EXPECT_EQ( a.container_count(), 0U );
    expect_valid( a );
    c |= b;
    EXPECT_EQ( values_of( c ), values_of( b ) );
    expect_valid( c );

    a = from_range<Bitmap>( 1'048'565, 1'048'569, 1 );
    a.optimize_keep_bitsets();
    a -= of<Bitmap>( { 1'048'566, 1'048'567 } );
    EXPECT_EQ( values_of( a ), ( values{ 1'048'565, 1'048'568 } ) );
    expect_valid( a );
}

// tests/toplevel_unit.c: test_full_range_mask_with_mixed_data (Go TestRoaring_AndMask / TestRoaring_OrThenAnd)
TYPED_TEST( UpstreamCases, test_full_range_mask_with_mixed_data ) {
    using Bitmap = TypeParam;
    std::mt19937 rng{ 123'456'789 };
    Bitmap rb;
    for ( int i{ 0 }; i < 1'000'000; ++i ) {
        std::ignore = rb.add( 5'000'000 + rng() % 10'000'000 );
    }
    auto mask{ from_range<Bitmap>( 1, 10'000'001, 1 ) };
    auto const held{ values_of( rb ) };
    values expected( std::ranges::lower_bound( held, 1U ), std::ranges::lower_bound( held, 10'000'001U ) );
    EXPECT_EQ( values_of( rb & mask ), expected );
    mask &= rb;
    expect_valid( mask );
    EXPECT_EQ( values_of( mask ), expected );

    // many runs in one container at key 4272, then a union and an intersection with a range
    for ( auto const optimize : { false, true } ) {
        rb = Bitmap{};
        constexpr std::uint32_t base{ 4272U << 16 };
        for ( std::uint32_t i{ 0 }; i < 1135; ++i ) {
            auto const start{ base + i * 57 };
            rb.add_closed_range( start, start + i % 20 );
        }
        rb.optimize_keep_bitsets();
        rb |= of<Bitmap>( { 279'981'785, 279'982'923, 279'988'809, 279'995'913 } );
        if ( optimize ) {
            rb.optimize_keep_bitsets();
        }
        expect_valid( rb );
        mask = from_range<Bitmap>( 279'900'001, 280'000'001, 1 );
        auto const anded{ rb & mask };
        auto const ored{ rb | mask };
        expect_valid( anded );
        expect_valid( ored );
        auto const rb_values{ values_of( rb ) };
        EXPECT_EQ( anded.size(), count_in( rb_values, 279'900'001, 280'000'001 ) );
        EXPECT_EQ( ored.size(), rb.size() + 100'000 - anded.size() );
        mask &= rb;
        expect_valid( mask );
        EXPECT_EQ( values_of( mask ), values_of( anded ) );
        rb |= anded;
        expect_valid( rb );
        EXPECT_EQ( values_of( rb ), rb_values );
    }
}

// tests/toplevel_unit.c: test_self_aliasing_ops and tests/cpp_unit.cpp: test_cpp_self_aliasing (Java testWithYourself):
// x op x for every container kind, the in-place forms included
TYPED_TEST( UpstreamCases, test_self_aliasing_ops ) {
    using Bitmap = TypeParam;
    std::vector<Bitmap> inputs;
    inputs.emplace_back( one_by_one<Bitmap>( 1, 11 ) );   // array
    inputs.emplace_back( from_range<Bitmap>( 0, 20'000, 3 ) );   // bitset
    inputs.emplace_back( from_range<Bitmap>( 10, 5000, 1 ) );   // run
    inputs[ 2 ].optimize_keep_bitsets();
    inputs.emplace_back( from_range<Bitmap>( 10, 5000, 1 ) );   // tests/cpp_unit.cpp: a run and an array
    std::ignore = inputs[ 3 ].add( 1U << 20 );
    inputs[ 3 ].optimize_keep_bitsets();
    for ( auto & x : inputs ) {
        auto const orig{ values_of( x ) };
        auto const card{ x.size() };
        EXPECT_EQ( values_of( x & x ), orig );
        EXPECT_EQ( values_of( x | x ), orig );
        EXPECT_TRUE( ( x ^ x ).empty() );
        EXPECT_TRUE( ( x - x ).empty() );
        EXPECT_EQ( x.and_cardinality( x ), card );
        EXPECT_EQ( x.or_cardinality( x ), card );
        EXPECT_EQ( x.xor_cardinality( x ), 0U );
        EXPECT_EQ( x.andnot_cardinality( x ), 0U );
        EXPECT_TRUE( x.intersects( x ) );
        EXPECT_TRUE( x.is_subset_of( x ) );
        EXPECT_FALSE( x.is_strict_subset_of( x ) );
        EXPECT_EQ( x.jaccard_index( x ), 1.0 );
        x &= x;
        EXPECT_EQ( values_of( x ), orig );
        expect_valid( x );
        x |= x;
        EXPECT_EQ( values_of( x ), orig );
        expect_valid( x );
        auto y{ x };
        y ^= y;
        EXPECT_TRUE( y.empty() );
        expect_valid( y );
        y = x;
        y -= y;
        EXPECT_TRUE( y.empty() );
        expect_valid( y );
        EXPECT_EQ( values_of( x ), orig );
    }
    Bitmap const empty;
    EXPECT_FALSE( empty.intersects( empty ) );
    EXPECT_TRUE( empty.is_subset_of( empty ) );
}

// tests/toplevel_unit.c: test_andnot_inplace_bitset_to_array_then_serialize (Go TestRoaringInPlaceAndNotBitmapContainer)
TYPED_TEST( UpstreamCases, test_andnot_inplace_bitset_to_array_then_serialize ) {
    using Bitmap = TypeParam;
    Bitmap bm;
    for ( std::uint32_t i{ 0 }; i < 8192; ++i ) {
        std::ignore = bm.add( i );
    }
    if constexpr ( reference_kinds<Bitmap> ) {
        ASSERT_EQ( bm.statistics().bitset_containers, 1U );
    }
    auto const subtrahend{ from_range<Bitmap>( 128, 8192, 1 ) };
    auto const difference{ bm - subtrahend };
    bm -= subtrahend;
    EXPECT_EQ( bm.size(), 128U );
    EXPECT_EQ( values_of( bm ), values_of( from_range<Bitmap>( 0, 128, 1 ) ) );
    EXPECT_EQ( values_of( difference ), values_of( bm ) );
    expect_valid( bm );
    std::vector<std::byte> buffer( bm.portable_size_in_bytes() );
    ASSERT_EQ( bm.portable_serialize( buffer ), buffer.size() );
    auto const back{ Bitmap::portable_deserialize( buffer ) };
    ASSERT_TRUE( back.has_value() );
    EXPECT_EQ( values_of( *back ), values_of( bm ) );
    EXPECT_EQ( back->portable_size_in_bytes(), buffer.size() );
    EXPECT_EQ( back->statistics().array_containers, 1U );   // the portable format holds 128 values as an array
    expect_valid( *back );
}

// tests/toplevel_unit.c: test_andnot_inplace_bitset_to_array_then_serialize, the kind of the result: a bitset that a
// subtraction leaves with no more values than the array limit becomes an array, as in the reference
TYPED_TEST( UpstreamCases, subtraction_leaves_a_sparse_bitset_an_array ) {
    using Bitmap = TypeParam;
    auto bm{ one_by_one<Bitmap>( 0, 8192 ) };
    auto const subtrahend{ from_range<Bitmap>( 128, 8192, 1 ) };
    auto const difference{ bm - subtrahend };
    bm -= subtrahend;
    for ( auto const & result : { difference, bm } ) {
        EXPECT_EQ( values_of( result ), values_of( one_by_one<Bitmap>( 0, 128 ) ) );
        EXPECT_EQ( result.validate( true ), nullptr );
    }
    auto const dense{ one_by_one<Bitmap>( 0, 5000 ) };
    auto const sparse{ one_by_one<Bitmap>( 1000, 5000 ) };
    if constexpr ( reference_kinds<Bitmap> ) {
        ASSERT_EQ( dense.statistics().bitset_containers, 1U );
        ASSERT_EQ( sparse.statistics().array_containers, 1U );
    }
    auto in_place{ dense };
    in_place -= sparse;
    for ( auto const & result : { dense - sparse, in_place } ) {
        EXPECT_EQ( values_of( result ), values_of( one_by_one<Bitmap>( 0, 1000 ) ) );
        EXPECT_EQ( result.validate( true ), nullptr );
    }
}

// tests/toplevel_unit.c: test_or_inplace_copy_on_write_isolation (Go TestBitmapOrBulkMergeCopyOnWrite{,Tail}Ownership)
TYPED_TEST( UpstreamCases, test_or_inplace_copy_on_write_isolation ) {
    using Bitmap = TypeParam;
    Bitmap left, right;
    for ( std::uint32_t i{ 0 }; i < 64; ++i ) {
        std::ignore = left.add( ( 2 * i ) << 16 );
        std::ignore = right.add( ( ( 2 * i + 1 ) << 16 ) | 1 );
    }
    auto const left_orig{ values_of( left ) };
    auto const right_orig{ values_of( right ) };
    auto receiver{ left };
    receiver |= right;
    EXPECT_EQ( receiver.size(), 128U );

    constexpr std::uint32_t tail{ 127U << 16 };
    std::ignore = receiver.add( ( 1U << 16 ) | 10 );
    std::ignore = receiver.add( tail | 10 );
    std::ignore = receiver.add( 10 );
    EXPECT_EQ( values_of( right ), right_orig );
    EXPECT_EQ( values_of( left ), left_orig );
    auto const receiver_snapshot{ values_of( receiver ) };
    std::ignore = right.add( ( 1U << 16 ) | 11 );
    std::ignore = right.add( tail | 11 );
    std::ignore = left.add( 11 );
    EXPECT_EQ( values_of( receiver ), receiver_snapshot );
    EXPECT_FALSE( receiver.contains( ( 1U << 16 ) | 11 ) );
    EXPECT_FALSE( receiver.contains( 11 ) );
    expect_valid( receiver );
    expect_valid( left );
    expect_valid( right );
}

// tests/toplevel_unit.c: test_ops_with_empty_operands
TYPED_TEST( UpstreamCases, test_ops_with_empty_operands ) {
    using Bitmap = TypeParam;
    auto const r{ from_range<Bitmap>( 100, 70'000, 3 ) };
    Bitmap const e;
    auto const orig{ values_of( r ) };
    auto const card{ r.size() };
    for ( auto const & same : { r | e, e | r, r ^ e, e ^ r, r - e } ) {
        EXPECT_EQ( values_of( same ), orig );
    }
    for ( auto const & empty : { r & e, e & r, e - r } ) {
        EXPECT_TRUE( empty.empty() );
    }
    // in place, with an empty receiver
    Bitmap x;
    x |= r;
    EXPECT_EQ( values_of( x ), orig );
    std::ignore = x.add( 1 );
    std::ignore = x.remove( 100 );
    EXPECT_EQ( values_of( r ), orig );
    x = Bitmap{};
    x ^= r;
    EXPECT_EQ( values_of( x ), orig );
    std::ignore = x.add( 1 );
    std::ignore = x.remove( 100 );
    EXPECT_EQ( values_of( r ), orig );
    x = Bitmap{};
    x &= r;
    EXPECT_TRUE( x.empty() );
    x -= r;
    EXPECT_TRUE( x.empty() );
    // in place, with an empty argument
    x = r;
    x |= e;
    x ^= e;
    x -= e;
    EXPECT_EQ( values_of( x ), orig );
    x &= e;
    EXPECT_TRUE( x.empty() );
    // cardinalities
    EXPECT_EQ( r.and_cardinality( e ), 0U );
    EXPECT_EQ( e.and_cardinality( r ), 0U );
    EXPECT_EQ( r.or_cardinality( e ), card );
    EXPECT_EQ( e.or_cardinality( r ), card );
    EXPECT_EQ( r.xor_cardinality( e ), card );
    EXPECT_EQ( e.xor_cardinality( r ), card );
    EXPECT_EQ( r.andnot_cardinality( e ), card );
    EXPECT_EQ( e.andnot_cardinality( r ), 0U );
    EXPECT_FALSE( r.intersects( e ) );
    EXPECT_FALSE( e.intersects( r ) );
    EXPECT_TRUE( e.is_subset_of( r ) );
    EXPECT_FALSE( r.is_subset_of( e ) );
    EXPECT_TRUE( e.empty() );
    EXPECT_EQ( values_of( r ), orig );
}

// tests/toplevel_unit.c: test_wide_aggregations_zero_or_one_input and tests/cpp_unit.cpp:
// test_cpp_fastunion_zero_or_one_input (Go "aggregate nothing" / "single bitmap"): a fold over one input is a result
// of its own
TYPED_TEST( UpstreamCases, test_wide_aggregations_zero_or_one_input ) {
    using Bitmap = TypeParam;
    Bitmap union_of_none;
    union_of_none.or_many_in_place( {} );
    EXPECT_TRUE( union_of_none.empty() );
    std::vector<Bitmap const *> none;
    union_of_none.or_many_heap_in_place( none );
    EXPECT_TRUE( union_of_none.empty() );
    EXPECT_TRUE( Bitmap::xor_many( {} ).empty() );

    auto in{ from_range<Bitmap>( 0, 100'000, 3 ) };
    in.add_closed_range( 200'000, 299'999 );
    std::ignore = in.add( 1U << 30 );
    in.optimize_keep_bitsets();
    auto const orig{ values_of( in ) };
    Bitmap const * const inputs[]{ &in };
    std::vector<Bitmap> results;
    results.emplace_back().or_many_in_place( inputs );
    std::vector<Bitmap const *> heap_inputs{ &in };
    results.emplace_back().or_many_heap_in_place( heap_inputs );
    results.emplace_back( Bitmap::xor_many( inputs ) );
    for ( auto & result : results ) {
        EXPECT_EQ( values_of( result ), orig );
        std::ignore = result.remove( 3 );
        std::ignore = result.remove( 250'000 );
        std::ignore = result.add( 1 );
        std::ignore = result.add( 200'000 - 1 );
        EXPECT_EQ( values_of( in ), orig );
        expect_valid( result );
    }
}

// A ranked image of a bitmap, with the view that reads it.
struct ranked_table {
    template <typename Bitmap>
    explicit ranked_table( Bitmap const & source ) : image{ source, { .ranked = true } }, view{ ranked_view<std::uint32_t>::from( image.bytes() ) } {}

    // The values from the first one >= `from`, through the view, with their indexes checked.
    [[nodiscard]] values from( std::uint32_t const from ) const {
        values seen;
        auto expected_index{ view.count_below( from ) };
        view.for_each_indexed_from( from, [ & ]( std::uint32_t const id, std::uint32_t const index ) {
            EXPECT_EQ( index, expected_index );
            ++expected_index;
            seen.emplace_back( id );
        } );
        return seen;
    }

    frozen_image image;
    ranked_view<std::uint32_t> view;
};

// tests/toplevel_unit.c: test_iterator_move_to_uint32_max (Java advanceIfNeeded(-1)), a seek through the ranked image
TEST( UpstreamCasesRanked, test_iterator_move_to_uint32_max ) {
    using Bitmap = bitmap<std::uint32_t>;
    for ( auto const variant : { 0, 1 } ) {
        auto r{ of<Bitmap>( { 0, 65'535, 65'536 } ) };
        if ( variant == 0 ) {
            std::ignore = r.add( max32 );
        } else {
            r.or_with( key_range::half_open( 0xFFFF'FF00U, universe ) );
            r.optimize_keep_bitsets();
        }
        ranked_table const table{ r };
        ASSERT_TRUE( static_cast<bool>( table.view ) );
        EXPECT_EQ( table.from( max32 ), ( values{ max32 } ) );
        EXPECT_EQ( table.view.index_of( max32 ), r.size() - 1 );
        EXPECT_EQ( table.view.back(), max32 );
    }
    ranked_table const pair{ of<Bitmap>( { 1, 2 } ) };
    EXPECT_TRUE( pair.from( max32 ).empty() );
    ranked_table const empty{ Bitmap{} };
    EXPECT_TRUE( empty.from( max32 ).empty() );
    EXPECT_TRUE( empty.from( 0 ).empty() );
}

// test_iterator_generate_data of tests/toplevel_unit.c: every shape of container (the random one with a generator of
// this test's own)
[[nodiscard]] values iterator_test_data() {
    values out;
    out.emplace_back( 0 );   // the smallest value
    std::uint32_t base{ 1234 };   // container index
    out.emplace_back( base * chunk );   // only the very first value in the container is set
    base += 2;
    out.emplace_back( base * chunk + 65'535 );   // only the very last value in the container is set
    base += 2;
    for ( std::uint32_t i{ 0 }; i < chunk; ++i ) {   // a full container
        out.emplace_back( base * chunk + i );
    }
    base += 2;
    for ( std::uint32_t i{ 0 }; i < chunk; i += 2 ) {   // even values
        out.emplace_back( base * chunk + i );
    }
    base += 2;
    for ( std::uint32_t i{ 1 }; i < chunk; i += 2 ) {   // odd values
        out.emplace_back( base * chunk + i );
    }
    base += 2;
    for ( std::uint32_t i{ 0 }; i < chunk; i += 65 ) {   // each next 64-bit word rotated by one
        out.emplace_back( base * chunk + i );
    }
    base += 2;
    for ( std::uint32_t i{ 0 }, run_index{ 0 }; i < chunk; ++i ) {   // runs of increasing length: 0, 1,0, 1,1,0, ...
        if ( i != ( run_index + 1 ) * ( run_index + 2 ) / 2 - 1 ) {
            out.emplace_back( base * chunk + i );
        } else {
            ++run_index;
        }
    }
    base += 2;
    for ( std::uint32_t i{ chunk - 100 }; i < chunk; ++i ) {   // 00000XX, XXXXXX, XX0000
        out.emplace_back( base * chunk + i );
    }
    base += 1;
    for ( std::uint32_t i{ 0 }; i < chunk; ++i ) {
        out.emplace_back( base * chunk + i );
    }
    base += 1;
    for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
        out.emplace_back( base * chunk + i );
    }
    base += 2;
    std::mt19937 rng{ 123'456'789 };
    for ( std::uint32_t i{ 0 }; i < chunk; i += rng() % 10 + 1 ) {   // random
        out.emplace_back( base * chunk + i );
    }
    out.emplace_back( max32 );   // the largest value
    return out;
}

// tests/toplevel_unit.c: test_read_uint32_iterator_after_seek (Java RoaringBitmapBatchIteratorTest), the values from
// a seek through the ranked image (a batched read has no counterpart)
TEST( UpstreamCasesRanked, test_read_uint32_iterator_after_seek ) {
    using Bitmap = bitmap<std::uint32_t>;
    auto const data{ iterator_test_data() };
    auto r{ of_values<Bitmap>( data ) };
    auto const expect_from{ [ & ]( ranked_table const & table, std::uint32_t const target ) {
        values expected( std::ranges::lower_bound( data, target ), data.end() );
        EXPECT_EQ( table.from( target ), expected ) << target;
    } };
    for ( auto const optimize : { false, true } ) {
        if ( optimize ) {
            r.optimize_keep_bitsets();
        }
        ranked_table const table{ r };
        ASSERT_TRUE( static_cast<bool>( table.view ) );
        auto const mid{ table.view.select( static_cast<std::uint32_t>( r.size() / 2 ) ) };
        EXPECT_EQ( mid, data[ data.size() / 2 ] );
        std::uint32_t first_absent{ 0 };
        while ( r.contains( first_absent ) ) {
            ++first_absent;
        }
        expect_from( table, mid );
        expect_from( table, first_absent );
        expect_from( table, 1234U << 16 );
        expect_from( table, ( 1234U << 16 ) + 1 );
    }

    ranked_table const single{ of<Bitmap>( { 8511 } ) };
    EXPECT_TRUE( single.from( 8512 ).empty() );
    ranked_table const one_chunk{ of<Bitmap>( { 3 << 16, ( 3 << 16 ) + 5, ( 3 << 16 ) + 10 } ) };
    EXPECT_EQ( one_chunk.from( 6 ), ( values{ 3 << 16, ( 3 << 16 ) + 5, ( 3 << 16 ) + 10 } ) );

    r = from_range<Bitmap>( 10, 16, 1 );
    std::ignore = r.add( 18 );
    r.add_closed_range( 20, 24 );
    r.optimize_keep_bitsets();
    ranked_table const runs{ r };
    for ( std::uint32_t const probe : { 10U, 11U, 12U, 13U, 14U, 15U, 18U, 20U, 21U, 23U, 24U } ) {
        auto const seen{ runs.from( probe ) };
        ASSERT_FALSE( seen.empty() ) << probe;
        EXPECT_EQ( seen.front(), probe );
    }
}

// tests/toplevel_unit.c: test_read_uint32_iterator_empty_bitmap (Java TestEmptyRoaringBatchIterator)
TYPED_TEST( UpstreamCases, test_read_uint32_iterator_empty_bitmap ) {
    using Bitmap = TypeParam;
    Bitmap r;
    EXPECT_TRUE( r.begin() == r.end() );
    EXPECT_TRUE( values_of( r ).empty() );
    std::ignore = r.add( 1 );
    auto it{ r.begin() };
    ASSERT_FALSE( it == r.end() );
    EXPECT_EQ( *it, 1U );
    ++it;
    EXPECT_TRUE( it == r.end() );
}

// tests/toplevel_unit.c: test_uint32_iterator_copy_is_independent (Java TestIntIteratorFlyweight testClone; the
// iterator here goes forward only)
TYPED_TEST( UpstreamCases, test_uint32_iterator_copy_is_independent ) {
    using Bitmap = TypeParam;
    auto const r{ of<Bitmap>( { 10, 20, 30, 65'576, 65'586, 65'596 } ) };
    auto it{ r.begin() };
    auto cp{ it };
    for ( int i{ 0 }; i < 3; ++i ) {
        ++cp;
    }
    EXPECT_EQ( *cp, 65'576U );
    EXPECT_EQ( *it, 10U );
    ++it;
    EXPECT_EQ( *it, 20U );
    cp = it;
    values read;
    for ( int i{ 0 }; i < 3; ++i ) {
        read.emplace_back( *cp++ );
    }
    EXPECT_EQ( read, ( values{ 20, 30, 65'576 } ) );
    EXPECT_EQ( *it, 20U );
    EXPECT_EQ( values( it, r.end() ), ( values{ 20, 30, 65'576, 65'586, 65'596 } ) );
}

// tests/toplevel_unit.c: test_reverse_iteration_many_containers (Java ReverseIntIteratorFlyweightManyContainersTest;
// iteration goes forward only here)
TYPED_TEST( UpstreamCases, test_reverse_iteration_many_containers ) {
    using Bitmap = TypeParam;
    Bitmap r;
    for ( std::uint32_t i{ 0 }; i < 32'769; ++i ) {
        std::ignore = r.add( i << 16 );
    }
    EXPECT_EQ( r.back(), 32'768U << 16 );
    std::uint32_t count{ 0 };
    for ( auto const value : r ) {
        ASSERT_EQ( value, count << 16 );
        ++count;
    }
    EXPECT_EQ( count, 32'769U );
}

// tests/toplevel_unit.c: test_iterate_runs_ending_at_signed_and_unsigned_max (Java testCorruptionInfiniteLoop,
// issue 475)
TYPED_TEST( UpstreamCases, test_iterate_runs_ending_at_signed_and_unsigned_max ) {
    using Bitmap = TypeParam;
    for ( std::uint32_t const end : { 0x7FFF'FFFFU, 0xFFFF'FFFFU } ) {
        auto r{ from_range<Bitmap>( end - 3, std::uint64_t{ end } + 1, 1 ) };
        r.optimize_keep_bitsets();
        values seen;
        EXPECT_TRUE( r.iterate( [ & ]( std::uint32_t const value ) {
            seen.emplace_back( value );
            return true;
        } ) );
        EXPECT_EQ( seen, ( values{ end - 3, end - 2, end - 1, end } ) );
        values walked;
        r.for_each( [ & ]( std::uint32_t const value ) { walked.emplace_back( value ); } );
        EXPECT_EQ( walked, seen );
    }
}

// tests/toplevel_unit.c: test_iterate_halts_early and tests/cpp_unit.cpp: test_cpp_iterate_halts_early (Go
// TestIterateHalt)
TYPED_TEST( UpstreamCases, test_iterate_halts_early ) {
    using Bitmap = TypeParam;
    auto r{ from_range<Bitmap>( 150'000, 450'000, 1 ) };
    std::uint64_t const card{ r.size() };
    for ( auto const optimize : { false, true } ) {
        if ( optimize ) {
            r.optimize_keep_bitsets();
        }
        for ( auto const stop_after : { card - 1, max64, std::uint64_t{ 1 } } ) {
            std::uint64_t calls{ 0 };
            auto const completed{ r.iterate( [ & ]( std::uint32_t ) {
                ++calls;
                return calls < stop_after;
            } ) };
            EXPECT_EQ( completed, stop_after > card );
            EXPECT_EQ( calls, std::min( stop_after, card ) );
        }
    }
}

// The bitmap of the reference's copy-on-write edge cases: an array, a run and a bitset container.
template <typename Bitmap>
[[nodiscard]] Bitmap three_containers() {
    auto r{ of<Bitmap>( { 1, 2, 3 } ) };
    r.add_closed_range( 131'072, 132'071 );
    for ( std::uint32_t v{ 262'144 }; v < 272'144; v += 2 ) {
        std::ignore = r.add( v );
    }
    r.optimize_keep_bitsets();
    return r;
}

// tests/toplevel_unit.c: test_copy_on_write_writes_after_index_shift (Java TestCopyOnWriteEdgeCases)
TYPED_TEST( UpstreamCases, test_copy_on_write_writes_after_index_shift ) {
    using Bitmap = TypeParam;
    for ( auto const c : { 0, 1, 2 } ) {
        auto const source{ three_containers<Bitmap>() };
        auto copy{ source };
        auto plain{ three_containers<Bitmap>() };
        for ( auto * const x : { &copy, &plain } ) {
            if ( c == 0 ) {
                std::ignore = x->add( 65'536 );
                std::ignore = x->remove( 262'146 );
            } else if ( c == 1 ) {
                std::ignore = x->add( 65'536 );
                *x |= of<Bitmap>( { 262'145 } );
            } else {
                std::ignore = x->add( 5 );
                for ( std::uint32_t const removed : { 1U, 2U, 3U, 5U, 131'072U } ) {
                    std::ignore = x->remove( removed );
                }
            }
        }
        EXPECT_EQ( values_of( source ), values_of( three_containers<Bitmap>() ) ) << c;
        EXPECT_EQ( values_of( copy ), values_of( plain ) ) << c;
        expect_valid( source );
        expect_valid( copy );
    }
}

// tests/toplevel_unit.c: test_copy_on_write_clear_and_overwrite (Java TestCopyOnWriteEdgeCases)
TYPED_TEST( UpstreamCases, test_copy_on_write_clear_and_overwrite ) {
    using Bitmap = TypeParam;
    for ( auto const c : { 0, 1, 2 } ) {
        auto const source{ three_containers<Bitmap>() };
        auto copy{ source };
        auto const other{ of<Bitmap>( { 7, 70'000 } ) };
        if ( c == 0 ) {
            copy.clear();
        } else if ( c == 1 ) {
            copy = other;
        } else {
            auto const other_copy{ other };
            copy = other_copy;
        }
        copy.add_closed_range( 0, ( 1U << 17 ) - 1 );
        copy.add_closed_range( 262'144, 262'149 );
        std::ignore = copy.remove( 5 );
        std::ignore = copy.remove( 262'146 );
        EXPECT_EQ( values_of( source ), values_of( three_containers<Bitmap>() ) ) << c;
        expect_valid( source );
        expect_valid( copy );
        EXPECT_EQ( copy.size(), ( 1U << 17 ) - 1 + 5 ) << c;
    }
}

// tests/toplevel_unit.c: test_stats_per_container_type (Go TestStats, Java BitmapAnalyserTest): the container counts
// (the per-kind value and byte counts have no counterpart)
TYPED_TEST( UpstreamCases, test_stats_per_container_type ) {
    using Bitmap = TypeParam;
    auto statistics{ Bitmap{}.statistics() };
    EXPECT_EQ( statistics.container_count(), 0U );
    EXPECT_EQ( statistics.array_containers, 0U );
    EXPECT_EQ( statistics.run_containers, 0U );
    EXPECT_EQ( statistics.bitset_containers, 0U );

    Bitmap r;
    for ( std::uint32_t i{ 0 }; i < 60'000; ++i ) {
        std::ignore = r.add( i );
    }
    statistics = r.statistics();
    EXPECT_EQ( statistics.container_count(), 1U );
    if constexpr ( reference_kinds<Bitmap> ) {
        EXPECT_EQ( statistics.bitset_containers, 1U );
    }
    EXPECT_EQ( r.size(), 60'000U );
    EXPECT_EQ( r.front(), 0U );
    EXPECT_EQ( r.back(), 59'999U );

    statistics = of<Bitmap>( { 2, 4 } ).statistics();
    EXPECT_EQ( statistics.container_count(), 1U );
    EXPECT_EQ( statistics.array_containers, 1U );

    r = from_range<Bitmap>( 0, 1000, 1 );
    r.optimize_keep_bitsets();
    statistics = r.statistics();
    EXPECT_EQ( statistics.container_count(), 1U );
    EXPECT_EQ( statistics.run_containers, 1U );

    r = of<Bitmap>( { 1, 3, 6, 26, 110, 1024 } );
    r.add_closed_range( 70'000, 79'999 );
    for ( std::uint32_t i{ 0 }; i < chunk; ++i ) {
        std::ignore = r.add( ( 5 << 16 ) + i );
    }
    statistics = r.statistics();
    EXPECT_EQ( statistics.container_count(), 3U );
    EXPECT_EQ( statistics.array_containers, 1U );
    // the reference holds the added range as a run; the lazy policy here picks no run outside optimize()
    EXPECT_EQ( statistics.run_containers + statistics.bitset_containers, 2U );
    EXPECT_EQ( r.size(), 75'542U );
}

// tests/cpp_unit.cpp: test_cpp_end_iterator_is_per_bitmap (CRoaring issue 945): end() is returned by value, and end
// iterators taken from distinct bitmaps, or twice from one, stay independent (decrementing one has no counterpart:
// the iterator goes forward only)
static_assert( std::same_as<decltype( std::declval<bitmap<std::uint32_t> const &>().end() ), bitmap<std::uint32_t>::const_iterator> );

TYPED_TEST( UpstreamCases, test_cpp_end_iterator_is_per_bitmap ) {
    using Bitmap = TypeParam;
    static_assert( std::same_as<decltype( std::declval<Bitmap const &>().end() ), typename Bitmap::const_iterator> );
    auto const a{ of<Bitmap>( { 1, 2 } ) };
    auto const b{ of<Bitmap>( { 100, 200 } ) };
    std::ignore = a.end();
    std::uint32_t n{ 0 };
    for ( auto i{ b.begin() }; i != b.end(); ++i ) {
        ++n;
    }
    EXPECT_EQ( n, 2U );
    EXPECT_EQ( values( a.begin(), a.end() ), ( values{ 1, 2 } ) );
    EXPECT_EQ( values( b.begin(), b.end() ), ( values{ 100, 200 } ) );
    auto const wide{ of<Bitmap>( { 1, 70'000, max32 } ) };
    std::ignore = a.end();
    EXPECT_EQ( values( wide.begin(), wide.end() ), ( values{ 1, 70'000, max32 } ) );
    {
        auto const gone{ of<Bitmap>( { 7 } ) };
        std::ignore = gone.end();
    }
    EXPECT_EQ( values( b.begin(), b.end() ), ( values{ 100, 200 } ) );
    Bitmap const empty;
    std::ignore = b.end();
    EXPECT_TRUE( empty.begin() == empty.end() );
}

} // namespace
