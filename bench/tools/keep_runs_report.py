"""Tabulates the bands that keep runs under subtraction: the minimum time over all the samples given and
the serialized size of the final bitmap, for the shipped arm and CRoaring with and without run_optimize.
Usage: keep_runs_report.py <bench-output> [<bench-output> ...] [--markdown] [--check]

Every bench-output file holds result lines ("<name>\t<n> us/op ...") and, for the end-to-end bands, "#bytes"
lines. A band takes the minimum time over every file given. Bands with a "-opt" CRoaring arm (the end-to-end
ones) show the ratio against it too; the single-chunk bands have only the plain CRoaring arm.

--check is the regression gate: a band is expected at or below 1.05x each CRoaring arm (the margin is the
noise of a minimum over a few repeats) unless it is listed in known_above, which holds a ceiling for each band that
is slower than that today; the exit status is 1 when any band is over its ceiling. The ceilings are the worst of
four builds (clang and g++, glibc and mimalloc) plus ten percent, so a band that improves is not reported and
a band that gets worse than its ceiling is. Lower a ceiling when a band is made faster."""
import re
import sys

frsr = 'frsr-rc-lazy'
plain = 'cpp-cow'
optimized = 'cpp-cow-opt'
families = ( 'KeepRuns', 'RunAndNot', 'AndAfterRunAndNot', 'AndInplaceAfterRunAndNot' )

# Bands expected above 1.05x the plain CRoaring arm, and the ceiling of each. Every KeepRunsRemoveMany band is under
# 1.0x and has no entry. The end-to-end rounds that keep runs are slower than plain CRoaring where its deletion leaves
# the chunk a bitset that later rounds clear a word at a time; against the arm that runs run_optimize after each
# deletion (the fair baseline) only the four-filter intersection is above 1.05x.
known_above = {
    'AndAfterRunAndNot/run_bitset/residue=dense': 1.39,
    'AndAfterRunAndNot/run_bitset/residue=nine_values': 1.22,
    'AndInplaceAfterRunAndNot/run_bitset/residue=dense': 1.31,
    # Noise, not a regression: the whole band is 55 to 60 ns, so a nanosecond or two moves the ratio by several percent.
    # Measured against the build before the batch removal and the array difference (same session, minimum of 9,
    # SCHED_FIFO, plain and 64-byte-aligned builds): the shipped arm is 0.97x to 1.00x of it, and the unchanged
    # CRoaring arm moves by the same amount (0.95x to 1.00x); the ratio between the two arms is about 1.04.
    'AndAfterRunAndNot/run_run/one_cut': 1.25,
    'AndInplaceAfterRunAndNot/run_bitset/residue=nine_values': 1.18,
    'AndInplaceAfterRunAndNot/run_bitset/residue=three_stretches': 1.26,
    'AndInplaceAfterRunAndNot/run_run/one_cut': 1.25,
    'KeepRunsRemove/dense_chunks': 1.31,
    'KeepRunsRemove/mixed': 1.98,
    'KeepRunsRemove/ranges': 1.75,
    'KeepRunsRemove/singles': 3.93,
    'KeepRunsRemoveAnd/mixed': 1.36,
    'KeepRunsRemoveAnd/ranges': 1.34,
    'KeepRunsRemoveAnd/singles': 2.89,
    'KeepRunsRemoveAndMany/ranges': 1.19,
    'KeepRunsRemoveAndMany/singles': 1.83,
    'RunAndNot/run_bitset/residue=dense': 1.28,
    'RunAndNot/run_bitset/residue=one_stretch': 1.17,
    'RunAndNot/run_run/one_cut': 1.19,
    'RunAndNotInplace/run_bitset/residue=dense': 1.3,
    # A 0.25 to 0.28 us band: the same kernel as the not-in-place band above (ceiling 1.17), measured at 0.9x to 1.09x.
    'RunAndNotInplace/run_bitset/residue=one_stretch': 1.2,
    'RunAndNotInplace/run_run/one_cut': 1.25,
}
# The same, against the CRoaring arm that runs run_optimize after each deletion.
known_above_optimized = {
}
margin = 1.05

markdown = '--markdown' in sys.argv
check = '--check' in sys.argv
paths = [ a for a in sys.argv[ 1 : ] if not a.startswith( '--' ) ]

times = {}
sizes = {}
for path in paths:
    for line in open( path, encoding = 'utf-8', errors = 'replace' ):
        m = re.match( r'^(\S+)\t([0-9.]+) us/op', line )
        if m:
            times[ m.group( 1 ) ] = min( times.get( m.group( 1 ), 1e300 ), float( m.group( 2 ) ) )
            continue
        m = re.match( r'^#bytes\t(\S+)\t(\d+)\t', line )
        if m:
            sizes[ m.group( 1 ) ] = int( m.group( 2 ) )

rows = []
for name, t in times.items():
    parts = name.split( '/', 2 )
    if len( parts ) < 3 or not parts[ 1 ].startswith( frsr ):
        continue
    band = parts[ 1 ][ len( frsr ) : ]
    if not any( band.startswith( f ) for f in families ):
        continue
    ref = lambda arm: f'{parts[ 0 ]}/{arm}{band}/{parts[ 2 ]}'
    p = times.get( ref( plain ) )
    o = times.get( ref( optimized ) )
    if p is None:
        continue
    rows.append( ( t / p, band + '/' + parts[ 2 ], t, p, o, sizes.get( name ), sizes.get( ref( plain ) ), sizes.get( ref( optimized ) ) ) )

rows.sort( key = lambda r: r[ 1 ] )
def fmt_t( x ):
    return '-' if x is None else f'{x:.2f}'
def fmt_b( x ):
    return '-' if x is None else str( x )

head = ( 'band', 'frsr us', 'CRoaring us', 'CRoaring+opt us', 'frsr/CRoaring', 'frsr/CRoaring+opt', 'frsr B', 'CRoaring B', 'CRoaring+opt B' )
if markdown:
    print( '| ' + ' | '.join( head ) + ' |' )
    print( '|' + '---|' * len( head ) )
else:
    print( '\t'.join( head ) )
for ratio, band, t, p, o, bf, bp, bo in rows:
    cells = ( band, fmt_t( t ), fmt_t( p ), fmt_t( o ), f'{ratio:.2f}', '-' if o is None else f'{t / o:.2f}', fmt_b( bf ), fmt_b( bp ), fmt_b( bo ) )
    print( ( '| ' + ' | '.join( cells ) + ' |' ) if markdown else '\t'.join( cells ) )
above = [ r for r in rows if r[ 0 ] > 1.0 ]
print( f'\n{len( rows )} bands, {len( above )} above 1.0x the plain CRoaring arm', file = sys.stderr )
if check:
    failed = 0
    for ratio, band, t, p, o, bf, bp, bo in rows:
        if ratio > known_above.get( band, margin ):
            print( f'over its ceiling against CRoaring: {band}  {ratio:.2f}x > {known_above.get( band, margin ):.2f}x', file = sys.stderr )
            failed += 1
        if o is not None and t / o > known_above_optimized.get( band, margin ):
            print( f'over its ceiling against CRoaring + run_optimize: {band}  {t / o:.2f}x > {known_above_optimized.get( band, margin ):.2f}x', file = sys.stderr )
            failed += 1
        if bf is not None and bo is not None and bf > bo:
            print( f'larger than CRoaring + run_optimize: {band}  {bf} > {bo} bytes', file = sys.stderr )
            failed += 1
    print( f'{failed} band checks failed', file = sys.stderr )
    sys.exit( 1 if failed else 0 )
