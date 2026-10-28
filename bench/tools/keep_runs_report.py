"""Tabulates the bands that keep runs under subtraction: the minimum time over all the samples given and
the serialized size of the final bitmap, for the shipped arm and CRoaring with and without run_optimize.
Usage: keep_runs_report.py <bench-output> [<bench-output> ...] [--markdown]

Every bench-output file holds result lines ("<name>\t<n> us/op ...") and, for the end-to-end bands, "#bytes"
lines. A band takes the minimum time over every file given. Bands with a "-opt" CRoaring arm (the end-to-end
ones) show the ratio against it too; the single-chunk bands have only the plain CRoaring arm."""
import re
import sys

frsr = 'frsr-rc-lazy'
plain = 'cpp-cow'
optimized = 'cpp-cow-opt'
families = ( 'KeepRuns', 'RunAndNot', 'AndAfterRunAndNot', 'AndInplaceAfterRunAndNot' )

markdown = '--markdown' in sys.argv
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
