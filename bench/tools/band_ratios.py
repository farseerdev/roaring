"""Pairs every band that exists for both the shipped arm and a reference arm and
prints the ratio distribution. Usage: band_ratios.py <bench-output> [frsr-arm] [reference-arm ...]

With no reference given, each band takes the first arm that exists for it, in order:
cpp-cow, r64, cpp. The 64-bit bands carry no cpp-cow arm at all - CRoaring's 64-bit
API has no copy-on-write - so a single fixed reference silently drops them, which is
what hid the whole synthetic family from this report. r64 (CRoaring's 64-bit C API)
precedes cpp because the bench's own Roaring64Map shim serializes through a copy, so
it flatters the shipped arm on the frozen bands."""
import sys, re, statistics

path = sys.argv[ 1 ]
frsr = sys.argv[ 2 ] if len( sys.argv ) > 2 else 'frsr-rc-lazy'
refs = sys.argv[ 3 : ] if len( sys.argv ) > 3 else [ 'cpp-cow', 'r64', 'cpp' ]

samples = {}
for line in open( path, encoding = 'utf-8', errors = 'replace' ):
    m = re.match( r'^(\S+)\t([0-9.]+) us/op', line )
    if not m:
        continue
    samples.setdefault( m.group( 1 ), [] ).append( float( m.group( 2 ) ) )
# A run that repeats a band (several repeats, several lanes) lists it once per
# sample; the band's time is their median, not whichever sample came last.
vals = { k: statistics.median( v ) for k, v in samples.items() }
reps = sorted( { len( v ) for v in samples.values() } )

pairs   = []
used    = {}
unpaired = []
for name, v in vals.items():
    # family/<arm><Band>/... - split the arm out of the second path element
    parts = name.split( '/' )
    if len( parts ) < 2:
        continue
    head = parts[ 1 ]
    if not head.startswith( frsr ):
        continue
    band = head[ len( frsr ) : ]
    for ref in refs:
        other = '/'.join( [ parts[ 0 ], ref + band ] + parts[ 2 : ] )
        if other in vals and vals[ other ] > 0:
            pairs.append( ( v / vals[ other ], name ) )
            used[ ref ] = used.get( ref, 0 ) + 1
            break
    else:
        unpaired.append( name )

if not pairs:
    print( 'no pairs' )
    sys.exit( 0 )

pairs.sort()
ratios = [ r for r, _ in pairs ]
print( f"pairs={len(pairs)}  samples/band=" +
       ( str( reps[ 0 ] ) if len( reps ) == 1 else f"{reps[0]}..{reps[-1]}" ) +
       f"  median={statistics.median(ratios):.3f}  "
       f"faster(<0.98)={sum(1 for r in ratios if r < 0.98)}  "
       f"flat={sum(1 for r in ratios if 0.98 <= r <= 1.02)}  "
       f"slower(>1.02)={sum(1 for r in ratios if r > 1.02)}" )
print( "reference arm used: " + ', '.join( f"{k} x{v}" for k, v in used.items() ) +
       ( f"  |  unpaired {frsr} bands: {len(unpaired)}" if unpaired else '' ) )
print( "\nworst 12:" )
for r, n in pairs[ -12 : ][ ::-1 ]:
    print( f"  {r:6.2f}x  {n}" )
print( "\nbest 8:" )
for r, n in pairs[ : 8 ]:
    print( f"  {r:6.3f}x  {n}" )
if unpaired:
    print( f"\nno reference arm ({', '.join(refs)}) for {len(unpaired)} bands:" )
    for n in sorted( unpaired )[ : 24 ]:
        print( '  ' + n )
