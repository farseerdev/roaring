"""Pairs every band that exists for both the shipped arm and a CRoaring arm and
prints the ratio distribution. Usage: band_ratios.py <bench-output> [frsr-arm] [cpp-arm]"""
import sys, re, statistics

path = sys.argv[ 1 ]
frsr = sys.argv[ 2 ] if len( sys.argv ) > 2 else 'frsr-rc-lazy'
cpp  = sys.argv[ 3 ] if len( sys.argv ) > 3 else 'cpp-cow'

vals = {}
for line in open( path, encoding = 'utf-8', errors = 'replace' ):
    m = re.match( r'^(\S+)\t([0-9.]+) us/op', line )
    if not m:
        continue
    vals[ m.group( 1 ) ] = float( m.group( 2 ) )

pairs = []
for name, v in vals.items():
    # family/<arm><Band>/... — split the arm out of the second path element
    parts = name.split( '/' )
    if len( parts ) < 2:
        continue
    head = parts[ 1 ]
    if not head.startswith( frsr ):
        continue
    band = head[ len( frsr ) : ]
    other = '/'.join( [ parts[ 0 ], cpp + band ] + parts[ 2 : ] )
    if other in vals and vals[ other ] > 0:
        pairs.append( ( v / vals[ other ], name ) )

if not pairs:
    print( 'no pairs' )
    sys.exit( 0 )

pairs.sort()
ratios = [ r for r, _ in pairs ]
print( f"pairs={len(pairs)}  median={statistics.median(ratios):.3f}  "
       f"faster(<0.98)={sum(1 for r in ratios if r < 0.98)}  "
       f"flat={sum(1 for r in ratios if 0.98 <= r <= 1.02)}  "
       f"slower(>1.02)={sum(1 for r in ratios if r > 1.02)}" )
print( "\nworst 12:" )
for r, n in pairs[ -12 : ][ ::-1 ]:
    print( f"  {r:6.2f}x  {n}" )
print( "\nbest 8:" )
for r, n in pairs[ : 8 ]:
    print( f"  {r:6.3f}x  {n}" )
