# test data

Copied from CRoaring's `tests/testdata` (https://github.com/RoaringBitmap/CRoaring, Apache-2.0 OR MIT), which shares
them with the Java (RoaringBitmap) and Go (roaring) libraries:

- `bitmapwithruns.bin`, `bitmapwithoutruns.bin`: the same 32-bit bitmap in the portable format, run-optimized and
  not (from Java's SerializeToDiskExample).
- `64maprunbitsetvals.bin`: a 64-bit portable serialization holding run, bitset and array containers (from Java).
- `crashproneinput*.bin`, `crashwithinvalidcookie.bin`, `crashwithoutcookie.bin`: malformed portable inputs that must
  be rejected (from Go).
- `bad-bitmap.bin`: an array container whose values are not sorted; well-formed in length but invalid (from Java).
- `testfrozendata/*.portable`: portable serializations made with gocroaring; `*.stats` describes each (from Go).
