from crystal_field.crystallography.io import _canonical_hash_key, _u01_for_key


def test_friedel_pair_hashes_together():
    assert _canonical_hash_key((1, 2, -3), True) == _canonical_hash_key((-1, -2, 3), True)


def test_hash_deterministic_and_seeded():
    key = _canonical_hash_key((3, 1, 4), True)
    assert _u01_for_key(key, 10) == _u01_for_key(key, 10)
    assert _u01_for_key(key, 10) != _u01_for_key(key, 11)
