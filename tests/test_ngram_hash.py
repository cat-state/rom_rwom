from rom_rwom import NgramHashConfig, NgramHasher


def test_hash_batch_shape_and_determinism():
    config = NgramHashConfig(
        vocab_size=128,
        table_size=(31, 43),
        layer_ids=(2, 15),
        min_ngram=2,
        max_ngram=3,
        heads_per_ngram=4,
        pad_id=0,
        seed=7,
    )

    hasher_a = NgramHasher(config)
    hasher_b = NgramHasher(config)
    input_ids = [[10, 11, 12], [20, 21, 22]]

    hashes = hasher_a.hash_batch(input_ids, layer_id=2)

    assert hashes == hasher_b.hash_batch(input_ids, layer_id=2)
    assert len(hashes) == 2
    assert len(hashes[0]) == 3
    assert len(hashes[0][0]) == hasher_a.heads_per_token == 8


def test_layers_use_distinct_addresses():
    config = NgramHashConfig(
        vocab_size=128,
        table_size=31,
        layer_ids=(2, 15),
        min_ngram=2,
        max_ngram=2,
        heads_per_ngram=2,
        seed=7,
    )
    hasher = NgramHasher(config)

    layer_2 = hasher.hash_batch([[10, 11, 12]], layer_id=2)
    layer_15 = hasher.hash_batch([[10, 11, 12]], layer_id=15)

    assert layer_2 != layer_15


def test_rejects_non_rectangular_batches():
    hasher = NgramHasher(NgramHashConfig(vocab_size=128, table_size=31))

    try:
        hasher.hash_batch([[1, 2], [3]], layer_id=2)
    except ValueError as exc:
        assert "rectangular" in str(exc)
    else:
        raise AssertionError("expected non-rectangular batch to fail")
