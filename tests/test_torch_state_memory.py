import torch

from rom_rwom import RomGatedDeltaMemory, RomStateMemory


def test_state_memory_read_only_does_not_return_next_state():
    memory = RomStateMemory(num_rows=7, num_heads=2, key_dim=3, value_dim=4)
    addresses = torch.tensor([[[1, 2], [3, 4]]])
    q = torch.randn(1, 2, 2, 3)
    k = torch.randn(1, 2, 2, 3)
    v = torch.randn(1, 2, 2, 4)
    beta = torch.ones(1, 2, 2)
    decay = torch.zeros(1, 2, 2)

    output, next_state = memory(addresses, q, k, v, beta, decay, write=False)

    assert output.shape == (1, 2, 2, 4)
    assert next_state is None
    assert torch.count_nonzero(memory.state) == 0


def test_state_memory_write_returns_updated_table():
    memory = RomStateMemory(num_rows=7, num_heads=2, key_dim=3, value_dim=4)
    addresses = torch.tensor([[[1, 2], [3, 4]]])
    q = torch.randn(1, 2, 2, 3)
    k = torch.randn(1, 2, 2, 3)
    v = torch.randn(1, 2, 2, 4)
    beta = torch.ones(1, 2, 2)
    decay = torch.zeros(1, 2, 2)

    _, next_state = memory(addresses, q, k, v, beta, decay, write=True)

    assert next_state is not None
    assert next_state.shape == memory.state.shape
    assert torch.count_nonzero(next_state) > 0
    assert torch.count_nonzero(memory.state) == 0


def test_state_memory_write_affects_later_same_address_read():
    memory = RomStateMemory(num_rows=3, num_heads=1, key_dim=2, value_dim=2)
    addresses = torch.tensor([[[1], [1]]])
    q = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]])
    beta = torch.tensor([[[1.0], [0.0]]])
    decay = torch.zeros(1, 2, 1)

    read_only, _ = memory(addresses, q, k, v, beta, decay, write=False)
    written, _ = memory(addresses, q, k, v, beta, decay, write=True)

    assert read_only[0, 1, 0, 0] == 0.0
    assert written[0, 0, 0, 0] == 2.0
    assert written[0, 1, 0, 0] == 2.0


def test_gated_delta_memory_starts_as_residual_noop():
    torch.manual_seed(0)
    layer = RomGatedDeltaMemory(
        hidden_size=8,
        num_rows=11,
        num_heads=2,
        key_dim=3,
        value_dim=4,
    )
    hidden = torch.randn(2, 3, 8)
    addresses = torch.randint(0, 11, (2, 3, 2))

    output, info = layer(hidden, addresses, write=False)

    assert torch.allclose(output, hidden)
    assert torch.count_nonzero(info.residual) == 0
    assert info.read.shape == (2, 3, 2, 4)
    assert info.next_state is None


def test_gated_delta_memory_write_gate_can_disable_updates():
    torch.manual_seed(0)
    layer = RomGatedDeltaMemory(
        hidden_size=8,
        num_rows=11,
        num_heads=2,
        key_dim=3,
        value_dim=4,
    )
    torch.nn.init.constant_(layer.write_gate_proj.weight, 0.0)
    torch.nn.init.constant_(layer.write_gate_proj.bias, -100.0)
    hidden = torch.randn(2, 3, 8)
    addresses = torch.randint(0, 11, (2, 3))

    _, info = layer(hidden, addresses, write=True)

    assert info.next_state is not None
    assert torch.allclose(info.next_state, layer.memory.state)
