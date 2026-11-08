import torch

from src.models import (
    MyGRUCell, MyLSTMCell, MyRNNCell, recurrent_parameter_counts,
    scaled_dot_product_attention,
)


def copy_cell_weights(custom, reference):
    custom.load_state_dict({
        "weight_ih": reference.weight_ih.detach().clone(),
        "weight_hh": reference.weight_hh.detach().clone(),
        "bias_ih": reference.bias_ih.detach().clone(),
        "bias_hh": reference.bias_hh.detach().clone(),
    })


def test_custom_cells_match_library():
    torch.manual_seed(7)
    batch, d, h = 3, 9, 64
    x, hidden = torch.randn(batch, d), torch.randn(batch, h)
    for custom, library in [(MyRNNCell(d, h), torch.nn.RNNCell(d, h)),
                            (MyLSTMCell(d, h), torch.nn.LSTMCell(d, h)),
                            (MyGRUCell(d, h), torch.nn.GRUCell(d, h))]:
        copy_cell_weights(custom, library)
        if isinstance(custom, MyLSTMCell):
            expected = library(x, (hidden, hidden))
            actual = custom(x, (hidden, hidden))
            assert torch.allclose(actual[0], expected[0], atol=1e-5)
        else:
            assert torch.allclose(custom(x, hidden), library(x, hidden), atol=1e-5)


def test_custom_cells_match_library_over_sequence():
    torch.manual_seed(11)
    batch, steps, d, h = 2, 5, 9, 64
    sequence = torch.randn(batch, steps, d)
    for custom, library in [(MyRNNCell(d, h), torch.nn.RNNCell(d, h)),
                            (MyLSTMCell(d, h), torch.nn.LSTMCell(d, h)),
                            (MyGRUCell(d, h), torch.nn.GRUCell(d, h))]:
        copy_cell_weights(custom, library)
        custom_h = library_h = torch.randn(batch, h)
        custom_c = library_c = torch.randn(batch, h)
        for step in range(steps):
            if isinstance(custom, MyLSTMCell):
                custom_h, custom_c = custom(sequence[:, step], (custom_h, custom_c))
                library_h, library_c = library(sequence[:, step], (library_h, library_c))
                assert torch.allclose(custom_h, library_h, atol=1e-5)
                assert torch.allclose(custom_c, library_c, atol=1e-5)
            else:
                custom_h = custom(sequence[:, step], custom_h)
                library_h = library(sequence[:, step], library_h)
                assert torch.allclose(custom_h, library_h, atol=1e-5)


def test_parameter_derivations():
    assert recurrent_parameter_counts() == {"RNN": 4736, "LSTM": 18944, "GRU": 14208}


def test_attention_rows_and_causality():
    q = torch.randn(2, 3, 4)
    out, weights = scaled_dot_product_attention(q, q, q)
    assert out.shape == q.shape
    assert torch.allclose(weights.sum(-1), torch.ones(2, 3), atol=1e-6)
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    _, causal_weights = scaled_dot_product_attention(q, q, q, mask)
    assert torch.equal(causal_weights.masked_select(~mask), torch.zeros_like(causal_weights.masked_select(~mask)))
