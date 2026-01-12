"""
Unit tests for profiling analyzers.
"""

import pytest
from hyperlora.profiling.analyzers import (
    GradientEnergyAnalyzer,
    FisherInformationAnalyzer,
    ProxyFineTuneAnalyzer,
)
from hyperlora.config import ProfilingConfig
from hyperlora.meta import LayerSpec


@pytest.fixture
def analyzer_config():
    return ProfilingConfig()


@pytest.fixture
def sample_layer():
    return LayerSpec(
        name="test_layer",
        hidden_size=768,
        layer_type="transformer",
        supports_attention=True,
    )


def test_gradient_energy_analyzer(analyzer_config, sample_layer):
    """Test GradientEnergyAnalyzer scoring."""
    analyzer = GradientEnergyAnalyzer(analyzer_config)
    activations = {"grad_norm": 100.0}
    
    score = analyzer.score(sample_layer, activations)
    # Should normalize by hidden_size
    expected = 100.0 / 768.0
    assert abs(score - expected) < 1e-6


def test_fisher_information_analyzer(analyzer_config, sample_layer):
    """Test FisherInformationAnalyzer scoring."""
    analyzer = FisherInformationAnalyzer(analyzer_config)
    activations = {"fisher_trace": 50.0}
    
    score = analyzer.score(sample_layer, activations)
    # Should normalize by sqrt(hidden_size)
    expected = 50.0 / (768.0 ** 0.5)
    assert abs(score - expected) < 1e-6


def test_proxy_finetune_analyzer(analyzer_config, sample_layer):
    """Test ProxyFineTuneAnalyzer scoring."""
    analyzer = ProxyFineTuneAnalyzer(analyzer_config)
    activations = {"proxy_gain": 0.15}
    
    score = analyzer.score(sample_layer, activations)
    assert score == 0.15


def test_analyzers_missing_activations(analyzer_config, sample_layer):
    """Test analyzers handle missing activation keys gracefully."""
    gradient_analyzer = GradientEnergyAnalyzer(analyzer_config)
    fisher_analyzer = FisherInformationAnalyzer(analyzer_config)
    proxy_analyzer = ProxyFineTuneAnalyzer(analyzer_config)
    
    empty_activations = {}
    
    # Should return 0.0 for missing keys
    assert gradient_analyzer.score(sample_layer, empty_activations) == 0.0
    assert fisher_analyzer.score(sample_layer, empty_activations) == 0.0
    assert proxy_analyzer.score(sample_layer, empty_activations) == 0.0

