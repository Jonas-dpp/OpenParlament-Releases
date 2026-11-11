from unittest.mock import patch, MagicMock
from scripts.nlp_session import NLPSession, _choose_cuda_device_interactive


def test_choose_cuda_device_interactive():
    # Test valid and invalid input paths in _choose_cuda_device_interactive
    with patch("builtins.input", return_value="1"):
        assert _choose_cuda_device_interactive(max_devices=3) == 1
    # default/empty input
    with patch("builtins.input", return_value=""):
        assert _choose_cuda_device_interactive(max_devices=3) == 0
    # invalid integer format
    with patch("builtins.input", return_value="invalid"):
        assert _choose_cuda_device_interactive(max_devices=3) == 0
    # out of bounds choice
    with patch("builtins.input", return_value="5"):
        assert _choose_cuda_device_interactive(max_devices=3) == 0


def test_resolve_device_explicit():
    # Explicit device set on NLPSession bypasses everything
    sess = NLPSession(device=2)
    assert sess._resolve_device() == 2


def test_resolve_device_no_cuda():
    # If cuda is not requested, device is -1
    sess = NLPSession(use_cuda=False)
    assert sess._resolve_device() == -1


def test_resolve_device_torch_unavailable():
    # If torch is not available, falls back to -1
    sess = NLPSession(use_cuda=True)
    with patch("scripts.nlp_session._TORCH_AVAILABLE", False):
        assert sess._resolve_device() == -1


def test_resolve_device_cuda_unavailable():
    sess = NLPSession(use_cuda=True)
    # Mock torch.cuda.is_available() returning False
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch):
        assert sess._resolve_device() == -1


def test_resolve_device_cuda_zero_count():
    sess = NLPSession(use_cuda=True)
    # Mock torch.cuda.device_count() returning 0
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 0
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch):
        assert sess._resolve_device() == -1


def test_resolve_device_interactive_tty():
    # Device count = 2, device_index = None, isatty = True
    sess = NLPSession(use_cuda=True, device_index=None)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("scripts.nlp_session._choose_cuda_device_interactive", return_value=1) as mock_choose:
        assert sess._resolve_device() == 1
        mock_choose.assert_called_once_with(2)


def test_resolve_device_non_interactive_non_tty():
    # Device count = 2, device_index = None, isatty = False
    sess = NLPSession(use_cuda=True, device_index=None)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch), \
         patch("sys.stdin.isatty", return_value=False):
        assert sess._resolve_device() == 0


def test_resolve_device_out_of_bounds_index():
    # Explicit index provided but out of range, fallback to 0
    sess = NLPSession(use_cuda=True, device_index=5)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch):
        assert sess._resolve_device() == 0


def test_resolve_device_exception_handling():
    sess = NLPSession(use_cuda=True)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = ValueError("Unexpected torch error")
    with patch("scripts.nlp_session._TORCH_AVAILABLE", True), \
         patch("scripts.nlp_session.torch", mock_torch):
        assert sess._resolve_device() == -1


def test_nlp_session_enter_exit_fp16_warning():
    # test fp16 requested but torch unavailable warning (line 160-164)
    sess = NLPSession(fp16=True, device=0) # positive device index allows fp16 pathway
    with patch("scripts.nlp_session._TORCH_AVAILABLE", False), \
         patch("scripts.nlp_session.SentimentEngine") as mock_sent, \
         patch("scripts.nlp_session.ToneClassifier"), \
         patch("scripts.nlp_session.AddresseeDetector"):
        with sess:
            pass
        # Should not pass torch.float16 as torch_dtype since torch is unavailable
        mock_sent.assert_called_once_with(batch_size=100, device=0, torch_dtype=None)


def test_nlp_session_engine_creation_failures():
    # Test line 169-171, 175-177, 181-183: creation exceptions caught and logged
    sess = NLPSession()
    with patch("scripts.nlp_session.SentimentEngine", side_effect=Exception("Sent init fail")), \
         patch("scripts.nlp_session.ToneClassifier", side_effect=Exception("Tone init fail")), \
         patch("scripts.nlp_session.AddresseeDetector", side_effect=Exception("Addr init fail")):
        with sess:
            assert sess.sentiment_engine is None
            assert sess.tone_classifier is None
            assert sess.addressee_detector is None
