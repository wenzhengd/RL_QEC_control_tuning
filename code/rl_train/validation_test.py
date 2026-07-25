"""Regression tests for user-facing training configuration validation."""

import contextlib
import io
import unittest

from .config import PPOConfig
from .train import parse_args


class PPOConfigValidationTest(unittest.TestCase):
    def test_valid_defaults(self) -> None:
        PPOConfig().validate()

    def test_reports_multiple_invalid_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "obs_dim must be greater than 0") as raised:
            PPOConfig(obs_dim=0, learning_rate=float("nan"), gamma=2.0).validate()
        message = str(raised.exception)
        self.assertIn("learning_rate must be a finite value greater than 0", message)
        self.assertIn("gamma must be between 0 and 1 inclusive", message)


class CLIValidationTest(unittest.TestCase):
    def test_valid_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.backend, "steane")

    def test_cli_reports_multiple_errors_with_flag_names(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            parse_args(["--max-steps", "0", "--ppo-learning-rate", "nan"])
        self.assertEqual(raised.exception.code, 2)
        feedback = stderr.getvalue()
        self.assertIn("--max-steps must be greater than 0", feedback)
        self.assertIn("--ppo-learning-rate must be a finite value greater than 0", feedback)

    def test_probability_range_is_explained(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(["--steane-measurement-bitflip-prob", "1.2"])
        self.assertIn("must be between 0 and 1 inclusive", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
