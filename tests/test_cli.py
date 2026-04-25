"""Tests for the chronogrammer CLI."""

import subprocess
import sys

import pytest

from chronogrammer.cli import cmd_score, cmd_rewrite, main


class TestMain:
    def test_score_subcommand(self, capsys):
        exit_code = main(["score", "Viva the view!"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Score" in captured.out or "score" in captured.out.lower()
        # V+i+v+a + t+h+e + v+i+e+w+! → V=5, I=1, V=5 (Viva) + V=5, I=1 (view) = 17
        assert "17" in captured.out

    def test_score_known_value(self, capsys):
        exit_code = main(["score", "Viva the view!"])
        assert exit_code == 0
        captured = capsys.readouterr()
        # V+i+v = 5+1+5 = 11, a=0, the=0, view = v+i = 5+1 = 6, ! = 0 → total = 17
        assert "17" in captured.out

    def test_score_with_target(self, capsys):
        exit_code = main(["score", "--target", "20", "Viva the view!"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "20" in captured.out

    def test_rewrite_returns_result(self, capsys):
        exit_code = main(["rewrite", "--target", "17", "Viva the view!"])
        # exit code 0 means exact match, 1 means near-miss — both are acceptable
        assert exit_code in (0, 1)
        captured = capsys.readouterr()
        assert "rewrite" in captured.out.lower() or "candidate" in captured.out.lower()

    def test_rewrite_exact_when_already_matches(self, capsys):
        # If target equals the source score, the source itself should be returned as best
        # "I" = 1
        exit_code = main(["rewrite", "--target", "1", "I"])
        assert exit_code == 0  # exact match

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_no_args_exits_with_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_unknown_command_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["notacommand"])
        assert exc_info.value.code != 0


class TestScoreBreakdownOutput:
    def test_breakdown_shows_letters(self, capsys):
        main(["score", "VIVA"])
        captured = capsys.readouterr()
        # V and I should be mentioned
        assert "V" in captured.out or "I" in captured.out

    def test_no_roman_letters(self, capsys):
        main(["score", "hello"])
        captured = capsys.readouterr()
        assert "0" in captured.out  # score should be 0


class TestCLIEntryPoint:
    """Test that the CLI works as a subprocess (i.e., the entry point wires up)."""

    def test_score_via_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "chronogrammer.cli", "score", "Viva the view!"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "17" in result.stdout

    def test_rewrite_via_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "chronogrammer.cli", "rewrite",
             "--target", "17", "Viva the view!"],
            capture_output=True,
            text=True,
        )
        assert result.returncode in (0, 1)
