import math
from pathlib import Path

import pytest

from juno.domain.values import Direction
from juno.eval.grading import GradingError, Leaderboard, Medal, Metrics, Table


class Csv:
    @staticmethod
    def table(tmp_path: Path, text: str, name: str = "table.csv") -> Table:
        path = tmp_path / name
        path.write_text(text)
        return Table.read(path)


class TestTable:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("id\n1\n", "at least one value column"),
            ("id,label\n1,a\n1,b\n", "repeats id '1'"),
            ("id,label\n1,a,extra\n", "3 of 2 cells"),
        ],
    )
    def test_bad_files_are_named(self, tmp_path: Path, text: str, message: str) -> None:
        with pytest.raises(GradingError, match=message):
            Csv.table(tmp_path, text)

    def test_a_submission_must_cover_exactly_the_test_ids(self, tmp_path: Path) -> None:
        truth = Csv.table(tmp_path, "id,label\n1,a\n2,b\n", "truth.csv")

        with pytest.raises(GradingError, match="1 ids are missing"):
            Csv.table(tmp_path, "id,label\n1,a\n").matching(truth)

        with pytest.raises(GradingError, match="not in the test set"):
            Csv.table(tmp_path, "id,label\n1,a\n2,b\n3,c\n").matching(truth)


class TestMetrics:
    def score(self, tmp_path: Path, metric: str, truth: str, predicted: str) -> float:
        expected = Csv.table(tmp_path, truth, "truth.csv")
        submitted = Csv.table(tmp_path, predicted, "submission.csv").matching(expected)
        return Metrics.get(metric).score(expected, submitted)

    def test_accuracy_compares_numbers_by_value(self, tmp_path: Path) -> None:
        score = self.score(
            tmp_path,
            "accuracy",
            "id,label\n1,1\n2,0\n3,1\n4,0\n",
            ("id,label\n1,1.0\n2,0\n3,0\n4,0\n"),
        )

        assert score == 0.75

    def test_rmse_and_mae(self, tmp_path: Path) -> None:
        truth, predicted = "id,y\n1,0\n2,0\n", "id,y\n1,3\n2,-1\n"

        assert self.score(tmp_path, "rmse", truth, predicted) == pytest.approx(math.sqrt(5))
        assert self.score(tmp_path, "mae", truth, predicted) == 2

    def test_binary_log_loss_reads_one_probability_column(self, tmp_path: Path) -> None:
        score = self.score(tmp_path, "log_loss", "id,label\n1,1\n2,0\n", "id,label\n1,0.8\n2,0.4\n")

        assert score == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)

    def test_multiclass_log_loss_normalises_each_row(self, tmp_path: Path) -> None:
        score = self.score(
            tmp_path, "logloss", "id,author\n1,EAP\n2,HPL\n", "id,EAP,HPL\n1,2,2\n2,0,1\n"
        )

        assert score == pytest.approx(-(math.log(0.5) + math.log(1 - 1e-15)) / 2)

    def test_log_loss_needs_a_column_per_class(self, tmp_path: Path) -> None:
        with pytest.raises(GradingError, match="class 'MWS'"):
            self.score(tmp_path, "log_loss", "id,author\n1,MWS\n", "id,EAP,HPL\n1,0.5,0.5\n")

    def test_auc_counts_tied_scores_as_half(self, tmp_path: Path) -> None:
        truth = "id,label\n1,1\n2,0\n3,1\n4,0\n"

        assert self.score(tmp_path, "auc", truth, "id,label\n1,0.9\n2,0.1\n3,0.8\n4,0.2\n") == 1
        assert (
            self.score(tmp_path, "roc_auc", truth, "id,label\n1,0.5\n2,0.5\n3,0.5\n4,0.5\n") == 0.5
        )
        assert self.score(tmp_path, "auc", truth, "id,label\n1,0.9\n2,0.9\n3,0.1\n4,0.1\n") == 0.5

    def test_bad_cells_and_unknown_metrics_raise(self, tmp_path: Path) -> None:
        with pytest.raises(GradingError, match="not a number"):
            self.score(tmp_path, "rmse", "id,y\n1,0\n", "id,y\n1,high\n")

        with pytest.raises(GradingError, match="not finite"):
            self.score(tmp_path, "rmse", "id,y\n1,0\n", "id,y\n1,nan\n")

        with pytest.raises(GradingError, match="no grader for metric 'f1'"):
            Metrics.get("f1")


class TestLeaderboard:
    def test_percentile_is_the_share_beaten_with_ties_as_half(self) -> None:
        board = Leaderboard([0.9, 0.8, 0.7, 0.7], Direction.MAXIMIZE)

        assert board.percentile(0.95) == 100
        assert board.percentile(0.7) == 25
        assert board.percentile(0.1) == 0

    def test_lower_is_better_when_minimizing(self) -> None:
        board = Leaderboard([0.3, 0.5], Direction.MINIMIZE)

        assert board.percentile(0.4) == 50
        assert board.medal(0.2) is Medal.GOLD

    @pytest.mark.parametrize(
        ("teams", "cutoffs"),
        [(50, (5, 10, 20)), (200, (10, 40, 80)), (500, (11, 50, 100)), (2_000, (14, 100, 200))],
    )
    def test_medal_cutoffs_follow_kaggle(self, teams: int, cutoffs: tuple[int, int, int]) -> None:
        assert Leaderboard.cutoffs(teams) == cutoffs

    def test_medal_comes_from_the_rank(self) -> None:
        board = Leaderboard([float(score) for score in range(10)], Direction.MAXIMIZE)

        assert board.medal(9) is Medal.GOLD
        assert board.medal(8) is Medal.SILVER
        assert board.medal(6) is Medal.BRONZE
        assert board.medal(5) is None
