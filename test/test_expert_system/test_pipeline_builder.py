"""Unit tests for PipelineBuilder class."""
import pytest
from unittest.mock import MagicMock, patch

from bukka.expert_system.pipeline_builder import PipelineBuilder


class TestPipelineBuilderInitialization:
    """Test suite for PipelineBuilder initialization."""

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_initialization_creates_problem_identifier(self, mock_problem_identifier_class):
        """Test that PipelineBuilder creates a ProblemIdentifier instance.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     assert builder.problem_identifier is not None
        """
        mock_identifier = MagicMock()
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target", problem_type="auto")
        
        # Verify ProblemIdentifier was created
        mock_problem_identifier_class.assert_called_once_with(
            dataset, "target", problem_type="auto"
        )
        assert builder.problem_identifier is mock_identifier

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_initialization_calls_identify_problems(self, mock_problem_identifier_class):
        """Test that initialization calls identify_problems on ProblemIdentifier.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier') as mock:
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     mock.return_value.identify_problems.assert_called_once()
        """
        mock_identifier = MagicMock()
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        
        # Verify identify_problems was called
        mock_identifier.identify_problems.assert_called_once()

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_initialization_with_custom_problem_type(self, mock_problem_identifier_class):
        """Test initialization with custom problem_type parameter.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier') as mock:
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target", problem_type="regression")
        ...     # Should pass problem_type to ProblemIdentifier
        """
        mock_identifier = MagicMock()
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target", problem_type="regression")
        
        mock_problem_identifier_class.assert_called_once_with(
            dataset, "target", problem_type="regression"
        )

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    @patch('bukka.expert_system.pipeline_builder.random.shuffle')
    def test_initialization_shuffles_ml_solutions(self, mock_shuffle, mock_problem_identifier_class):
        """Test that initialization shuffles ML problem solutions.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     with patch('bukka.expert_system.pipeline_builder.random.shuffle') as mock_shuffle:
        ...         dataset = MagicMock()
        ...         builder = PipelineBuilder(dataset, "target")
        ...         # Shuffle should be called on ml_problem.solutions
        """
        mock_identifier = MagicMock()
        mock_ml_problem = MagicMock()
        mock_ml_problem.solutions = [MagicMock(), MagicMock()]
        mock_identifier.ml_problem = mock_ml_problem
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        
        # Verify shuffle was called
        mock_shuffle.assert_called_once()


class TestPipelineBuilderBuildPipeline:
    """Test suite for PipelineBuilder.build_pipeline method."""

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_build_pipeline_returns_list(self, mock_problem_identifier_class):
        """Test that build_pipeline returns a list of pipeline steps.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     result = builder.build_pipeline()
        ...     assert isinstance(result, list)
        """
        mock_identifier = MagicMock()
        mock_identifier.ml_problem = None
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = []
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder.build_pipeline()
        
        assert isinstance(result, list)

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_build_pipeline_includes_processor_steps(self, mock_problem_identifier_class):
        """Test that build_pipeline includes processor steps from problems.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     # Mock some problems with solutions
        ...     result = builder.build_pipeline()
        ...     # Result should include processor steps
        """
        mock_identifier = MagicMock()
        
        # Create mock problems with solutions
        mock_solution = MagicMock()
        mock_problem = MagicMock()
        mock_problem.solutions = [mock_solution]
        
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = [mock_problem]
        mock_identifier.ml_problem = None
        
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder.build_pipeline()
        
        # Should have at least one step from the processor
        assert len(result) >= 1
        # First item should be a tuple of (solution, problem)
        assert isinstance(result[0], tuple)

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_build_pipeline_includes_ml_solution(self, mock_problem_identifier_class):
        """Test that build_pipeline includes ML solution when available.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     result = builder.build_pipeline()
        ...     # Should include ML solution as final step
        """
        mock_identifier = MagicMock()
        
        # Mock ML problem with solutions
        mock_ml_solution = MagicMock()
        mock_ml_problem = MagicMock()
        mock_ml_problem.solutions = [mock_ml_solution]
        
        mock_identifier.ml_problem = mock_ml_problem
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = []
        
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder.build_pipeline()
        
        # Should have ML solution
        assert len(result) == 1
        # ML solution should be in the result
        assert result[0][0] == mock_ml_solution

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_build_pipeline_stores_steps_in_attribute(self, mock_problem_identifier_class):
        """Test that build_pipeline stores steps in pipeline_steps attribute.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     builder.build_pipeline()
        ...     assert hasattr(builder, 'pipeline_steps')
        """
        mock_identifier = MagicMock()
        mock_identifier.ml_problem = None
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = []
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder.build_pipeline()
        
        assert hasattr(builder, 'pipeline_steps')
        assert builder.pipeline_steps == result


class TestPipelineBuilderProcessorSelection:
    """Test suite for PipelineBuilder._processor_selection method."""

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    @patch('bukka.expert_system.pipeline_builder.random.choice')
    def test_processor_selection_returns_list(self, mock_choice, mock_problem_identifier_class):
        """Test that _processor_selection returns a list.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     result = builder._processor_selection()
        ...     assert isinstance(result, list)
        """
        mock_identifier = MagicMock()
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = []
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder._processor_selection()
        
        assert isinstance(result, list)

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    @patch('bukka.expert_system.pipeline_builder.random.choice')
    def test_processor_selection_chooses_one_solution_per_problem(
        self, mock_choice, mock_problem_identifier_class
    ):
        """Test that _processor_selection chooses one solution per problem.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     with patch('bukka.expert_system.pipeline_builder.random.choice') as mock_choice:
        ...         dataset = MagicMock()
        ...         builder = PipelineBuilder(dataset, "target")
        ...         # Should call random.choice for each problem
        """
        mock_identifier = MagicMock()
        
        # Create multiple problems
        mock_solution1 = MagicMock()
        mock_solution2 = MagicMock()
        mock_problem1 = MagicMock()
        mock_problem1.solutions = [mock_solution1]
        mock_problem2 = MagicMock()
        mock_problem2.solutions = [mock_solution2]
        
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = [mock_problem1, mock_problem2]
        
        mock_choice.side_effect = [mock_solution1, mock_solution2]
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder._processor_selection()
        
        # Should have selected one solution for each problem
        assert len(result) == 2
        # random.choice should be called twice
        assert mock_choice.call_count == 2

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_processor_selection_returns_solution_problem_tuples(self, mock_problem_identifier_class):
        """Test that _processor_selection returns (solution, problem) tuples.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     result = builder._processor_selection()
        ...     # Each item should be a tuple
        ...     for item in result:
        ...         assert isinstance(item, tuple)
        ...         assert len(item) == 2
        """
        mock_identifier = MagicMock()
        
        mock_solution = MagicMock()
        mock_problem = MagicMock()
        mock_problem.solutions = [mock_solution]
        
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = [mock_problem]
        
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder._processor_selection()
        
        # Should return tuples
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    @patch('bukka.expert_system.pipeline_builder.ProblemIdentifier')
    def test_processor_selection_skips_problems_without_solutions(self, mock_problem_identifier_class):
        """Test that _processor_selection skips problems with no solutions.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.expert_system.pipeline_builder.ProblemIdentifier'):
        ...     dataset = MagicMock()
        ...     builder = PipelineBuilder(dataset, "target")
        ...     # Problems without solutions should be skipped
        """
        mock_identifier = MagicMock()
        
        # Problem with solutions
        mock_solution = MagicMock()
        mock_problem1 = MagicMock()
        mock_problem1.solutions = [mock_solution]
        
        # Problem without solutions
        mock_problem2 = MagicMock()
        mock_problem2.solutions = []
        
        mock_identifier.problems_to_solve = MagicMock()
        mock_identifier.problems_to_solve.problems = [mock_problem1, mock_problem2]
        
        mock_problem_identifier_class.return_value = mock_identifier
        
        dataset = MagicMock()
        builder = PipelineBuilder(dataset, "target")
        result = builder._processor_selection()
        
        # Should only have one result (from problem1)
        assert len(result) == 1
