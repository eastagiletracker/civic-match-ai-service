import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from app.services.matching_engine_service import MatchingEngineService
from app.schemas.candidate_schema import CandidateResponseSchema, CandidateResponseItemSchema
from app.schemas.policy_matching_schema import EnhancedMatchResult
from app.schemas.voters_schema import VoterSubmissionSchema, VoterResponseItemSchema


VALID_CATEGORIES = {"TOP", "OTHER", "UNMATCH"}


@pytest.fixture
def matching_engine():
    return MatchingEngineService()


def enhanced_result(candidate_id, percentage):
    return EnhancedMatchResult(
        voter_id="v001",
        candidate_id=candidate_id,
        overall_match_percentage=percentage,
        confidence_weighted_percentage=percentage,
        dimension_matches=[],
        consistency_penalty_applied=0.0,
        match_explanation="",
        top_aligned_dimensions=[]
    )


def calculated_match(engine, candidate_id, percentage):
    """A match as produced for an eligible candidate (starts out as PENDING)."""
    return engine._convert_to_legacy_format(enhanced_result(candidate_id, percentage))


def candidate(candidate_id, eligible=True):
    return CandidateResponseSchema(
        candidate_id=candidate_id,
        election_id="e001",
        responses=[
            CandidateResponseItemSchema(
                id=f"r-{candidate_id}",
                question="Should the city expand public transit?",
                answer="Agree",
                comment="",
                election_id="e001"
            )
        ],
        has_completed_profile=eligible,
        has_completed_questionnaire=eligible
    )


class TestAssignMatchCategories:

    def test_eligible_candidate_with_zero_match_is_unmatch(self, matching_engine):
        matches = [
            calculated_match(matching_engine, "c001", 72),
            calculated_match(matching_engine, "c002", 0),
            matching_engine._create_zero_match_result(candidate("c003", eligible=False)),
        ]
        matches.sort(key=lambda m: m.match_percentage, reverse=True)

        matching_engine._assign_match_categories(matches)

        categories = {m.candidate_id: m.match_category for m in matches}
        assert categories == {"c001": "TOP", "c002": "UNMATCH", "c003": "UNMATCH"}

    def test_top_other_and_unmatch_split(self, matching_engine):
        matches = [calculated_match(matching_engine, f"c{i}", pct)
                   for i, pct in enumerate([90, 80, 70, 60, 50, 0, 0])]

        matching_engine._assign_match_categories(matches)

        assert [m.match_category for m in matches] == [
            "TOP", "TOP", "TOP", "OTHER", "OTHER", "UNMATCH", "UNMATCH"
        ]

    def test_only_zero_matches(self, matching_engine):
        matches = [calculated_match(matching_engine, "c001", 0),
                   matching_engine._create_zero_match_result(candidate("c002", eligible=False))]

        matching_engine._assign_match_categories(matches)

        assert all(m.match_category == "UNMATCH" for m in matches)

    def test_empty_matches(self, matching_engine):
        matching_engine._assign_match_categories([])


class TestProcessVoterSubmissionCategories:

    @pytest.mark.asyncio
    async def test_no_pending_category_in_response(self, matching_engine):
        """An eligible candidate scoring 0% must come back as UNMATCH, not PENDING."""
        submission = VoterSubmissionSchema(
            election_id="e001",
            citizen_id="v001",
            responses=[VoterResponseItemSchema(
                question_id="q001",
                question="Should the city expand public transit?",
                answer="Strongly Agree"
            )]
        )
        candidates = [candidate("c001"), candidate("c002"), candidate("c003", eligible=False)]
        scores = {"c001": 64, "c002": 0}

        async def calculate(voter_profile, candidate_profile):
            return enhanced_result(candidate_profile.person_id, scores[candidate_profile.person_id])

        async def candidate_profiles(eligible, election_analysis):
            return [SimpleNamespace(person_id=c.candidate_id) for c in eligible]

        with patch("app.services.matching_engine_service.candidate_service.get_candidates_for_election",
                   new=AsyncMock(return_value=candidates)), \
             patch("app.services.matching_engine_service.policy_dimension_discovery_service.discover_election_policy_dimensions",
                   new=AsyncMock(return_value=SimpleNamespace(discovered_dimensions=[]))), \
             patch("app.services.matching_engine_service.enhanced_matching_calculator_service.calculate_enhanced_match",
                   new=calculate), \
             patch.object(matching_engine, "_create_voter_policy_profile", new=AsyncMock(return_value=None)), \
             patch.object(matching_engine, "_create_candidate_policy_profiles", new=candidate_profiles), \
             patch.object(matching_engine, "_generate_voter_values_profile", new=AsyncMock(return_value=[])), \
             patch.object(matching_engine, "_determine_processing_quality", return_value=("enhanced", 0.8)):
            result = await matching_engine.process_voter_submission(submission)

        assert result.processing_method == "enhanced"
        categories = {m.candidate_id: m.match_category for m in result.matches}
        assert categories == {"c001": "TOP", "c002": "UNMATCH", "c003": "UNMATCH"}
        assert {m.match_category for m in result.matches} <= VALID_CATEGORIES
