from few_nerd_prompting.prompt_building_utils import extract_predicted_entities, output_well_formed
from few_nerd_prompting.prompt_building_utils import TAG_START, TAG_END


class TestExtractPredictedEntities:
    """
    Tests for the extract_predicted_entities function
    """

    def test_extract_predicted_entities_single_word(self):
        # Test with single-word entity
        snt = f"We live in {TAG_START}Estonia{TAG_END}."
        result = ["Estonia"]
        assert extract_predicted_entities(snt) == result

    def test_extract_predicted_entities_multiple_word(self):
        # Test with multiple-word entity
        snt = f"We live in {TAG_START}New York{TAG_END}."
        result = ["New York"]
        assert extract_predicted_entities(snt) == result

    def test_extract_predicted_entities_multiple_entities(self):
        # Test with multiple different entities
        snt = f"We live in {TAG_START}Manhattan{TAG_END}, in {TAG_START}New York{TAG_END}, on {TAG_START}Madison Avenue{TAG_END}."
        result = ["Manhattan", "New York", "Madison Avenue"]
        assert extract_predicted_entities(snt) == result

    def test_extract_predicted_entities_no_entities(self):
        # Test with no entities
        snt = "We live in the swamp."
        result = []
        assert extract_predicted_entities(snt) == result


class TestOutputWellFormed:
    """
    Tests for the output_well_formed function
    """

    def test_output_well_formed_no_tags(self):
        # Test with no entity tags
        snt = "We live in the swamp."
        assert output_well_formed(snt) is True

    def test_output_well_formed_multiple_entities(self):
        # Test well-formed output with multiple entities
        snt = f"We live in {TAG_START}Manhattan{TAG_END}, in {TAG_START}New York{TAG_END}, on {TAG_START}Madison Avenue{TAG_END}."
        assert output_well_formed(snt) is True

    def test_output_well_formed_start_but_no_end(self):
        # Test with start tag but no end tag
        snt = f"We live in the{TAG_START} swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_end_but_no_start(self):
        # Test with end tag but not start tag
        snt = f"We live in the{TAG_END} swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_end_before_start(self):
        # Test with end tag before start tag
        snt = f"We {TAG_END}live in {TAG_START}the swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_no_end_after_second_start(self):
        # Test with no end tag for second start tag
        snt = f"{TAG_START}We {TAG_END}live in {TAG_START}the swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_second_start_before_first_end(self):
        # Test with second start tag before first end tag
        snt = f"{TAG_START}We live in {TAG_START}the {TAG_END}swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_third_start_before_second_end(self):
        # Test with third start tag before second end tag
        snt = f"{TAG_START}We{TAG_END} live {TAG_START}in {TAG_START}the {TAG_END}swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_start_end_end(self):
        # Test with start - end - end
        snt = f"{TAG_START}We {TAG_END}live in {TAG_END}the swamp."
        assert output_well_formed(snt) is False

    def test_output_well_formed_end_start_end(self):
        # Test with end - start - end
        snt = f"{TAG_END}We {TAG_START}live in {TAG_END}the swamp."
        assert output_well_formed(snt) is False

