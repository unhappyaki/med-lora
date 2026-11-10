import unittest

from evaluate_v5 import (
    merge_grounded_pairs,
    recover_unique_whitespace_span,
    salvage_prediction,
)


class GroundingRecoveryTests(unittest.TestCase):
    def test_recovers_one_unique_whitespace_variant(self):
        source = "诊疗计划：1\n、因妊娠期间完善检查。"
        self.assertEqual(
            recover_unique_whitespace_span("1、因妊娠期间完善检查。", source),
            "1\n、因妊娠期间完善检查。",
        )

    def test_refuses_ambiguous_normalized_match(self):
        source = "1\n、完善检查。\n其他：1 、完善检查。"
        self.assertIsNone(recover_unique_whitespace_span("1、完善检查。", source))

    def test_salvage_returns_literal_source_span(self):
        source = "诊疗计划：1\n、因妊娠期间完善检查。"
        payload = {
            "extractions": [
                {
                    "slot": "1、因妊娠期间完善检查。",
                    "slot_attributes": {"field_name": "诊疗计划"},
                }
            ]
        }
        pairs, schema_valid, errors = salvage_prediction(payload, source, 128)
        self.assertTrue(schema_valid)
        self.assertEqual(errors, [])
        self.assertEqual(pairs, [("诊疗计划", "1\n、因妊娠期间完善检查。")])
        self.assertIn(pairs[0][1], source)

    def test_chunk_merge_keeps_one_value_per_field(self):
        source = "年龄：60岁\n年龄描述：61岁\n记录者：医师甲"
        pairs = merge_grounded_pairs(
            [[("年龄", "60岁")], [("年龄", "61岁"), ("记录者", "医师甲")]],
            source,
        )
        self.assertEqual(pairs, [("年龄", "60岁"), ("记录者", "医师甲")])


if __name__ == "__main__":
    unittest.main()
