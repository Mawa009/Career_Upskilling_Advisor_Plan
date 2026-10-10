import unittest
from src.career_paths import CareerTransitionModel
from src.preprocessing import records_to_trajectories
from src.evaluation import evaluate


class TestCareerPaths(unittest.TestCase):
    def test_predictions_and_probabilities(self):
        model = CareerTransitionModel().fit([['A','B'], ['A','B'], ['A','C']])
        scores = model.predict_next_occupations('A')
        self.assertEqual(scores[0]['occupation_id'], 'B')
        self.assertAlmostEqual(scores[0]['probability'], 2/3)
        self.assertEqual(model.predict_next_occupations('Z'), [])

    def test_same_start_ambiguity_and_missing(self):
        rows = [
            dict(person_id=1, esco_code='A', start_date='Q1 2020'),
            dict(person_id=1, esco_code='B', start_date='Q1 2020'),
            dict(person_id=1, esco_code='C', start_date='Q3 2021'),
            dict(person_id=2, esco_code='D', start_date='Q1 2019'),
            dict(person_id=2, esco_code='E', start_date='Q1 2020')]
        self.assertEqual(list(records_to_trajectories(rows)), [['C'], ['D', 'E']])

    def test_eval(self):
        m = CareerTransitionModel().fit([['A','B'], ['A','B']])
        metrics = evaluate(m, [['A','B'], ['A','C']])
        self.assertEqual(metrics['n_test'], 2)
        self.assertEqual(metrics['recall@5'], .5)


if __name__ == '__main__':
    unittest.main()
