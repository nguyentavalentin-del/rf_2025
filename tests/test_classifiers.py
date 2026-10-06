import os
import unittest

from rf_2025.src import utils
from rf_2025.src import classifiers


class ClassifierTests(unittest.TestCase):
    def test_knn_handles_descriptor_dimensions_and_scaling(self):
        train = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [10.0, 10.0, 10.0]]
        labels = [1, 1, 2]
        self.assertEqual(classifiers.knn(train, labels, [[9.0, 9.0, 9.0]], 1), [2])

    def test_knn_rejects_invalid_neighbor_count(self):
        with self.assertRaises(ValueError):
            classifiers.knn([[0.0]], [1], [[1.0]], 2)

    def test_knn_returns_empty_predictions_for_empty_test_set(self):
        self.assertEqual(classifiers.knn([[0.0]], [1], [], 1), [])

    def test_leave_one_out_predictions_match_sample_count(self):
        vectors = [[0.0], [0.1], [10.0], [10.1]]
        labels = [1, 1, 2, 2]
        self.assertEqual(
            classifiers.knn_leave_one_out(vectors, labels, 1),
            labels,
        )

    def test_kmeans_is_repeatable_and_does_not_mutate_input(self):
        vectors = [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]
        original = [point[:] for point in vectors]
        first = classifiers.kmeans(vectors, 2)
        second = classifiers.kmeans(vectors, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 2)
        self.assertEqual(vectors, original)


class UtilityTests(unittest.TestCase):
    def test_distances_support_arbitrary_dimensions(self):
        self.assertAlmostEqual(utils.distance_euclidienne([0, 0, 0], [1, 2, 2]), 3.0)
        self.assertEqual(utils.distance_manhattan([0, 0, 0], [1, 2, 2]), 5.0)
        with self.assertRaises(ValueError):
            utils.distance_manhattan([0], [0, 1])

    def test_confusion_metrics_handle_zero_divisions(self):
        matrix, labels = utils.matrice_confusion([1, 2], [1, 1])
        self.assertEqual(labels, [1, 2])
        self.assertEqual(utils.taux_reco(matrix), 0.5)
        self.assertAlmostEqual(utils.f_mesure(matrix, 2), 1.0 / 3.0)

    def test_loader_keeps_working_directory_and_returns_every_descriptor(self):
        original_directory = os.getcwd()
        methods = utils.load_methods()
        self.assertEqual(os.getcwd(), original_directory)
        self.assertEqual(len(methods), 5)
        self.assertTrue(all(len(method) == 99 for method in methods))
        self.assertEqual(
            {len(vector) for _, vector in methods[0]},
            {16},
        )
        self.assertEqual(
            [{len(vector) for _, vector in method} for method in methods],
            [{16}, {128}, {128}, {100}, {90}],
        )


if __name__ == "__main__":
    unittest.main()
