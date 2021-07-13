import unittest
from wine import data_knn3, data_knn5, data_kmeans


class Test_KNN3(unittest.TestCase):

	def test_knn3(self):
		expected_n3 = 65 <= data_knn3(self) <= 70
		assert data_knn3(self), expected_n3
		print("\nK nearest Neighbor 3\n..........Hit")

	def test_knn5(self):
		expected_n5 = 65 <= data_knn5(self) <= 70
		assert data_knn5(self), expected_n5
		print("\nK nearest Neighbor 5\n..........Hit")

	def test_kmeans(self):
		expected_km = 30 <= data_kmeans(self) <= 70
		assert data_kmeans(self), expected_km
		print("K-means Clustering\n..........Hit")
			
if __name__ == '__main__':
	unittest.main()