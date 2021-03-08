from unittest import TestCase

from django_mock_queries.mocks import mocked_relations
from django_mock_queries.storage import clear_storage
from tests.mock_models import Car, Manufacturer


class TestStorage(TestCase):

    def tearDown(self) -> None:
        super().tearDown()
        clear_storage()

    def test_save_model(self):
        with mocked_relations(Car, Manufacturer):
            man = Manufacturer(name='test')
            man.save()
            car = Car(speed=10, make=man)
            car.save()
            self.assertEqual(Car.objects.count(), 1)
            self.assertEqual(car.pk, 1)
            self.assertIs(Car.objects.get(pk=car.pk), car)
            self.assertEqual(Car.objects.filter(speed=10).count(), 1)
            self.assertIs(Car.objects.filter(speed=10).first(), car)
            self.assertIs(Car.objects.filter(speed__lt=11).first(), car)
            self.assertIsNone(Car.objects.filter(speed__lt=9).first())
            self.assertIs(Car.objects.filter(make__name='test').first(), car)
            self.assertIsNone(Car.objects.filter(make__name='ne-test').first())
            car.save()  # check not failed

            assert car.make is man

            another_car = Car()
            another_car.pk = car.pk
            with self.assertRaises(ValueError):
                another_car.save()

    def test_bulk_create(self):
        with mocked_relations(Car):
            Car.objects.bulk_create([Car(), Car()])
            self.assertEqual(Car.objects.count(), 2)
