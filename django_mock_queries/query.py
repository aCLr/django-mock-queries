import datetime
import random
from collections import OrderedDict, namedtuple
from copy import copy

from django.db.models import Q, BooleanField, Value
from django.db.models.functions import Cast
from six import with_metaclass

from .storage import get_models_from_storage, remove_from_storage, add_to_storage, STORAGE

try:
    from unittest.mock import Mock, MagicMock, PropertyMock
except ImportError:
    from mock import Mock, MagicMock, PropertyMock

from .constants import *
from .exceptions import *
from .utils import (
    matches, merge, intersect, get_attribute, validate_mock_set, is_list_like_iter, flatten_list, truncate,
    hash_dict, filter_results, filter_objects
)


class MockSetMeta(type):
    def __call__(cls, *initial_items, **kwargs):
        obj = super(MockSetMeta, cls).__call__(**kwargs)
        # obj.add(*initial_items)
        return obj

class MockSet(with_metaclass(MockSetMeta, MagicMock)):
    EVENT_ADDED = 'added'
    EVENT_UPDATED = 'updated'
    EVENT_SAVED = 'saved'
    EVENT_DELETED = 'deleted'
    SUPPORTED_EVENTS = [EVENT_ADDED, EVENT_UPDATED, EVENT_SAVED, EVENT_DELETED]
    RETURN_SELF_METHODS = [
        'all',
        'only',
        'defer',
        'using',
        'select_related',
        'prefetch_related',
        'select_for_update',
        'iterator'
    ]

    def none(self):
        new = self._clone()
        return new.annotate(x=Value(False, BooleanField())).filter(Q(x=True))

    def __deepcopy__(self, memodict=None):
        return self._clone()

    def _get_items(self):
        if self._cached is not None and self._transaction_id == STORAGE.current_transaction_id:
            return self._cached
        results = filter_objects(get_models_from_storage(self.model), self._filters)

        if self._exclusions:
            excluded = set(filter_objects(results, self._exclusions))
            results = [item for item in results if item not in excluded]

        for key, value in self._annotations.items():
            for x, row in enumerate(results):
                row = copy(row)
                if not hasattr(row, '_annotated_fields'):
                    row._annotated_fields = []
                row._annotated_fields.append(key)
                setattr(row, key, get_attribute(row, value)[0])
                results[x] = row

        for field in reversed(self._order_by):
            if field == '?':
                random.shuffle(results)
                break
            if isinstance(field, Cast):
                attr = field.source_expressions[0].deconstruct()[1][0]
                getter = lambda r: field.field.to_python(get_attribute(r, attr))
                is_reversed = False
            else:
                is_reversed = field.startswith('-')
                attr = field[1:] if is_reversed else field
                getter = lambda r: get_attribute(r, attr)
            results = sorted(results,
                             key=getter,
                             reverse=is_reversed)

        if self._distincts:
            distinct_results = OrderedDict()
            for item in results:
                key = hash_dict(item, *self._distincts)
                if key not in distinct_results:
                    distinct_results[key] = item
            results = list(distinct_results.values())

        if self._values:
            as_dict = self._values_params['as_dict']
            as_values = []
            for row in results:
                as_values.extend(self._item_values(row, self._values))
            if as_dict:
                results = as_values
            else:
                results = []
                for row in as_values:
                    results.append(self._values_row(row, self._values, **self._values_params))
        self._cached = results
        self._transaction_id = STORAGE.current_transaction_id
        return results

    def _clone(self):
        new = MockSet(model=self.model)
        new._filters = self._filters.copy()
        new._exclusions = self._exclusions.copy()
        new._values_params = self._values_params.copy()
        new._values = self._values.copy()
        new._annotations = self._annotations.copy()
        new._order_by = self._order_by.copy()
        new._distincts = self._distincts.copy()
        return new

    def __init__(self, **kwargs):
        clone = kwargs.pop('clone', None)
        model = kwargs.pop('model', None)

        for x in self.RETURN_SELF_METHODS:
            kwargs.update({x: self._return_self})

        super(MockSet, self).__init__(spec=DjangoQuerySet, **kwargs)

        self._values = []
        self._values_params = {}
        self._annotations = {}
        self._exclusions = []
        self._filters = []
        self._order_by = []
        self._transaction_id = None
        self._distincts = []

        self._cached = None
        self.model = getattr(clone, 'model', model)
        self.clone = clone
        self.events = {}

        self.__len__ = lambda s: len(s._get_items())
        self.__iter__ = lambda s: iter(s._get_items())
        self.__getitem__ = lambda s, k: self._get_items()[k]
        self.__bool__ = self.__nonzero__ = lambda s: len(s._get_items()) > 0

    def _return_self(self, *_, **__):
        return self

    def count(self):
        return len(self._get_items())

    def fire(self, obj, *events):
        for name in events:
            for handler in self.events.get(name, []):
                handler(obj)

    def on(self, event, handler):
        assert event in self.SUPPORTED_EVENTS, event
        self.events[event] = self.events.get(event, []) + [handler]

    def _register_fields(self, obj):
        if not (isinstance(obj, MockModel) or isinstance(obj, Mock)):
            return

        for f in self.model._meta.fields:
            if f.name not in obj.keys():
                setattr(obj, f.name, None)

    def add(self, *models):
        if self.model:
            # Initialize MockModel default fields from MockSet model fields if defined
            for obj in models:
                if isinstance(obj, dict):
                    continue
                self._register_fields(obj)

        for model in models:
            if isinstance(obj, dict):
                continue
            add_to_storage(model)
            self.fire(model, self.EVENT_ADDED, self.EVENT_SAVED)

    def filter(self, *args, **attrs):
        new = self._clone()
        new._filters += list(args)
        new._filters += list(attrs.items())
        return new

    def exclude(self, *args, **attrs):
        new = self._clone()
        new._exclusions += list(args)
        new._exclusions += list(attrs.items())
        return new

    def exists(self):
        return len(self._get_items()) > 0

    def in_bulk(self, id_list=None, *, field_name='pk'):
        result = {}
        for model in self._get_items():
            if id_list is None or model.pk in id_list:
                result[getattr(model, field_name)] = model
        return result

    def annotate(self, **kwargs):
        new = self._clone()
        new._annotations.update(kwargs)
        return new

    def aggregate(self, *args, **kwargs):
        result = {}

        for expr in set(args):
            kwargs['{0}__{1}'.format(expr.source_expressions[0].name, expr.function).lower()] = expr
        items = list(self._get_items())
        for alias, expr in kwargs.items():
            values = []
            expr_result = None

            for x in items:
                val = get_attribute(x, expr.source_expressions[0].name)[0]
                if val is None:
                    continue
                values.extend(val if is_list_like_iter(val) else [val])

            if len(values) > 0:
                expr_result = {
                    AGGREGATES_SUM: lambda: sum(values),
                    AGGREGATES_COUNT: lambda: len(values),
                    AGGREGATES_MAX: lambda: max(values),
                    AGGREGATES_MIN: lambda: min(values),
                    AGGREGATES_AVG: lambda: sum(values) / len(values)
                }[expr.function]()

            if len(values) == 0 and expr.function == AGGREGATES_COUNT:
                expr_result = 0

            result[alias] = expr_result

        return result

    def order_by(self, *fields):
        new = self._clone()
        new._order_by += fields
        return new

    def distinct(self, *fields):
        new = self._clone()
        new._distincts += list(fields)
        return new

    def _raise_does_not_exist(self):
        does_not_exist = getattr(self.model, 'DoesNotExist', ObjectDoesNotExist)
        raise does_not_exist()

    def _get_order_fields(self, fields, field_name):
        if fields and field_name is not None:
            raise ValueError('Cannot use both positional arguments and the field_name keyword argument.')

        if field_name is not None:
            # The field_name keyword argument is deprecated in favor of passing positional arguments.
            order_fields = (field_name,)
        elif fields:
            order_fields = fields
        else:
            order_fields = self.model._meta.get_latest_by
            if order_fields and not isinstance(order_fields, (tuple, list)):
                order_fields = (order_fields,)

        if order_fields is None:
            raise ValueError(
                "earliest() and latest() require either fields as positional "
                "arguments or 'get_latest_by' in the model's Meta."
            )

        return order_fields

    def _earliest_or_latest(self, *fields, **field_kwargs):
        """
        Mimic Django's behavior
        https://github.com/django/django/blob/746caf3ef821dbf7588797cb2600fa81b9df9d1d/django/db/models/query.py#L560
        """
        field_name = field_kwargs.get('field_name', None)
        reverse = field_kwargs.get('reverse', False)
        order_fields = self._get_order_fields(fields, field_name)

        results = sorted(
            self._get_items(),
            key=lambda obj: tuple(get_attribute(obj, key) for key in order_fields),
            reverse=reverse,
        )

        if len(results) == 0:
            self._raise_does_not_exist()

        return results[0]

    def earliest(self, *fields, **field_kwargs):
        return self._earliest_or_latest(*fields, **field_kwargs)

    def latest(self, *fields, **field_kwargs):
        return self._earliest_or_latest(*fields, reverse=True, **field_kwargs)

    def first(self):
        items = self._get_items()
        if items:
            return items[0]

    def last(self):
        items = self._get_items()
        if items:
            return items[-1]

    def create(self, **attrs):
        validate_mock_set(self, **attrs)

        obj = self.model(**attrs)
        obj.save()

        return obj

    def update(self, **attrs):
        validate_mock_set(self, for_update=True, **attrs)

        count = 0
        for item in self._get_items():
            count += 1
            for k, v in attrs.items():
                setattr(item, k, v)
                self.fire(item, self.EVENT_UPDATED, self.EVENT_SAVED)
            add_to_storage(item)

        return count

    def bulk_create(self, items, batch_size=None, ignore_conflicts=False):
        for item in items:
            add_to_storage(item)

    def bulk_update(self, items, fields, batch_size=None):
        for item in items:
            item.save()

    def _delete_recursive(self, *items_to_remove, **attrs):
        for item in matches(*items_to_remove, **attrs):
            remove_from_storage(item)
            self.fire(item, self.EVENT_DELETED)

        if self.clone is not None:
            self.clone._delete_recursive(*items_to_remove, **attrs)

    def delete(self, **attrs):
        # Delete normally doesn't take **attrs - they're only needed for remove
        self._delete_recursive(*self._get_items(), **attrs)

    # The following 2 methods were kept for backwards compatibility and
    # should be removed in the future since they are covered by filter & delete
    def clear(self, **attrs):
        return self.delete(**attrs)

    def remove(self, **attrs):
        return self.delete(**attrs)

    def get(self, *args, **attrs):
        results = self.filter(*args, **attrs)
        if not results.exists():
            self._raise_does_not_exist()
        elif results.count() > 1:
            raise MultipleObjectsReturned()
        else:
            return results[0]

    def get_or_create(self, defaults=None, **attrs):
        if defaults is not None:
            validate_mock_set(self)
        defaults = defaults or {}
        lookup = attrs.copy()
        attrs.update(defaults)
        results = self.filter(**lookup)
        if not results.exists():
            return self.create(**attrs), True
        elif results.count() > 1:
            raise MultipleObjectsReturned()
        else:
            return results[0], False

    def update_or_create(self, defaults=None, **attrs):
        if defaults is not None:
            validate_mock_set(self)
        defaults = defaults or {}
        lookup = attrs.copy()
        attrs.update(defaults)
        results = self.filter(**lookup)
        if not results.exists():
            return self.create(**attrs), True
        elif results.count() > 1:
            raise MultipleObjectsReturned()
        else:
            obj = results[0]
            for k, v in attrs.items():
                setattr(obj, k, v)
                self.fire(obj, self.EVENT_UPDATED, self.EVENT_SAVED)
            return obj, False

    def _item_values(self, item, fields):
        field_buckets = {}
        result_count = 1

        if len(fields) == 0:
            field_names = [f.attname for f in item._meta.concrete_fields]
        else:
            field_names = list(fields)

        for field in sorted(field_names, key=lambda k: k.count('__')):
            if isinstance(field, str) and hasattr(item, field + '_id'):
                value = get_attribute(item, field + '_id')[0]
            else:
                value = get_attribute(item, field)[0]

            if is_list_like_iter(value):
                value = flatten_list(value)
                result_count = max(result_count, len(value))

                for bucket, data in field_buckets.items():
                    while len(data) < result_count:
                        data.append(data[-1])

                field_buckets[field] = value
            else:
                field_buckets[field] = [value]

        item_values = []
        for i in range(result_count):
            item_values.append({k: v[i] for k, v in field_buckets.items()})

        return item_values

    def _item_values_list(self, values_dict, fields, flat):
        if flat:
            return values_dict[fields[0]]
        else:
            data = []
            for key in sorted(values_dict.keys(), key=lambda k: fields.index(k)):
                data.append(values_dict[key])
            return tuple(data)

    def _values_row(self, values_dict, fields, **kwargs):
        flat = kwargs.pop('flat', False)
        named = kwargs.pop('named', False)
        kwargs.pop('as_dict', None)

        if kwargs:
            raise TypeError('Unexpected keyword arguments to values_list: %s' % (list(kwargs),))
        if flat and len(fields) > 1:
            raise TypeError('`flat` is not valid when values_list is called with more than one field.')
        if flat and named:
            raise TypeError('`flat` and `named` can\'t be used together.')

        if named:
            Row = namedtuple('Row', fields)
            row = Row(**values_dict)
        else:
            row = self._item_values_list(values_dict, fields, flat)

        return row

    def values(self, *fields):
        new = self._clone()
        new._values = list(fields)
        new._values_params = {'as_dict': True}
        return new

    def values_list(self, *fields, **kwargs):
        # Django doesn't complain about this:
        # https://github.com/django/django/blob/a4e6030904df63b3f10aa0729b86dc6942b0458e/django/db/models/query.py#L845
        # if len(fields) == 0:
        #     raise NotImplementedError('values_list() with no arguments is not implemented')
        new = self._clone()
        new._values = list(fields)
        new._values_params = {'as_dict': False, **kwargs}
        return new

    def _date_values(self, field, kind, order, key_func):
        initial_values = list(self.values_list(field, flat=True))

        return MockSet(*sorted(
            {truncate(x, kind) for x in initial_values},
            key=key_func,
            reverse=True if order == 'DESC' else False
        ), clone=self)

    def dates(self, field, kind, order='ASC'):
        assert kind in ("year", "month", "day"), "'kind' must be one of 'year', 'month' or 'day'."
        assert order in ('ASC', 'DESC'), "'order' must be either 'ASC' or 'DESC'."

        return self._date_values(field, kind, order, lambda y: datetime.date.timetuple(y)[:3])

    def datetimes(self, field, kind, order='ASC'):
        # TODO: Handle `tzinfo` parameter
        assert kind in ("year", "month", "day", "hour", "minute", "second"), \
            "'kind' must be one of 'year', 'month', 'day', 'hour', 'minute' or 'second'."
        assert order in ('ASC', 'DESC'), "'order' must be either 'ASC' or 'DESC'."

        return self._date_values(field, kind, order, lambda y: datetime.datetime.timetuple(y)[:6])


class MockModel(dict):
    def __init__(self, *args, **kwargs):
        super(MockModel, self).__init__(*args, **kwargs)

        # self.save = PropertyMock()
        self.__meta = MockOptions(*self.get_fields())

    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __hash__(self):
        return hash_dict(self)

    def __call__(self, *args, **kwargs):
        return MockModel(*args, **kwargs)

    def get_fields(self):
        skip_keys = ['save', '_MockModel__meta']
        return [key for key in self.keys() if key not in skip_keys]

    @property
    def _meta(self):
        self.__meta.load_fields(*self.get_fields())
        return self.__meta

    def __repr__(self):
        return self.get('mock_name', None) or super(MockModel, self).__repr__()


def create_model(*fields):
    if len(fields) == 0:
        raise ValueError('create_model() is called without fields specified')
    return MockModel(**{f: None for f in fields})


class MockOptions(object):
    def __init__(self, *field_names):
        self.load_fields(*field_names)
        self.get_latest_by = None

    def load_fields(self, *field_names):
        fields = {name: MockField(name) for name in field_names}

        for key in ('_forward_fields_map', 'parents', 'fields_map'):
            self.__dict__[key] = {}

            if key == '_forward_fields_map':
                for name, obj in fields.items():
                    self.__dict__[key][name] = obj

        for key in ('local_concrete_fields', 'concrete_fields', 'fields'):
            self.__dict__[key] = []

            for name, obj in fields.items():
                self.__dict__[key].append(obj)


class MockField(object):
    def __init__(self, field):
        for key in ('name', 'attname'):
            self.__dict__[key] = field
