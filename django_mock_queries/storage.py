from collections import defaultdict
from typing import Dict, Type, List, Any, Iterable

from django.db.models import Model, fields


def make_storage() -> Dict[Type[Model], Dict[Any, Model]]:
    return defaultdict(dict)


def copy_global_storage():
    return _storage.copy()

_storage = make_storage()


def _set_obj_pk(obj):
    pk = obj.pk
    if pk is None:
        pk = obj._meta.pk.get_pk_value_on_save(obj)
    if pk is None:
        for klass in obj._meta.pk.__class__.__mro__:
            if klass in (fields.IntegerField,):
                pk = max(_storage[obj.__class__].keys() or [0]) + 1
                break
    assert pk is not None
    existed = _storage[obj.__class__].get(pk)
    if existed is not None and existed is not obj:
        raise ValueError('primary key %s for %s already registered' % (obj.pk, obj.__class__))
    obj.pk = pk


def add_to_storage(obj: Model) -> None:
    if obj.pk is None:
        _set_obj_pk(obj)
    _storage[obj.__class__][obj.pk] = obj


def remove_from_storage(obj: Model) -> None:
    _storage[obj.__class__].pop(obj.pk, None)


def clear_storage():
    _storage.clear()


def get_models_from_storage(model: Type[Model]) -> Iterable[Model]:
    return _storage[model].values()