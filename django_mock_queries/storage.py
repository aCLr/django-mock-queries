from collections import defaultdict
from copy import deepcopy
from typing import Type, Iterable
from uuid import uuid4

from django.db.models import Model, fields


class _Transaction:
    def __init__(self):
        self._id = uuid4()
        self.storage = defaultdict(dict)
        self.deleted = defaultdict(set)


def make_storage() -> _Transaction:
    return _Transaction()


class Storage:
    def __init__(self):
        self._global = make_storage()
        self._transactions = []
        self._need_rollback = False

    @property
    def current_transaction_id(self):
        return self.get_last()._id

    def set_rollback(self, value):
        self._need_rollback = value

    def begin_transaction(self):
        self._need_rollback = False
        self._transactions.append(make_storage())

    def commit_transaction(self):
        self._need_rollback = False
        last_transaction = self._transactions.pop()
        deleted = defaultdict(set)
        if not self._transactions:
            prev_transaction = self._global
        else:
            prev_transaction = self._transactions[-1]

        deleted.update({m: d.copy() for m, d in last_transaction.deleted.items()})
        for model, objects in last_transaction.storage.items():
            for obj_pk, obj in objects.items():
                if obj_pk not in deleted[model]:
                    prev_transaction.storage[model][obj_pk] = obj

    def rollback_transaction(self):
        self._transactions.pop()
        self._need_rollback = False

    def get_last(self):
        return self._transactions[-1] if self._transactions else self._global

    def get_object_from_storage(self, model: Type[Model], pk):
        for t in self._transactions[::-1]:
            if pk in t.deleted[model]:
                return None
            elif pk in t.storage[model]:
                return t.storage[model][pk]

    def get_model_objects(self, model):
        if self._transactions:
            last = self._transactions[-1]
            res = {t.pk: t for t in last.storage[model].values()}
        else:
            res = defaultdict(dict)
        deleted = set()
        for r in self._transactions[-2::-1]:
            deleted.update(r.deleted[model])
            for obj in r.storage[model].values():
                if obj.pk not in res and obj.pk not in deleted:
                    res[obj.pk] = _copy_object(obj)
        for obj in self._global.storage[model].values():
            if obj.pk not in res and obj.pk not in deleted:
                res[obj.pk] = _copy_object(obj)

        return list(res.values())


STORAGE = Storage()


def _set_obj_pk(obj):
    pk = obj.pk
    if pk is None:
        pk = obj._meta.pk.get_pk_value_on_save(obj)
    if pk is None:
        for klass in obj._meta.pk.__class__.__mro__:
            if klass in (fields.IntegerField,):
                pk = max(STORAGE.get_last().storage[obj.__class__].keys() or [0]) + 1
                break
    assert pk is not None
    existed = STORAGE.get_last().storage[obj.__class__].get(pk)
    if existed is not None and existed is not obj:
        raise ValueError('primary key %s for %s already registered' % (obj.pk, obj.__class__))
    obj.pk = pk


def add_to_storage(obj: Model) -> None:
    if obj.pk is None:
        _set_obj_pk(obj)
    STORAGE.get_last().storage[obj.__class__][obj.pk] = obj


def remove_from_storage(obj: Model) -> None:
    STORAGE.get_last().storage[obj.__class__].pop(obj.pk, None)
    STORAGE.get_last().deleted[obj.__class__].add(obj.pk)


def _copy_object(model_object):
    return model_object.__class__(**{f.attname: getattr(model_object, f.attname) for f in model_object._meta.fields})


def get_models_from_storage(model: Type[Model]) -> Iterable[Model]:
    return STORAGE.get_model_objects(model)


def get_obj_from_storage(model: Type[Model], pk) -> Model:
    return STORAGE.get_object_from_storage(model, pk)