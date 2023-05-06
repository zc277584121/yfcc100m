import gzip
import json
import itertools as it
from string import hexdigits
from queue import Empty
from operator import itemgetter
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.queues import Queue

from tqdm import tqdm
from datadings.tools.cached_property import cached_property


class Sentinel:
    pass


class IterableQueue(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize, ctx=mp.get_context())
        self._sentinel = object()

    def get(self, block=True, timeout=1):
        while True:
            try:
                return super().get(block=block, timeout=timeout)
            except Empty:
                pass

    def __iter__(self):
        while True:
            item = self.get()
            if isinstance(item, Sentinel):
                break
            yield item

    def close(self):
        self.put(Sentinel())


class WorkerBase:
    def __init__(
            self,
            shards,
            indir,
            metadir,
            outdir,
            kinds,
            filter_code,
            processes,
            threads,
    ):
        self.positions = None
        self.shards = shards
        self.indir = indir
        self.metadir = metadir
        self.outdir = outdir
        self.kinds = kinds
        self.filter_code = filter_code
        self.processes = processes
        self.threads = threads

    @cached_property
    def filter_fun(self):
        return eval(self.filter_code)

    @contextmanager
    def positioned(self):
        with mp.Manager() as manager:
            if self.processes > 0:
                self.positions = manager.Queue()
                for p in range(1, self.processes+1):
                    self.positions.put(p)
            yield

    @contextmanager
    def position(self):
        if self.positions:
            position = self.positions.get()
            try:
                yield position
            finally:
                self.positions.put(position)
        else:
            yield 0

    def prepare_metadata(self, shard):
        # load metadata and sort by key
        metadata = load_metadata(self.metadir, shard)
        metadata = filter(lambda s: s['marker'] in self.kinds, metadata)
        metadata = filter(self.filter_fun, metadata)
        return sorted(metadata, key=itemgetter('key'))

    def tqdm(self, iterable, desc, position, length=None):
        if length is None:
            length = len(iterable)
        return tqdm(
            iterable,
            desc=f'{"├" if position < self.processes else "└"} {desc}',
            total=length,
            smoothing=0,
            position=position,
            leave=position == 0,
        )

    def pool(self):
        write_lock = mp.Lock()
        tqdm.set_lock(write_lock)
        return mp.Pool(
            self.processes,
            initializer=tqdm.set_lock,
            initargs=(write_lock,),
            maxtasksperchild=1,
        )


def load_metadata(indir, shard):
    with gzip.open(indir / (shard + '.gz'), 'rt', encoding='utf-8') as fp:
        for line in fp:
            yield json.loads(line)


def load_finished_shards(outdir):
    try:
        with (outdir / 'finished_shards').open('rt', encoding='utf-8') as fp:
            return {line.strip('\n') for line in fp}
    except FileNotFoundError:
        return set()


def shard_finished(shard, outdir):
    with (outdir / 'finished_shards').open('at', encoding='utf-8') as fp:
        fp.write(shard+'\n')


DATAKEYS = 'image', 'video'
PREFIX = 'data/images', 'data/videos/mp4'
EXTENSION = 'jpg', 'mp4'


def get_possible_keys(
        sample,
        __prefix=PREFIX,
        __extension=EXTENSION,
):
    prefix = __prefix[sample['marker']]
    h = sample['key']
    stem = f"{prefix}/{h[:3]}/{h[3:6]}/{h}."
    canonical_key = stem + __extension[sample['marker']]
    yield canonical_key, canonical_key
    yield canonical_key, stem + sample['ext']


def make_shard_names():
    digits = set(hexdigits.lower())
    return {''.join(c) for c in it.product(digits, digits, digits)}


def make_meta_shard_names():
    return {name+'.gz' for name in make_shard_names()}


def make_output_shard_names():
    return {name+'.zip' for name in make_shard_names()}


def find_meta_shards(indir):
    return set(p.stem for p in indir.glob('*.gz')) & make_shard_names()


def find_output_shards(outdir):
    return set(p.stem for p in outdir.glob('*.zip')) & make_shard_names()
