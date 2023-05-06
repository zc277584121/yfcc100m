"""
Convert YFCC100m dataset shards into the datadings msgpack format.
"""
import io
import zipfile
from pathlib import Path
from multiprocessing.pool import ThreadPool
from math import ceil

from datadings.writer import FileWriter

from .tools import find_meta_shards
from .tools import WorkerBase
from .tools import DATAKEYS
from .tools import load_finished_shards
from .tools import shard_finished

import numpy as np
from PIL import Image
from tqdm import tqdm
from simplejpeg import decode_jpeg
from simplejpeg import decode_jpeg_header
from simplejpeg import encode_jpeg
from datadings.tools import yield_threaded


def encode_fast(
        arr,
        quality=85,
        colorspace='RGB',
        colorsubsampling='422',
        long_side=500,
        target_pixels=500*375,
):
    h, w = arr.shape[:2]
    # enforce full color resolution for small images
    if h*w <= 0.5*target_pixels:
        colorsubsampling = '444'
    # downscale big images
    if h*w > 1.5*target_pixels:
        s = max(w, h)
        r = long_side/s
        w, h = int(round(r*w)), int(round(r*h))
        pil = Image.fromarray(arr, 'RGB')
        arr = np.array(pil.resize((w, h), resample=Image.LANCZOS))
    return encode_jpeg(
        arr,
        quality=quality,
        colorspace=colorspace,
        colorsubsampling=colorsubsampling,
    )


def decode_fast(data):
    try:
        # decode JPEGs at reduced size for speedup
        return data, False, decode_jpeg(
            data,
            'gray',
            fastdct=True,
            fastupsample=True,
            min_height=100,
            min_width=100,
            min_factor=1,
        )
    except ValueError:
        # use pillow in case anything goes wrong
        # and re-encode the image
        bio = io.BytesIO(data)
        im = Image.open(bio)
        data = encode_fast(np.array(im.convert('RGB')))
        return data, True, np.array(im.convert('L'))


def validate_image(data):
    try:
        data, compressed, im = decode_fast(data)
        # if the compressed image is very small
        # and less than 5% of all lines have significant variance
        # the image is most likely garbage
        if len(data) < 20000 and np.percentile(im.var(0), 95) < 50:
            return None, False
        return data, compressed
    except (ValueError, IOError, OSError):
        return None, False


def sharp_path(indir, shard):
    return indir / (shard + '.zip')


class Converter(WorkerBase):
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
            compress,
            subsampling,
            target_size,
    ):
        super().__init__(
            shards,
            indir,
            metadir,
            outdir,
            kinds,
            filter_code,
            processes,
            threads,
        )
        self.compress = compress
        self.subsampling = subsampling
        self.target_size = target_size
        self.short_side = int(ceil(target_size / 4 * 3))
        self.target_pixels = target_size * self.short_side

    def yield_samples(self, shard):
        metadata = {
            meta['key']: meta
            for meta in self.prepare_metadata(shard)
        }
        with zipfile.ZipFile(sharp_path(self.indir, shard), 'r') as z:
            for file in z.infolist():
                key = Path(file.filename).stem
                if key in metadata:
                    sample = metadata.pop(key)
                    datakey = DATAKEYS[sample['marker']]
                    sample[datakey] = z.open(file).read()
                    yield sample

    def convert_file(self, sample):
        datakey = DATAKEYS[sample['marker']]
        # handle image recompression
        if sample['marker'] == 0:
            data, compressed = validate_image(sample[datakey])
            if data is None:
                return None
            if self.compress and not compressed:
                h, w, _, _ = decode_jpeg_header(data)
                # do not compress small images
                if h * w > 0.5 * self.target_pixels:
                    arr = decode_jpeg(
                        data,
                        min_width=self.short_side,
                        min_height=self.short_side,
                    )
                    data = encode_fast(
                        arr,
                        quality=self.compress,
                        colorsubsampling=self.subsampling,
                        target_pixels=self.target_pixels,
                        long_side=self.target_size,
                    )
            sample[datakey] = data
        return sample

    def convert_files(self, shard):
        with zipfile.ZipFile(sharp_path(self.indir, shard), 'r') as z:
            num_files = len(z.infolist())
        pool = ThreadPool(self.threads)
        gen = yield_threaded(self.yield_samples(shard))
        with self.position() as position:
            yield from self.tqdm(
                pool.imap(self.convert_file, gen),
                shard,
                position,
                length=num_files,
            )

    def convert_shard(self, shard):
        writer = FileWriter(
            self.outdir / (shard + '.msgpack'),
            overwrite=True,
            disable=True,
        )
        with writer:
            for sample in self.convert_files(shard):
                if sample:
                    writer.write(sample)
        return shard

    def convert_shards(self):
        with self.positioned(), self.pool() as pool:
            yield from pool.imap_unordered(self.convert_shard, self.shards)


def convert_parallel(
        indir,
        metadir,
        outdir,
        shards=None,
        kinds=(0, 1),
        filter_code='lambda x: True',
        processes=8,
        threads=8,
        overwrite=False,
        compress=85,
        subsampling='422',
        target_size=500,
):
    if not shards:
        shards = find_meta_shards(metadir)
        shards = {shard for shard in shards
                  if sharp_path(indir, shard).exists()}
        finished_shards = load_finished_shards(outdir)
        if not overwrite:
            shards -= finished_shards
        shards = sorted(shards)

    converter = Converter(
        shards, indir, metadir, outdir, kinds, filter_code, processes, threads,
        compress, subsampling, target_size,
    )
    gen = tqdm(
        converter.convert_shards(),
        desc='total',
        total=len(shards),
        smoothing=0,
        position=0,
    )
    for shard in gen:
        shard_finished(shard, outdir)


def main():
    from datadings.tools.argparse import make_parser
    from datadings.tools.argparse import argument_outdir

    parser = make_parser(
        __doc__,
        no_confirm=False,
        skip_verification=False,
        shuffle=False,
        outdir=False,
    )
    parser.add_argument(
        '-m', '--metadir',
        type=str,
        default='.',
        metavar='METADIR',
        help='Metadata directory. Defaults to indir.'
    )
    argument_outdir(parser)
    parser.add_argument(
        '-p', '--processes',
        default=8,
        type=int,
        help='Number of shards downloaded in parallel.'
    )
    parser.add_argument(
        '-t', '--threads',
        default=8,
        type=int,
        help='Number of threads to download each shard.'
    )

    def _kind(v):
        return {'images': 0, 'videos': 1}[v]

    # noinspection PyTypeChecker
    parser.add_argument(
        '--kind',
        nargs='+',
        type=_kind,
        choices=('images', 'videos'),
        default=(0,),
        help='Kinds of files to download. Defaults to images.'
    )
    parser.add_argument(
        '--shard',
        default=(),
        type=str,
        nargs='+',
        help='Specify individual shards to download.'
    )

    def _evalable(v):
        eval(v)
        return v
    parser.add_argument(
        '--filter',
        type=_evalable,
        default='lambda x: True',
        help='Lambda function to select samples.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite finished shards.'
    )
    parser.add_argument(
        '--compress',
        type=int,
        default=None,
        help='Re-compress images with this quality 85.'
    )
    parser.add_argument(
        '--subsampling',
        type=str,
        choices=('444', '422', '411', '420'),
        default='422',
        help='Use this color subsampling method when compressing.'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=500,
        help='Longer side of compressed images',
    )
    args = parser.parse_args()
    indir = Path(args.indir)
    metadir = Path(args.metadir or args.indir)
    outdir = Path(args.outdir or args.indir)

    try:
        convert_parallel(
            indir,
            metadir,
            outdir,
            args.shard,
            args.kind,
            args.filter,
            args.processes,
            args.threads,
            args.overwrite,
            args.compress,
            args.subsampling,
            args.target_size,
        )
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
