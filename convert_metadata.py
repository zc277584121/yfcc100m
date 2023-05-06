"""
This tool performs the first step of the download process. The SQLite database
available from AWS must be converted into a format that can be used to quickly
and efficiently download and store shards of samples.

The result of this conversion is 4096 files stored in the output directory
with a combined size of ~20GiB. You can check that the conversion went OK by
counting the number of exported samples with:
    zcat <meta dir>/*.gz | wc -l
It should show 100,000,000 lines across all files.

IMPORTANT: After progress reaches 100% all remaining cached samples are flushed
           to files before the program exits. This may take a few minutes.

IMPORTANT: Needs AWS credentials to work. Create an AWS account (credit card
           required, but will not be charged). You can install aws-cli and run
           aws configure or create these files manually:

           ~/.aws/config
           [default]
           region = us-west-2
           s3 =
               max_concurrent_requests = 100
               max_queue_size = 10000
               max_bandwidth = 50MB/s
               multipart_threshold = 64MB
               multipart_chunksize = 16MB
               use_accelerate_endpoint = false

           ~/.aws/credentials
           [default]
           aws_access_key_id = <key>
           aws_secret_access_key = <secret>
"""
import gzip
import json
import sqlite3
from multiprocessing import Process
from hashlib import md5
from collections import defaultdict
from pathlib import Path

import boto3
from tqdm import tqdm
from datadings.tools import yield_threaded

from .vars import FILES
from .vars import TOTAL
from .vars import AWS_KEY
from .vars import BUCKET_NAME
from .tools import IterableQueue
from .tools import make_meta_shard_names


def download_db(files):
    outpath = Path(files['db']['path'])
    if outpath.exists():
        return
    name = outpath.name
    session = boto3.Session()
    bucket = session.resource('s3').Bucket(BUCKET_NAME)
    obj = bucket.Object(AWS_KEY)
    total = obj.content_length
    with tqdm(total=total, unit='B', unit_scale=True, desc=name) as p:
        obj.download_file(str(outpath), Callback=lambda n: p.update(n))


BYTE_MAP = {'%02x' % v: '%x' % v for v in range(256)}


# noinspection PyDefaultArgument
def yfcc_hash(url, __bm=BYTE_MAP):
    h = md5(url.encode('utf-8')).hexdigest()
    return ''.join(__bm[h[x:x+2]] for x in range(0, 32, 2))


def get_cols(path, table):
    conn = sqlite3.connect(path)
    result = conn.execute(f'select * from {table} limit 0').description
    return [c[0] for c in result]


def generate_rows_from_db(path, table):
    conn = sqlite3.connect(path)
    # get column names
    # some settings that hopefully speed up the queries
    conn.execute(f'PRAGMA query_only = YES')
    conn.execute(f'PRAGMA journal_mode = OFF')
    conn.execute(f'PRAGMA locking_mode = EXCLUSIVE')
    conn.execute(f'PRAGMA page_size = 4096')
    conn.execute(f'PRAGMA mmap_size = {4*1024*1024}')
    conn.execute(f'PRAGMA cache_size = 10000')
    # retrieve rows in order
    yield from conn.execute(f'select * from {table}')


def _int_none(v):
    return int(v) if v else None


def _float_none(v):
    return float(v) if v else None


def create_sample(row, cols):
    row = dict(zip(cols, row))
    row['key'] = yfcc_hash(row['downloadurl'])
    row['longitude'] = _float_none(row['longitude'])
    row['latitude'] = _float_none(row['latitude'])
    row['accuracy'] = _int_none(row['accuracy'])
    return row


def write_buckets(queue):
    gen = yield_threaded(
        (outdir, key, '\n'.join(json.dumps(sample) for sample in samples))
        for outdir, key, samples in queue
    )
    for outdir, key, chunk in gen:
        path = outdir / (key + '.gz')
        with gzip.open(path, 'at', encoding='utf-8', compresslevel=1) as fp:
            fp.write(chunk)
            fp.write('\n')


def clear_outdir(outdir: Path, __shards=make_meta_shard_names()):
    for path in outdir.glob('*'):
        if path.name in __shards:
            path.unlink()


def convert_metadata(files, outdir, chunk_size=100):
    table = 'yfcc100m_dataset'
    path = files['db']['path']
    cols = get_cols(path, table)

    write_queue = IterableQueue(maxsize=4096)
    buckets = defaultdict(list)
    writer = Process(target=write_buckets, args=(write_queue,))
    writer.start()
    gen = yield_threaded(generate_rows_from_db(path, table))

    clear_outdir(outdir)

    try:
        for row in tqdm(gen, total=TOTAL['db']):
            sample = create_sample(row, cols)
            key = sample['key'][:3]
            buckets[key].append(sample)
            if len(buckets[key]) >= chunk_size:
                write_queue.put((outdir, key, buckets.pop(key)))
        # send remaining buckets to be written
        for key, bucket in buckets.items():
            write_queue.put((outdir, key, bucket))
    finally:
        write_queue.close()


def main():
    from datadings.tools.argparse import make_parser
    from datadings.tools import locate_files
    from datadings.tools import prepare_indir

    parser = make_parser(__doc__, no_confirm=False, shuffle=False)
    args = parser.parse_args()
    args.outdir = Path(args.outdir or args.indir)

    files = locate_files(FILES, args.indir)
    # download DB file with AWS tools
    download_db(files)
    # check indir correctness
    files = prepare_indir(FILES, args)

    try:
        convert_metadata(files, args.outdir)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
