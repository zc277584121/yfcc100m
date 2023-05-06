"""
Download the YFCC100m dataset from AWS. Requires the metadata files produced by
the convert_metadata tool as input. Files are stored in 4096 ZIP-files. For
images shards are ~27000 files each and occupy 11 TiB. The download can be
stopped at any time and will ignore fully downloaded shards when resumed.

WARNING:

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
import io
import zipfile
from pathlib import Path
import threading as th
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
from datadings.tools.cached_property import cached_property

from download_repair_error import hex_range
from vars import BUCKET_NAME
from tools import find_meta_shards
from tools import get_possible_keys
from tools import WorkerBase
from tools import load_finished_shards
from tools import shard_finished


class Downloader(WorkerBase):
    @cached_property
    def bucket(self):
        loc = th.local()
        if not hasattr(loc, 'bucket'):
            loc.session = boto3.Session()
            loc.bucket = loc.session.resource('s3').Bucket(BUCKET_NAME)
        return loc.bucket

    def file_exists(self, sample):
        for _, key in get_possible_keys(sample):
            try:
                self.bucket.Object(key).load()
                return True
            except ClientError:
                pass
        return False

    def download_file(self, sample):
        for canonical_key, key in get_possible_keys(sample):
            try:
                bio = io.BytesIO()
                for try_time in range(20):
                    try:
                        self.bucket.download_fileobj(key, bio)
                        break
                    except:
                        continue
                return key, bio.getvalue()
            except ClientError:
                pass
        else:
            # noinspection PyUnboundLocalVariable
            return canonical_key, None

    def download_files(self, shard, files):
        pool = ThreadPool(self.threads)
        with self.position() as position:
            yield from self.tqdm(
                pool.imap(self.download_file, files),
                shard,
                position,
                length=len(files),
            )

    def download_shard(self, shard):
        metadata = self.prepare_metadata(shard)
        errpath = self.outdir / (shard + '.err')
        with zipfile.ZipFile(self.outdir / (shard + '.zip'), 'w') as z:
            for key, data in self.download_files(shard, metadata):
                if data:
                    z.writestr(key, data)
                else:
                    with errpath.open('at', encoding='utf-8') as err:
                        err.write(key+'\n')
        return shard

    def download_shards(self):
        with self.positioned(), self.pool() as pool:
            yield from pool.imap_unordered(self.download_shard, self.shards)

    def find_missing(self, shard, metadata, existing_keys):
        missing = []
        # for each sample, check if any of the possible keys is in the shard
        # if not, check if file exists on AWS and add to list of missing samples
        with self.position() as position:
            for sample in self.tqdm(metadata, shard, position):
                for _, key in get_possible_keys(sample):
                    if key in existing_keys:
                        break
                else:
                    if self.file_exists(sample):
                        missing.append(sample)
        return missing

    def check_shard(self, shard):
        shardpath = self.outdir / (shard + '.zip')
        if not shardpath.exists():
            return shard
        # get named of all files in shard
        with zipfile.ZipFile(shardpath, 'r') as z:
            existing_keys = set(z.namelist())
        metadata = self.prepare_metadata(shard)
        missing = self.find_missing(shard, metadata, existing_keys)
        if missing:
            shardpath = self.outdir / (shard + '_missing.zip')
            with zipfile.ZipFile(shardpath, 'w') as z:
                for key, data in self.download_files(shard, missing):
                    if data:
                        z.writestr(key, data)
        return shard

    def check_shards(self):
        with self.positioned(), self.pool() as pool:
            yield from pool.imap_unordered(self.check_shard, self.shards)


def download_parallel(
        indir,
        outdir,
        shards=None,
        kinds=(0, 1),
        filter_code='lambda x: True',
        processes=8,
        threads=8,
        # overwrite=False,
        check=False,
        shard_start='000',
        shard_end='fff'
):
    if not shards:
        shards = find_meta_shards(indir)
        shard_range = hex_range(shard_start, shard_end)

        shards = list(set(shards) & set(shard_range))
        # finished_shards = load_finished_shards(outdir)
        # if check:
        #     shards = finished_shards
        # elif not overwrite:
        #     shards -= finished_shards
        shards = sorted(shards)

    downloader = Downloader(
        shards, indir, indir, outdir, kinds, filter_code, processes, threads
    )
    gen = tqdm(
        downloader.check_shards() if check else downloader.download_shards(),
        desc='total',
        total=len(shards),
        smoothing=0,
        position=0,
    )
    for shard in gen:
        if not check:
            pass
            # shard_finished(shard, outdir)


def main():
    from datadings.tools.argparse import make_parser

    parser = make_parser(
        __doc__,
        no_confirm=False,
        skip_verification=False,
        shuffle=False,
    )
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

    parser.add_argument(
        '--start',
        type=str,
        default='000',
        help='start shard index'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='fff',
        help='end shard index'
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
    # parser.add_argument(
    #     '--overwrite',
    #     action='store_true',
    #     help='Overwrite finished shards.'
    # )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check existing shard and attempt to add missing files.'
    )
    args = parser.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) or args.indir

    try:
        print(indir)
        print(outdir)
        print(args.shard)
        print(args.kind)
        print(args.filter)
        print(args.processes)
        print(args.threads)
        print(args.check)
        print(args.start)
        print(args.end)
        download_parallel(
            indir,
            outdir,
            args.shard,
            args.kind,
            args.filter,
            args.processes,
            args.threads,
            # args.overwrite,
            args.check,
            args.start,
            args.end
        )

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()


# (.yfcc100m) zilliz@zilliz:/mnt/disk2/yfcc100m$
#python -m yfcc100m.download --threads=32 metadata -o images
#


# (.yfcc100m) zilliz@zilliz:/mnt/disk2/yfcc100m$ python -m yfcc100m.download --threads=32 metadata -o images
# total:   0%|▏                                                      | 9/3839 [1:27:55<623:40:19, 586.22s/it]
# multiprocessing.pool.RemoteTraceback: █████████████████▊             | 19962/25983 [38:52<11:43,  8.56it/s]
# """09:  78%|████████████████████████████████████████████▌            | 20245/25878 [39:02<10:51,  8.64it/s]
# Traceback (most recent call last):
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connection.py", line 174, in _new_conn
#     conn = connection.create_connection(██████████████▊              | 19460/25922 [38:54<12:55,  8.34it/s]
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/util/connection.py", line 72, in create_connect
# ion11:  60%|██████████████████████████████████▎                      | 16578/27571 [32:26<21:30,  8.52it/s]
#     for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):01/25973 [38:58<11:53,  8.51it/s]
#   File "/usr/local/anaconda3/lib/python3.9/socket.py", line 954, in getaddrinfo
#     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
# socket.gaierror: [Errno -2] Name or service not known
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/httpsession.py", line 455, in send
#     urllib_response = conn.urlopen(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connectionpool.py", line 787, in urlopen
#     retries = retries.increment(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/util/retry.py", line 525, in increment
#     raise six.reraise(type(error), error, _stacktrace)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/packages/six.py", line 770, in reraise
#     raise value
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connectionpool.py", line 703, in urlopen
#     httplib_response = self._make_request(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connectionpool.py", line 386, in _make_request
#     self._validate_conn(conn)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connectionpool.py", line 1042, in _validate_con
# n
#     conn.connect()
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connection.py", line 363, in connect
#     self.sock = conn = self._new_conn()
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/urllib3/connection.py", line 186, in _new_conn
#     raise NewConnectionError(
# urllib3.exceptions.NewConnectionError: <botocore.awsrequest.AWSHTTPSConnection object at 0x7f57d6a7d280>: Failed to establish a new connection: [Errno -2] Name or service not known
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/usr/local/anaconda3/lib/python3.9/multiprocessing/pool.py", line 125, in worker
#     result = (True, func(*args, **kwds))
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 92, in download_shard
#     for key, data in self.download_files(shard, metadata):
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 81, in download_files
#     yield from self.tqdm(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/tqdm/std.py", line 1178, in __iter__
#     for obj in iterable:
#   File "/usr/local/anaconda3/lib/python3.9/multiprocessing/pool.py", line 870, in next
#     raise value
#   File "/usr/local/anaconda3/lib/python3.9/multiprocessing/pool.py", line 125, in worker
#     result = (True, func(*args, **kwds))
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 70, in download_file
#     self.bucket.download_fileobj(key, bio)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/boto3/s3/inject.py", line 837, in bucket_download_fileobj
#     return self.meta.client.download_fileobj(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/boto3/s3/inject.py", line 795, in download_fileobj
#     return future.result()
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/s3transfer/futures.py", line 103, in result
#     return self._coordinator.result()
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/s3transfer/futures.py", line 266, in result
#     raise self._exception
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/s3transfer/tasks.py", line 269, in _main
#     self._submit(transfer_future=transfer_future, **kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/s3transfer/download.py", line 354, in _submit
#     response = client.head_object(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/client.py", line 530, in _api_call
#     return self._make_api_call(operation_name, kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/client.py", line 943, in _make_api_call
#     http, parsed_response = self._make_request(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/client.py", line 966, in _make_request
#     return self._endpoint.make_request(operation_model, request_dict)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/endpoint.py", line 119, in make_request
#     return self._send_request(request_dict, operation_model)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/endpoint.py", line 202, in _send_request
#     while self._needs_retry(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/endpoint.py", line 354, in _needs_retry
#     responses = self._event_emitter.emit(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/hooks.py", line 412, in emit
#     return self._emitter.emit(aliased_event_name, **kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/hooks.py", line 412, in emit
#     return self._emitter.emit(aliased_event_name, **kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/hooks.py", line 256, in emit
#     return self._emit(event_name, kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/hooks.py", line 239, in _emit
#     response = handler(**kwargs)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 207, in __call__
#     if self._checker(**checker_kwargs):
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 284, in __call__
#     should_retry = self._should_retry(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 320, in _should_retry
#     return self._checker(attempt_number, response, caught_exception)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 363, in __call__
#     checker_response = checker(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 247, in __call__
#     return self._check_caught_exception(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/retryhandler.py", line 416, in _check_caught_exception
#     raise caught_exception
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/endpoint.py", line 281, in _do_get_response
#     http_response = self._send(request)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/endpoint.py", line 377, in _send
#     return self.http_session.send(request)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/botocore/httpsession.py", line 484, in send
#     raise EndpointConnectionError(endpoint_url=request.url, error=e)
# botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL: "https://multimedia-commons.s3.us-west-2.amazonaws.com/data/images/10b/c59/10bc598be4bc1fb543ba389a9858f9fa.jpg"
#
# The above exception was the direct cause of the following exception:
#
# Traceback (most recent call last):
#   File "/usr/local/anaconda3/lib/python3.9/runpy.py", line 197, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "/usr/local/anaconda3/lib/python3.9/runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 257, in <module>
#     main()
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 241, in main
#     download_parallel(
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 170, in download_parallel
#     for shard in gen:
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/tqdm/std.py", line 1178, in __iter__
#     for obj in iterable:
#   File "/home/zilliz/fzliu/.yfcc100m/lib/python3.9/site-packages/yfcc100m/download.py", line 102, in download_shards
#     yield from pool.imap_unordered(self.download_shard, self.shards)
#   File "/usr/local/anaconda3/lib/python3.9/multiprocessing/pool.py", line 870, in next
#     raise value
# botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL: "https://multimedia-commons.s3.us-west-2.amazonaws.com/data/images/10b/c59/10bc598be4bc1fb543ba389a9858f9fa.jpg"