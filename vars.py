AWS_URL_PREFIX = 'https://multimedia-commons.s3-us-west-2.amazonaws.com'
AWS_KEY = 'tools/etc/yfcc100m_dataset.sql'
FILES = {
    'db': {
        'url': AWS_URL_PREFIX+'/'+AWS_KEY,
        'path': 'yfcc100m_dataset.sql',
        'md5': 'b8e3fa7f40ee2f309d63b8de8d755495',
    },
}
TOTAL = {
    'db': 100_000_000,
}
BUCKET_NAME = 'multimedia-commons'
