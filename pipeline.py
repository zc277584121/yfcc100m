import os

from download_repair_error import hex_range

pre_start = '51'
pre_end = '5f'

# start = '500'
# end = '50f'


pre_list = [more[1:] for more in hex_range(pre_start, pre_end)]
# start_list = pre_list[:-1]
# end_list = pre_list[1:]
print(pre_list)

start_list = [pre + '0' for pre in pre_list]
end_list = [pre + 'f' for pre in pre_list]
print(start_list)
print(end_list)

python_interpreter = '/Users/zilliz/zhangchen_workspace/yfcc100m/.yfcc/bin/python'
for start, end in zip(start_list, end_list):

    cmd = f'{python_interpreter} download.py --threads=32 /Users/zilliz/yfcc100m_metadata -o images --start {start} --end {end}'
    print(cmd)
    os.system(cmd)

    cmd = f'{python_interpreter} extract_embeddings_v2.py --device_id -1 --batch_size 64 --start {start} --end {end}'
    print(cmd)
    os.system(cmd)


    feature_folder = f'features/features_{start[:2]}'
    print(cmd)
    os.system(cmd)

    cmd = f'mkdir {feature_folder}'
    print(cmd)
    os.system(cmd)
    cmd = f'mv features/features/{start[:2]}* {feature_folder}'
    print(cmd)
    os.system(cmd)

    cmd = f'kaggle datasets init -p {feature_folder}'
    print(cmd)
    os.system(cmd)
    cmd = f'kaggle datasets create -p {feature_folder}'
    print(cmd)
    os.system(cmd)


    # rm
    cmd = f'rm images/{start[:2]}*.zip'
    print(cmd)
    os.system(cmd)
    print('\n')