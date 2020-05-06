import zhconv
import os

for path, dir_list, file_list in os.walk('zh'):
    for file_name in file_list:
        src_path = os.path.join(path, file_name)
        f = open(src_path, 'r', encoding='utf8')
        content = ''.join(f.readlines())

        dst_dir = path.replace('zh', 'zh_hant')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, file_name)
        f_ = open(dst_path, 'w', encoding='utf8')
        f_.write(zhconv.convert_for_mw(content, 'zh-tw'))
        print(src_path + ' -> ' + dst_path)

        dst_dir = path.replace('zh', 'zh_hans')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, file_name)
        f_ = open(dst_path, 'w', encoding='utf8')
        f_.write(zhconv.convert_for_mw(content, 'zh-hans'))
        print(src_path + ' -> ' + dst_path)