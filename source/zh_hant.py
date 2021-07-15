# import zhconv
import os
import re

for path, dir_list, file_list in os.walk('zh'):
    for file_name in file_list:
        src_path = os.path.join(path, file_name)
        f = open(src_path, 'r', encoding='utf8')
        content = ''.join(f.readlines())

        # dst_dir = path.replace('zh', 'zh_hant')
        # os.makedirs(dst_dir, exist_ok=True)
        # dst_path = os.path.join(dst_dir, file_name)
        # f_ = open(dst_path, 'w', encoding='utf8')
        # content_zh_hant = re.sub(r'\.\. _(.+?):', r'.. _zh_hant_\1:', content)
        # content_zh_hant = re.sub(r':ref:`(.+?) <(.+?)>`', r':ref:`\1 <zh_hant_\2>`', content_zh_hant)
        # content_zh_hant = re.sub(r':label: (.+?)', r':label: zh_hant_\1', content_zh_hant)
        # content_zh_hant = re.sub(r':eq:`(.+?)`', r':eq:`zh_hant_\1`', content_zh_hant)
        # f_.write(zhconv.convert_for_mw(content_zh_hant, 'zh-tw'))
        # print(src_path + ' -> ' + dst_path)

        dst_dir = path.replace('zh', 'zh_hans')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, file_name)
        f_ = open(dst_path, 'w', encoding='utf8')
        content_zh_hans = re.sub(r'\.\. _(.+?):', r'.. _zh_hans_\1:', content)
        content_zh_hans = re.sub(r':ref:`(.+?) <(.+?)>`', r':ref:`\1 <zh_hans_\2>`', content_zh_hans)
        content_zh_hans = re.sub(r':label: (.+?)', r':label: zh_hans_\1', content_zh_hans)
        content_zh_hans = re.sub(r':eq:`(.+?)`', r':eq:`zh_hans_\1`', content_zh_hans)
        # f_.write(zhconv.convert_for_mw(content_zh_hans, 'zh-hans'))
        f_.write(content_zh_hans)
        print(src_path + ' -> ' + dst_path)