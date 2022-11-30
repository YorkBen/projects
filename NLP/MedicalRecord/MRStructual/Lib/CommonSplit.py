"""
通用病历结构化代码。
加载病历字段数据： ../data/template/all.json
处理说明：
  1. level 1为叶子节点字段，level 1以上的为父节点字段
  2. level 2以上 先检查DATE和DESC
  3. 两个以上空格作为字段切分符号
  4. 字段名后到冒号中间的括号，使用通用匹配，如“科室(包括入院时科别及转科科别): ”
"""
import re
import json
import logging
import time
import sys
import copy
import logging

# logging.basicConfig(level=logging.debug)
LOG_FILE = 'mylog.log'

file_handler = logging.FileHandler(LOG_FILE) #输出到文件
console_handler = logging.StreamHandler()  #输出到控制台
# file_handler.setLevel('DEBUG')     #error以上才输出到文件
# console_handler.setLevel('DEBUG')   #info以上才输出到控制台

fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
file_handler.setFormatter(formatter) #设置输出内容的格式
console_handler.setFormatter(formatter)

logger = logging.getLogger('updateSecurity')
logger.setLevel('DEBUG')     #设置了这个才会把debug以上的输出到控制台

# logger.addHandler(file_handler)    #添加handler
# logger.addHandler(console_handler)


sys.path.append('../Lib')

from RegexUtil import RegexUtil

class CommonSplit:
    def __init__(self, template_file=r'data/template/all.json'):
        self.utils = RegexUtil()
        self.key_brack_re = '([ ]*[(（][^(（）):：]+[）)])'
        self.load_template(template_file)

    def load_template(self, template_file):
        """
        加载正则匹配模板
        """
        template = None
        with open(template_file, encoding="utf-8") as f:
            template = json.load(f, strict=False)
            logger.debug('load template: %s' % str(template))
        self.template = sorted(template.items(), key=lambda x: x[0], reverse=True)

        # 匹配字典
        match_dict = {}
        for key, val in self.template:
            match_keys = []
            match_vals = copy.deepcopy(val)
            if "match" in val:
                match_keys = val["match"]
                match_vals.pop("match")
            else:
                match_keys = [key]

            match_vals['key'] = key
            match_vals['level'] = match_vals['level'] if 'level' in match_vals else 1
            for mk in sorted(match_keys, reverse=True):
                match_dict[mk] = match_vals
        self.match_dict = match_dict

        # 父节点
        parents_dict = {}
        for key, val in self.template:
            if key not in parents_dict:
                parents_dict[key] = []

            if "children" in val:
                for c in val['children']:
                    if c not in parents_dict:
                        parents_dict[c] = []
                    parents_dict[c].append(key)

        # 最顶层的节点的上层为root
        for key, val in parents_dict.items():
            if len(val) == 0 and not key == 'root':
                parents_dict[key] = ['root']
            else:
                parents_dict[key] = self.sort_keys_by_level(val)
                match_dict[key]['parents'] = parents_dict[key].copy()
        self.parents_dict = parents_dict

        # 匹配正则表达式
        match_keys = []
        for key in sorted(self.match_dict.keys(), key=lambda x: [len(x), x], reverse=True):
            match_keys.append('(' + key + ')' + self.key_brack_re + '?')
        self.match_keys_str = "(" + ")|(".join([key for key in match_keys]) + ")"


    def sort_keys_by_level(self, keys):
        """
        根据keys的level从小到大排序
        """
        keys_level = sorted([(p, self.match_dict[p]['level']) for p in keys], key=lambda x: x[1])
        return [p for p, l in keys_level]

    def write_result(self, file_path, data=None):
        """
        将Json数据写入文件
        """
        with open(file_path, "w", encoding="utf-8") as f:
            if data is None:
                f.write(json.dumps(self.results, indent=1, separators=(',', ':'), ensure_ascii=False))
            else:
                f.write(json.dumps(data, indent=1, separators=(',', ':'), ensure_ascii=False))


    # def post_process_txt(self, val):
    #     if len(val) > 0 and val[0] in [':', '：']:
    #         val = val[1:]
    #
    #     # 去除病历中的[***]内容
    #     val = self.utils.remove_squarebracket_cnt(val)
    #     # 去掉包含在移除键数组中的key
    #     # val = self.utils.remove_after_keys(val, self.remove_keys)
    #     # 去掉末尾的（
    #     val = re.sub(r'（\s*\n?$', '', val)
    #     # 去掉一个（
    #     val = re.sub(r'^\s*）\s*\n?$', '', val)
    #     # 去掉末尾的序号1、2、等
    #     val = re.sub(r'[1-9]+[、. ]+$', '', val)
    #     # 去掉;;;
    #     val = re.sub(r';[; ]+;', '', val)
    #
    #     return val.strip()

    def process_val_field(self, key, val_str):
        """
        处理字段的值，如果是叶子节点，返回一个值；如果是非叶子节点，返回字典。
        """
        def comp_raw_str(new_str, raw_str):
            p_str = self.utils.format_by_type(self.utils.format_by_type(raw_str, "text").replace(new_str, ''), 'text')
            if len(p_str) > 10:
                return p_str
            else:
                return None

        level = self.match_dict[key]['level']
        val_type = self.match_dict[key]['valtype'] if 'valtype' in self.match_dict[key] else 'text'

        if val_type == 'regex':
            val_regex = self.match_dict[key]['regex']
            val_str_ = self.utils.format_by_type(val_str, val_type, val_regex)
            return {'desc': val_str_}, comp_raw_str(val_str_, val_str)
        elif val_type == 'date_desc':
            val_date, val_str_ = self.utils.format_by_type(val_str, val_type)
            return {'date': val_date, 'desc': val_str_}, comp_raw_str(val_str_, val_str)
        else:
            val_str_ = self.utils.format_by_type(val_str, val_type)
            if val_type != 'text':
                return val_str_, comp_raw_str(val_str_, val_str)
            else:
                return {'desc': val_str_}, comp_raw_str(val_str_, val_str)


    def get_ancestors(self, key):
        """
        获取key的所有祖先节点
        """
        results = self.parents_dict[key].copy()
        for parent in self.parents_dict[key]:
            if parent != 'root':
                results.extend(self.get_ancestors(parent))

        return results


    def process_exclude(self, match_results):
        """
        确定exclude信息
        """
        match_results_ = []
        logger.debug('processing exclude...')
        for idx, mr in enumerate(match_results):
            logger.debug('key: [%s], parents: %s' % (mr['key'], mr['parents']))
            if mr['exclude'] == 1:
                logger.debug('exclude')
                match_results_.append(mr)
            # elif len(match_results_) == 0:
            #     logger.debug('first one, continue..')
            #     continue
            elif len(match_results_) > 0 and len(mr['val_raw_str']) > 0 and mr['val_raw_str'][0] in ['。', '；', ';']:
                logger.debug('ends with 。 append text')
                match_results_[-1]['val_end'] = mr['val_end']
                match_results_[-1]['val_raw_str'] = match_results_[-1]['val_raw_str'] + mr['match_str'] + mr['val_raw_str']
            else:
                logger.debug('not exclude..')
                pset = set(mr['parents'])
                pre_match_parent = False
                pre_match_brother = False
                post_match_child = False
                post_match_brother = False
                # 向前检查当前节点的祖先节点
                # if len(match_results_) > 0:
                #     logger.debug('searching parents forward...')
                #     for k in range(len(match_results_)-1, -1, -1):
                #         mr_k = match_results_[k]
                #         logger.debug('forward key: %s' % mr_k['key'])
                #         if mr_k['key'] in pset:
                #             logger.debug('parent matched!')
                #             match_parent = True
                #             break
                #         else:
                #             pset = set(self.get_ancestors(mr_k['key'])).intersection(pset)
                #             if 'root' in pset:
                #                 pset.remove('root')
                #             logger.debug('parent intersection: %s' % pset)
                #             if len(pset) == 0:
                #                 logger.debug('break loop.')
                #                 break
                #             else:
                #                 match_common_parent = match_common_parent + 1
                #                 logger.debug('match parent')

                if len(match_results_) > 0 and \
                    match_results_[-1]['exclude'] == 1:
                    pre_key = match_results_[-1]['key']
                    # 向前匹配父节点
                    if pre_key in pset:
                        logger.debug('pre match parent')
                        mr['parents'] = [match_results_[-1]['key']]
                        mr['exclude'] = 1
                        match_results_.append(mr)
                        continue

                    # 向前匹配兄弟节点
                    if set(match_results_[-1]['parents']) == pset:
                        logger.debug('pre match brother')
                        mr['exclude'] = 1
                        match_results_.append(mr)
                        continue

                if idx < len(match_results) - 1 and \
                    match_results[idx+1]['exclude'] == 1:
                    # 向后检查节点的祖先节点，检查一个
                    post_key = match_results[idx+1]['key']
                    if mr['key'] in self.get_ancestors(post_key):
                        logger.debug('post match child')
                        mr['exclude'] = 1
                        match_results_.append(mr)
                        continue

                    # 向后检查一个兄弟节点，需直接父节点集合相同
                    if pset == set(match_results[idx+1]['parents']):
                        logger.debug('post match brother')
                        mr['exclude'] = 1
                        match_results_.append(mr)
                        continue

                # 写结果
                if len(match_results_) > 0:
                    logger.debug('not matched, append to pre item.')
                    match_results_[-1]['val_end'] = mr['val_end']
                    match_results_[-1]['val_raw_str'] = match_results_[-1]['val_raw_str'] + mr['match_str'] + mr['val_raw_str']

        return match_results_


    def update_val(self, oldval, newval):
        """
        跟新结构化数据。如果之前数组中存在该键、值，按照下述规则更新。
        """
        if isinstance(oldval, str) and isinstance(newval, str):
            if oldval != newval and len(newval) > len(oldval):
                return newval
            else:
                return oldval
        elif isinstance(oldval, dict) and isinstance(newval, dict):
            for k, v in newval.items():
                if k not in oldval:
                    oldval[k] = v
                elif k == "desc":
                    oldval[k] = oldval[k] + v
                else:
                    oldval[k] = self.update_val(oldval[k], v)
            return oldval
        else:
            raise TypeError("oldval: %s and newval: %s type not match!" % (oldval, newval))

    def stack_oper(self, node_stack, pop_num, key, val):
        """
        操作堆栈，从末尾第end_idx个位置添加key: val作为match_key的子节点，并作为最后的节点。
        end_idx从0开始
        """
        logger.debug('stack pop and append...')
        logger.debug('stack pop num: [%d]' % pop_num)
        for k in range(0, pop_num):
            node_stack.pop()

        # 加入当前节点到栈
        logger.debug('stack append: [%s, %s]' % (key, val))
        if key == node_stack[-1][0]:
            logger.debug('key in stack as last node, compare values..')
            val_ = node_stack[-1][1]
            logger.debug('old value: %s, new value: %s' % (val_, val))
            up_val = self.update_val(val_, val)
            logger.debug('updated value: %s' % up_val)
            node_stack[-1][1] = up_val
        else:
            if key in node_stack[-1][1]:
                logger.debug('key in stack as last node child, compare values..')
                val_ = node_stack[-1][1][key]
                logger.debug('old value: %s, new value: %s' % (val_, val))
                up_val = self.update_val(val_, val)
                logger.debug('updated value: %s' % up_val)
                node_stack[-1][1][key] = up_val
            else:
                node_stack[-1][1][key] = val

            node_stack.append([key, node_stack[-1][1][key]])

        return node_stack


    def stack_append(self, node_stack, match_key, key, val, max_match=10):
        """
        检查match_key是否在栈的最后两个里面，如果在，将key, val插入相应位置
        """
        processed = False
        for k in range(-1, -(min(len(node_stack), max_match)+1), -1):
            if match_key == node_stack[k][0]:
                logger.debug('stack matched! Index: [%d], stack key: [%s]' % (k, node_stack[k][0]))
                node_stack = self.stack_oper(node_stack, -k - 1, key, val)

                processed = True
                logger.debug('processed! break loop')
                break
            else:
                logger.debug('stack not match, Index: [%d], stack key: [%s]' % (k, node_stack[k][0]))

        return processed, node_stack


    def struc_results(self, match_results, root_node):
        """
        结构化结果
        """
        logger.debug('structring results...')
        results = {root_node: {'desc': ''}}
        node_stack = [[root_node, results[root_node]]] # 节点键和节点引用
        for idx, mr in enumerate(match_results):
            key = mr['key']
            val, post_str = self.process_val_field(key, mr['val_raw_str'])
            level = self.match_dict[key]['level']
            # pset = set(mr['parents'])
            logger.debug('processing item: [key: %s], [level: %s], [raw str: %s], [str: %s], [post str: %s], [parents: %s]' % (key, level, mr['val_raw_str'], val, post_str, mr['parents']))
            processed = False

            ### 如果当前节点的文字处理后有很多，而且下一个节点不是兄弟节点，则添加到父节点的desc上。

            # 下一个节点是否兄弟节点
            # has_next_brother = False if idx < len(match_results) - 1 and len(set(mr['parents']).intersection(set(match_results[idx+1]['parents']))) == 0 else True

            # 当前节点在队列中
            processed, node_stack = self.stack_append(node_stack, key, key, val)

            # 当前节点的父节点在队列中
            if not processed:
                for parent in mr['parents']:
                    logger.debug('checking parent: [%s] in stack...' % parent)
                    processed, node_stack = self.stack_append(node_stack, parent, key, val)
                    if processed:
                        # 追加文字到父节点
                        if post_str is not None: # not has_next_brother and
                            node_stack[-2][1]['desc'] = node_stack[-2][1]['desc'] + post_str
                        break

            # 当前节点的父节点的父节点在队列中
            # if not processed:
            #     logger.debug('checking parent`s parent in stack...')
            #     pset = set(parents)
            #     pset_len, match_results_len = len(pset), len(match_results)
            #     k = idx - 1
            #     logger.debug('current parents: %s' % pset)
            #     while pset_len > 0 and k < match_results_len:
            #         if pset_len == 1:
            #             logger.debug('parents set length: [%d] == 1' % len(pset))
            #             parent = list(pset)[0]
            #             for pparent in self.parents_dict[parent]:
            #                 if pparent == 'root' and parent == '体格检查':
            #                     continue
            #                 logger.debug('checking parent`s parent: [%s] in stack...' % pparent)
            #                 processed, node_stack = self.stack_append(node_stack, pparent, parent, {'desc': ''})
            #                 if processed:
            #                     logger.debug('append current node to stack')
            #                     node_stack = self.stack_oper(node_stack, 0, key, val)
            #                     break
            #             break
            #         else:
            #             logger.debug('parents set length: [%d] > 1' % len(pset))
            #             while k < 0 or k == idx:
            #                 k = k + 1
            #             logger.debug('Make uniqe parents set, step: [%d]' % k)
            #             logger.debug('neighbor parents: %s' % match_results[k]["parents"])
            #             pset = set(match_results[k]["parents"]).intersection(pset)
            #             logger.debug('merging results: %s' % pset)
            #             pset_len = len(pset)
            #             k = k + 1
                #
                # if pset_len == 0:
                #     logger.debug('parents set length: [%d] == 0, break loop' % len(pset))


            # 当前节点和下一个节点的共同父节点的父节点在栈中
            if not processed and key != '其他' and idx < len(match_results) - 1:
                logger.debug('checking parent`s parent in stack...')
                p2set = set(mr['parents']).intersection(set(match_results[idx+1]['parents']))
                for p2arent in self.sort_keys_by_level(list(p2set)):
                    logger.debug('parent[%s]`s parents: %s' % (p2arent, self.parents_dict[p2arent]))
                    for pparent in self.parents_dict[p2arent]:
                        if pparent == 'root' and p2arent == '体格检查':
                            continue
                        logger.debug('checking parent[%s]`s parent[%s]:  in stack...' % (p2arent, pparent))
                        processed, node_stack = self.stack_append(node_stack, pparent, p2arent, {'desc': ''})
                        if processed:
                            logger.debug('append current node to stack')
                            node_stack = self.stack_oper(node_stack, 0, key, val)
                            break
                    if processed:
                        break

            # 还未处理，就将节点文本拼接到上一个节点字段
            if not processed:
                logger.debug('append current node to leading node in stack...')
                if isinstance(node_stack[-1][1], dict):
                    node_stack[-1][1]['desc'] = node_stack[-1][1]['desc'] + mr['match_str'] + mr['val_raw_str']

            # logger.debug('stack now: %s' % node_stack)

        return results


    def process(self, text, root_node='root'):
        """
        主函数
        """
        text = self.utils.remove_squarebracket_cnt(text)

        match_results = []
        for match in re.finditer(self.match_keys_str, text):
            match_key = re.sub(self.key_brack_re, '', match.group(0))
            start, end = match.span()
            key_exclude = self.match_dict[match_key]['exclude'] if 'exclude' in self.match_dict[match_key] else 1
            if key_exclude == 0:
                if len(text) > end:
                    if text[end] in ['\t', '\n', ':', '：']: #, '(', '（', ' ',
                        key_exclude = 1

            match_results.append({
                'match_str': match.group(0),
                'match_key': match_key,
                'key': self.match_dict[match_key]['key'],
                'key_start': start,
                'key_end': end,
                'exclude': key_exclude,
                'parents': self.parents_dict[self.match_dict[match_key]['key']]
            })

        for idx in range(len(match_results)-1):
            match_results[idx]['val_start'] = match_results[idx]['key_end']
            match_results[idx]['val_end'] = match_results[idx+1]['key_start']
            match_results[idx]['val_raw_str'] = text[match_results[idx]['val_start']:match_results[idx]['val_end']]
            # match_results[idx]['val_str'] = self.post_process_txt(match_results[idx]['val_raw_str'])

        match_results[-1]['val_start'] = match_results[-1]['key_end']
        match_results[-1]['val_end'] = len(text)
        match_results[-1]['val_raw_str'] = text[match_results[-1]['val_start']:match_results[-1]['val_end']]
        # match_results[-1]['val_str'] = self.post_process_txt(match_results[-1]['val_raw_str'])

        # logger.debug('match results:')
        # for m in match_results:
        #     logger.debug(m)

        match_results = self.process_exclude(match_results)

        result = self.struc_results(match_results, root_node)
        # s = json.dumps(results, indent=1, separators=(',', ':'), ensure_ascii=False)
        # print(s)
        # print(text)
        return result
#
