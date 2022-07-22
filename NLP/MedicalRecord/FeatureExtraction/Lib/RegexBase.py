"""
正则匹配基类
"""
class RegexBase:
    def __init__(self):
        # inner_neg
        self.inner_neg = '[^，；、。无未不]{,4}'
        self.inner_neg_x = '[^，；、。无未不]{,8}'
        self.inner = '[^，；、。]*?'
        self.regex_suspect = '((？)|[?]|(怀疑)|(待排)|(可能))'
