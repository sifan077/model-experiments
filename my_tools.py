import os
import warnings


# 去除使用警告
def remove_warning():
    warnings.filterwarnings("ignore", category=UserWarning)
    print('---Warning removed.---')


# 使用代理进行请求
def request_proxy():
    # 设置代理。这里 7890 既可以是 http 代理的端口，也可以是 socks5 代理的端口
    proxy = 'http://127.0.0.1:7890'
    os.environ['http_proxy'] = proxy
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy
    print('---Proxy set to:', proxy, '---')


# 文件windows路径转linux路径
def windows_path_to_linux(path):
    return path.replace('\\', '/')
