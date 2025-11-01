import requests

def getNotes():
    # 设置 API 密钥和基础 URL
    api_key = '5310f9a7f0a45fa211d6ab889557b7be988984ca87e0171a5d7614fc7f0e90d7'
    base_url = 'http://127.0.0.1:27123'
    # 读取笔记
    response = requests.get(f'{base_url}/vault/all/ai/', headers={'Authorization': f'Bearer {api_key}'})
    notes = response.json()
    # 打印笔记列表
    for file in notes['files']:
        print(file)


def main():
    print("Hello from uvdemo!")
    # getNotes()

if __name__ == "__main__":
    main()


