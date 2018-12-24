# coding: utf-8
from urllib.parse import quote
from Crypto.Cipher import AES
import base64

class WebProxyTool():
    def __init__(self):
        self.padding = '\0'
        self.bs = 16
        self.pad = lambda s: s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)
        self.key = b'\xde\xed\x10B\x1c\x1aUcr\xb8X\xc0%p\xde\xd1\xf1\x18\xaf\x90\xad5i\x1d\x18\x1a\x11\xda\x83\xec5\xd1'
        self.vector = b'\x92,eoB cw\xe7y\xd3XM h\x9c'

    def encrypt(self, decrypted_text):
        generator = AES.new(self.key, AES.MODE_CBC, self.vector)
        encrypted_text = generator.encrypt(self.pad(decrypted_text).encode("utf-8"))
        result = base64.b64encode(encrypted_text)
        return result

    def decrypt(self, encrypted_text):
        content = base64.b64decode(encrypted_text)
        generator = AES.new(self.key, AES.MODE_CBC, self.vector)
        recovery = generator.decrypt(content)
        return recovery.rstrip(self.padding.encode('utf-8'))

    def getCrawlUrl(self, url):
        format = 'http://www.bing.com/dict/proxy/proxy?k={k}'
        k = self.encrypt(url).decode("utf-8")
        return format.format(k=quote(k))


def W():
    url = 'http://stackoverflow.com/questions/3297030/whats-the-equivalent-of-cs-getbytes-in-python'
    tool = WebProxyTool()
    crawl_url = tool.getCrawlUrl(url)
    print(crawl_url)


if __name__ == "__main__":
    W()