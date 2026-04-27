import urllib.request, re

def check_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        links = re.findall(r'href="(.*?\.zip)"', html)
        links += re.findall(r'href="(.*?\.tar\.gz)"', html)
        print(f"Links found on {url}:")
        for l in set(links):
            print(" -", l)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

check_url("https://www.openslr.org/64/")
check_url("https://www.openslr.org/103/")
