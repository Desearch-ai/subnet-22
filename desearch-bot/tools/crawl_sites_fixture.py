"""Write tests/data/crawl_sites/fixture.json: the replay server's synthetic hosts and the crawl-sites parity runs."""

import base64, gzip, json
from pathlib import Path

H = {}
def html(body, head="", lang=None, **kw):
    l = f' lang="{lang}"' if lang else ""
    return {"status": 200, "type": "text/html", "body": f"<html{l}><head><title>t</title>{head}</head><body>{body}</body></html>", **kw}
def links(*paths, **kw):
    return html("".join(f'<a href="{p}">{p}</a>' for p in paths), **kw)
def robots(text, status=200):
    return {"status": status, "type": "text/plain", "body": text}
def redirect(to, status=302, **kw):
    return {"status": status, "headers": [["Location", to]], **kw}
def status(code, **kw):
    return {"status": code, **kw}
NOFOLLOW_META = '<meta name="robots" content="nofollow">'
NOINDEX_META = '<meta name="robots" content="noindex">'

H["plain.test"] = {
  "/robots.txt": status(404),
  "/": html('<a href="/a">a</a><a href="/b">b</a><a href="/c">c</a><a href="/redir">r</a><a href="/redir2">r2</a>'
            '<a href="/gone">g</a><a href="/away">o</a><a href="/f.pdf">f</a><a href="/hdr">h</a><a href="/xhtml">x</a>'
            '<a href="/data.txt">t</a><a href="/empty">e</a><a href="/none">n</a><a href="https://other.test/x">ext</a>'
            '<a href="/a?utm_source=x">dup</a><a href="/q?b=2&amp;a=1&amp;gclid=z">q</a><a href="mailto:x@plain.test">m</a>'
            '<a href="/A%20b">sp</a><a href="/caf%C3%A9">cafe</a><a href="/xmldecl">xd</a><a href="/srv">srv</a>'),
  "/a": html('<p>a</p><a href="/hidden" rel="nofollow">h</a><a href="/a2">a2</a>'),
  "/a2": html("<p>a2</p>"),
  "/b": html('<a href="/b2">b2</a>', NOINDEX_META),
  "/b2": html("<p>b2</p>"),
  "/c": html('<a href="/d">d</a>', NOFOLLOW_META),
  "/d": html("<p>d</p>"),
  "/hidden": html("<p>hidden</p>"),
  "/redir": redirect("/a"),
  "/redir2": redirect("/e", 301),
  "/e": html("<p>e</p>"),
  "/away": redirect("https://elsewhere.test/landing"),
  "/hdr": html("<p>hdr</p>", headers=[["X-Robots-Tag", "noindex"]]),
  "/xhtml": {"status": 200, "type": "application/xhtml+xml", "body": '<html xmlns="http://www.w3.org/1999/xhtml"><body><a href="/x2">x2</a></body></html>'},
  "/x2": html("<p>x2</p>"),
  "/data.txt": {"status": 200, "type": "text/plain", "body": "plain text"},
  "/empty": {"status": 200, "type": "text/html", "body": ""},
  "/none": html('<a href="/n2">n2</a>', '<meta name="ROBOTS" content="None">'),
  "/n2": html("<p>n2</p>"),
  "/q?a=1&b=2": html("<p>q sorted</p>"),
  "/q?b=2&a=1": html("<p>q</p>"),
  "/A%20b": html("<p>space</p>"),
  "/caf%C3%A9": html("<p>cafe</p>"),
  "/xmldecl": {"status": 200, "type": "text/html", "body": '<?xml version="1.0" encoding="utf-8"?><html><body><a href="/xd2">x</a></body></html>'},
  "/srv": status(500),
}
H["elsewhere.test"] = {"/robots.txt": robots("User-agent: *\nAllow: /\n"), "/landing": html("<p>elsewhere</p>")}
H["robots403.test"] = {"/robots.txt": status(403), "/": links("/private/x", "/y"), "/private/x": html("<p>p</p>"), "/y": html("<p>y</p>")}
H["robots503.test"] = {"/robots.txt": status(503), "/": links("/a")}
H["robots401.test"] = {"/robots.txt": status(401), "/": links("/a"), "/a": html("<p>a</p>")}
H["robots429.test"] = {"/robots.txt": status(429, headers=[["Retry-After", "1"]]), "/": links("/a"), "/a": html("<p>a</p>")}
H["robots410.test"] = {"/robots.txt": status(410), "/": links("/a"), "/a": html("<p>a</p>")}
H["slow.test"] = {
  "/robots.txt": robots("User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n"),
  "/": links("/p1", "/p2", "/private/x", "/p3"), "/p1": html("<p>1</p>"), "/p2": html("<p>2</p>"), "/p3": html("<p>3</p>"),
}
H["named.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n\nUser-agent: DesearchBot\nDisallow: /no/\nDisallow: /*.php$\nAllow: /no/yes\n"),
  "/": links("/no/a", "/no/yes", "/page.php", "/page.php?x=1", "/ok"),
  "/no/yes": html("<p>yes</p>"), "/page.php?x=1": html("<p>php</p>"), "/ok": html("<p>ok</p>"),
}
H["ratelimit.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": links("/p1", "/p2", "/p3", "/p4"),
  "/p1": status(429, headers=[["Retry-After", "6"]]), "/p2": html("<p>2</p>"), "/p3": status(429), "/p4": html("<p>4</p>"),
}
H["midforbid.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": links("/a", "/f1", "/b", "/f2", "/f3", "/f4", "/c"),
  "/a": html("<p>a</p>"), "/b": html("<p>b</p>"), "/c": html("<p>c</p>"),
  "/f1": status(403), "/f2": status(403), "/f3": status(403), "/f4": status(403),
}
H["refuse401.test"] = {"/robots.txt": robots("User-agent: *\nAllow: /\n"), "/": links("/a", "/b", "/c"), "/a": html("<p>a</p>"), "/b": status(401), "/c": html("<p>c</p>")}
H["errors.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": links("/ok", "/e1", "/e2", "/ok2", "/e3", "/e4", "/e5", "/e6", "/e7", "/after"),
  "/ok": html("<p>ok</p>"), "/ok2": html("<p>ok2</p>"), "/after": html("<p>after</p>"),
  "/e1": status(500), "/e2": status(502), "/e3": status(503), "/e4": status(500), "/e5": status(504), "/e6": status(500), "/e7": {"close": True},
}
H["bare.test"] = {"/robots.txt": {"close": True}, "/": {"close": True}}
H["www.bare.test"] = {"/robots.txt": robots("User-agent: *\nAllow: /\n"), "/": links("/w1"), "/w1": html("<p>w1</p>")}
H["homeredirect.test"] = {"/robots.txt": status(404), "/": redirect("https://www.homeredirect.test/", 301)}
H["www.homeredirect.test"] = {"/robots.txt": status(404), "/": links("/h1"), "/h1": html("<p>h1</p>")}
H["cookie.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": links("/art", "/nocookie", "/blk", "/loop"),
  "/art": {"need_cookie": "session", "else": redirect("https://idp.cookie.test/authorize?back=/art", 303), **html("<p>art</p>")},
  "/nocookie": {"need_cookie": "other", "else": redirect("https://idp.cookie.test/plain?back=/nocookie", 303), **html("<p>nocookie</p>")},
  "/blk": {"need_cookie": "third", "else": redirect("https://idp.cookie.test/blocked?back=/blk", 303), **html("<p>blk</p>")},
  "/loop": {"need_cookie": "fourth", "else": redirect("https://idp.cookie.test/elsewhere", 302), **html("<p>loop</p>")},
}
H["idp.cookie.test"] = {
  "/robots.txt": robots("User-agent: *\nDisallow: /blocked\n"),
  "/authorize?back=/art": redirect("/transit?back=/art", headers=[["Location", "/transit?back=/art"], ["Set-Cookie", "session=1; Domain=cookie.test; Path=/"]]),
  "/transit?back=/art": redirect("https://cookie.test/art"),
  "/plain?back=/nocookie": redirect("https://cookie.test/nocookie"),
  "/elsewhere": redirect("https://elsewhere.test/landing", headers=[["Location", "https://elsewhere.test/landing"], ["Set-Cookie", "fourth=1; Domain=cookie.test; Path=/"]]),
}
fr = lambda b: html(b, lang="fr-CA")
en = lambda b: html(b, lang="en")
H["lang.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": en('<a href="/fr/a">fr</a><a href="/en/a">en</a><a href="/x">x</a>'),
  "/fr/a": fr('<a href="/fr/b">b</a><a href="/fr/c">c</a><a href="/y">y</a>'),
  "/en/a": en('<a href="/fr/d">d</a><a href="/en/b">b</a>'),
  "/en/b": en("<p>b</p>"), "/x": fr("<p>x</p>"),
}
H["frhome.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\n"),
  "/": fr('<a href="/en/a">en</a><a href="/de/a">de</a>'),
  "/en/a": en('<a href="/en/b">b</a>'), "/en/b": html("<p>b</p>", lang="EN-us"), "/de/a": html("<p>de</p>", lang="de"),
}
H["variants.test"] = {"/robots.txt": robots("User-agent: *\nAllow: /\n"), "/": links(*[f"/list?related={i}" for i in range(12)], "/list?utm_source=a")}
for i in range(12):
    H["variants.test"][f"/list?related={i}"] = html(f"<p>{i}</p>")
H["capped.test"] = {"/robots.txt": status(404), "/": links(*[f"/c{i}" for i in range(8)])}
for i in range(8):
    H["capped.test"][f"/c{i}"] = html(f"<p>{i}</p>")
H["near.test"] = {"/robots.txt": status(404), "/": links("/blog/1", "/docs/1", "/blog/2", "/docs/2"), "/docs/1": html("<p>d1</p>"), "/docs/2": html("<p>d2</p>"), "/blog/1": html("<p>b1</p>"), "/blog/2": html("<p>b2</p>")}
H["deep.test"] = {"/robots.txt": status(404), "/": links("/d1")}
for i in range(1, 10):
    H["deep.test"][f"/d{i}"] = links(f"/d{i+1}")
smindex = '<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://sitemap.test/sm1.xml</loc></sitemap><sitemap><loc>https://sitemap.test/sm2.xml.gz</loc></sitemap><sitemap><loc>https://sitemap.test/broken.xml</loc></sitemap><sitemap><loc>https://sitemap.test/missing.xml</loc></sitemap></sitemapindex>'
sm1 = '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://sitemap.test/s1</loc></url><url><loc> https://sitemap.test/s2?b=1&amp;utm_medium=x </loc></url><url><loc>https://other.test/s3</loc></url><url><loc>https://sitemap.test/s4.pdf</loc></url></urlset>'
sm2 = '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://sitemap.test/s5</loc></url></urlset>'
broken = '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://sitemap.test/s6?a&b</loc></url></urlset>'
H["sitemap.test"] = {
  "/robots.txt": robots("User-agent: *\nAllow: /\nSitemap: /smindex.xml\n"),
  "/smindex.xml": {"status": 200, "type": "application/xml", "body": smindex},
  "/sm1.xml": {"status": 200, "type": "application/xml", "body": sm1},
  "/sm2.xml.gz": {"status": 200, "type": "application/x-gzip", "b64": base64.b64encode(gzip.compress(sm2.encode(), mtime=0)).decode()},
  "/broken.xml": {"status": 200, "type": "application/xml", "body": broken},
  "/": links("/s1"), "/s1": html("<p>s1</p>"), "/s2?b=1": html("<p>s2</p>"), "/s5": html("<p>s5</p>"), "/s6?a=&b=": html("<p>s6</p>"),
}
H["skip.test"] = {"/robots.txt": status(404), "/": links("/held", "/fresh"), "/held": html("<p>held</p>"), "/fresh": html("<p>fresh</p>")}
H["charset.test"] = {
  "/robots.txt": status(404), "/": links("/latin", "/cp1252", "/bad"),
  "/latin": {"status": 200, "type": "text/html; charset=ISO-8859-1", "b64": base64.b64encode("<html><body><p>café \x80</p></body></html>".encode("latin-1")).decode()},
  "/cp1252": {"status": 200, "type": "text/html; charset=windows-1252", "b64": base64.b64encode(b"<html><body><p>\x80 \x81 \x93q\x94</p></body></html>").decode()},
  "/bad": {"status": 200, "type": "text/html; charset=utf-8", "b64": base64.b64encode(b"<html><body><p>ok \xff\xfe \xe2\x82 end</p></body></html>").decode()},
}
H["badbase.test"] = {"/robots.txt": status(404), "/": html('<base href="http://[broken/"><a href="/x">x</a>')}
H["slots.test"] = {"/robots.txt": status(404), "/": links(*[f"/s{i}" for i in range(6)])}
for i in range(6):
    H["slots.test"][f"/s{i}"] = html(f"<p>{i}</p>")
H["seeded.test"] = {"/robots.txt": robots("User-agent: *\nDisallow: /blocked\n"), "/": links("/a"), "/a": html("<p>a</p>"), "/start?x=1": html('<a href="/from-seed">s</a>'), "/from-seed": html("<p>s</p>")}
H["prefixed.test"] = {"/robots.txt": status(404), "/": links("/docs/a", "/blog/b")}

del H["prefixed.test"]
B = {}
B["setwhole.test"] = {"/robots.txt": status(404), "/": links("/lit/a", "/blog/x"), "/lit/": links("/lit/b"), "/lit/a": html("<p>a</p>"), "/lit/b": links("/lit/c"), "/lit/c": links("/lit/d"), "/lit/d": html("<p>d</p>"), "/blog/x": html("<p>x</p>"), "/lit/seeded": html("<p>seeded</p>")}
B["setland.test"] = {"/robots.txt": status(404), "/": links("/about", "/a/b/c", "/about?x=1", "/fr-fr/about"), "/about": html("<p>about</p>"), "/extra-seed": html("<p>extra</p>")}
B["setnews.test"] = {"/robots.txt": status(404), "/": links("/2026/11/story", "/section"), "/2026/11/story": html("<p>s</p>"), "/section": html("<p>sec</p>")}
B["unlisted.test"] = {"/robots.txt": status(404), "/": links("/a"), "/a": html("<p>a</p>")}
B["setout.test"] = {"/robots.txt": status(404), "/": links("/a"), "/a": html("<p>a</p>")}
C = {"prefixed.test": {"/robots.txt": status(404), "/": links("/docs/a"), "/docs/": links("/docs/a", "/blog/b", "/docs/b"), "/docs/a": html("<p>a</p>"), "/docs/b": html("<p>b</p>"), "/blog/b": html("<p>b</p>")}}
for k, v in {**B, **C}.items():
    H[k] = v
first = [h for h in H if h not in B and h not in C and h not in ("elsewhere.test", "idp.cookie.test", "www.bare.test", "www.homeredirect.test")]
lines = []
for h in first:
    lines.append({"capped.test": "capped.test\t3", "slots.test": "slots.test\t\t2", "near.test": "near.test\t2\t\t/docs/"}.get(h, h))
runs = {
 "a": {"hosts_file": lines, "args": ["--hosts-at-once", "40", "--seed", "https://seeded.test/start?x=1"], "skip": ["https://skip.test/held", "https://plain.test/e extra words"], "found": True},
 "b": {"hosts_file": list(B), "args": ["--no-sitemaps", "--max-depth", "2", "--hosts-at-once", "2"], "set": {"sites": [
      {"host": "setwhole.test", "rule": "whole", "paths": ["/lit/"]}, {"host": "setland.test", "rule": "landing", "depth": 1},
      {"host": "setnews.test", "rule": "news", "since": "2026-10-01"}, {"host": "setout.test", "rule": "out"}]},
      "seeds": ["https://setland.test/extra-seed", "https://setwhole.test/lit/seeded", "https://elsewhere.test/nothing"]},
 "c": {"hosts_file": list(C), "args": ["--prefix", "/docs/", "--seed", "https://prefixed.test/docs/", "--max-pages", "2"]},
}
out = Path(__file__).resolve().parent.parent / "tests/data/crawl_sites/fixture.json"
json.dump({"hosts": H, "runs": runs}, open(out, "w"), indent=1, sort_keys=True)
